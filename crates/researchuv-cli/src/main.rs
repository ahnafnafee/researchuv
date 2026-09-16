//! The `researchuv` command-line tool — mesh inspection (`info`) and the
//! full unwrap pipeline (`unwrap`) over OBJ/STL files or built-in fixtures,
//! exporting OBJ atlases and SVG previews.

use researchuv_math::{Vec2, Vec3};
use researchuv_unwrap::meshgen;
use researchuv_unwrap::pipeline::{run, Packer, PipelineOptions};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "\
ResearchUV — explore the path from triangle meshes to texture space.

USAGE:
    researchuv <COMMAND> [OPTIONS]

COMMANDS:
    info <INPUT>                 Import a mesh (.obj/.stl), weld it, and print
                                 topology + validation findings.
    unwrap <INPUT>               Run the full unwrap pipeline and export the
                                 packed atlas as an OBJ (+ optional SVG).
    fixtures                     List the built-in fixture meshes.
    help                         Show this help.

    Instead of a file, both commands accept a fixture:
        researchuv unwrap --fixture cube 6 -o cube-uv.obj

OPTIONS (unwrap):
    -o, --output <PATH>          Output OBJ (default: <input>-uv.obj).
    --svg <PATH>                 Also render the atlas to an SVG.
    --stl <PATH>                 Also export the welded mesh as binary STL.
    --angle <DEG>                Sharp-edge cut angle (default 30).
    --weld <TOL>                 Weld tolerance (default 1e-12).
    --iters <N>                  Unfold driver iterations (default 100).
    --packer <shelf|islands>     Final packer (default: islands).
    --seam-cut|--no-seam-cut     Cut closed charts open (default: off).
    --recut|--no-recut           Distortion-driven re-cut of folding charts
                                 (default: off).
    --threads <N>                Unfold worker threads, 0 = all cores.
    --gpu|--no-gpu               CUDA conjugate-gradient unfold solve with
                                 CPU fallback (default: off).
    --padding <F>                Shelf packer gutter (default 0.01).
    --margin <F>                 Island packer margin (default 0.003).
    --rotation-step <DEG>        Island rotation step (default 90).
    --no-rotate                  Disable island rotation.
    --heuristic <SECONDS>        Enable the heuristic search with a budget.
    --raster <RES>               Rasterizer placement resolution (GPU; 256,
                                 512, 1024 — 0 disables).
    --tiles <COLS>               Tile targets on the raster path: a grid of
                                 COLS unit-tile columns, rows added as needed.
    --quiet                      Skip the per-chart table.

OPTIONS (info):
    --angle <DEG>                Chart threshold for the chart count.
    --weld <TOL>                 Weld tolerance.

Exit codes: 0 success, 1 runtime error, 2 usage error.
";

fn researchuv_pack_params_tile_dynamic() -> researchuv_pack::params::TileTargetMode {
    researchuv_pack::params::TileTargetMode::DynamicTiles
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("help") | Some("--help") | Some("-h") => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("--version") | Some("-V") => {
            println!("researchuv {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("fixtures") => fixtures(),
        Some("info") => cmd_info(&args[1..]),
        Some("unwrap") => cmd_unwrap(&args[1..]),
        Some(other) => {
            eprintln!("error: unknown command {other:?}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn fixtures() -> ExitCode {
    println!("cube     <n>   subdivided cube (6·(n+1)² raw verts)");
    println!("sphere   <n>   UV sphere n×n/2 (duplicated poles/seam, welded)");
    println!("torus    <n>   closed torus (major 8n, minor 5n)");
    println!("annulus  <n>   torus with the inner band removed (2 border loops)");
    println!("grid     <n>   n×n flat grid plane");
    println!("cylinder <n>   open cylinder (2 border loops)");
    ExitCode::SUCCESS
}

struct CommonOpts {
    angle: f64,
    weld_tol: f64,
}

/// Split `args` into (positional, flags). Errors on a flag without its value.
fn parse_common(args: &[String], i: &mut usize, common: &mut CommonOpts) -> Result<(), String> {
    let a = &args[*i];
    let need_value = |name: &str, i: &mut usize| -> Result<String, String> {
        *i += 1;
        args.get(*i)
            .cloned()
            .ok_or_else(|| format!("{name} requires a value"))
    };
    match a.as_str() {
        "--angle" => {
            let v = need_value("--angle", i)?;
            common.angle = v.parse().map_err(|_| format!("bad --angle {v:?}"))?;
        }
        "--weld" => {
            let v = need_value("--weld", i)?;
            common.weld_tol = v.parse().map_err(|_| format!("bad --weld {v:?}"))?;
        }
        _ => return Err(format!("unknown option {a:?}")),
    }
    Ok(())
}

fn load_input(input: &str) -> Result<(Vec<Vec3>, Vec<[u32; 3]>), String> {
    let path = Path::new(input);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "obj" => researchuv_core::io::read_obj(path).map_err(|e| e.to_string()),
        "stl" => researchuv_core::io::read_stl(path).map_err(|e| e.to_string()),
        other => Err(format!(
            "unsupported input {input:?} (.{other}; expected .obj or .stl, or use --fixture)"
        )),
    }
}

fn build_fixture(name: &str, n: usize) -> Result<(Vec<Vec3>, Vec<[u32; 3]>), String> {
    meshgen::fixture(name, n).ok_or_else(|| {
        format!("unknown fixture {name:?} (expected cube, sphere, torus, annulus, grid, cylinder)")
    })
}

/// Resolve the mesh source: an explicit `--fixture NAME [N]`, or a positional
/// file path. Returns `(positions, faces, label)`.
fn resolve_source(args: &[String]) -> Result<Option<(Vec<Vec3>, Vec<[u32; 3]>, String)>, String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--fixture" {
            let name = args
                .get(i + 1)
                .ok_or_else(|| "--fixture requires a name".to_string())?
                .clone();
            let n: usize = match args.get(i + 2) {
                Some(v) => v
                    .parse()
                    .map_err(|_| format!("bad fixture size {v:?}"))?,
                None => 6,
            };
            let (p, f) = build_fixture(&name, n)?;
            return Ok(Some((p, f, format!("fixture:{name}({n})"))));
        }
        i += 1;
    }
    // Positional: the first non-flag argument that is not a flag value.
    let mut positional: Vec<&String> = Vec::new();
    let mut skip_next = false;
    for a in args.iter() {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a.starts_with('-') {
            // Options that consume a value (shared with the unwrap parser).
            if matches!(
                a.as_str(),
                "-o" | "--output" | "--svg" | "--stl" | "--angle" | "--weld" | "--iters" | "--packer"
                    | "--padding" | "--margin" | "--rotation-step" | "--heuristic" | "--threads"
                    | "--raster" | "--tiles"
            ) {
                skip_next = true;
            }
            continue;
        }
        positional.push(a);
    }
    match positional.first() {
        Some(path) => {
            let (p, f) = load_input(path)?;
            Ok(Some((p, f, path.to_string())))
        }
        None => Ok(None),
    }
}

fn cmd_info(args: &[String]) -> ExitCode {
    let mut common = CommonOpts { angle: 30.0, weld_tol: 1e-12 };
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--fixture" {
            // Consumed by resolve_source (name + optional integer size).
            i += 1;
            if args.get(i + 1).map(|v| v.parse::<usize>().is_ok()).unwrap_or(false) {
                i += 1;
            }
            i += 1;
            continue;
        }
        if !args[i].starts_with('-') {
            i += 1;
            continue; // positional input (resolved by resolve_source)
        }
        if let Err(e) = parse_common(args, &mut i, &mut common) {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
        i += 1;
    }
    let source = match resolve_source(args) {
        Ok(Some(s)) => s,
        Ok(None) => {
            eprintln!("error: info needs an input file or --fixture\n\n{USAGE}");
            return ExitCode::from(2);
        }
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let (positions, faces, label) = source;
    let raw_report = researchuv_unwrap::validate_raw(&positions, &faces);
    if raw_report.has_errors() {
        for e in raw_report.errors() {
            eprintln!("error: [{}] {}", e.code, e.detail);
        }
        return ExitCode::from(1);
    }
    let mesh = researchuv_unwrap::weld::weld(positions, faces, common.weld_tol);
    let report = researchuv_unwrap::validate_mesh(&mesh);
    let (charts, _) = researchuv_unwrap::segment::segment(&mesh, common.angle);
    println!("input          {label}");
    println!("vertices       {} (welded)", mesh.positions.len());
    println!("faces          {} (welded)", mesh.faces.len());
    println!("charts         {} (at {}°)", charts.len(), common.angle);
    println!("closed charts  {}", charts.iter().filter(|c| c.is_borderless()).count());
    for f in report.warnings() {
        println!("warning: [{}] {}", f.code, f.detail);
    }
    if report.is_ok() {
        println!("validation     clean");
    }
    ExitCode::SUCCESS
}

fn cmd_unwrap(args: &[String]) -> ExitCode {
    let mut opts = PipelineOptions::default();
    // The CLI defaults to the island packer: it is the higher-quality engine.
    opts.packer = Packer::Islands;
    let mut output: Option<PathBuf> = None;
    let mut svg: Option<PathBuf> = None;
    let mut stl: Option<PathBuf> = None;
    let mut quiet = false;
    let mut heuristic: Option<f64> = None;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let value = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{} requires a value", args[*i - 1]))
        };
        let r = match a.as_str() {
            "-o" | "--output" => value(&mut i).map(|v| output = Some(PathBuf::from(v))),
            "--svg" => value(&mut i).map(|v| svg = Some(PathBuf::from(v))),
            "--stl" => value(&mut i).map(|v| stl = Some(PathBuf::from(v))),
            "--iters" => value(&mut i)
                .and_then(|v| v.parse::<usize>().map_err(|e| e.to_string()))
                .map(|v| opts.unfold.max_iter = v.max(1)),
            "--packer" => value(&mut i).and_then(|v| match v.as_str() {
                "shelf" => Ok(opts.packer = Packer::Shelf),
                "islands" => Ok(opts.packer = Packer::Islands),
                other => Err(format!("unknown packer {other:?}")),
            }),
            "--seam-cut" => Ok(opts.seam_cut.enable = true),
            "--no-seam-cut" => Ok(opts.seam_cut.enable = false),
            "--recut" => Ok(opts.recut.enable = true),
            "--no-recut" => Ok(opts.recut.enable = false),
            "--threads" => value(&mut i)
                .and_then(|v| v.parse::<u32>().map_err(|e| e.to_string()))
                .map(|v| opts.threads = v.min(1024)),
            "--gpu" => Ok(opts.unfold.solver = researchuv_unwrap::SolverBackend::Gpu),
            "--no-gpu" => Ok(opts.unfold.solver = researchuv_unwrap::SolverBackend::Cpu),
            "--padding" => value(&mut i)
                .and_then(|v| v.parse::<f64>().map_err(|e| e.to_string()))
                .map(|v| opts.padding = v),
            "--margin" => value(&mut i)
                .and_then(|v| v.parse::<f64>().map_err(|e| e.to_string()))
                .map(|v| opts.island_pack.margin = v),
            "--rotation-step" => value(&mut i)
                .and_then(|v| v.parse::<u32>().map_err(|e| e.to_string()))
                .map(|v| {
                    opts.island_pack.rotation_step = v.clamp(1, 180);
                    opts.island_pack.island_rot_step = v.clamp(1, 180);
                }),
            "--no-rotate" => Ok(opts.island_pack.rotation_enable = false),
            "--heuristic" => value(&mut i)
                .and_then(|v| v.parse::<f64>().map_err(|e| e.to_string()))
                .map(|v| heuristic = Some(v.max(0.0))),
            "--raster" => value(&mut i)
                .and_then(|v| v.parse::<u32>().map_err(|e| e.to_string()))
                .map(|v| {
                    opts.island_pack.raster_resolution =
                        if v == 0 { 0 } else { v.clamp(64, 4096) / 32 * 32 }
                }),
            "--tiles" => value(&mut i)
                .and_then(|v| v.parse::<u32>().map_err(|e| e.to_string()))
                .map(|v| {
                    opts.island_pack.tiles_in_row = v.clamp(1, 100);
                    opts.island_pack.tile_target =
                        researchuv_pack_params_tile_dynamic();
                }),
            "--quiet" => Ok(quiet = true),
            "--fixture" => {
                // Consumed by resolve_source (name + optional integer size).
                i += 1;
                if args.get(i + 1).map(|v| v.parse::<usize>().is_ok()).unwrap_or(false) {
                    i += 1;
                }
                Ok(())
            }
            "--angle" | "--weld" => value(&mut i).and_then(|v| {
                v.parse::<f64>()
                    .map_err(|_| format!("bad {a} {v:?}"))
                    .map(|parsed| {
                        if a == "--angle" {
                            opts.angle_min_deg = parsed;
                        } else {
                            opts.weld_tol = parsed;
                        }
                    })
            }),
            other if !other.starts_with('-') => Ok(()), // positional input (resolve_source)
            other => Err(format!("unknown option {other:?}")),
        };
        if let Err(e) = r {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
        i += 1;
    }
    if let Some(secs) = heuristic {
        opts.island_pack.heuristic_enable = true;
        opts.island_pack.heuristic_search_time = secs;
        if secs == 0.0 {
            // A zero budget means "continuous" — bound it for a CLI run.
            opts.island_pack.heuristic_max_wait_time = 1.0;
        }
    }
    let source = match resolve_source(args) {
        Ok(Some(s)) => s,
        Ok(None) => {
            eprintln!("error: unwrap needs an input file or --fixture\n\n{USAGE}");
            return ExitCode::from(2);
        }
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let (positions, faces, label) = source;
    // Fixture labels carry "fixture:name(n)" — sanitize path-hostile
    // characters (':' is a drive separator on Windows).
    let stem: String = label
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '-' })
        .collect();
    let default_out = PathBuf::from(format!("{stem}-uv.obj"));
    let output = output.unwrap_or(default_out);

    let started = std::time::Instant::now();
    let result = match run(positions, faces, &opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let elapsed = started.elapsed().as_secs_f64();
    let placed = result.placed.iter().filter(|p| p.is_some()).count();
    if !quiet {
        println!(
            "{:<6} {:>7} {:>6} {:>10} {:>10} {:>9} {:>6}",
            "chart", "verts", "tris", "conf-mean", "conf-max", "area-r", "flips"
        );
        for (i, cr) in result.charts.iter().enumerate() {
            println!(
                "{:<6} {:>7} {:>6} {:>10.3} {:>10.1} {:>9.3} {:>6}",
                i,
                cr.chart.vertex_ids.len(),
                cr.chart.tris.len(),
                cr.metrics.conformal_mean,
                cr.metrics.conformal_max,
                cr.metrics.area_ratio_mean,
                cr.metrics.flips
            );
        }
    }
    // Per-face UV triplets for the OBJ export.
    let mut face_uvs: Vec<[Vec2; 3]> = Vec::with_capacity(result.mesh.faces.len());
    'faces: for fi in 0..result.mesh.faces.len() {
        for (ci, cr) in result.charts.iter().enumerate() {
            if cr.chart.face_ids.contains(&fi) {
                let ch = &cr.chart;
                let island = &result.multi.islands[ci];
                let mut tri = [Vec2::new(0.0, 0.0); 3];
                for k in 0..3 {
                    let v = result.mesh.faces[fi][k];
                    match ch.local_of(v) {
                        Some(li) => tri[k] = island.uv[li],
                        None => continue 'faces,
                    }
                }
                face_uvs.push(tri);
                continue 'faces;
            }
        }
    }
    let face_uvs_opt = if face_uvs.len() == result.mesh.faces.len() {
        Some(face_uvs.as_slice())
    } else {
        None
    };
    if let Err(e) = researchuv_core::io::write_obj_file(&output, &result.mesh, face_uvs_opt) {
        eprintln!("error: cannot write {}: {e}", output.display());
        return ExitCode::from(1);
    }
    if let Some(svg_path) = &svg {
        let island_tris = researchuv_core::io::atlas_tris(&result.multi.islands);
        if let Err(e) = researchuv_core::io::write_atlas_svg_file(svg_path, &island_tris, 512.0) {
            eprintln!("error: cannot write {}: {e}", svg_path.display());
            return ExitCode::from(1);
        }
        println!("svg            {}", svg_path.display());
    }
    if let Some(stl_path) = &stl {
        if let Err(e) = researchuv_core::io::write_stl_file(stl_path, &result.mesh) {
            eprintln!("error: cannot write {}: {e}", stl_path.display());
            return ExitCode::from(1);
        }
        println!("stl            {}", stl_path.display());
    }
    println!("source         {label}");
    println!("solver         {}", match opts.unfold.solver {
        researchuv_unwrap::SolverBackend::Cpu => "cpu",
        researchuv_unwrap::SolverBackend::Gpu => "gpu (cuda, cpu fallback)",
    });
    println!("packer         {}", match opts.packer {
        Packer::Shelf => "shelf",
        Packer::Islands => "islands",
    });
    println!("charts         {} ({} placed)", result.charts.len(), placed);
    println!("scale          {:.6}", result.scale);
    println!("time           {elapsed:.2}s");
    println!("output         {}", output.display());
    let mut failed = false;
    for f in result.mesh_report.findings.iter().chain(result.atlas_report.findings.iter()) {
        let kind = if f.severity == researchuv_unwrap::validate::Severity::Error {
            failed = true;
            "error"
        } else {
            "warning"
        };
        eprintln!("{kind}: [{}] {}", f.code, f.detail);
    }
    if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
