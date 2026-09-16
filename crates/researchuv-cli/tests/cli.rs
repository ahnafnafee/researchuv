//! End-to-end CLI tests: drive the compiled `researchuv` binary over
//! fixtures and files, checking exit codes and outputs.

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_researchuv"))
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join("researchuv-cli-test");
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

#[test]
fn help_and_version_exit_zero() {
    for args in [&["help"][..], &["--version"][..]] {
        let out = bin().args(args).output().unwrap();
        assert!(out.status.success(), "{args:?} failed");
        assert!(!out.stdout.is_empty());
    }
}

#[test]
fn unknown_command_is_a_usage_error() {
    let out = bin().arg("frobnicate").output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn info_reports_cube_topology() {
    let out = bin().args(["info", "--fixture", "cube", "6"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("218"), "welded cube vertex count in:\n{text}");
    assert!(text.contains("432"), "welded cube face count in:\n{text}");
    assert!(text.contains("charts         6"));
    assert!(text.contains("validation     clean"));
}

#[test]
fn unwrap_cube_writes_obj_svg_and_stl() {
    let obj = tmp("cli-cube.obj");
    let svg = tmp("cli-cube.svg");
    let stl = tmp("cli-cube.stl");
    let _ = std::fs::remove_file(&obj);
    let _ = std::fs::remove_file(&svg);
    let _ = std::fs::remove_file(&stl);
    let out = bin()
        .args([
            "unwrap",
            "--fixture",
            "cube",
            "6",
            "-o",
        ])
        .arg(&obj)
        .arg("--svg")
        .arg(&svg)
        .arg("--stl")
        .arg(&stl)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}\nstdout: {}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    let text = std::fs::read_to_string(&obj).unwrap();
    assert!(text.contains("vt "), "OBJ carries UVs");
    let (p, f) = researchuv_core::io::parse_obj(&text).unwrap();
    assert_eq!(p.len(), 218);
    assert_eq!(f.len(), 432);
    let svg_text = std::fs::read_to_string(&svg).unwrap();
    assert!(svg_text.contains("<svg"));
    // Six islands → six colored groups.
    assert_eq!(svg_text.matches("<g ").count(), 6);
    // STL export: binary framing (84 + 50 × facets).
    let stl_bytes = std::fs::read(&stl).unwrap();
    assert_eq!(stl_bytes.len(), 84 + 50 * 432);
}

#[test]
fn unwrap_over_an_obj_file() {
    // Round-trip: fixture → OBJ file → unwrap the file.
    let src = tmp("cli-torus-src.obj");
    let (p, f) = researchuv_unwrap::meshgen::torus_annulus(2.0, 0.7, 32, 20);
    let mesh = researchuv_unwrap::weld::weld(p, f, 1e-12);
    researchuv_core::io::write_obj_file(&src, &mesh, None).unwrap();
    let dst = tmp("cli-torus-out.obj");
    let out = bin()
        .args(["unwrap", "--quiet", "-o"])
        .arg(&dst)
        .arg(&src)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("charts         1"), "{text}");
}

#[test]
fn malformed_mesh_fails_with_an_error() {
    let bad = tmp("cli-bad.obj");
    std::fs::write(&bad, "v 0 0 0\nv 1 0 0\nf 1 2 9\n").unwrap();
    let out = bin().arg("info").arg(&bad).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    // The OBJ reader rejects out-of-range indices at parse time.
    assert!(String::from_utf8_lossy(&out.stderr).contains("out of range"));
}

#[test]
fn shelf_packer_flag_is_honored() {
    let dst = tmp("cli-shelf.obj");
    let out = bin()
        .args(["unwrap", "--quiet", "--packer", "shelf", "--fixture", "cube", "4", "-o"])
        .arg(&dst)
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("packer         shelf"), "{text}");
}

#[test]
fn raster_flag_produces_a_valid_atlas() {
    // GPU-dependent: on machines without a device the pipeline falls back
    // to the exact planner and the run still succeeds.
    let obj = tmp("cli-raster.obj");
    let out = bin()
        .args([
            "unwrap",
            "--fixture",
            "cube",
            "4",
            "--packer",
            "islands",
            "--raster",
            "256",
            "--quiet",
            "-o",
        ])
        .arg(&obj)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}
stdout: {}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    let text = std::fs::read_to_string(&obj).unwrap();
    assert!(text.contains("vt "), "OBJ carries UVs");
    let (p, f) = researchuv_core::io::parse_obj(&text).unwrap();
    assert_eq!(p.len(), 98);
    assert_eq!(f.len(), 192);
}

#[test]
fn missing_input_is_a_usage_error() {
    let out = bin().arg("unwrap").output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}
