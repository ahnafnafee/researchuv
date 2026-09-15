//! Run the cube fixture and export its packed UV triangles as an SVG.

use researchuv_unwrap::{meshgen, run, PipelineOptions};
use std::fmt::Write as _;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("cube-uv.svg"));
    let (positions, triangles) = meshgen::cube(6);
    let result = run(positions, triangles, &PipelineOptions::default())
        .expect("the cube fixture must run cleanly");
    if result.placed.iter().any(Option::is_none) {
        return Err("the cube fixture could not be fully packed".into());
    }

    let mut svg = String::from(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="620" viewBox="0 0 1100 620" role="img" aria-labelledby="title desc">
<title id="title">A cube unfolded into six UV islands</title>
<desc id="desc">Packed triangle coordinates from the ResearchUV cube example.</desc>
<rect width="1100" height="620" rx="20" fill="#f8f3e9"/>
<g font-family="Arial, Helvetica, sans-serif" fill="#263e35">
<text x="52" y="84" font-size="12" font-weight="700" letter-spacing="3">THE CUBE FIXTURE</text>
<text x="50" y="150" font-size="43" font-weight="700">Six faces.</text>
<text x="50" y="202" font-size="43" font-weight="700">One atlas.</text>
<text x="52" y="249" font-size="16" fill="#53655b">Weld, chart, unfold, and pack.</text>
<path d="M52 282H102" stroke="#d36b42" stroke-width="3"/>
"##,
    );
    let stats = [
        format!("{} welded vertices", result.mesh.positions.len()),
        format!("{} source triangles", result.mesh.faces.len()),
        format!("{} packed islands", result.multi.islands.len()),
        format!("{:.6} packing scale", result.scale),
    ];
    for (index, line) in stats.iter().enumerate() {
        writeln!(
            svg,
            r#"<text x="52" y="{}" font-size="17">{line}</text>"#,
            329 + index * 34
        )?;
    }
    svg.push_str(
        r##"<text x="52" y="550" font-size="12" fill="#53655b">Generated from the included Rust example.</text>
</g>
<rect x="470" y="60" width="520" height="520" fill="#fffdf8" stroke="#bbc4b8"/>
<text x="470" y="42" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="#53655b" letter-spacing="2">UV SPACE / [0, 1]²</text>
"##,
    );
    let colors = [
        "#a7c5b0", "#e8ac87", "#ebcb83", "#98b7c8", "#c5b1a0", "#bfc795",
    ];
    for (index, island) in result.multi.islands.iter().enumerate() {
        writeln!(
            svg,
            r##"<g fill="{}" stroke="#304d43" stroke-width="0.65" stroke-linejoin="round">"##,
            colors[index % colors.len()]
        )?;
        for triangle in &island.tris {
            let points = triangle
                .iter()
                .map(|&vertex| {
                    let uv = island.uv[vertex as usize];
                    format!("{:.3},{:.3}", 470.0 + 520.0 * uv.u, 580.0 - 520.0 * uv.v)
                })
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(svg, r#"<polygon points="{points}"/>"#)?;
        }
        svg.push_str("</g>\n");
    }
    svg.push_str("</svg>\n");
    std::fs::write(&output, svg)?;
    println!(
        "{} charts, {} triangles, {} packed islands -> {}",
        result.charts.len(),
        result.mesh.faces.len(),
        result.multi.islands.len(),
        output.display()
    );
    Ok(())
}
