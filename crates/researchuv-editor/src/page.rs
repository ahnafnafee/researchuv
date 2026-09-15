//! The editor page — a single-file HTML/JS canvas front end served at `/`.

/// The index document (no external assets, no frameworks).
pub const INDEX: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ResearchUV — atlas editor</title>
<style>
  :root { color-scheme: light; }
  body { font-family: system-ui, sans-serif; margin: 0; background: #f8f3e9; color: #263e35; }
  header { padding: 14px 22px; background: #304d43; color: #f8f3e9; display: flex; gap: 18px; align-items: baseline; }
  header h1 { font-size: 17px; margin: 0; letter-spacing: 2px; }
  header span { font-size: 12px; opacity: .8; }
  main { display: flex; gap: 18px; padding: 18px 22px; flex-wrap: wrap; }
  #panel { background: #fffdf8; border: 1px solid #bbc4b8; border-radius: 10px; padding: 14px 16px; min-width: 250px; }
  #panel h2 { font-size: 13px; margin: 4px 0 10px; letter-spacing: 1px; text-transform: uppercase; color: #53655b; }
  .row { display: flex; justify-content: space-between; gap: 10px; margin: 7px 0; font-size: 13px; align-items: center; }
  .row label { flex: 1; }
  .row input, .row select { width: 120px; }
  button { width: 100%; margin-top: 10px; padding: 9px; background: #d36b42; color: #fffdf8; border: 0;
           border-radius: 7px; font-size: 14px; cursor: pointer; }
  button:disabled { background: #a08b7e; }
  #stage { position: relative; }
  canvas { background: #fffdf8; border: 1px solid #bbc4b8; border-radius: 10px; display: block; }
  #table { margin-top: 10px; font-size: 12px; border-collapse: collapse; max-height: 200px; overflow: auto; display: block; }
  #table td, #table th { border-bottom: 1px solid #e4e0d5; padding: 2px 8px; text-align: right; }
  #table th:first-child, #table td:first-child { text-align: left; }
  #status { font-size: 12px; color: #53655b; margin-top: 8px; min-height: 16px; }
  #err { color: #b3402a; }
</style>
</head>
<body>
<header>
  <h1>RESEARCHUV</h1>
  <span>atlas editor — live re-unwrap over the engine API</span>
</header>
<main>
  <div id="panel">
    <h2>Mesh</h2>
    <div class="row"><label>fixture</label>
      <select id="fixture">
        <option>cube</option><option>sphere</option><option>torus</option>
        <option>annulus</option><option>grid</option><option>cylinder</option>
      </select></div>
    <div class="row"><label>subdivision n</label><input id="n" type="number" value="12" min="2" max="64"></div>
    <h2>Unwrap</h2>
    <div class="row"><label>angle °</label><input id="angle" type="number" value="30" step="1"></div>
    <div class="row"><label>packer</label>
      <select id="packer"><option>islands</option><option>shelf</option></select></div>
    <div class="row"><label>seam-cut closed</label><input id="seam" type="checkbox"></div>
    <div class="row"><label>re-cut folds</label><input id="split" type="checkbox" checked></div>
    <div class="row"><label>threads (0=auto)</label><input id="threads" type="number" value="0" min="0" max="128"></div>
    <div class="row"><label>CUDA solve</label><input id="gpu" type="checkbox"></div>
    <h2>Pack</h2>
    <div class="row"><label>margin</label><input id="margin" type="number" value="0.003" step="0.001" min="0" max="0.2"></div>
    <div class="row"><label>rotation step °</label><input id="rot" type="number" value="90" min="1" max="180"></div>
    <button id="run">Unwrap</button>
    <div id="status"></div>
  </div>
  <div id="stage">
    <canvas id="cv" width="640" height="640"></canvas>
    <table id="table"></table>
  </div>
</main>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
const COLORS = ["#a7c5b0","#e8ac87","#ebcb83","#98b7c8","#c5b1a0","#bfc795",
                "#d8b4c4","#a8c5a0","#c9b88a","#9fb8c8","#c7a9a1","#b5bcd0"];
let islands = [], selected = -1;

async function api(entry, params) {
  const res = await fetch("/api", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entry: entry, params: params || {} }),
  });
  const env = await res.json();
  if (!env.Ok) throw new Error(env.Error);
  return env.Payload;
}

function draw() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.save();
  ctx.scale(cv.width, cv.height);
  ctx.lineWidth = 1.2 / cv.width;
  for (let i = 0; i < islands.length; i++) {
    const isl = islands[i];
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.strokeStyle = "#304d43";
    ctx.beginPath();
    for (const t of isl.tris) {
      ctx.moveTo(isl.uv[2*t[0]], 1 - isl.uv[2*t[0]+1]);
      for (let k = 1; k <= 3; k++) {
        const v = t[k % 3];
        ctx.lineTo(isl.uv[2*v], 1 - isl.uv[2*v+1]);
      }
    }
    ctx.fill();
    if (i === selected) { ctx.strokeStyle = "#d36b42"; ctx.lineWidth = 3 / cv.width; }
    ctx.stroke();
  }
  ctx.restore();
}

function renderTable(run) {
  let html = "<tr><th>island</th><th>conf μ</th><th>conf max</th><th>area r</th><th>folds</th></tr>";
  islands.forEach((isl, i) => {
    html += `<tr data-i="${i}"><td>${i}</td><td>${isl.conf[0].toFixed(3)}</td><td>${isl.conf[1].toFixed(1)}</td>` +
            `<td>${isl.area.toFixed(3)}</td><td>${isl.folds}</td></tr>`;
  });
  $("table").innerHTML = html;
  $("table").querySelectorAll("tr[data-i]").forEach((tr) => {
    tr.onclick = () => { selected = +tr.dataset.i; draw(); };
  });
}

async function runPipeline() {
  const btn = $("run");
  btn.disabled = true;
  $("status").textContent = "running…";
  $("status").id = "status";
  try {
    await api("Doc.Fixture", { Fixture: $("fixture").value, N: +$("n").value });
    const run = await api("Unwrap.Run", {
      Angle: +$("angle").value,
      Packer: $("packer").value,
      SeamCut: $("seam").checked,
      SplitCut: $("split").checked,
      Threads: +$("threads").value,
      Gpu: $("gpu").checked,
    });
    await api("Config.Set", { Key: "Editor.Margin", Value: +$("margin").value });
    const atlas = await api("Atlas.Get", {});
    islands = atlas.Islands.map((o) => ({
      uv: o.Uv,
      tris: [],
      conf: [o.ConformalMean, o.ConformalMax],
      area: o.AreaRatio,
      folds: o.Folds,
    }));
    // Tris arrive interleaved [a,b,c]*T; Uv interleaved [u,v]*N.
    atlas.Islands.forEach((o, i) => {
      const t = [];
      for (let k = 0; k < o.Tris.length; k += 3) t.push([o.Tris[k], o.Tris[k+1], o.Tris[k+2]]);
      islands[i].tris = t;
    });
    selected = -1;
    draw();
    renderTable(run);
    $("status").textContent =
      `charts ${run.Charts} · placed ${run.Placed} · scale ${run.Scale.toFixed(4)} · folds Σ` +
      islands.reduce((s, i) => s + i.folds, 0);
  } catch (e) {
    $("status").innerHTML = `<span id="err">${e}</span>`;
  }
  btn.disabled = false;
}

$("run").onclick = runPipeline;
runPipeline();
</script>
</body>
</html>
"##;
