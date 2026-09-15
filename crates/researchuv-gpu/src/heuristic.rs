//! The GPU packing heuristic — multi-restart stochastic box relocation.
//!
//! The CPU heuristic ([`researchuv_pack`]'s `heuristic_refine`) relocates one
//! island at a time against neighbor-edge anchors. The GPU analog runs R
//! independent restarts **concurrently, one CUDA block per restart**: each
//! block walks its islands in a seed-derived order and scores candidate
//! anchors in parallel across its threads (clearance against every other box
//! plus target containment, Manhattan distance from the start corner as the
//! objective), keeping strictly-improving moves. The best restart wins.
//!
//! Deterministic by construction: every random draw is a counter-based
//! splitmix hash of `(seed, restart, pass, island, candidate)` — no device
//! RNG state, bitwise-stable across runs and driver versions.

use crate::solver::{DeviceMem, GpuSolver, BLOCK};
use crate::ffi::CUDA_SUCCESS;
use std::ffi::c_void;

/// One relocatable island box: fixed size `(w, h)` at position `(x, y)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutBox {
    pub w: f64,
    pub h: f64,
    pub x: f64,
    pub y: f64,
}

/// Refine `layout` inside the target box `[tx0, tx1] × [ty0, ty1]` with the
/// relative `margin` semantics (gap = margin · max extent of the pair, border
/// = margin · the island's extent).
///
/// Returns the best-of-restarts layout and its score (total start-corner
/// distance), or `None` when no device is available or a driver error
/// occurred. The result is validated on the host before being returned.
pub fn refine(
    layout: &[LayoutBox],
    (tx0, ty0, tx1, ty1): (f64, f64, f64, f64),
    margin: f64,
    seed: u64,
    restarts: u32,
    passes: u32,
    anchors: u32,
) -> Option<(Vec<LayoutBox>, f64)> {
    let gpu = GpuSolver::global()?;
    if !gpu.make_current() {
        return None;
    }
    let n = layout.len();
    if n == 0 || restarts == 0 {
        return None;
    }
    let n = n as i32;
    let restarts = restarts.clamp(1, 4096);
    let passes = passes.clamp(1, 512);
    let anchors = anchors.clamp(1, 4096);
    let c = &gpu.cuda;
    let mut mem = DeviceMem::new(c);
    let w: Vec<f64> = layout.iter().map(|b| b.w).collect();
    let h: Vec<f64> = layout.iter().map(|b| b.h).collect();
    let x0: Vec<f64> = layout.iter().map(|b| b.x).collect();
    let y0: Vec<f64> = layout.iter().map(|b| b.y).collect();
    let d_w = mem.upload(c, &w)?;
    let d_h = mem.upload(c, &h)?;
    let d_x0 = mem.upload(c, &x0)?;
    let d_y0 = mem.upload(c, &y0)?;
    let d_px = mem.alloc_f64(c, restarts as usize * n as usize)?;
    let d_py = mem.alloc_f64(c, restarts as usize * n as usize)?;
    let d_scores = mem.alloc_f64(c, restarts as usize)?;

    let p_u64 = |v: &u64| v as *const u64 as *mut c_void;
    let p_i32 = |v: &i32| v as *const i32 as *mut c_void;
    let p_f64 = |v: &f64| v as *const f64 as *mut c_void;
    let p_u64v = |v: &u64| v as *const u64 as *mut c_void;
    let seed64 = seed;
    // SAFETY: one launch, parameters point at locals that outlive it.
    let ok = unsafe {
        (c.cuLaunchKernel)(
            gpu.k_heur,
            restarts,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            std::ptr::null_mut(),
            [
                p_u64(&d_w),
                p_u64(&d_h),
                p_u64(&d_x0),
                p_u64(&d_y0),
                p_u64v(&d_px),
                p_u64v(&d_py),
                p_u64(&d_scores),
                p_i32(&n),
                p_i32(&(passes as i32)),
                p_i32(&(anchors as i32)),
                p_f64(&tx0),
                p_f64(&ty0),
                p_f64(&tx1),
                p_f64(&ty1),
                p_f64(&margin),
                p_u64(&seed64),
            ]
            .as_mut_ptr(),
            std::ptr::null_mut(),
        ) == CUDA_SUCCESS
            && (c.cuCtxSynchronize)() == CUDA_SUCCESS
    };
    if !ok {
        eprintln!("heuristic: launch/sync failed");
        return None;
    }
    let mut scores = vec![0.0f64; restarts as usize];
    let mut px = vec![0.0f64; restarts as usize * n as usize];
    let mut py = vec![0.0f64; restarts as usize * n as usize];
    // SAFETY: host buffers sized for the copies.
    unsafe {
        if (c.cuMemcpyDtoH)(scores.as_mut_ptr().cast(), d_scores, scores.len() * 8) != CUDA_SUCCESS
            || (c.cuMemcpyDtoH)(px.as_mut_ptr().cast(), d_px, px.len() * 8) != CUDA_SUCCESS
            || (c.cuMemcpyDtoH)(py.as_mut_ptr().cast(), d_py, py.len() * 8) != CUDA_SUCCESS
        {
            return None;
        }
    }
    if std::env::var("RUV_DBG").is_ok() {
        eprintln!("heuristic scores: {:?}", &scores[..scores.len().min(8)]);
    }
    // Best restart by score (smallest id on ties), then host-validate.
    let mut best: Option<(usize, f64)> = None;
    for (r, &s) in scores.iter().enumerate() {
        if !s.is_finite() {
            continue;
        }
        best = match best {
            Some((br, bs)) if s >= bs => Some((br, bs)),
            _ => Some((r, s)),
        };
    }
    let (r, score) = best?;
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        out.push(LayoutBox {
            w: w[i],
            h: h[i],
            x: px[r * n as usize + i],
            y: py[r * n as usize + i],
        });
    }
    if !validate(&out, (tx0, ty0, tx1, ty1), margin) {
        if std::env::var("RUV_DBG").is_ok() {
            eprintln!("heuristic: host validation failed, restart {r}, score {score}");
            for (i, b) in out.iter().enumerate() {
                eprintln!("  box {i}: {:?}", b);
            }
        }
        return None;
    }
    Some((out, score))
}

/// Feasibility of a whole layout: every box inside the target (border gap)
/// and disjoint from every other (pair gap). The GPU enforces this per move;
/// the host check is the trust boundary.
fn validate(layout: &[LayoutBox], (tx0, ty0, tx1, ty1): (f64, f64, f64, f64), margin: f64) -> bool {
    for b in layout {
        let e = b.w.max(b.h);
        let g = margin * e;
        if b.x < tx0 + g - 1e-9 || b.y < ty0 + g - 1e-9 || b.x + b.w > tx1 - g + 1e-9 || b.y + b.h > ty1 - g + 1e-9 {
            if std::env::var("RUV_DBG").is_ok() {
                eprintln!("validate: containment failed for {b:?} g={g}");
            }
            return false;
        }
    }
    for i in 0..layout.len() {
        for j in (i + 1)..layout.len() {
            let a = &layout[i];
            let b = &layout[j];
            let g = margin * a.w.max(a.h).max(b.w.max(b.h));
            let clear = a.x + a.w + g <= b.x + 1e-9
                || b.x + b.w + g <= a.x + 1e-9
                || a.y + a.h + g <= b.y + 1e-9
                || b.y + b.h + g <= a.y + 1e-9;
            if !clear {
                if std::env::var("RUV_DBG").is_ok() {
                    eprintln!("validate: pair ({i},{j}) overlap: a={a:?} b={b:?} g={g}");
                }
                return false;
            }
        }
    }
    true
}

/// The total start-corner score of a layout (lower = tighter toward BL).
pub fn score(layout: &[LayoutBox], (tx0, ty0): (f64, f64)) -> f64 {
    layout.iter().map(|b| (b.x - tx0) + (b.y - ty0)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn available() -> bool {
        if GpuSolver::global().is_none() {
            eprintln!("skipping: no CUDA device/PTX available");
            return false;
        }
        true
    }

    fn sq(s: f64) -> LayoutBox {
        LayoutBox { w: s, h: s, x: 0.0, y: 0.0 }
    }

    #[test]
    fn improves_a_scattered_layout() {
        if !available() {
            return;
        }
        // Four 0.2 squares parked far from the start corner.
        let scattered: Vec<LayoutBox> = vec![
            LayoutBox { x: 0.7, y: 0.7, ..sq(0.2) },
            LayoutBox { x: 0.7, y: 0.4, ..sq(0.2) },
            LayoutBox { x: 0.4, y: 0.7, ..sq(0.2) },
            LayoutBox { x: 0.5, y: 0.5, ..sq(0.2) },
        ];
        let target = (0.0, 0.0, 1.0, 1.0);
        let before = score(&scattered, (0.0, 0.0));
        let (out, after) = refine(&scattered, target, 0.003, 42, 16, 8, 256)
            .expect("GPU refine must run");
        assert!(after < before - 0.1, "score {after} must improve on {before}");
        // The returned layout is valid (validated inside refine, but assert again).
        assert!(validate(&out, target, 0.003));
    }

    #[test]
    fn never_worsens_and_is_deterministic() {
        if !available() {
            return;
        }
        // An already-tight BL layout: no move should raise the score. The
        // border gap (margin · extent = 0.0012) applies to the input too.
        let tight: Vec<LayoutBox> = vec![
            LayoutBox { x: 0.0013, y: 0.0013, ..sq(0.4) },
            LayoutBox { x: 0.4513, y: 0.0013, ..sq(0.4) },
            LayoutBox { x: 0.0013, y: 0.4513, ..sq(0.4) },
        ];
        let target = (0.0, 0.0, 1.0, 1.0);
        let before = score(&tight, (0.0, 0.0));
        let (out, after) = refine(&tight, target, 0.003, 7, 16, 8, 256).expect("runs");
        assert!(after <= before + 1e-9, "score {after} vs {before}");
        assert!(validate(&out, target, 0.003));
        // Same seed → bitwise-identical result.
        let (out2, after2) = refine(&tight, target, 0.003, 7, 16, 8, 256).expect("runs");
        assert_eq!(out, out2);
        assert_eq!(after, after2);
    }

    #[test]
    fn infeasible_input_is_rejected_not_corrupted() {
        if !available() {
            return;
        }
        // Two boxes stacked on top of each other: the GPU never validates the
        // INITIAL state, only moves — the host validator must reject a result
        // that leaves an overlap (here: no move can fix box 2 without moving
        // box 1, which the greedy pass may or may not do; either way the
        // return contract holds).
        let stacked = vec![sq(0.3), LayoutBox { x: 0.1, y: 0.1, ..sq(0.3) }];
        let r = refine(&stacked, (0.0, 0.0, 1.0, 1.0), 0.003, 1, 4, 4, 128);
        if let Some((out, _)) = r {
            assert!(validate(&out, (0.0, 0.0, 1.0, 1.0), 0.003));
        }
    }
}
