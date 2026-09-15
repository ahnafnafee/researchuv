//! Greedy shelf packing into the unit square.
//!
//! Islands are sorted by descending area and placed left to right in rows with
//! padding between them. When a layout does not fit, the packer scales every
//! island by 0.8 and retries. [`PackBackend::Gpu`] is a reserved backend option;
//! this module implements the CPU path.

use researchuv_math::{SBox2, SVec2};

/// A packable rectangle (island extent, pre-scale).
#[derive(Clone, Copy, Debug)]
pub struct PackRect {
    pub w: f64,
    pub h: f64,
}

/// A placed island in the unit square.
#[derive(Clone, Copy, Debug)]
pub struct Placed {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Placed {
    pub fn box2(&self) -> SBox2 {
        SBox2::new(SVec2::new(self.x, self.y), SVec2::new(self.x + self.w, self.y + self.h))
    }
    #[inline]
    pub fn area(&self) -> f64 {
        self.w * self.h
    }
}

/// Greedy shelf into the unit square (reference `pack_charts`).
///
/// Returns the uniform scale `s` and the per-island placements (in input order;
/// `None` where an island could not be placed in a failed iteration).
pub fn pack_charts(rects: &[PackRect], padding: f64, max_iters: usize) -> (f64, Vec<Option<Placed>>) {
    let n = rects.len();
    // Order by descending area (largest first); ties by index for determinism.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        let ai = rects[i].w.abs() * rects[i].h.abs();
        let aj = rects[j].w.abs() * rects[j].h.abs();
        aj.total_cmp(&ai).then(i.cmp(&j))
    });
    let mut s = 1.0f64;
    let mut placed: Vec<Option<Placed>> = vec![None; n];
    for _ in 0..max_iters.max(1) {
        let mut x = 0.0f64;
        let mut y = 0.0f64;
        let mut row_h = 0.0f64;
        let mut ok = true;
        placed = vec![None; n];
        for &i in &order {
            let w = rects[i].w * s;
            let h = rects[i].h * s;
            if x + w > 1.0 - padding {
                x = 0.0;
                y += row_h + padding;
                row_h = 0.0;
            }
            if y + h > 1.0 - padding {
                ok = false;
                break;
            }
            placed[i] = Some(Placed { x, y, w, h });
            x += w + padding;
            row_h = row_h.max(h);
        }
        if ok {
            return (s, placed);
        }
        s *= 0.8;
    }
    (s, placed)
}

/// Pack backend selection — mirrors `Prefs.PackOptions.UseGPU`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackBackend {
    /// CPU `CFinalPack` (this crate).
    Cpu,
    /// Reserved GPU backend; implementation pending.
    Gpu,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_quarter_rects_pack_into_the_unit_square() {
        // Two 0.5×0.5 rects: at s=1 the gutter (0.5 + 0.01 + 0.5 > 0.99) forces one
        // extra row, so the first fitting scale is s=0.8 (0.4 + 0.01 + 0.4 ≤ 0.99),
        // exactly as the reference `pack_charts` returns.
        let rects = vec![PackRect { w: 0.5, h: 0.5 }, PackRect { w: 0.5, h: 0.5 }];
        let (s, placed) = pack_charts(&rects, 0.01, 200);
        assert!((s - 0.8).abs() < 1e-12, "expected s=0.8, got s={s}");
        assert!(placed.iter().all(|p| p.is_some()));
        // No overlap: check box separation (including the gutter).
        let a = placed[0].unwrap().box2();
        let b = placed[1].unwrap().box2();
        let sep = a.max.u < b.min.u || b.max.u < a.min.u || a.max.v < b.min.v || b.max.v < a.min.v;
        assert!(sep, "islands overlap");
    }

    #[test]
    fn too_many_rects_scale_down() {
        // 100 unit squares cannot fit at s=1.
        let rects: Vec<PackRect> = (0..100).map(|_| PackRect { w: 1.0, h: 1.0 }).collect();
        let (s, placed) = pack_charts(&rects, 0.0, 200);
        assert!(s < 1.0, "expected a scale-down, got s={s}");
        // The final placement is complete (ok) at the returned scale.
        assert!(placed.iter().all(|p| p.is_some()));
        let total: f64 = placed.iter().map(|p| p.unwrap().area()).sum();
        assert!(total <= 1.0 + 1e-9, "placed area {total} exceeds the unit square");
    }
}
