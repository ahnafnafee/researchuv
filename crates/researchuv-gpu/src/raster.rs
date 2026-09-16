//! The free-space rasterizer — GPU occupancy-grid placement search.
//!
//! The packer's alternative to exact candidate enumeration: placed islands
//! are rasterized into occupancy bitmasks (one bit per texel of the target
//! box, even-odd scanline fill — outer and hole rings XOR-compose), the
//! grids are dilated by margin radii with separable horizontal and vertical
//! bitblock passes, and a placement for the next island is found by scanning
//! its footprint mask against the dilated grids: one thread per candidate
//! anchor, word-level AND of the (unaligned) occupancy window,
//! strategy-scored, reduced on the host (lowest candidate index wins ties —
//! deterministic).
//!
//! # Per-island-extent margins
//!
//! The packer's relative margin is a *pair* quantity:
//! `gap(a, b) = margin · max(extent(a), extent(b))`. A single dilated grid
//! cannot express that, so islands are bucketed into [`EXTENT_CLASSES`]
//! extent classes (fractions of the target's larger side). Each class `j`
//! keeps its own occupancy grid `O_j`, and a candidate of class `k` is
//! checked against grids dilated by `max(r_j, r_k)`:
//!
//! - for `j ≤ k`: `Q[k][j] = dilate(O_j, r_k)` (the candidate's own class
//!   radius dominates),
//! - for `j > k`: `Q[j][j] = dilate(O_j, r_j)` (the neighbor's class
//!   radius dominates).
//!
//! That enforces exactly `margin · max(E_j, E_k)` per pair — up to one
//! power-of-two of class rounding, and never less than the relative-margin
//! intent. A small island no longer clears its small neighbors by the
//! largest islands' margin.

use crate::ffi::{Cuda, CUDA_SUCCESS};
use crate::solver::{GpuSolver, BLOCK};
use std::ffi::c_void;

/// Extent-class bounds: fractions of the target's larger side. An island
/// whose placed extent fraction falls into `(EXTENT_CLASSES[j-1],
/// EXTENT_CLASSES[j]]` belongs to class `j`.
pub const EXTENT_CLASSES: [f64; 4] = [0.125, 0.25, 0.5, 1.0];

/// The number of extent classes.
pub const N_CLASSES: usize = EXTENT_CLASSES.len();

/// The extent class of a placed-extent fraction (of the target's larger
/// side): the smallest bound the fraction fits under (the largest class on
/// overflow — oversized islands are rejected before they reach here).
pub fn extent_class(frac: f64) -> usize {
    for (j, &b) in EXTENT_CLASSES.iter().enumerate() {
        if frac <= b + 1e-12 {
            return j;
        }
    }
    N_CLASSES - 1
}

/// Placement scoring modes for [`RasterState::find_best`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RasterMode {
    /// Manhattan distance from the start corner (square / automatic).
    Corner,
    /// Side-to-side vertical: rows dominate.
    SideToSideVert,
    /// Side-to-side horizontal: columns dominate.
    SideToSideHori,
}

impl RasterMode {
    fn code(self) -> i32 {
        match self {
            RasterMode::Corner => 0,
            RasterMode::SideToSideVert => 1,
            RasterMode::SideToSideHori => 2,
        }
    }
}

/// An owned device allocation (freed on drop).
struct DevBuf {
    ptr: u64,
    free: crate::ffi::FnCuMemFree,
}

impl DevBuf {
    fn alloc(c: &Cuda, bytes: usize) -> Option<DevBuf> {
        let mut p: u64 = 0;
        // SAFETY: checked allocation of `bytes`.
        unsafe {
            if bytes == 0 || (c.cuMemAlloc)(&mut p, bytes) != CUDA_SUCCESS {
                return None;
            }
        }
        Some(DevBuf { ptr: p, free: c.cuMemFree })
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        // SAFETY: the pointer came from a successful cuMemAlloc.
        unsafe { (self.free)(self.ptr) };
    }
}

/// A persistent occupancy-grid family for one pack run.
pub struct RasterState {
    solver: &'static GpuSolver,
    /// Grid side in texels.
    pub resolution: u32,
    /// u32 words per row: `resolution / 32 + 1` (the guard word absorbs the
    /// unaligned window overread in the search kernel).
    words: u32,
    /// Per-class undilated occupancy `O_j`.
    occ: Vec<DevBuf>,
    /// Dilated grid table `Q[k][j]` for `j ≤ k` (10 of them at 4 classes).
    q: Vec<Vec<DevBuf>>,
    /// Dilation scratch.
    tmp: DevBuf,
    /// Device pointer table for the search (one candidate class's grids).
    table: DevBuf,
    scores: DevBuf,
}

impl RasterState {
    /// Allocate the grids on the device (`resolution` must be a multiple of
    /// 32; 256–1024 are the practical sizes). Requires a CUDA device.
    pub fn new(resolution: u32) -> Option<RasterState> {
        let solver = GpuSolver::global()?;
        if !solver.make_current() || resolution < 32 || resolution % 32 != 0 || resolution > 4096 {
            return None;
        }
        let c = &solver.cuda;
        let words = resolution / 32 + 1;
        let grid_bytes = words as usize * resolution as usize * 4;
        let mut occ = Vec::with_capacity(N_CLASSES);
        let mut q = Vec::with_capacity(N_CLASSES);
        for _ in 0..N_CLASSES {
            occ.push(DevBuf::alloc(c, grid_bytes)?);
        }
        for k in 0..N_CLASSES {
            let mut row = Vec::with_capacity(k + 1);
            for _ in 0..=k {
                row.push(DevBuf::alloc(c, grid_bytes)?);
            }
            q.push(row);
        }
        let tmp = DevBuf::alloc(c, grid_bytes)?;
        let table = DevBuf::alloc(c, N_CLASSES * 8)?;
        let scores = DevBuf::alloc(c, resolution as usize * resolution as usize * 8)?;
        Some(RasterState { solver, resolution, words, occ, q, tmp, table, scores })
    }

    fn launch(
        &self,
        f: crate::ffi::CUfunction,
        threads: usize,
        params: &mut [*mut c_void],
    ) -> bool {
        let c = &self.solver.cuda;
        // SAFETY: launch parameters point at locals that outlive the
        // (synchronous-parameter) call, per the driver contract.
        unsafe {
            (c.cuLaunchKernel)(
                f,
                threads.div_ceil(BLOCK as usize) as u32,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            ) == CUDA_SUCCESS
        }
    }

    fn sync(&self) -> bool {
        // SAFETY: plain context synchronize.
        unsafe { (self.solver.cuda.cuCtxSynchronize)() == CUDA_SUCCESS }
    }

    /// Clear every occupancy grid.
    pub fn reset(&mut self) -> bool {
        let c = &self.solver.cuda;
        let n = self.words as usize * self.resolution as usize;
        let zero = vec![0u32; n];
        for g in &self.occ {
            // SAFETY: host buffer sized for the copy.
            unsafe {
                if (c.cuMemcpyHtoD)(g.ptr, zero.as_ptr().cast(), n * 4) != CUDA_SUCCESS {
                    return false;
                }
            }
        }
        true
    }

    /// Rasterize one ring (absolute cell coordinates) into the occupancy
    /// grid of `class`. Rings XOR-compose (holes).
    pub fn rasterize_ring(&mut self, cell_pts: &[(f64, f64)], class: usize) -> bool {
        if cell_pts.len() < 3 {
            return true;
        }
        let class = class.min(N_CLASSES - 1);
        let c = &self.solver.cuda;
        let n = cell_pts.len();
        let xs: Vec<f64> = cell_pts.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = cell_pts.iter().map(|p| p.1).collect();
        let mem = match DevBuf::alloc(c, n * 16) {
            Some(m) => m,
            None => return false,
        };
        // SAFETY: uploads of host slices sized `n`; launch and sync per the
        // driver contract.
        unsafe {
            let ok_upload = (c.cuMemcpyHtoD)(mem.ptr, xs.as_ptr().cast(), n * 8) == CUDA_SUCCESS
                && (c.cuMemcpyHtoD)(mem.ptr + n as u64 * 8, ys.as_ptr().cast(), n * 8)
                    == CUDA_SUCCESS;
            if !ok_upload {
                return false;
            }
            let words = self.words as i32;
            let rows = self.resolution as i32;
            let npts = n as i32;
            let base = mem.ptr;
            let ys_at = mem.ptr + n as u64 * 8;
            let out = self.occ[class].ptr;
            let ok = self.launch(
                self.solver.k_rst_raster,
                words as usize * rows as usize,
                &mut [
                    (&base) as *const u64 as *mut c_void,
                    (&ys_at) as *const u64 as *mut c_void,
                    (&npts) as *const i32 as *mut c_void,
                    (&out) as *const u64 as *mut c_void,
                    (&words) as *const i32 as *mut c_void,
                    (&rows) as *const i32 as *mut c_void,
                ],
            );
            ok && self.sync()
        }
    }

    /// Rebuild the dilated table `Q[k][j] = dilate(O_j, r_k)` for every
    /// `j ≤ k`. `radii` carries one texel radius per class.
    pub fn dilate_all(&mut self, radii: &[u32]) -> bool {
        if radii.len() != N_CLASSES {
            return false;
        }
        let words = self.words as i32;
        let rows = self.resolution as i32;
        for (k, &rk) in radii.iter().enumerate() {
            let radius = rk.min(90) as i32;
            for j in 0..=k {
                let src = self.occ[j].ptr;
                let mid = self.tmp.ptr;
                let dst = self.q[k][j].ptr;
                if !self.launch(
                    self.solver.k_rst_dil_h,
                    rows as usize,
                    &mut [
                        (&src) as *const u64 as *mut c_void,
                        (&mid) as *const u64 as *mut c_void,
                        (&words) as *const i32 as *mut c_void,
                        (&rows) as *const i32 as *mut c_void,
                        (&radius) as *const i32 as *mut c_void,
                    ],
                ) {
                    return false;
                }
                if !self.launch(
                    self.solver.k_rst_dil_v,
                    words as usize * rows as usize,
                    &mut [
                        (&mid) as *const u64 as *mut c_void,
                        (&dst) as *const u64 as *mut c_void,
                        (&words) as *const i32 as *mut c_void,
                        (&rows) as *const i32 as *mut c_void,
                        (&radius) as *const i32 as *mut c_void,
                    ],
                ) {
                    return false;
                }
            }
        }
        self.sync()
    }

    /// Find the best anchor cell for a footprint mask of extent `class`
    /// against the per-class dilated grids (the candidate checks
    /// `Q[class][j]` for `j < class` and `Q[j][j]` for `j ≥ class`).
    /// Returns `(cell_x, cell_y, score)`, or `None` when the mask fits
    /// nowhere or a driver error occurs.
    pub fn find_best(
        &mut self,
        mask: &DeviceMask,
        mode: RasterMode,
        class: usize,
    ) -> Option<(u32, u32, f64)> {
        if mask.w == 0 || mask.h == 0 || mask.w > self.resolution || mask.h > self.resolution {
            return None;
        }
        let class = class.min(N_CLASSES - 1);
        // The candidate's grid set: Q[class][j] (j < class) + Q[j][j].
        let mut ptrs: Vec<u64> = Vec::with_capacity(N_CLASSES);
        for j in 0..class {
            ptrs.push(self.q[class][j].ptr);
        }
        for j in class..N_CLASSES {
            ptrs.push(self.q[j][j].ptr);
        }
        let c = &self.solver.cuda;
        // SAFETY: upload of the small pointer table.
        unsafe {
            if (c.cuMemcpyHtoD)(self.table.ptr, ptrs.as_ptr().cast(), ptrs.len() * 8)
                != CUDA_SUCCESS
            {
                return None;
            }
        }
        let cand_w = (self.resolution - mask.w + 1) as i32;
        let cand_h = (self.resolution - mask.h + 1) as i32;
        let cands = cand_w as usize * cand_h as usize;
        let g_words = self.words as i32;
        let m_words = mask.words as i32;
        let m_rows = mask.h as i32;
        let n_grids = ptrs.len() as i32;
        let mode_c = mode.code();
        let grids = self.table.ptr;
        let msk = mask.buf.ptr;
        let out = self.scores.ptr;
        if !self.launch(
            self.solver.k_rst_find,
            cands,
            &mut [
                (&grids) as *const u64 as *mut c_void,
                (&n_grids) as *const i32 as *mut c_void,
                (&msk) as *const u64 as *mut c_void,
                (&g_words) as *const i32 as *mut c_void,
                (&m_words) as *const i32 as *mut c_void,
                (&m_rows) as *const i32 as *mut c_void,
                (&cand_w) as *const i32 as *mut c_void,
                (&cand_h) as *const i32 as *mut c_void,
                (&mode_c) as *const i32 as *mut c_void,
                (&out) as *const u64 as *mut c_void,
            ],
        ) {
            return None;
        }
        if !self.sync() {
            return None;
        }
        let mut scores = vec![0.0f64; cands];
        // SAFETY: host buffer sized for the copy.
        unsafe {
            if (c.cuMemcpyDtoH)(scores.as_mut_ptr().cast(), self.scores.ptr, cands * 8)
                != CUDA_SUCCESS
            {
                return None;
            }
        }
        let mut best: Option<(usize, f64)> = None;
        for (t, &s) in scores.iter().enumerate() {
            if !s.is_finite() || s >= 1e300 {
                continue;
            }
            best = match best {
                Some((bt, bs)) if s >= bs => Some((bt, bs)),
                _ => Some((t, s)),
            };
        }
        let (t, score) = best?;
        Some(((t % cand_w as usize) as u32, (t / cand_w as usize) as u32, score))
    }
}

/// An island footprint mask: the rasterized ring set in its cell bounding
/// box (w × h texels, `w/32 + 1` words per row with a guard word). Points
/// are relative to the mask origin (subtract the footprint's cell min
/// before rasterizing).
pub struct DeviceMask {
    pub w: u32,
    pub h: u32,
    words: u32,
    buf: DevBuf,
}

impl DeviceMask {
    /// Rasterize `rings` (cell coordinates relative to the mask origin)
    /// into a new mask of size `w × h`.
    pub fn rasterize(rings: &[&[(f64, f64)]], w: u32, h: u32) -> Option<DeviceMask> {
        let solver = GpuSolver::global()?;
        if !solver.make_current() {
            return None;
        }
        if w == 0 || h == 0 || w > 4096 || h > 4096 {
            return None;
        }
        let c = &solver.cuda;
        let words = w.div_ceil(32) + 1; // guard word for the unaligned window
        let total = words as usize * h as usize;
        let zero = vec![0u32; total];
        let buf = DevBuf::alloc(c, total * 4)?;
        // SAFETY: host buffer sized for the copy.
        unsafe {
            if (c.cuMemcpyHtoD)(buf.ptr, zero.as_ptr().cast(), total * 4) != CUDA_SUCCESS {
                return None;
            }
        }
        let mask = DeviceMask { w, h, words, buf };
        for ring in rings {
            if ring.len() < 3 {
                continue;
            }
            let n = ring.len();
            let xs: Vec<f64> = ring.iter().map(|p| p.0).collect();
            let ys: Vec<f64> = ring.iter().map(|p| p.1).collect();
            let mem = DevBuf::alloc(c, n * 16)?;
            // SAFETY: uploads sized `n`.
            unsafe {
                let ok_upload =
                    (c.cuMemcpyHtoD)(mem.ptr, xs.as_ptr().cast(), n * 8) == CUDA_SUCCESS
                        && (c.cuMemcpyHtoD)(mem.ptr + n as u64 * 8, ys.as_ptr().cast(), n * 8)
                            == CUDA_SUCCESS;
                if !ok_upload {
                    return None;
                }
                let words_i = mask.words as i32;
                let rows = h as i32;
                let npts = n as i32;
                let base = mem.ptr;
                let ys_at = mem.ptr + n as u64 * 8;
                let out = mask.buf.ptr;
                let ok = mask_launch(
                    solver,
                    solver.k_rst_raster,
                    words_i as usize * rows as usize,
                    &mut [
                        (&base) as *const u64 as *mut c_void,
                        (&ys_at) as *const u64 as *mut c_void,
                        (&npts) as *const i32 as *mut c_void,
                        (&out) as *const u64 as *mut c_void,
                        (&words_i) as *const i32 as *mut c_void,
                        (&rows) as *const i32 as *mut c_void,
                    ],
                );
                if !ok {
                    return None;
                }
            }
        }
        // SAFETY: plain context synchronize.
        unsafe {
            if (c.cuCtxSynchronize)() != CUDA_SUCCESS {
                return None;
            }
        }
        Some(mask)
    }
}

fn mask_launch(
    solver: &GpuSolver,
    f: crate::ffi::CUfunction,
    threads: usize,
    params: &mut [*mut c_void],
) -> bool {
    let c = &solver.cuda;
    // SAFETY: same launch contract as RasterState::launch.
    unsafe {
        (c.cuLaunchKernel)(
            f,
            threads.div_ceil(BLOCK as usize) as u32,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        ) == CUDA_SUCCESS
    }
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

    fn square_ring(x: f64, y: f64, s: f64) -> Vec<(f64, f64)> {
        vec![(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    }

    fn uniform(r: u32) -> [u32; N_CLASSES] {
        [r; N_CLASSES]
    }

    #[test]
    fn extent_classes_bucket_by_bound() {
        assert_eq!(extent_class(0.05), 0);
        assert_eq!(extent_class(0.2), 1);
        assert_eq!(extent_class(0.5), 2);
        assert_eq!(extent_class(0.9), 3);
        assert_eq!(extent_class(4.0), 3);
    }

    #[test]
    fn empty_grid_places_at_the_start_corner() {
        if !available() {
            return;
        }
        let mut st = RasterState::new(256).expect("state");
        assert!(st.reset());
        assert!(st.dilate_all(&uniform(2)));
        let mask = DeviceMask::rasterize(&[&square_ring(0.0, 0.0, 64.0)], 64, 64).expect("mask");
        let (x, y, score) = st.find_best(&mask, RasterMode::Corner, 0).expect("fits");
        assert_eq!((x, y), (0, 0), "empty grid: the corner wins");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn occupied_region_is_avoided() {
        if !available() {
            return;
        }
        let mut st = RasterState::new(256).expect("state");
        assert!(st.reset());
        // A 64-cell block covering the bottom-left corner (cells [0,64)²).
        assert!(st.rasterize_ring(&square_ring(0.0, 0.0, 64.0), 0));
        assert!(st.dilate_all(&uniform(2)));
        let mask = DeviceMask::rasterize(&[&square_ring(0.0, 0.0, 64.0)], 64, 64).expect("mask");
        let (x, y, _) = st.find_best(&mask, RasterMode::Corner, 0).expect("fits beside");
        assert!(
            (x >= 66 && y == 0) || (y >= 66 && x == 0),
            "got anchor ({x}, {y})"
        );
    }

    #[test]
    fn small_islands_clear_small_neighbors_by_the_small_margin() {
        if !available() {
            return;
        }
        // One SMALL occupied block; a small candidate (class 0) must clear
        // it by r_0 only — while the same mask declared large (class 3)
        // clears by r_3.
        let mut st = RasterState::new(256).expect("state");
        assert!(st.reset());
        assert!(st.rasterize_ring(&square_ring(0.0, 0.0, 32.0), 0)); // class-0 block
        let radii = [1u32, 8, 16, 32];
        assert!(st.dilate_all(&radii));
        let small = DeviceMask::rasterize(&[&square_ring(0.0, 0.0, 32.0)], 32, 32).expect("mask");
        let (x, y, _) = st
            .find_best(&small, RasterMode::Corner, 0)
            .expect("small fits near");
        // Small-vs-small: separation r_0 = 1 → the neighbor sits at 33.
        assert!(
            (x == 33 && y == 0) || (y == 33 && x == 0),
            "small candidate got ({x}, {y}), expected the r0=1-texel spot"
        );
        // The same footprint declared large (class 3): clears by r_3 = 32.
        let (lx, ly, _) = st
            .find_best(&small, RasterMode::Corner, 3)
            .expect("large-class fits farther");
        assert!(
            (lx >= 64 && ly == 0) || (ly >= 64 && lx == 0),
            "large-class candidate got ({lx}, {ly}), expected ≥ 64"
        );
    }

    #[test]
    fn oversize_mask_fits_nowhere() {
        if !available() {
            return;
        }
        let mut st = RasterState::new(128).expect("state");
        assert!(st.reset());
        assert!(st.dilate_all(&uniform(1)));
        let big = DeviceMask::rasterize(&[&square_ring(0.0, 0.0, 200.0)], 200, 200);
        assert!(big.is_none() || st.find_best(&big.unwrap(), RasterMode::Corner, 0).is_none());
    }
}
