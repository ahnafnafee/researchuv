//! Deterministic parallel execution — the engine's multi-core stage runner.
//!
//! The reference engine parallelizes its heavy stages across GPU kernels
//! (`thread_count` defaults to the host core count in the addon's prefs).
//! GPU execution is out of scope under this workspace's no-`unsafe`,
//! no-external-dependency constraints; this module is the CPU analog of the
//! same architecture: independent stage work (per-chart unfolds, per-island
//! checks) runs on scoped worker threads over **contiguous chunks**, and the
//! results are joined in input order — the output is byte-identical to the
//! serial run regardless of the worker count.

/// Resolve a requested worker count: `0` = all available cores, otherwise
/// the request clamped to at least 1.
pub fn worker_count(requested: u32) -> usize {
    if requested == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1)
    } else {
        requested as usize
    }
}

/// Map `f` over `items` on `workers` threads (0/1 = serial). Deterministic:
/// items are split into contiguous chunks, each chunk is mapped on one
/// worker, and the chunk results are concatenated in input order.
pub fn par_map<T, R, F>(workers: usize, items: Vec<T>, f: F) -> Vec<R>
where
    T: Send,
    R: Send,
    F: Fn(T) -> R + Send + Sync,
{
    if workers <= 1 || items.len() <= 1 {
        return items.into_iter().map(f).collect();
    }
    let workers = workers.min(items.len());
    // Contiguous chunks (front-loaded so the first chunk takes the remainder).
    let chunk_len = items.len().div_ceil(workers);
    let mut chunks: Vec<Vec<T>> = Vec::with_capacity(workers);
    let mut rest = items;
    while !rest.is_empty() {
        let take = chunk_len.min(rest.len());
        let tail = rest.split_off(take);
        chunks.push(rest);
        rest = tail;
    }
    let results: Vec<Vec<R>> = std::thread::scope(|scope| {
        let f = &f;
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| scope.spawn(move || chunk.into_iter().map(f).collect::<Vec<R>>()))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker thread must not panic"))
            .collect()
    });
    results.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_count_resolution() {
        assert!(worker_count(0) >= 1);
        assert_eq!(worker_count(1), 1);
        assert_eq!(worker_count(7), 7);
    }

    #[test]
    fn par_map_matches_serial_for_every_worker_count() {
        let items: Vec<u64> = (0..1000)
            .scan(0x9E3779B97F4A7C15u64, |s, i| {
                *s = s.wrapping_add(i as u64).wrapping_mul(6364136223846793005);
                Some(*s)
            })
            .collect();
        let serial: Vec<u64> = items.iter().map(|&i| i.wrapping_mul(3)).collect();
        for workers in [0usize, 1, 2, 3, 7, 16, 2000] {
            let out = par_map(workers, items.clone(), |i| i.wrapping_mul(3));
            assert_eq!(out, serial, "workers = {workers}");
        }
    }

    #[test]
    fn par_map_handles_empty_and_singleton_inputs() {
        let empty: Vec<u32> = Vec::new();
        assert!(par_map(4, empty, |x| x + 1).is_empty());
        assert_eq!(par_map(4, vec![41], |x| x + 1), vec![42]);
    }

    #[test]
    fn par_map_chunks_cover_every_item_once() {
        // A fold over the output must see every input exactly once,
        // independent of the worker count.
        for workers in [1usize, 2, 5, 64] {
            let items: Vec<usize> = (0..997).collect();
            let out = par_map(workers, items, |i| 1usize << (i % 63));
            let sum: usize = out.iter().map(|_| 1).sum();
            assert_eq!(sum, 997, "workers = {workers}");
        }
    }
}
