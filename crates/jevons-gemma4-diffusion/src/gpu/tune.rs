//! Device profile and measured launch plans for the quantized matrix kernels.
//!
//! The best tile shape and split-K factor depend on the device (compute units, clocks, cache
//! and register file sizes) and on the matrix shape. Instead of hardcoding choices for one GPU,
//! [`Tuner::tune`] times the candidate plans on the device at startup, for every weight shape the
//! model uses and a set of row-count buckets. Results are stored next to CubeCL's kernel cache
//! and reused while the device profile and kernel version stay the same.
//!
//! Tuning never changes results a caller relies on being exact: prefill plans are chosen only
//! among row-invariant configurations (matrix tiles without split-K), whose per-row
//! accumulation order does not depend on the tile shape.

use super::gemm::{self, Groups, Plan, QMatrix, SplitScratch};
use super::{Buf, Gpu, ops};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// Bump when kernels change in a way that invalidates stored measurements.
const TUNE_VERSION: u32 = 2;
const FILE_NAME: &str = "autotune.txt";

/// What the runtime exposes about the device, plus the compute unit count (HIP in CubeCL does
/// not report it; `DIFFUSION_CUBECL_CUS` overrides the default of 40).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceProfile {
    pub device: usize,
    pub compute_units: usize,
    pub plane_size: u32,
    pub shared_memory: usize,
    pub max_allocation: u64,
    pub load_width: u32,
}

impl DeviceProfile {
    pub fn detect(gpu: &Gpu, device: usize) -> Self {
        let props = gpu.client.properties();
        let hw = &props.hardware;
        let compute_units = std::env::var("DIFFUSION_CUBECL_CUS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v: &usize| v > 0)
            .or(hw.num_streaming_multiprocessors.map(|v| v as usize))
            .unwrap_or(40);
        Self {
            device,
            compute_units,
            plane_size: hw.plane_size_max,
            shared_memory: hw.max_shared_memory_size,
            max_allocation: props.memory.max_page_size,
            load_width: hw.load_width,
        }
    }

    /// Workgroups needed to keep every compute unit busy with a few resident groups.
    pub fn groups_target(&self) -> usize {
        4 * self.compute_units
    }

    fn key(&self) -> String {
        format!(
            "v{TUNE_VERSION} device={} cus={} plane={} lds={} page={} load={}",
            self.device,
            self.compute_units,
            self.plane_size,
            self.shared_memory,
            self.max_allocation,
            self.load_width
        )
    }
}

/// Row-count bucket: plans measured at `m = bucket` serve every `m` in `(bucket/2, bucket]`.
pub fn bucket(m: usize) -> usize {
    m.next_power_of_two().max(4)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DenseKey {
    fmt: u32,
    n: usize,
    k: usize,
    bucket: usize,
    row_invariant: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GroupKey {
    fmt: u32,
    n: usize,
    k: usize,
    experts: usize,
    bucket: usize,
}

/// Tile shape of the expert-grouped product (grouping must use the same `bm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupPlan {
    pub bm: usize,
    pub bn: usize,
}

impl GroupPlan {
    pub const ROW_TILES: [usize; 2] = [32, 64];

    fn heuristic(w: &QMatrix) -> Self {
        Self {
            bm: 32,
            bn: gemm::tile_n(w.n),
        }
    }
}

/// Weight shapes and row counts the model will run, gathered before tuning.
pub struct Workload<'a> {
    /// Dense matrices (deduplicated by shape and format internally).
    pub dense: Vec<&'a QMatrix>,
    /// Expert matrices used by the grouped product.
    pub grouped: Vec<&'a QMatrix>,
    pub top_k: usize,
    /// Largest prompt chunk.
    pub prefill_rows: usize,
    /// Largest canvas.
    pub canvas_rows: usize,
}

/// Launch plans keyed by weight shape and row bucket, falling back to heuristics.
pub struct Tuner {
    pub profile: DeviceProfile,
    dense: HashMap<DenseKey, Plan>,
    grouped: HashMap<GroupKey, GroupPlan>,
    path: Option<PathBuf>,
    announced: bool,
}

impl Tuner {
    /// Loads stored plans for this device profile. `DIFFUSION_CUBECL_AUTOTUNE=0` disables
    /// stored and measured plans (heuristics only).
    pub fn new(profile: DeviceProfile) -> Self {
        let path = match std::env::var("DIFFUSION_CUBECL_AUTOTUNE").as_deref() {
            Ok("0") | Ok("off") => None,
            _ => super::cache_dir().map(|d| d.join(FILE_NAME)),
        };
        let mut tuner = Self {
            profile,
            dense: HashMap::new(),
            grouped: HashMap::new(),
            path,
            announced: false,
        };
        if std::env::var("DIFFUSION_CUBECL_AUTOTUNE").as_deref() != Ok("retune") {
            tuner.load();
        }
        tuner
    }

    pub fn enabled(&self) -> bool {
        self.path.is_some()
    }

    /// Plan for `out[m, n] = x[m, k] . W^T`. Row-invariant plans never split K.
    pub fn dense(&self, w: &QMatrix, m: usize, row_invariant: bool) -> Plan {
        let key = DenseKey {
            fmt: w.fmt,
            n: w.n,
            k: w.k,
            bucket: bucket(m),
            row_invariant,
        };
        match self.dense.get(&key) {
            Some(&p) if p.valid_for(w, m) && (!row_invariant || invariant(p)) => p,
            _ => Plan::heuristic(w, m, row_invariant, self.profile.groups_target()),
        }
    }

    /// Tile shape for the grouped product of `rows` tokens.
    pub fn grouped(&self, w: &QMatrix, rows: usize) -> GroupPlan {
        let key = GroupKey {
            fmt: w.fmt,
            n: w.n,
            k: w.k,
            experts: w.experts,
            bucket: bucket(rows),
        };
        match self.grouped.get(&key) {
            Some(&p) if w.n.is_multiple_of(p.bn) => p,
            _ => GroupPlan::heuristic(w),
        }
    }

    /// Measures missing plans for `work`, then stores the table. `x`, `out` and the grouping
    /// buffers are caller scratch large enough for the largest row count; `jobs` must hold
    /// enough tiles for 32-row groups of the largest route count.
    pub fn tune(&mut self, gpu: &Gpu, work: &Workload, bufs: &TuneBuffers) -> usize {
        if !self.enabled() {
            return 0;
        }
        let start = Instant::now();
        FIRST_RUN_NS.store(0, std::sync::atomic::Ordering::Relaxed);
        let before = self.dense.len() + self.grouped.len();
        let split = SplitScratch::new();
        for ws in same_shape(&work.dense) {
            if ws[0].experts != 1 {
                continue;
            }
            for m in buckets(4, work.prefill_rows) {
                self.tune_dense(gpu, &ws, m, true, bufs, &split);
            }
            for m in buckets(4, work.canvas_rows) {
                self.tune_dense(gpu, &ws, m, false, bufs, &split);
            }
        }
        for ws in same_shape(&work.grouped) {
            for m in buckets(4, work.prefill_rows.max(work.canvas_rows)) {
                self.tune_grouped(gpu, &ws, m, work.top_k, bufs, &split);
            }
        }
        let added = self.dense.len() + self.grouped.len() - before;
        if added > 0 {
            let first = FIRST_RUN_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9;
            eprintln!(
                "cubecl: tuned {added} launch plans in {:.1}s ({first:.1}s compiling, {} compute units)",
                start.elapsed().as_secs_f64(),
                self.profile.compute_units
            );
            self.save();
        }
        added
    }

    fn tune_dense(
        &mut self,
        gpu: &Gpu,
        ws: &[&QMatrix],
        m: usize,
        row_invariant: bool,
        bufs: &TuneBuffers,
        split: &SplitScratch,
    ) {
        let w = ws[0];
        let key = DenseKey {
            fmt: w.fmt,
            n: w.n,
            k: w.k,
            bucket: m,
            row_invariant,
        };
        if self.dense.contains_key(&key)
            || bufs.x.len() < m * w.k
            || bufs.out.len() < m * w.n
            || !w.k.is_multiple_of(64)
        {
            return;
        }
        self.announce();
        let baseline = Plan::heuristic(w, m, row_invariant, self.profile.groups_target());
        let mut candidates = dense_candidates(w, m, row_invariant);
        if !candidates.contains(&baseline) {
            candidates.push(baseline);
        }
        let best = fastest(gpu, &candidates, baseline, |p| {
            for w in ws {
                gemm::matmul_plan(gpu, &bufs.x, m, w, &bufs.out, &bufs.dummy, split, *p);
            }
        });
        if let Some(p) = best {
            self.dense.insert(key, p);
        }
    }

    fn tune_grouped(
        &mut self,
        gpu: &Gpu,
        ws: &[&QMatrix],
        m: usize,
        top_k: usize,
        bufs: &TuneBuffers,
        split: &SplitScratch,
    ) {
        let w = ws[0];
        let key = GroupKey {
            fmt: w.fmt,
            n: w.n,
            k: w.k,
            experts: w.experts,
            bucket: m,
        };
        let a = m * top_k;
        if self.grouped.contains_key(&key)
            || bufs.route_capacity < a
            || bufs.x.len() < a * w.k
            || bufs.out.len() < a * w.n
        {
            return;
        }
        self.announce();
        // Synthetic balanced routing: each token picks `top_k` distinct experts. The buffer has
        // the model's route capacity so the grouping kernel variant is the one requests use.
        let ids: Vec<u32> = (0..bufs.route_capacity)
            .map(|i| (((i / top_k) * 7 + (i % top_k) * 13) % w.experts) as u32)
            .collect();
        let ids = gpu.upload_u32(&ids);
        let mut candidates = Vec::new();
        for bm in GroupPlan::ROW_TILES {
            for bn in Plan::COL_TILES {
                if w.n.is_multiple_of(bn) {
                    candidates.push(GroupPlan { bm, bn });
                }
            }
        }
        let baseline = GroupPlan::heuristic(w);
        let mut times = Vec::new();
        for p in candidates {
            ops::group_routes(
                gpu,
                &ids,
                &bufs.sorted,
                &bufs.offsets,
                &bufs.jobs,
                a,
                w.experts,
                p.bm,
            );
            let max_jobs = a.div_ceil(p.bm) + w.experts;
            if bufs.jobs.len() < 1 + 2 * max_jobs {
                continue;
            }
            let g = Groups {
                ids: &bufs.sorted,
                offsets: &bufs.offsets,
                jobs: &bufs.jobs,
                max_jobs,
                in_div: 1,
                rows: a,
            };
            let t = time(gpu, || {
                for w in ws {
                    gemm::matmul_grouped(gpu, &bufs.x, w, &g, &bufs.out, p.bm, p.bn, 1, split);
                }
            });
            times.push((t, p));
        }
        if let Some(p) = pick(&times, baseline) {
            self.grouped.insert(key, p);
        }
    }

    fn announce(&mut self) {
        if !self.announced {
            self.announced = true;
            eprintln!(
                "cubecl: measuring launch plans for this device (the first start also compiles \
                 kernel variants and can take a few minutes; DIFFUSION_CUBECL_AUTOTUNE=0 skips it)"
            );
        }
    }

    fn load(&mut self) {
        let Some(text) = self
            .path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
        else {
            return;
        };
        let mut lines = text.lines();
        if lines.next() != Some(self.profile.key().as_str()) {
            return;
        }
        for line in lines {
            let v: Vec<&str> = line.split_whitespace().collect();
            let num = |i: usize| v.get(i).and_then(|s| s.parse::<usize>().ok());
            match (v.first().copied(), v.len()) {
                (Some("dense"), 9) => {
                    if let (Some(fmt), Some(n), Some(k), Some(b), Some(ri), Some(bm), Some(bn)) =
                        (num(1), num(2), num(3), num(4), num(5), num(6), num(7))
                        && let Some(splits) = num(8)
                    {
                        self.dense.insert(
                            DenseKey {
                                fmt: fmt as u32,
                                n,
                                k,
                                bucket: b,
                                row_invariant: ri == 1,
                            },
                            Plan { bm, bn, splits },
                        );
                    }
                }
                (Some("grouped"), 8) => {
                    if let (Some(fmt), Some(n), Some(k), Some(e), Some(b), Some(bm), Some(bn)) =
                        (num(1), num(2), num(3), num(4), num(5), num(6), num(7))
                    {
                        self.grouped.insert(
                            GroupKey {
                                fmt: fmt as u32,
                                n,
                                k,
                                experts: e,
                                bucket: b,
                            },
                            GroupPlan { bm, bn },
                        );
                    }
                }
                _ => {}
            }
        }
    }

    fn save(&self) {
        let Some(path) = &self.path else { return };
        let mut out = self.profile.key();
        out.push('\n');
        let mut dense: Vec<_> = self.dense.iter().collect();
        dense.sort_by_key(|(k, _)| (k.fmt, k.n, k.k, k.row_invariant, k.bucket));
        for (k, p) in dense {
            out += &format!(
                "dense {} {} {} {} {} {} {} {}\n",
                k.fmt, k.n, k.k, k.bucket, k.row_invariant as u8, p.bm, p.bn, p.splits
            );
        }
        let mut grouped: Vec<_> = self.grouped.iter().collect();
        grouped.sort_by_key(|(k, _)| (k.fmt, k.n, k.k, k.experts, k.bucket));
        for (k, p) in grouped {
            out += &format!(
                "grouped {} {} {} {} {} {} {}\n",
                k.fmt, k.n, k.k, k.experts, k.bucket, p.bm, p.bn
            );
        }
        let tmp = path.with_extension("tmp");
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|_| std::fs::write(&tmp, out))
            .and_then(|_| std::fs::rename(&tmp, path));
        if let Err(e) = written {
            eprintln!(
                "cubecl: cannot store tuned plans in {}: {e}",
                path.display()
            );
        }
    }
}

/// Scratch used while tuning (contents are overwritten).
pub struct TuneBuffers {
    pub x: Buf,
    pub out: Buf,
    pub dummy: Buf,
    /// Length of the model's route buffer (fixes the grouping kernel variant).
    pub route_capacity: usize,
    pub sorted: Buf,
    pub offsets: Buf,
    pub jobs: Buf,
}

/// Weight bytes each timed run cycles through: more than the GPU's last-level cache, so
/// candidates are measured reading cold weights as a forward pass does, not re-reading one
/// cache-resident matrix.
const ROTATION_BYTES: usize = 64 << 20;

/// Matrices grouped by kernel variant (format and shape), each group truncated to the
/// rotation needed to exceed [`ROTATION_BYTES`].
fn same_shape<'a>(all: &[&'a QMatrix]) -> Vec<Vec<&'a QMatrix>> {
    let mut groups: Vec<Vec<&QMatrix>> = Vec::new();
    for &w in all {
        let same = |g: &&mut Vec<&QMatrix>| {
            let o = g[0];
            (o.fmt, o.n, o.k, o.experts) == (w.fmt, w.n, w.k, w.experts)
        };
        match groups.iter_mut().find(|g| same(g)) {
            Some(g) => g.push(w),
            None => groups.push(vec![w]),
        }
    }
    for g in &mut groups {
        let per = g[0].bytes().max(1);
        g.truncate(ROTATION_BYTES.div_ceil(per).max(1));
    }
    groups
}

fn invariant(p: Plan) -> bool {
    p.bm > 0 && p.splits == 1
}

fn buckets(min: usize, max: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut b = min;
    while b <= bucket(max.max(1)) && b <= 1024 {
        out.push(b);
        b *= 2;
    }
    out
}

fn dense_candidates(w: &QMatrix, m: usize, row_invariant: bool) -> Vec<Plan> {
    let mut out = Vec::new();
    if !row_invariant && m <= gemm::GEMV_ROWS {
        out.push(Plan {
            bm: 0,
            bn: gemm::tile_n(w.n),
            splits: 1,
        });
    }
    let steps = w.k / 64;
    for bm in Plan::ROW_TILES {
        // Taller tiles than the row count only waste work.
        if bm > 32 && bm > m {
            continue;
        }
        for bn in Plan::COL_TILES {
            // 128-row tiles only pay off with wide columns and few K slices; every other
            // variant costs a kernel compilation on the first start.
            if !w.n.is_multiple_of(bn) || (bm == 128 && bn != 128) {
                continue;
            }
            let split_options: &[usize] = if row_invariant { &[1] } else { &[1, 2, 4, 8] };
            for &splits in split_options {
                if (splits > 1 && steps / splits < 4) || (bm == 128 && splits > 4) {
                    continue;
                }
                out.push(Plan { bm, bn, splits });
            }
        }
    }
    out
}

/// Fastest candidate by the minimum of a few timed batches (see [`pick`]).
fn fastest<P: Copy + PartialEq>(
    gpu: &Gpu,
    candidates: &[P],
    baseline: P,
    run: impl Fn(&P),
) -> Option<P> {
    let times: Vec<(f64, P)> = candidates
        .iter()
        .map(|p| (time(gpu, || run(p)), *p))
        .collect();
    pick(&times, baseline)
}

/// The fastest plan, unless it beats the heuristic `baseline` by less than [`MARGIN`]: timing
/// noise then decides nothing and plans stay stable across runs and neighbouring buckets.
fn pick<P: Copy + PartialEq>(times: &[(f64, P)], baseline: P) -> Option<P> {
    let (best_t, best) = times.iter().copied().min_by(|a, b| a.0.total_cmp(&b.0))?;
    match times.iter().find(|(_, p)| *p == baseline) {
        Some(&(base_t, _)) if best_t > base_t * (1.0 - MARGIN) => Some(baseline),
        _ => Some(best),
    }
}

/// Relative speedup a measured plan needs over the heuristic to replace it.
const MARGIN: f64 = 0.05;

/// Time spent in first launches (kernel compilation or cache loads) while tuning.
static FIRST_RUN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Seconds per launch: one untimed run (compilation, clocks), then the best of three batches
/// sized to take at least ~2 ms so host synchronization does not dominate short kernels.
fn time(gpu: &Gpu, run: impl Fn()) -> f64 {
    let first = Instant::now();
    run();
    gpu.sync();
    FIRST_RUN_NS.fetch_add(
        first.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    let probe = Instant::now();
    run();
    gpu.sync();
    let once = probe.elapsed().as_secs_f64();
    let reps = ((2e-3 / once.max(1e-6)).ceil() as usize).clamp(1, 32);
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..reps {
            run();
        }
        gpu.sync();
        best = best.min(start.elapsed().as_secs_f64() / reps as f64);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_plan_is_kept_unless_clearly_slower() {
        assert_eq!(pick(&[(1.00, 'h'), (0.97, 'a')], 'h'), Some('h'));
        assert_eq!(pick(&[(1.00, 'h'), (0.90, 'a')], 'h'), Some('a'));
        assert_eq!(pick(&[(0.80, 'a'), (0.90, 'b')], 'h'), Some('a'));
        assert_eq!(pick::<char>(&[], 'h'), None);
    }

    #[test]
    fn row_buckets_cover_each_power_of_two_range() {
        assert_eq!(bucket(1), 4);
        assert_eq!(bucket(12), 16);
        assert_eq!(bucket(16), 16);
        assert_eq!(bucket(17), 32);
        assert_eq!(buckets(32, 300), vec![32, 64, 128, 256, 512]);
        assert_eq!(buckets(4, 1), vec![4]);
        assert_eq!(buckets(4, 12), vec![4, 8, 16]);
        assert_eq!(buckets(32, 4096), vec![32, 64, 128, 256, 512, 1024]);
    }
}
