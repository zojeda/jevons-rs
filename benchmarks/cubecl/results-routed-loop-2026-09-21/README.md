# Grouped Q4_K loop optimization — 2026-09-21

The revised CubeCL kernel lowers pooled median latency by **10.2–12.4%** with the
previously selected eight-token/four-wave configuration. All four workloads improve
in both run orders. It remains **8.1–16.3x slower** than patched llama.cpp by pooled
median; this is a useful local improvement, not evidence for replacing the engine.

| Workload | Patched native p50 ms | CubeCL before p50 ms | CubeCL after p50 ms | CubeCL latency reduction | After/native slowdown |
| --- | --- | --- | --- | --- | --- |
| routed_balanced_128 | 3.818 | 34.504 | 30.921 | 10.4% | 8.10x |
| routed_skewed_128 | 2.237 | 31.715 | 28.485 | 10.2% | 12.73x |
| routed_balanced_512 | 7.712 | 138.851 | 121.618 | 12.4% | 15.77x |
| routed_skewed_512 | 7.505 | 136.348 | 121.995 | 10.5% | 16.26x |

| Workload | Reduction in order A | Reduction in order B | Native p95 ms | Before p95 ms | After p95 ms |
| --- | --- | --- | --- | --- | --- |
| routed_balanced_128 | 7.5% | 12.7% | 5.722 | 41.522 | 36.775 |
| routed_skewed_128 | 10.3% | 11.1% | 2.993 | 38.025 | 33.069 |
| routed_balanced_512 | 11.6% | 13.1% | 8.283 | 159.751 | 138.074 |
| routed_skewed_512 | 8.2% | 12.8% | 7.829 | 157.696 | 139.255 |

The four-token configuration also appears in the raw results. Its 128-token balanced
case has mixed per-order results and only a 0.5% pooled reduction; do not generalize
the eight-token gain to every tile. Eight tokens was already the preferred grouped
configuration in the preceding experiment, rather than selected from these results.

## Change and evidence

The kernel resolves expert counts, assignment IDs, and input offsets once per tile.
It unrolls the eight subgroups within each Q4_K block. Inactive token slots use a
valid input-row offset during reduction and never scatter their sums, eliminating
per-token routing branches inside the reduction. The output calculation and routing
capacity are unchanged. Actual expert counts still determine jobs; the average does
not cap any expert. Bounds-checked launches and packed weights are retained.

[candidate.patch](candidate.patch) captures the complete kernel delta. These changes
were measured together; the data does not separately attribute gains to hoisting,
unrolling, or removing branches. Early debug smoke runs were used for correctness,
not for the reported release performance conclusion.

ROCm `rocprofv3` aborts on this WSL host because it discovers zero rocprofiler agents
while HSA reports two; `/sys/class/kfd/kfd/topology/nodes` is absent. The fallback
CubeCL 0.10 HIP profiler synchronizes before and after launches and uses host wall
time (`start_profile`/`end_profile` in its HIP server and `TimestampProfiler` in its
runtime). Its timings include dispatch/synchronization and are **not GPU event or
hardware-counter measurements**. They locate the expensive launch but cannot identify
memory stalls, instruction utilization, or pure GPU execution time.

Separate profiling runs used three warmups and two measured calls per case/variant.
The baseline eight-token product launch measured about 31–35 ms at 128 tokens and
128–130 ms at 512 tokens. Count reset, grouping, and job construction each measured
roughly 0.1–0.3 ms. This directs work toward the product kernel; eliminating grouping
alone cannot explain or close the observed gap. Do not subtract these serialized
profile timings from the unprofiled benchmark or sum them with overlapping CPU work.
The compressed raw logs, driver output, and warmup-excluding summaries are retained.
The after-change product-launch medians were 31.81/30.53 ms for balanced/skewed
128-token cases and 122.43/120.06 ms for the corresponding 512-token cases. The
product remains the dominant synchronized launch after the change.

## Method and validation

- Same real Q4_K gate/up weights for all 128 experts, K=2816, N=1408, top-8 routes.
  Balanced routes give 8/32 assignments per expert; skewed routes use eight experts
  for all 128/512 tokens and leave 120 empty. Activations and routes are synthetic.
- Patched llama.cpp executable and the preserved old CubeCL executable match their
  prior recorded hashes. Native build archive/overlay hashes were checked unchanged.
  Run metadata, fixture identity, old/new source hashes, and executable hashes are saved.
- Order: native A, old CubeCL A, new CubeCL A, new CubeCL B, old CubeCL B, native B.
  Three warmups and 50 measured calls per run yield 100 samples per case/variant.
  The two tile sizes together give 800 old and 800 new CubeCL measured outputs, plus
  400 native outputs. All values of every measured output passed reference checks.
- Upload, output readback, and CPU validation are excluded. CubeCL output/scratch
  allocation, count reset, grouping, job construction, compute, and scatter are timed.
  Native preallocates graph output and manages its internal temporary buffers.
- Compilation, GPU tests, and profiling ran outside the reported latency benchmark.
  Clocks/thermals are not pinned. Native differs substantially between orders; in
  particular, balanced 512 has per-run medians near 3.93 and 8.03 ms. Inspect per-order
  ratios as well as pooled medians. Native wins every workload in both orders.
- All seven CubeCL tests pass, including HIP tests for skew, empty experts, repeated
  resets, and partial token/output tiles. HIP clippy with warnings denied, rustfmt,
  Python syntax checks, and diff whitespace checks pass. The comparator reproduces
  the old report unchanged and returns unity for identical before/after samples.

Maximum new CubeCL relative RMSE is approximately 1.82e-7; native is about 1.28e-2
because its arithmetic includes activation quantization. Both pass the existing
2% relative RMSE / 5% normalized maximum-error limits against the CPU reference.
These limits do not establish exact full-model logits or probability parity.

Storage remains 285,474,816 bytes of raw weights and 380,633,088 bytes of prepared
CubeCL payload (1.33x); this is not a peak-memory measurement. No full FP16 weight
cache is introduced.

## Scope and decision

Keep this improvement in the experimental CubeCL worktree. Keep llama.cpp as the
service engine. The measurement covers only grouped gate/up multiplication with
preselected expert IDs. It excludes router logits/top-k selection, activation, down
projection, expert combination, attention, and the rest of the model. There is no
measured full-model CubeCL prefill or request-level improvement, and this percentage
must not be compared directly with the reported 14.1% native request-level gain.

The remaining gap calls for a different product kernel with substantially more
activation/weight reuse and efficient quantized arithmetic. Further scheduling-only
tweaks cannot be assumed to bridge it. Reproduction instructions and profiling
commands are in [the routed guide](../ROUTED.md).
