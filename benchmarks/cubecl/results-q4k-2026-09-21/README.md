# Q4_K comparison, 2026-09-21

**The current CubeCL path does not establish a general prefill speedup.** Its best
variant, cached FP16 weights, reduced median time for the 512-token attention
projection by 22.8%. It lost on the smaller attention and expert cases; the larger
expert case was roughly tied. Unpacking on every invocation was slower on every
tested shape. Full DiffusionGemma inference on CubeCL is not implemented yet.

## Controlled operator results

Both drivers used actual Q4_K weights from the same local checkpoint, identical
FP32 activation files, GPU 0 on the AMD Radeon 8060S (`gfx1151`), ROCm 7.2.1 and
release/optimized builds. All compilation and other experiment GPU jobs finished
before timing. Native/CubeCL/CubeCL/native order provided 100 measured samples per
case and variant, after three warmups per run. GPU clocks and thermals were not
locked. Readback and correctness checks are outside the synchronized host timing.

The native baseline is the pinned GGML/HIP build from this worktree, without the
sibling checkout's newer MMQ overlay. CubeCL uses Burn 0.21.0 / CubeCL 0.10.0 /
CubeK 0.2.0. [Run metadata](run.json) records executable hashes and order;
[fixture metadata](manifest.json) records checkpoint and selected weight hashes.

Median milliseconds, lower is better:

| Matrix product | Native Q4_K | CubeCL unpack→FP32 | CubeCL cached FP32 | CubeCL unpack→FP16 | CubeCL cached FP16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention Q, 128 tokens, K=2816/N=4096 | **0.758** | 4.354 | 3.248 | 3.014 | 1.131 |
| Attention Q, 512 tokens, K=2816/N=4096 | 3.063 | 13.077 | 11.722 | 4.376 | **2.365** |
| Expert 0 gate/up, 32 tokens, K=2816/N=1408 | **0.327** | 2.717 | 1.456 | 1.369 | 0.508 |
| Expert 0 gate/up, 128 tokens, K=2816/N=1408 | 0.609 | 2.403 | 1.704 | 1.353 | 0.622 |

The FP16 variants include converting inputs to FP16 and outputs back to FP32 in
the timed operation. Cached variants exclude one-time weight expansion. The
unpack variants prepare scale/min metadata once on the CPU, keep nibbles packed
as U8, and materialize expanded weights on the GPU each time; these are not fused
quantized matrix kernels. The expert cases measure one expert slice, without
routing, grouped execution, scatter/gather, or the full MoE layer.

Tail latency and run-order variation for the best CubeCL variant:

| Case | Native p95 ms | Cached FP16 p95 ms | Native/CubeCL median ratio, run A / run B |
| --- | ---: | ---: | ---: |
| Attention 128 | 1.147 | 1.440 | 0.392 / 0.895 |
| Attention 512 | 3.248 | 2.715 | 1.264 / 1.307 |
| Expert 32 | 0.444 | 0.735 | 0.690 / 0.598 |
| Expert 128 | 0.756 | 0.861 | 1.060 / 0.837 |

A ratio above one means CubeCL was faster. Attention 128 varied substantially in
the native runs (roughly 0.44 versus 1.04 ms medians), but CubeCL was slower in
both. The expert-128 result crossed parity with run order, so it is not evidence
of a repeatable win. Attention 512 improved in both orders. P95 is the nearest-rank
percentile of 100 pooled samples per case; it is not a service-latency percentile.

## Numerical checks and memory

Every output element was checked after every measured invocation against a CPU
f64 matrix product using the pinned GGUF dequantizer's weights (reference stored
as FP32). Thresholds were fixed before measurement: relative RMSE <= 2% and maximum
error normalized by maximum reference magnitude <= 5%. These thresholds validate
this operator experiment, not full-model probability equivalence.

- CubeCL GPU Q4_K unpack matched all reference weight values exactly.
- Native outputs had relative RMSE of 1.17–1.36% on these matrices.
- CubeCL FP32 outputs had relative RMSE around 0.000095%.
- CubeCL FP16 outputs had relative RMSE around 0.0359%.

The arithmetic paths are therefore not numerically identical. Smaller operator
error against this reference does not establish better answers from the model.

| Weight slice | Native packed bytes | CubeCL prepared packed bytes | Cached FP16 bytes | Cached FP32 bytes |
| --- | ---: | ---: | ---: | ---: |
| Attention Q | 6,488,064 | 8,650,752 | 23,068,672 | 46,137,344 |
| Expert gate/up | 2,230,272 | 2,973,696 | 7,929,856 | 15,859,712 |

Cached FP16 uses 3.56 times the Q4_K weight storage. These are tensor representation
sizes, not peak device memory: the comparison driver retains reference/cache
tensors, and runtime workspace adds allocations. Repeated resident weight slices
also have different cache behavior from cycling through a complete model.

## Full-request baseline

A separate, sequential 20-round Snake run measured 160 requests after eight
warmups. All 160 matched their warmup outputs exactly, and the reference outputs
also matched the earlier exploratory run. This uses the same native engine,
batch=512, context=8192, eight CPU threads, GPU offload, flash attention off,
steps=1, samples=1, think=0, and no cross-request prefix cache.

| Phase | p50 | p95 |
| --- | ---: | ---: |
| Direct engine request | 1402.6 ms | 1532.4 ms |
| Synchronized prefill | 1271.4 ms | 1388.6 ms |
| Canvas forward | 125.7 ms | 139.8 ms |

Model loading took 25.3 seconds and is excluded. These are direct engine timings,
without HTTP or queue time. Phase percentiles do not necessarily sum to the
request percentile. Prefill still accounts for roughly 91% of median request time.
This fresh run differs from the earlier five-round exploratory measurement; it
does not represent a native implementation change or a controlled regression
comparison. Clocks/thermals were not pinned. See [raw requests](native-full-prefill.json)
and [overall summary](native-full-summary.json) for complete data.

There is no full CubeCL request result to compare against this baseline; the
operator speedups above must not be presented as an end-to-end migration speedup.

## Artifacts and decision

- [Computed comparison](comparison.json), containing all variants and paired ratios.
- [Native A](native-a.json), [CubeCL A](cubecl-a.json),
  [CubeCL B](cubecl-b.json), [Native B](native-b.json).
- [Method and reproduction commands](../Q4K.md).

The current prepared/unpack path is a correctness reference, not a competitive
replacement for the native quantized kernels. The next performance experiment
should avoid materializing full weight matrices on each invocation, fuse unpack
with multiplication where practical, and measure grouped expert execution. A
cached FP16 path is worth testing on larger projections if the full model's memory
budget permits it. Keep the native backend as the service default until a complete
CubeCL forward meets both numerical and prefill latency gates.

Validation: the two Q4_K layout/input-validation tests passed; 20 regular workspace
tests passed; HIP-feature Clippy and formatting checks passed. Both optimized
drivers completed all operator checks. The native inference implementation was
not modified by this comparison.
