# Explicit expert tiles and scalar fused Q4_K — initial sweep

On gfx1151, tested six explicit FP16 tile configurations (five supported) and eight scalar fused Q4_K configurations. The original fusion-enabled cached-FP16 operation is the control. Each case/variant has 100 measured samples across native/CubeCL/CubeCL/native order; all output checks passed.

`tile_1_2_2_planes2` was selected from this sweep for independent confirmation at 8 and 32 tokens. It means a 16x16x16 instruction, partition 1x2x2, two planes along the stage M dimension. It still expands weights to FP16. The explicit path also bypasses fusion scheduling, so its improvement cannot be attributed entirely to tile selection.

| Expert tokens | Native Q4_K p50 ms | Original cached FP16 p50 ms | Selected explicit FP16 p50 ms | Best scalar fused p50 ms |
| --- | --- | --- | --- | --- |
| 4 | 0.2197 | 0.4156 | 0.3786 | 0.5855 |
| 8 | 0.2718 | 0.5558 | 0.3960 | 0.9653 |
| 16 | 0.2696 | 0.5514 | 0.3934 | 1.5769 |
| 32 | 0.2820 | 0.5036 | 0.3821 | 2.0890 |
| 64 | 0.3847 | 0.4933 | 0.4867 | 2.6122 |
| 128 | 0.4687 | 0.5980 | 0.5988 | 5.2971 |

The selected FP16 configuration lost to native at 8 and 32 tokens in both run orders. Scalar fused Q4_K was slower still. Per-case best values are exploratory, not independently confirmed wins. The complete variant timings, p95, numerical errors, unsupported-config reason and paired ratios are in `comparison.json`.

Native was linked against the existing RDNA 3.5 overlay build in the original worktree. Its overlay was checked against the pinned source: only the predicate extension differs. Source/archive hashes are recorded in `native-build.json`. This driver uses dense `ggml_mul_mat`, so the routed tile heuristic is not exercised. Neither the user-reported 14.1% full-request improvement nor grouped execution is measured here.

The scalar kernel uses 192 bytes per Q4_K block, including expanded scale/min metadata, versus GGUF's 144. Cached FP16 uses 3.56x GGUF storage. All fixtures remain ignored local model assets. No full-model CubeCL inference is implemented.

This initial sweep preceded the matrix-instruction fused kernel. The binary hash and source hashes identify this earlier experiment; the current source also contains the later CMMA variants.
