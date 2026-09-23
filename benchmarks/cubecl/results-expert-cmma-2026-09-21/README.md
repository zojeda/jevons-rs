# Explicit FP16 confirmation and fused matrix-instruction sweep

The explicit FP16 configuration was selected before this run from the preceding sweep (`selection.json`). Six fused Q4_K CMMA configurations were also explored. The GPU is gfx1151; the model/fixtures and tolerances match the earlier experiments. No compilation ran during timing. Fifty samples in each of two run orders yield 100 measured samples per case/variant.

| Expert tokens | Native Q4_K p50 ms | Original cached FP16 p50 ms | Fixed explicit FP16 p50 ms | Best fused CMMA p50 ms |
| --- | --- | --- | --- | --- |
| 4 | 0.2232 | 0.4017 | 0.3677 | 1.2363 |
| 8 | 0.2796 | 0.4915 | 0.3689 | 1.3838 |
| 16 | 0.2542 | 0.5116 | 0.3833 | 1.5212 |
| 32 | 0.2906 | 0.4945 | 0.3807 | 1.6719 |
| 64 | 0.4027 | 0.4865 | 0.4824 | 1.9143 |
| 128 | 0.4832 | 0.6520 | 0.6212 | 2.1899 |

At 8 and 32 expert tokens, fixed explicit FP16 is approximately 32% and 31% slower than native. Pooled medians improve over the original CubeCL control, but the 8-token improvement was not consistent across run orders: the first order was roughly tied and the second improved. None of the CMMA fused configurations beat native. Its best-per-case values above are exploratory minima, not independently confirmed wins.

All warmup and measured outputs passed the fixed error limits. Full errors, p95, configuration names and per-order ratios are preserved in `comparison.json`. The explicit FP16 path requires 3.56x GGUF weight storage. Fused variants require 1.33x for packed nibbles and expanded metadata, plus small shared-memory tiles; this is representation size, not peak memory.

Explicit FP16 bypasses fusion scheduling as well as choosing a tile, so the speed difference is not a pure tiling measurement. Fused CMMA converts input and unpacked weight tiles to FP16, accumulates in FP32, and writes FP32. It does not create a persistent FP16 weight matrix. These are single-stage kernels without asynchronous loading or double buffering.

Native is the existing build with the RDNA 3.5 predicate overlay; source/archive hashes are recorded. This dense driver does not exercise expert routing. The later routed benchmark measures the grouped operator with that heuristic active. Neither benchmark is full-model inference.

Clocks/thermals were not pinned and run-order variation is visible. Numerical tolerances establish operator agreement, not exact logits/probability parity.

## Validation

- Every output checked across 100 measured invocations per case/variant.
- GPU tests cover partial M/N tiles, distinct output features, and all fused configurations.
- HIP clippy with warnings denied and Rust formatting checks passed.
- An exploratory non-fusion autotuning path attempted unsupported HIP tensor-map operations and failed numerical checks; it was excluded, not used as a timing result. Explicit configurations and the original fusion-enabled control passed.

See [the operator guide](../Q4K.md) for commands. This run used `--variants` from `run.json`; initial selection, sources and binary identities are saved alongside the samples.
