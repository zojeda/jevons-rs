# Small expert diagnostic — 2026-09-21

This extends the existing Q4_K operator comparison to 4/8/16/32/64/128 tokens per expert. It changes benchmark coverage, not inference kernels. The same release binaries as the preceding comparison were reused; hashes are recorded in `run.json`.

Each row uses expert 0 of `blk.0.ffn_gate_up_exps.weight`, K=2816, N=1408, on the local gfx1151 GPU. Fifty measured invocations per run, three warmups, native/CubeCL/CubeCL/native order: 100 measured samples per case and variant. Inputs are seeded synthetic activations; every output is checked against the same CPU reference.

| Tokens per expert | Native Q4_K p50 ms | CubeCL unpack + FP16 p50 ms | CubeCL cached FP16 p50 ms | Native / cached FP16 p95 ms |
| --- | --- | --- | --- | --- |
| 4 | 0.2366 | 1.2290 | 0.4439 | 0.3270 / 0.7202 |
| 8 | 0.2693 | 1.3342 | 0.5382 | 0.3664 / 0.7263 |
| 16 | 0.2638 | 1.3117 | 0.5414 | 0.3996 / 0.7537 |
| 32 | 0.2939 | 1.3111 | 0.4990 | 0.4069 / 0.7474 |
| 64 | 0.4145 | 1.3208 | 0.4885 | 0.5282 / 0.7799 |
| 128 | 0.5278 | 1.4304 | 0.6141 | 0.7376 / 0.8710 |

Cached FP16 is approximately 2.00x slower at 8 tokens and 1.70x slower at 32 tokens, despite using 3.56x the packed weight storage. Per-call unpack plus FP16 matmul is approximately 4.95x slower at 8 tokens. All variants passed the previously defined operator error thresholds; GPU weight dequantization matched the CPU reference exactly. This is not exact model-output parity.

There is substantial run-order variation; clocks and thermals were not pinned. Cached FP16 lost the 8- and 32-token cases in both orders. The 128-token case changed from losing in one order to a slight win in the other. Raw samples and per-order ratios are retained in this directory. Do not interpret differences from the earlier report as a kernel regression or improvement: no kernel was changed.

The native driver uses dense `ggml_mul_mat`, not routed `ggml_mul_mat_id`. It uses this worktree's original pinned build without the other worktree's RDNA 3.5 overlay. This does not measure that patch or a complete MoE layer; it only supplies realistic small dimensions for the next kernel experiment. Both implementations already receive the individual expert token count.

Next: compare explicit small-M configurations against existing autotuning, then test fused Q4_K unpack/matmul with packed persistent weights. Validate the winning candidate with all experts, balanced/skewed routing, and the optimized native routed baseline. See [experiment details](../Q4K.md#small-expert-workloads).

## Reproduce

Use the small-expert export command in the linked guide, followed by the HIP environment and:

```bash
python3 benchmarks/cubecl/run_matmul.py /tmp/cubecl-native-matmul \
  "$CARGO_TARGET_DIR/release/examples/q4k_bench" \
  benchmarks/results/cubecl-small-experts \
  benchmarks/cubecl/results-small-experts-new --rounds 50
```

Weight/input/reference fixtures remain in the ignored `benchmarks/results/` directory. This report contains metadata and timings only.
