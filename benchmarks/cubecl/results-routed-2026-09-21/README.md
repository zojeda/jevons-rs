# Grouped Q4_K gate/up comparison — 2026-09-21

The implemented CubeCL grouped prototype is slower than the patched native baseline on every tested workload in both run orders. The faster of the two tested CubeCL configurations consistently reuses weights across 8 tokens with four waves per block. Its pooled median is 9.1–24.7x the native latency. This does not justify replacing the native expert operator.

| Workload | Native p50 ms | CubeCL tile 4 p50 ms | CubeCL tile 8 p50 ms | Native / tile 8 p95 ms | Tile 8 slowdown |
| --- | --- | --- | --- | --- | --- |
| routed_balanced_128 | 3.675 | 39.041 | 33.375 | 5.682 / 39.096 | 9.08x |
| routed_skewed_128 | 2.400 | 35.380 | 31.068 | 2.897 / 38.692 | 12.95x |
| routed_balanced_512 | 5.573 | 154.467 | 137.800 | 8.381 / 152.692 | 24.73x |
| routed_skewed_512 | 7.350 | 152.297 | 135.213 | 7.740 / 158.685 | 18.40x |

## What was measured

- Real `blk.0.ffn_gate_up_exps.weight` Q4_K weights for all 128 experts; K=2816, N=1408.
- 128/512 input tokens with top-8 expert IDs. Balanced routes give 8/32 assignments per expert. Skewed routes give every token to experts 0–7, leaving 120 experts empty.
- Native calls `ggml_mul_mat_id`. CubeCL resets counts, groups IDs on the GPU, builds tile jobs from actual counts, multiplies packed weights and scatters output to token/slot positions.
- The average never caps capacity. The grouped kernel uses the tested FP32 scalar wave-reduction implementation; the separate fused CMMA implementation was measured on isolated expert slices.
- Three warmups and 50 measured invocations per run, in native/CubeCL/CubeCL/native order. Thus 100 measured samples per workload/variant, 800 CubeCL samples and 400 native samples overall.
- All compilation and tests finished before timing. Transfers and CPU verification are excluded; normal CubeCL pooled output/scratch allocations are included. GGML preallocates graph output and manages its internal temporary buffers.
- Model hash, fixture counts, run order, executable hashes, source hashes and native archive/overlay hashes are retained alongside the raw samples.

Native was linked against the original worktree's existing RDNA 3.5 overlay build. The overlay was checked to match pinned `mmq.cu` plus only the added RDNA 3.5 predicate. Unlike the earlier dense-slice benchmarks, this routed operator exercises the affected code path. No original checkout source was modified.

## Correctness, storage, and limits

Every measured output element passed the predeclared operator limits (relative RMSE <= 2%, normalized maximum error <= 5%) against the same CPU f64 reference. Maximum CubeCL relative RMSE was approximately 1.82e-7. Different arithmetic paths are used: native quantizes activations internally, whereas this CubeCL kernel uses FP32 arithmetic. These checks do not establish exact full-model logits or probability parity.

Raw Q4_K weights occupy 285,474,816 bytes. CubeCL prepared nibbles/metadata occupy 380,633,088 bytes (1.33x), plus activation/output/routing scratch and runtime memory. It does not hold an expanded FP16 weight cache. These numbers describe payload representation, not measured peak memory.

Clocks and thermals were not pinned. Native timings vary substantially between the first and last run; the comparison file preserves those per-order ratios. CubeCL loses every workload in both orders, so the direction of the result is consistent despite that variation.

This is the routed gate/up linear operator only. Router logits/top-k selection, activation, down projection, expert combination, attention, and the rest of the model are excluded. Routes and activations are synthetic, not captured from model execution. There is still no full-model CubeCL prefill or request-level speedup. The user-reported 14.1% native full-request improvement is not remeasured here.

## Validation

- Seven CubeCL tests passed, including GPU checks for partial tiles, skewed/empty experts, and repeated route-count resets.
- Real-weight smoke checks and every measured output check passed for native and CubeCL.
- HIP clippy with warnings denied, formatting, and diff whitespace checks passed.
- Regular workspace tests: 20 passed, four model-dependent tests ignored. The service inference implementation was not changed in this experiment; native model reproducibility was validated in the earlier implementation slice.

## Decision

Explicit FP16 tiles improved some cases versus the original CubeCL control but remained slower than native on the primary small-expert workloads and retained the expanded-weight cost. Neither isolated fused implementation nor this grouped implementation produced a native performance win. Keep the current native engine; any further CubeCL work should begin with profiling these kernels and a bounded kernel-level improvement, not a claim that the full engine migration is already faster.

See [the routed guide](../ROUTED.md) for reproduction and [the isolated confirmation](../results-expert-cmma-2026-09-21/README.md) for the FP16/CMMA results. Weight/input fixtures remain in the ignored local results directory.
