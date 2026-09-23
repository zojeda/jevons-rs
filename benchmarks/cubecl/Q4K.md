# Q4_K operator comparison

This compares real local GGUF weights through pinned GGML/HIP and a new safe Rust
Burn/CubeCL implementation. It measures matrix products needed during prefill;
the CubeCL implementation does not yet run a complete DiffusionGemma forward.

See the [measured comparison](results-q4k-2026-09-21/README.md) for medians,
tail latency, numerical differences and memory costs.

## Matched workload

The fixture contains `blk.0.attn_q.weight` (2816 inputs, 4096 outputs) and expert 0
of `blk.0.ffn_gate_up_exps.weight` (2816 inputs, 1408 combined gate/up outputs).
Attention uses 128/512 tokens; the expert slice uses 32/128 tokens. It does not
exercise routing, grouping, scatter/gather, or all 128 experts. Synthetic signed
activations are generated once with fixed seeds and supplied byte-for-byte to both
drivers. Weights come from the same Q4_K blocks, not separately requantized models.

The CPU reference uses the pinned GGUF reader's dequantizer and f64 NumPy matrix
multiplication. Every output element is checked after every timed invocation.
Before measurement we fixed the acceptance thresholds at relative RMSE <= 2% and
maximum absolute error / maximum absolute reference <= 5%; these are operator
experiment tolerances, not model probability acceptance criteria. All dequantized
weight values in the CubeCL path are also compared to the reference (normalized
maximum error <= 1e-6).

Native GGML/HIP uses its normal Q4_K matrix path. CubeCL variants are:

| Variant | Timed work | Persistent weight representation |
| --- | --- | --- |
| `dequant_each_matmul` | GPU unpack to FP32, then matmul | Packed nibbles and expanded scale/min metadata |
| `cached_f32_matmul` | FP32 matmul | FP32 expanded weights |
| `dequant_each_f16_matmul` | GPU unpack, FP16 weight conversion, input FP16 conversion, matmul, FP32 output conversion | Packed nibbles and expanded scale/min metadata |
| `cached_f16_matmul` | Input FP16 conversion, FP16 matmul, FP32 output conversion | FP16 expanded weights |

Scale/min metadata is prepared on the CPU once at load, while nibbles remain U8
on the device until GPU unpack. This prepared packed payload is 192 bytes per
256 weights versus GGUF's 144 bytes. FP16 expansion costs 3.56 times the original
Q4_K storage; FP32 costs 7.11 times. Cached variants are performance/memory
experiments and do not establish that all model weights should be expanded.
The driver retains reference/cache tensors for comparison, so representation byte
counts in reports are not measurements of peak process or device memory.

Both drivers time synchronized host execution with inputs resident on the GPU.
Upload, output readback, CPU correctness checks and model-file parsing are outside
the measured region. Three warmup executions precede each case/variant; upload,
first unpack and warmup timings are reported separately. These are not a complete
measurement of process startup or cache preparation. Identical resident weight slices are reused
across repetitions, which differs from cycling through a full model's weights.
Different arithmetic paths may have different numerical errors; the report includes
those errors alongside speed and memory.

## Reproduce

Use the [HIP environment](README.md#local-prerequisites). Export only into the
ignored `benchmarks/results/` directory: these fixtures contain model weights and
must stay out of commits. Python requires NumPy and PyYAML in an isolated environment.

```bash
OPENBLAS_NUM_THREADS=8 uv run --no-project --with numpy==2.2.6 --with pyyaml==6.0.3 \
  python benchmarks/cubecl/export_matmul.py "$DIFFUSION_MODEL" \
  benchmarks/results/cubecl-matmul

cargo build --release --locked -p jevons-cubecl --features hip --example q4k_bench

# Use this worktree's HIP OUT_DIR/build containing cargo-link.txt.
python3 benchmarks/cubecl/build_native_matmul.py /path/to/native/out/build \
  /tmp/cubecl-native-matmul

# Finish all compilation and other inference before timing. Output must be new.
python3 benchmarks/cubecl/run_matmul.py /tmp/cubecl-native-matmul \
  "$CARGO_TARGET_DIR/release/examples/q4k_bench" \
  benchmarks/results/cubecl-matmul benchmarks/cubecl/results-q4k --rounds 50
```

The runner executes native/CubeCL/CubeCL/native sequentially to expose run-order
variation. It saves raw measurements, the fixture manifest and binary hashes,
then produces per-case p50/p95, speed ratios and numerical errors. A ratio greater
than one means CubeCL is faster. Source weights and input/output matrices remain
in the ignored fixture directory; published results contain only metadata and
aggregate errors.

The baseline is this worktree's pinned native implementation, without the original
checkout's subsequent MMQ heuristic overlay. Compare any later native optimization
as a separately named baseline. Operator speedups cannot be multiplied into the
whole-request timing or presented as a measured end-to-end CubeCL speedup.

## Small expert workloads

See the [measured small-expert sweep](results-small-experts-2026-09-21/README.md).

For 128 experts with top-8 routing, the average assignment count per expert is
`T * 8 / 128`; its ceiling is a tile-selection estimate, not a bound on actual
assignments. A 128-token prefill averages 8 tokens/expert and a 512-token prefill
averages 32. The original 32/128-token expert cases missed the smaller example.
Generate a wider diagnostic sweep with:

```bash
OPENBLAS_NUM_THREADS=8 uv run --no-project --with numpy==2.2.6 --with pyyaml==6.0.3 \
  python benchmarks/cubecl/export_matmul.py "$DIFFUSION_MODEL" \
  benchmarks/results/cubecl-small-experts --expert-only \
  --expert-tokens 4 8 16 32 64 128
```

Use the same paired runner above with the new fixture and a fresh output directory.
These remain individual dense expert slices. The native driver calls
`ggml_mul_mat`, not routed `ggml_mul_mat_id`, so this sweep does not reproduce or
measure the RDNA 3.5 routed tile-selection patch. CubeCL already receives the
individual expert token count as the matrix's M dimension. Burn 0.21's default
features also enable matmul autotuning; enabling it again is not an optimization.

The next kernel experiment should separate two questions:

1. Can explicitly selected small-M tiles/strategies beat the existing tuned FP16
   path with exactly the same conversions, inputs and persistent representation?
   Record chosen configurations, warmup separately, and p50/p95 by M. This isolates
   tiling but retains the 3.56x FP16 weight-storage cost.
2. Can a fused Q4_K dequantization/matmul keep weights packed and remove the full
   intermediate float matrix? Test small-M configurations and include dequantization
   in every timed operation. Quantization format compatibility must be implemented
   explicitly; generic low-bit support does not establish GGUF Q4_K support.

A routed benchmark is required before either result motivates an engine port.
Use all 128 experts, top-8 assignments, balanced and skewed counts (including empty
experts), then actual model routing traces. Compare grouped execution against
GGML's routed operator with the isolated RDNA 3.5 patch enabled. Include dispatch,
gather/scatter and the same precision conversions in each measured scope; do not
multiply isolated expert timings by 128. The average must never truncate or cap
an expert's actual assignments. Retain the existing numerical checks, then require
model-level logits/probability validation before claiming request-level parity.

The controlled balanced/skewed routed gate/up comparison is now implemented;
see the [routed operator guide](ROUTED.md). Captured model routing traces remain pending.

## Explicit tiles and fused Q4_K prototype

Measured results: [initial explicit/scalar sweep](results-expert-tune-2026-09-21/README.md)
and [FP16 confirmation / fused CMMA sweep](results-expert-cmma-2026-09-21/README.md).

`expert_tune` keeps the original fusion-enabled cached-FP16 operation as its
control. It tests explicit CubeK single-stage CMMA configurations with a
16x16x16 matrix instruction, varying partition sizes and planes per block. Input
F32-to-F16 and output F16-to-F32 conversions remain inside the measured operation.
The explicit path bypasses Burn's fusion scheduler as well as autotuning, so a
speed difference from the control cannot be attributed entirely to tile selection.
Unsupported configurations are reported rather than silently substituted.

The custom checked CubeCL kernel keeps Q4_K nibbles packed. One 32-lane wave
handles an output feature; each lane processes a different reduction element.
Unpacked weights are reused across 1/2/4/8 token rows, with 4/8 waves per block.
Inputs, accumulation and outputs are FP32. This first fused kernel uses scalar
arithmetic and wave reductions, not matrix instructions. It does not materialize
a full float weight matrix or require an FP16 weight cache. Scale/min metadata is
still expanded once at upload (192 bytes per block including nibbles, versus 144
bytes in the GGUF representation). It currently requires a 32-lane HIP device.

Run with the same fixtures and paired runner:

```bash
cargo build --release --locked -p jevons-cubecl --features hip --example expert_tune
python3 benchmarks/cubecl/run_matmul.py /tmp/cubecl-native-matmul-rdna35 \
  "$CARGO_TARGET_DIR/release/examples/expert_tune" \
  benchmarks/results/cubecl-small-experts benchmarks/cubecl/results-expert-tune-new \
  --rounds 50
```

The native binary above should be linked to the separately identified build with
the RDNA 3.5 overlay. For dense expert slices the routed heuristic is not exercised;
a routed comparison is still required to assess its benefit. Finish all compilation
before timing. Treat a per-shape minimum across candidate configurations as a tuning
result; validate a selected fixed configuration in a fresh run before claiming a win.

The model-free ignored HIP test `fused_tiles_preserve_partial_token_and_output_rows`
checks partial M/N tiles and every fused token/wave configuration against a scalar
reference. Real-weight runs check every output for every warmup and measured
invocation. Incorrect outputs abort the run. Handwritten unsafe Rust remains
forbidden in this crate.

The `fused_cmma_*` variants stage 16 token rows and 16 output features per wave,
with 2/4/8 waves per block and reduction stages of 32/64 elements. Input conversion
and Q4_K unpacking occur in the same kernel; small FP16 tiles live only in shared
memory. Matrix instructions accumulate in FP32 and the output remains FP32.
`fused_cmma_preserves_partial_tiles` checks these configurations with distinct
output features and partial token/output tiles. No FP16 persistent weight cache is
needed. These remain single-stage prototypes without asynchronous loading or
double buffering.

For a fresh confirmation of configurations chosen in an earlier run, the runner
accepts `--variants cached_f16_matmul,tile_1_2_2_planes2` (or other exact names).
It records the filter and forwards it only to `expert_tune`; native still receives
the identical fixtures. An optional third positional argument provides the same
comma-separated filter when invoking `expert_tune` directly.
