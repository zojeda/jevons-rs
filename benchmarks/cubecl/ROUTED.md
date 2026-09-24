# Routed expert comparison

> Historical record. The scripts and examples these commands use (llama.cpp matmul builds,
> `export_*.py`, `run_matmul.py`, `q4k_bench`, `routed_bench`) were removed with the llama.cpp
> backend; reproducing needs a checkout from before 2026-09-24.

See the [initial comparison](results-routed-2026-09-21/README.md) and the
[profile-guided loop optimization](results-routed-loop-2026-09-21/README.md).

This is the grouped gate/up matrix product for all 128 experts, with top-8 IDs
already selected. The native driver executes `ggml_mul_mat_id` using the RDNA 3.5
MMQ overlay. CubeCL runs count reset, atomic grouping of assignments, construction
of token-tile jobs, fused Q4_K multiplication, and scatter to `[token, slot, output]`.
Inputs and expert IDs are resident on the GPU before timing for both drivers.
Router logits/top-k selection, activation, down projection, and expert-output
combination are excluded for both. This is not a full MoE layer or model prefill.

The two CubeCL configurations reuse unpacked weights over 4 or 8 tokens, with four
32-lane waves per block. They use FP32 arithmetic and packed nibbles with expanded
scale/min metadata; no full float weight matrix is materialized. The matrix-instruction
fused kernel was measured separately on isolated expert slices. These grouped
configurations use the scalar wave-reduction kernel.

The current reduction resolves routing indices once per tile and unrolls the eight
subgroups of each Q4_K block. Inactive token slots read the valid first input row,
but their sums are never scattered. This removes per-token routing branches from
the reduction without changing which assignments produce outputs. Bounds-checked
launches remain enabled. No full floating-point weight cache is introduced.

Grouping uses actual counts. Each expert can receive all input tokens, and empty
experts produce no jobs. Distinct IDs per token are validated at upload, making a
capacity of `T` assignments per expert sufficient. The dispatch uses the safe bound
`ceil(T * top_k / tile_tokens) + expert_count`; blocks beyond the actual job count
exit. This avoids a host readback of the GPU-generated job count. The average
assignment count is never used as a capacity limit. Repeated calls reset counts;
the ignored GPU test checks skew, empty experts, partial tiles and repeated calls.

The synthetic workloads contain 128 and 512 tokens, each with two route patterns:

- Balanced: every expert receives 8 or 32 tokens, respectively.
- Skewed: every token selects experts 0 through 7, so those experts each receive
  128 or 512 tokens and the remaining 120 experts are empty.

Weights are all real Q4_K blocks from `blk.0.ffn_gate_up_exps.weight`. Signed seeded
activations are identical between drivers. Every measured output is checked
against pinned GGUF dequantization and CPU f64 multiplication, stored as f32.
The predeclared limits remain relative RMSE <= 2% and normalized maximum error <=
5%. Passing these does not establish exact model logits/probabilities.

## Reproduce

Use the HIP environment in [README](README.md#local-prerequisites).

```bash
OPENBLAS_NUM_THREADS=8 uv run --no-project --with numpy==2.2.6 --with pyyaml==6.0.3 \
  python benchmarks/cubecl/export_routed.py "$DIFFUSION_MODEL" \
  benchmarks/results/cubecl-routed
cargo build --release --locked -p jevons-cubecl --features hip --example routed_bench
python3 benchmarks/cubecl/build_native_matmul.py /path/to/patched/native/out/build \
  /tmp/cubecl-native-routed-rdna35
python3 benchmarks/cubecl/run_matmul.py /tmp/cubecl-native-routed-rdna35 \
  "$CARGO_TARGET_DIR/release/examples/routed_bench" \
  benchmarks/results/cubecl-routed benchmarks/cubecl/results-routed-new --rounds 50
```

The fixture output directory must be new and remain ignored: it contains model
weights. Finish compilation before timing. The runner uses native/CubeCL/CubeCL/native
order, with three warmups and raw samples retained. Weight upload, output readback,
and CPU validation are outside timing. CubeCL's normal pooled allocation calls for
output and routing scratch remain timed; GGML preallocates graph output and manages
its own internal temporary buffers. Both include GPU grouping on every operation.

For a before/after experiment, preserve the old release executable before rebuilding
and pass `--cubecl-before /path/to/preserved/routed_bench` to `run_matmul.py`. The order
becomes native/before/after/after/before/native. `comparison.json` includes pooled
and per-order before/after ratios alongside the native comparison.

On this WSL host, ROCm's profiler cannot discover the GPU through KFD. CubeCL's
`CUBECL_DEBUG_LOG=/tmp/profile.txt CUBECL_DEBUG_OPTION=profile-full` fallback records
host wall time around synchronized launches, not hardware event or counter timing.
Run profiling separately from the unprofiled benchmark, redirect driver stdout to
a JSON file, then use `summarize_routed_profile.py /tmp/profile.txt /tmp/driver.json`
to discard the three warmups and summarize individual launch times. Profiling adds
synchronization overhead; do not subtract those times from unprofiled measurements.

The scalar and CMMA [isolated experiments](Q4K.md) cannot be multiplied by expert
count to predict this grouped operator. Neither operator timing can be presented
as a measured request-level speedup. Full-model routing traces are still pending;
these balanced/skewed fixtures deliberately test controlled bounds instead.
