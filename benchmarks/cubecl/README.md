# CubeCL feasibility and prefill baseline

> Update 2026-09-23: the full model, including the vision encoder, now runs on CubeCL, and the
> llama.cpp backend and submodule have been removed. See the [backend guide](../../docs/cubecl.md)
> and the [end-to-end report](results-model-2026-09-23/README.md). The sections below document the
> earlier feasibility slices; commands that build llama.cpp or `llama-diffusion-sys`, and the
> Python helper scripts, need a checkout from before the removal.

This is the first implementation slice of the [migration plan](../../docs/cubecl-migration-plan.md).
The service still uses llama.cpp. The new `jevons-cubecl` crate runs a real HIP
compute probe; it does not yet implement GGUF loading or DiffusionGemma inference.
`jevons-core` holds shared Rust types, so `jevons-system-one` now builds without native
inference libraries. The low-level engine adapter remains future work.

The next slice adds [Q4_K GPU unpacking and a matched native/CubeCL operator
comparison](Q4K.md), using actual attention and expert weights. It includes FP32
and FP16 experiments and checks every output value. This still does not execute
the full model on CubeCL.

The experiment is based on commit `0e3a62f`, not the original checkout's subsequent
prefill/cache optimizations. Its native baseline recomputes the complete prompt at
every prefill call. Comparing it to the sibling checkout requires recording that
distinction. No speedup from CubeCL is claimed by the float probe.

## Local prerequisites

The optional HIP probe uses Burn 0.21.0 and CubeCL 0.10.0, locked in `Cargo.lock`.
`jevons-cubecl` declares Rust 1.92, including in full-workspace builds; the
existing crates retain Rust 1.88. Checks here used Rust 1.98.1. The HIP feature is opt-in and is not required to test the
protocol crate or run the ordinary CPU workspace checks.

Use a dedicated build directory to avoid replacing binaries in the other worktree:

```bash
export CARGO_TARGET_DIR=/dev/shm/cargo_target_cubecl
export ROCM_PATH=/opt/rocm-7.2.1
export HIP_PATH="$ROCM_PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export HSA_ENABLE_DXG_DETECTION=1
```

## HIP correctness probe

```bash
cargo run --locked -p jevons-cubecl --features hip --example hip_probe -- 5 0
```

The arguments are rounds and device index. The probe checks device upload/readback,
elementwise arithmetic, reduction, and float matrix multiplication. A small matrix
is checked exhaustively; larger matrices check 64 distributed outputs against CPU
f64 accumulation on every iteration. Nonfinite or out-of-tolerance results fail.
The larger shapes use dimensions from the local GGUF's attention and expert
projections, but do not model quantization or expert routing.

JSON reports initial execution separately from repeated synchronized matmul time.
The latter excludes upload and readback. Debug builds are sufficient to establish
functionality; use release builds and controlled runs for performance decisions.
JIT and autotuning can make first execution take seconds. This is not a full
prefill benchmark, an allocation-failure test, or proof that all model operators
are supported.

## Checkpoint inventory

Initialize the pinned submodule, then use an isolated Python environment:

```bash
git submodule update --init --recursive
uv run --no-project --with numpy==2.2.6 --with pyyaml==6.0.3 \
  python benchmarks/cubecl/inventory.py "$DIFFUSION_MODEL" --sha256
```

The reader maps the local model read-only and exports metadata, tensor shapes,
encoding counts, and storage sizes, not weight values. `--sha256` reads the entire
file to establish checkpoint identity. The FP16 estimate covers stored tensor
elements only; caches, activations, staging and workspace add to peak memory.
The inventory script requires Python 3.11+ for streaming file hashing.

## Native prefill benchmark

Configure the [native HIP build environment](../../docs/build.md#rocmhip), including
`CMAKE_PREFIX_PATH`, `CMAKE_HIP_COMPILER`, and `AMDGPU_TARGETS`. Then:

```bash
cargo run --release --locked -p jevons-rs --features hip,native \
  --example prefill_bench -- --model "$DIFFUSION_MODEL" --rounds 20
```

Default synthetic requests vary prompt length, repeat inputs, change their
beginning/end, and shrink a long prompt. Names describe request relationships,
not actual cache hits: this baseline has no cross-request prefix reuse.
`--requests PATH` accepts a JSON array of text-only System One requests with default
inference options, compiled before timing. Use only synthetic inputs: the output
includes candidate tokens, logits and probabilities as reference data.

`snake-requests.json` was copied as a synthetic fixture from the original
checkout's `benchmarks/interop/snake-requests.json` on 2026-09-21. Only this dataset
was adopted; its modified engine, timing, and cache implementation were not.
Use `--requests benchmarks/cubecl/snake-requests.json` to reproduce that workload.

`--batch-size`, `--context-size`, `--gpu-layers`, and `--flash-attention` select the
native configuration. Changing them requires a separately labeled comparison.
The benchmark loads once, warms every case, then checks every measured result
against that case's warmup output, including token positions, initial noise,
candidate IDs, logits, probabilities, and usage. This checks repeatability, not
cross-backend equivalence. It exits with an error on any mismatch.

`Engine::prefill_profile()` reports synchronized wall time, calls, actual batches,
processed tokens, and reused tokens for the latest read. It resets before each
read, including invalid reads. It counts completed prefill calls; a failed partial
call is not included. The native path already synchronized at these boundaries,
so instrumentation adds no new GPU barriers. Existing `forward_ms` and HTTP usage
semantics remain unchanged. Logical usage may count the same prompt for multiple
samples even though it was evaluated only once.

The report contains raw samples and nearest-rank p50/p95 per case. A few rounds
are a smoke check, not reliable tail-latency evidence. Run without other inference
workloads, compare fresh-prompt work separately from reuse, and retain model hash,
build configuration, workload and run order with results.

Recorded implementation checks and measurements:
[2026-09-21 report](results-2026-09-21/README.md).

The [Q4_K operator guide](Q4K.md) covers explicit expert tiles and fused
quantized kernels. The [routed operator guide](ROUTED.md) covers all-expert gate/up
products with GPU grouping. Complete MoE layers and full-model inference remain pending.
The [grouped results](results-routed-2026-09-21/README.md) remain slower than the
patched native implementation on all tested workloads.
The [profile-guided loop change](results-routed-loop-2026-09-21/README.md) reduces
the eight-token CubeCL configuration's median latency by 10–12% against a preserved
binary in the same session. It still loses to patched native on every workload.

## Checks

```bash
cargo test --locked -p jevons-core -p jevons-system-one
cargo tree --locked -p jevons-system-one --edges normal
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo clippy --locked -p jevons-cubecl --features hip --all-targets -- -D warnings
cargo test --locked -p jevons-cubecl --features hip --lib -- --include-ignored
cargo test --locked -p jevons-engine --features hip,native \
  native_reads_preserve_reproducibility_across_requests -- --ignored --nocapture
cargo test --locked -p jevons-engine --features hip,native \
  native_extensions_average_refine_think_and_chunk -- --ignored --nocapture
```

Full workspace checks still build the native sys crate. The CubeCL probe and
`jevons-system-one` package builds have no llama.cpp dependency. GPU/model checks require
the environment and local assets above; do not use `--all-features`.
