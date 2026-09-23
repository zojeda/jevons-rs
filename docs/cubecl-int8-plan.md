# CubeCL int8 plan

[Back to the CubeCL backend guide](cubecl.md)

Goal: faster prompt prefill on RDNA3 by multiplying 8-bit activations with the quantized weights
using integer instructions, as llama.cpp does, instead of converting weight tiles to FP16.

Today CubeCL 0.10's HIP backend exposes only FP16/BF16 matrix instructions, and its `dot` compiles
to scalar multiplies (`examples/dot4_probe.rs`). Prefill is dominated by the routed-expert
products, which are limited by the weight-to-FP16 conversion.

## Constraints

- No unsafe code, no FFI. Keep checked kernel launches.
- Exact prompt-cache reuse must keep working: activation quantization must be per row and
  deterministic, independent of the chunk and tile shape.
- Accuracy: answer labels must match the current backend on the Snake, JevBench and System One
  runs. Probability changes should stay within llama.cpp's spread between its flash-attention
  and non-flash-attention paths.
- The FP16 path stays, as the fallback and as a runtime option.

**Approach chosen: patch CubeCL** (a local copy wired in with `[patch.crates-io]`), rather than a
separate crate with HIP FFI.

## Phase 0: confirm the instructions

- [ ] Confirm the RDNA3 (`gfx1151`) instructions compile through hipRTC with CubeCL's compile
      flags: `__builtin_amdgcn_sdot4` (`v_dot4_i32_i8`) and
      `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`. Write a standalone HIP test file and check
      the ISA.
- [ ] Decide which to target first: `v_dot4` (simpler, VALU) or int8 WMMA (up to 2x the FP16
      WMMA rate, harder register layout). Default: `v_dot4` first, as a small proof.

## Phase 1: patch CubeCL

- [ ] Put the needed crates under `third_party/` (likely `cubecl-ir`, `cubecl-cpp`, maybe
      `cubecl-core` for the frontend) at exactly 0.10.0.
- [ ] Wire them in with `[patch.crates-io]` in the workspace `Cargo.toml`, and record the
      upstream version and a patch summary in `third_party/README.md`.
- [ ] Add a packed int8 dot-product operation (`i32 += dot4(u32, u32)`) to the IR, the HIP code
      generator and the frontend.
- [ ] Add `i8 x i8 -> i32` as an accepted type for the matrix-multiply operations on HIP RDNA3,
      with the correct register layout. Phase 3 only.
- [ ] Unit-test the new ops against CPU references, extending `dot4_probe`, and check the ISA
      contains `v_dot4` / `v_wmma_i32_16x16x16_iu8`.
- [ ] Prepare an upstream PR to tracel-ai/cubecl so the patch can eventually be dropped.

## Phase 2: int8 dot product in the expert products

- [ ] Activation quantization kernel: per row, blocks of 32 (Q8_1-like: int8 values, an f16
      scale and a sum for the min term), deterministic. Fuse it into the norm kernels that
      produce FP16 activations where possible.
- [ ] Integer inner loops for Q4_K, Q5_0, Q6_K and Q8_0 weights: unpack the nibbles into int8,
      use `dot4`, apply the block scales and mins in f32. Follow the formulas in llama.cpp's
      `vec_dot_q*_q8_1` so the math is known to be correct.
- [ ] Start with the grouped expert product (the prefill hotspot), then the dense projections.
- [ ] Kernel tests against CPU references for every weight format, plus a row-invariance test
      (bitwise identical rows across tile shapes and row counts).
- [ ] Benchmark against the FP16 kernel with `moe_bench` and `gemm_bench` at prefill sizes.
      **Go/no-go:** continue only if the expert products get at least 1.3x faster.

## Phase 3: int8 matrix instructions (if Phase 2 pays off)

- [ ] Tile kernel using `wmma_i32_16x16x16_iu8`: int8 activation tiles and int8 weight tiles
      in shared memory, i32 accumulators, block scales applied per 32-element K block.
- [ ] Add the new tile shapes to the autotuner's candidate lists (`gpu/tune.rs`) and bump
      `TUNE_VERSION`.
- [ ] Same tests and benchmarks as Phase 2.

## Phase 4: integrate into the model

- [ ] Choose FP16 or int8 through a model setting, surfaced as a CLI option
      (`--matmul fp16|int8`), with `auto` using int8 when the device supports it.
- [ ] Prefill uses int8. Decide separately for the canvas (bandwidth bound, smaller gain) based
      on measurements.
- [ ] Extend the tuning-table key with the precision mode.
- [ ] `examples/prefix_check.rs` must stay bitwise identical with int8 prefill.

## Phase 5: validate and report

- [ ] Kernel tests, `prefix_check`, and the engine reproducibility and extension tests.
- [ ] Snake prefill benchmark: interleaved in-process FP16 vs int8 comparison (`tune_ab`-style),
      plus `prefill_bench`.
- [ ] JevBench and System One corpus runs; compare labels and probabilities with the FP16
      snapshots.
- [ ] Update `docs/cubecl.md` (design notes, the "No int8 matrix path" note, tuning), the README
      benchmark tables and a new report under `benchmarks/cubecl/`.

## Risks

- Patch maintenance: every CubeCL upgrade needs the patch rebased until it is upstream.
- hipRTC may not accept a built-in the offline compiler does; Phase 0 checks this early.
- The int8 WMMA register layout on RDNA3 differs from FP16's; getting it wrong gives silently
  wrong results, so the CPU-reference tests are mandatory.
- Accuracy: 8-bit activations change probabilities; a label regression on the benchmarks would
  block making int8 the default.
