# jevons-burn

## Purpose

`jevons-burn` is the Burn 0.22 runtime that Nemotron-Labs-Diffusion and Parakeet TDT run on. It
opens the HIP device with a persistent kernel and autotune cache, streams checkpoint weights to the
device, and provides the transformer building blocks: RMS norm, rotary embedding, grouped-query
attention over a resident KV cache, the gated MLP and greedy reads. FP16 products go through the
tuned GEMM of `jevons-kernels`, registered as a Burn backend extension.

## Scope

It owns `device` (HIP device and cache directory), `weights` (the safetensors `Loader`), `layers`
(the building blocks) and `kernels` (the `f16_matmul` extension, `f16_linear` and
`f16_linear_scaled`).

It leaves to other crates:

- The GEMM kernel and empty-cache cleanup to `jevons-kernels`.
- Safetensors parsing to `jevons-formats`.
- Model graphs, configurations and parity with reference implementations to
  `jevons-nemotron-diffusion` and `jevons-parakeet`.

Numerics: the residual stream and norms are f32. Weights for the tuned GEMM, matmul inputs,
attention inputs and the KV cache are FP16. Principle P1 applies: the crate forbids unsafe code.

## Requirements

### R1 HIP device and cache

`device::hip(index)` returns Burn's ROCm device `index`. The first call in a process points CubeCL's
compiled-kernel and autotune cache at `JEVONS_BURN_CACHE`, else `~/.cache/jevons-burn` (home is
`HOME`, or `USERPROFILE` on Windows), after removing empty cache databases there (`jevons-kernels`
R3). A configuration that another runtime set before it stays.

Tests: none yet

### R2 Weights must have the declared shape and type

The loader reads a tensor when its shape equals the requested shape. A mismatch fails with an error
naming the tensor, the expected shape and the found shape. BF16 matrices (`matrix`, `stacked`) must
be BF16; f32 reads accept BF16, FP16 and F32. Any other dtype fails with an error naming the tensor.

Tests: none yet

### R3 Weights stay resident

Each weight matrix is read from the checkpoint and uploaded on its own, as a persistent device
allocation, so host memory holds one matrix (or one fused stack of matrices) at a time.

Tests: none yet

### R4 FP16 weights for the tuned GEMM

`stacked_f16` converts BF16, FP16 or F32 matrices to FP16, stacks them by rows and appends zero rows
up to the requested multiple. On real Nemotron projection weights, products with the FP16 copy stay
within 1e-3 of the largest output of the BF16 product.

Tests: `fp16_conversion_preserves_real_projection_weights`

### R5 Linear products

`linear(x, w)` returns `x · wᵀ` in f32 for weights stored `[out, in]`. FP16 weights on a CubeCL
device use the tuned GEMM; other weights, and the CPU device, use Burn's matmul in the weight's
dtype. Inputs to `linear` must fit the FP16 range. `linear_unbounded` divides each row whose largest
magnitude exceeds 16384 by `max / 16384` and multiplies its output back.

Tests: `f16_matmul_matches_burn_matmul_for_canvas_and_prefill_rows`, `gated_mlp_with_fp16_weights_matches_f32`

### R6 The tuned GEMM matches Burn's matmul

`f16_linear` takes FP16 weights `[N, K]` with `N` and `K` multiples of 64 and panics on other
operands. For canvas and prefill shapes (32 to 512 rows, up to 131136 columns and 14336 inner
values), each output stays within 2e-3 × max(largest output, 1) of an f32 matmul. Outputs beyond the
FP16 range stay within 1e-3 relative. Row, column and offset slices of larger tensors give results
within 2e-3 of dense copies.

Tests: `f16_matmul_matches_burn_matmul_for_canvas_and_prefill_rows`, `f16_matmul_outputs_beyond_the_fp16_range_stay_exact`, `f16_matmul_handles_views_of_larger_tensors`

### R7 Batch size does not change a row

The tuned GEMM uses fixed plans without split-K, so each output row is accumulated in the same order
whatever the number of rows in the product.

Tests: `every_tunable_dense_plan_matches_reference_and_tiles_are_row_invariant`

### R8 Gated MLP

`gated_mlp` computes `down(silu(gate(x)) * up(x))`, with `down` through `linear_unbounded`. With
FP16 weights it stays within 1e-2 of the same computation in f32.

Tests: `gated_mlp_with_fp16_weights_matches_f32`

### R9 RMS norm

`rms_norm` returns `x / sqrt(mean(x²) + eps) * weight` per row, in f32.

Tests: `rms_norm_scales_rows_to_unit_rms_times_weight`

### R10 Rotary embedding

`rotate_half` rotates `[rows, heads, head_dim]` with `cos` and `sin` rows of width `head_dim` whose
frequencies repeat across both halves: element `i` pairs with element `i + head_dim / 2`.

Tests: `rotate_half_rotates_pairs_across_the_two_halves`

### R11 Grouped-query attention

`grouped_attention` attends queries `[rows, heads, head_dim]` over keys and values
`[1, kv_heads, len, head_dim]`, query head `h` using KV head `h / (heads / kv_heads)`, and returns
`[rows, heads * head_dim]`. An `attention_mask` hides key `j` from row `i` where its predicate is
true; without a mask every query sees every key. With and without a causal mask, results stay within
2e-2 of f64 attention (FP16 inputs). Mismatched head dimensions, a head count that is not a multiple
of the KV head count, or a mask of the wrong shape panic.

Tests: `grouped_attention_matches_naive_attention_with_and_without_causality`

### R12 KV cache

A `KvCache` holds FP16 keys and values `[1, kv_heads, capacity, head_dim]`, zero at creation.
`write(start, keys, values)` stores rows at positions `start..start + rows`, and `view(len)` returns
positions `0..len`.

Tests: `kv_cache_writes_positions_and_views_a_prefix`

### R13 Greedy reads

`greedy` returns each row's argmax and that column's softmax probability, reading two values per row
back from the device.

Tests: `greedy_reads_each_rows_argmax_and_its_probability`

### R14 CPU device for tests

The `flex` feature enables Burn's CPU device. The building blocks run there with Burn's own matmul,
so model crates test their layers without a GPU.

Tests: `rms_norm_scales_rows_to_unit_rms_times_weight`, `kv_cache_writes_positions_and_views_a_prefix`

### R15 A release guard returns a dropped model's memory

`device::ReleaseOnDrop` syncs the device, returns its unused pooled memory and syncs again when
it is dropped. Burn queues a dropped tensor's release, so the first sync is what lets the
cleanup free its pages. Declared as a model's last field it runs once the model's tensors are
dropped: nothing of a dropped model stays reserved (on APUs device memory is system memory), and
a process that ends holds none of it. A device that fails then is left as it is: the guard never
panics. The cleanup runs also after a sync that failed. The guard logs what it did: the bytes
the device held reserved before and after and the time taken, or a warning with the sync's error
or what the caught panic said (`jevons-kernels` R13).

Tests: none yet
