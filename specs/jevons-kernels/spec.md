# jevons-kernels

## Purpose

`jevons-kernels` holds the tuned CubeCL kernels the model runtimes share on HIP: device buffers and
the matrix product over quantized or FP16 weights. The kernels run on the same CubeCL runtime and
memory pool as Burn, so DiffusionGemma calls them on its own buffers and the Burn models call them
on the buffers of Burn tensors.

## Scope

It owns:

- `Gpu`: the compute client of one HIP device, uploads, reads, synchronization and memory release.
- `Buf`: a device allocation with a logical element count.
- The kernel cache directory and the persistent compilation cache.
- `gemm`: `QMatrix` weights, launch `Plan`s with their validity rules and default heuristic, and the
  dense, split-K, small-row and expert-grouped products.

It leaves to other crates:

- GGML block packing and dequantization to `jevons-formats`.
- Measuring launch plans per device, and every DiffusionGemma kernel (attention, routing, norms,
  vision), to `jevons-gemma4-diffusion`.
- The Burn backend extension over these products to `jevons-burn`.

Principle P1 applies: the crate forbids unsafe code.

## Requirements

### R1 A device with 32-lane waves

`Gpu::new(index)` opens HIP device `index`. It fails with an error naming the device when the device
cannot be opened, and with "The CubeCL backend requires 32-lane waves" when the device's minimum or
maximum wave size is not 32.

Tests: none yet

### R2 Compiled kernels persist across processes

The first `Gpu::new` in a process turns on CubeCL's compilation cache in `cache_dir()`:
`DIFFUSION_CUBECL_CACHE`, else `$XDG_CACHE_HOME/diffusion-cubecl`, else `~/.cache/diffusion-cubecl`,
where the home directory is `HOME` or, on Windows, `USERPROFILE`. When another runtime in the
process (Burn) configured CubeCL before it, that configuration stays.

Tests: none yet

### R3 Empty cache databases are removed

Before the cache directory is used, every empty `.db` file in it is deleted together with its `-wal`
and `-shm` files. Databases with content stay. An empty file, left by a process that stopped before
it created its database, makes every lookup fail, so nothing tuned is saved.

Tests: `empty_cache_databases_are_removed_and_others_kept`

### R4 Weight encodings

`QMatrix` holds weights `[experts * n, k]` in one of five encodings: Q4_K, Q6_K, Q5_0 and Q8_0 GGML
blocks, dequantized to FP16 inside the product, and row-major FP16. The kernel tests exercise Q8_0
and FP16.

Tests: `every_tunable_dense_plan_matches_reference_and_tiles_are_row_invariant`, `f16_matmul_matches_burn_matmul_for_canvas_and_prefill_rows`

### R5 Unsupported weights are rejected

Building a `QMatrix` fails with a message naming the type and shape when the tensor type is none of
the five encodings, when `n` or `k` is not a multiple of 64 or `k` is not a multiple of the block
size, or when the raw byte count or packed regions do not match the shape.

Tests: none yet

### R6 Products match the dequantized weights

A product computes `out[m, n] = x[m, k] · Wᵀ` from FP16 rows `x` into FP32 `out`, accumulating in
FP32. For every valid plan, each output stays within 2e-3 of the largest output magnitude when
compared with an f64 product over the dequantized weights. Outputs larger than the FP16 range come
out in full.

Tests: `every_tunable_dense_plan_matches_reference_and_tiles_are_row_invariant`, `f16_matmul_outputs_beyond_the_fp16_range_stay_exact`

### R7 Valid plans

A plan is valid for a matrix and a row count `m` when its row tile is 32, 64 or 128, its column tile
is 64 or 128 and divides `n`, and its split-K factor lies in 1..=16 and does not exceed `k / 64`.
The small-row plan (row tile 0) needs `m <= 16` and no split.

Tests: `every_tunable_dense_plan_matches_reference_and_tiles_are_row_invariant`

### R8 Row-invariant plans

A plan with a row tile and no split-K accumulates each output row in the same order whatever `m` and
the tile shape are. Across such plans the outputs are bitwise identical, and the default heuristic
picks one when a caller asks for row invariance.

Tests: `every_tunable_dense_plan_matches_reference_and_tiles_are_row_invariant`

### R9 Expert-grouped products

The grouped product multiplies rows sorted by expert: each assignment reads input row `id / in_div`,
writes output row `id` and uses its own expert's weights. Results stay within 2e-3 of the f64
reference, and the tile shape does not change any row.

Tests: `grouped_products_use_each_assignments_expert_weights`

### R10 Kernels run on Burn tensors

`Buf::from_handle` views an existing allocation, such as a Burn tensor's buffer, and
`Gpu::from_client` wraps an existing client, so the products run on Burn tensors without copies.
Views of slices of larger tensors give the same results as dense copies.

Tests: `f16_matmul_matches_burn_matmul_for_canvas_and_prefill_rows`, `f16_matmul_handles_views_of_larger_tensors`

### R11 Checked launches

Kernels launch in CubeCL's checked mode. Every global access is bounded by the real allocation size,
so a wrong logical length cannot read or write outside a buffer.

Tests: none yet

### R12 Released memory returns to the device

`release_memory` waits for queued work and returns unused pooled allocations to the device. On APUs
device memory is system memory, so a dropped model gives its memory back.

Tests: `released_buffers_return_device_memory_after_cleanup`

### R13 A release guard runs after what it follows

`ReleaseOnDrop` releases the device's memory (R12) when it is dropped. Declared as the last
field of what owns a model's buffers, it runs once they are all dropped, so nothing of a dropped
model stays reserved and a process that ends holds none of it. A device that fails then is left
as it is: the guard never panics, since a panic in a release that runs while its thread unwinds
would abort the process in the middle of a call to the driver. The guard logs what it did: the
bytes the device held reserved before and after and the time taken, or a warning with what the
caught panic said. A release that failed never reads, in the log, as one that ran.

Tests: `the_release_guard_returns_what_was_dropped_before_it`, `a_caught_panic_is_reported_with_what_it_said`
