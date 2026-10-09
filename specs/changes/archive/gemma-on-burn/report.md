# DiffusionGemma on Burn: report

[Proposal](proposal.md) · [Design](design.md) · [Tasks](tasks.md) · [Issue #5](https://github.com/zojeda/jevons-rs/issues/5)

Measured on 2026-10-08 and 09 on the benchmark machine (Ryzen AI MAX+ 395, Radeon 8060S),
natively on Windows with the HIP SDK 7.2, release builds, the Q4_K_M checkpoint. The spike's
code is on the branch `spike/gemma-burn`.

The tray app ran idle beside the per-call runs and the one-layer run, with its models loaded.
At 23:58 on the 8th, while the attention comparison ran, its window lost its GPU device and
the app quit, unloading its models. The rest of the attention comparison, the six-layer run
whose times are in the table, and the wgpu tests had the GPU to themselves.

## Recommendation

**Keep the current runtime for the decoder. Do not port it to Burn, wrapped or rewritten.**

- **Wrapped, it loses a little and gains nothing.** The same kernels behind Burn backend
  extensions give the same bits and cost 20 to 40 µs more per call: 2% of a prefill and 3% of
  a canvas forward as measured here, and about 6% of the 57 ms canvas forward of
  `benchmarks/README.md`. Every kernel, the quantized weights, the KV caches, the launch plans
  and the tuner stay as they are, so no code goes away and nothing becomes portable.
- **Rewritten on Burn's own operations, it loses more.** Norms and residual adds cost 5% to 14%
  of a forward. Attention is 1.5 times slower at canvas size and at least 8 times at prefill
  size.
  The results stop being the current runtime's bits, and the difference grows with depth.
  Whether prompt rows would still be bitwise equal across chunkings, which prefix reuse
  needs, was not checked.
- **wgpu is not a step away.** Two of the fifteen kernel tests pass on Vulkan. The matrix
  products and attention do not compile there, and the rest return wrong numbers.
- **The two runtimes already work together.** A Burn tensor's buffer goes to the kernels as it
  is (`Buf::from_handle`), which is how Nemotron uses the FP16 product. A Burn module can
  hand DiffusionGemma its rows with no port.

One part is worth its own measured change: **the vision tower**. Its weights are FP16, its
attention needs no mask, and it has no routing, which is what `jevons-burn`'s layers already
serve for Pixtral. It would replace about 800 lines of kernels and glue. This spike did not
measure it; the rule of the [proposal](proposal.md) applies to it as it stands.

## What was measured

### A product called through Burn, per call

The FP16 product through the existing extension (`f16_matmul`) against the same kernel on raw
buffers with the same plan. Medians of 25 interleaved batches of 20 calls, in µs per call. The
results are bitwise equal.

| `[m, k] x [n, k]` | Direct | Through Burn | Difference | Through Burn, outputs kept |
| --- | ---: | ---: | ---: | ---: |
| `[32, 64] x [64, 64]` | 26.4 | 57.5 | +31.1 | 53.0 |
| `[32, 2816] x [4096, 2816]` | 110.9 | 136.4 | +25.6 | 111.2 |
| `[32, 2816] x [2048, 2816]` | 122.2 | 161.6 | +39.3 | 131.3 |
| `[32, 2816] x [4224, 2816]` | 117.5 | 136.4 | +19.0 | 116.2 |
| `[32, 4096] x [2816, 4096]` | 121.4 | 149.9 | +28.5 | 122.2 |
| `[512, 2816] x [4096, 2816]` | 811.3 | 873.4 | +62.1 | 859.4 |
| `[512, 2816] x [2048, 2816]` | 453.6 | 488.4 | +34.8 | 479.2 |
| `[512, 2816] x [4224, 2816]` | 834.4 | 905.4 | +71.1 | 899.7 |
| `[512, 4096] x [2816, 4096]` | 862.1 | 1061.0 | +198.9 | 1012.0 |

- **"Through Burn"** drops each output when the next is made, as a forward does. **"Outputs
  kept"** holds all twenty until the device is done.
- **Where the cost comes from.** Burn's lazy stream registers every operation, and every
  dropped tensor as an operation of its own, and may run what is queued at each. Kept outputs
  hide most of it at 32 rows because the device stays busy either way.
- **Fusion cannot be turned off for one model.** With the `fusion` feature, `burn-cubecl`'s
  backend type is `Fusion<CubeBackend>` for the whole build, and Nemotron and Parakeet need it
  for their own glue.
- The last row's +199 µs is not explained.

Burn's own RMS norm plus a residual add over 2816 columns costs 88 µs at 32 rows and 187 µs at
512. The current path does four norms, the sum over experts, the add and the scale in one
kernel launch.

### Layers of the checkpoint on three arms

The model was loaded as a prefix of its layers, so the reference is the shipped forward over
the same weights in the same process.

- **Current:** `Model::forward` as it ships.
- **Wrapped:** the same forward over Burn tensors, each of the 16 kernels of a layer as a
  backend extension with the tuned plan and the quantized weight passed as arguments.
- **Burn glue:** as wrapped, with `post_attention` and `post_ffn` replaced by Burn's norms,
  adds and casts.

Per forward, medians of 15 interleaved batches of 8 forwards.

| Layers | Rows | Current | Wrapped | Burn glue |
| --- | --- | ---: | ---: | ---: |
| 0 (sliding) | prefill, 512 | 16.31 ms | 16.54 ms, +1.4% | 17.57 ms, +7.8% |
| 0 (sliding) | canvas, 32 | 2.94 ms | 3.04 ms, +3.5% | 3.36 ms, +14.3% |
| 0 to 5 (five sliding, one full) | prefill, 512 | 111.82 ms | 113.94 ms, +1.9% | 120.43 ms, +7.7% |
| 0 to 5 | canvas, 32 | 21.91 ms | 22.57 ms, +3.0% | 23.01 ms, +5.0% |

- **Wrapped, bits:** the residual rows, the next layer's input rows and the cached keys are
  bitwise equal to the current path's, after one layer and after six, at both sizes.
- **Wrapped, time:** 110 µs more per layer at canvas size and 230 to 350 µs at prefill size.
  That is what the per-call table predicts for a layer's 16 calls. The layer test alone
  would not show it: the wrapped arm is slower in 10 of 15 rounds at each size, and one batch
  varies more than the difference (the current path's tenth and ninetieth percentiles over six
  layers are 107.9 and 128.7 ms at prefill, 21.6 and 31.2 ms at canvas).
- **Projected to 30 layers:** about 3.3 ms on a canvas forward and 7 to 11 ms on a prefill of
  512 rows. Against the 57 ms and 688 ms of `benchmarks/README.md`, that is about 6% and 1 to
  1.5%, if the cost per call is the same there.
- **Burn glue, bits:** the residual rows differ from the current path's by 4.5e-5 of their
  largest value after one layer and 2.7e-2 after six, where 162 of 512 prefill rows differ by
  more than 1e-3. A likely cause, not checked: the norms' outputs are rounded to FP16 for the
  products, and routing picks experts by rank, so a small difference can change a pick.
- **Burn glue, time:** slower in 12 or 13 of 15 rounds.

These absolute times are slower than the WSL figures in `benchmarks/README.md` (3.7 ms per
layer at canvas size here, 1.9 ms there). The arms are compared with each other, not with
those.

### Burn's attention against the visibility kernel

`jevons_burn::layers::grouped_attention` with a materialized mask against `flash_attention`,
which works visibility out per score from the positions. Synthetic rows, 512 prompt positions,
the two layer kinds. In ms per call; the two agree to within 3e-3 of the largest value.

| Layer kind | Rows x keys | Kernel | Burn | Building the mask, once |
| --- | --- | ---: | ---: | ---: |
| sliding (8 KV heads, head 256) | prefill, 512 x 512 | 0.74 | 5.79 | 68.8 |
| sliding | canvas, 32 x 544 | 0.26 | 0.42 | none needed |
| full (2 KV heads, head 512) | prefill, 512 x 512 | 0.95 | 70.91 | 10.2 |
| full | canvas, 32 x 544 | 0.23 | 0.36 | none needed |

- The 70.91 ms of the full layers at prefill size is not explained. With a head of 512, Burn's
  attention may leave the path it takes for 256; the spike did not look.
- The spike did not check whether Burn's attention keeps a row's bits across chunkings.

## The issue's questions

1. **Quantized products as Burn extensions.** Yes. An extension takes the `QMatrix` and the
   `Plan` as ordinary arguments, so the weights need not be Burn tensors. The spike's
   `g4_matmul` and `g4_grouped` are a few dozen lines together. Fusion treats them as opaque
   and adds the per-call cost above.
2. **Mixture of experts.** Routing, grouping and the grouped products wrap as they are, with
   the same bits. In Burn's own operations, routing needs a softmax, a top-k with the kernel's
   tie rule and a renormalization, and the grouped products have no counterpart at all.
3. **Attention and cache.** `KvCache` holds keys and values as `[1, kv_heads, capacity,
   head_dim]` and copies each write in with `slice_assign`; DiffusionGemma keeps values
   transposed and writes both in place from the kernel that also normalizes and rotates. The
   visibility rules fit a mask, at the cost in the table above.
4. **Vision tower.** Comparable to Pixtral: FP16 weights, no mask, no routing. It adds per-head
   Q/K norms, a weightless V norm, two-dimensional rotary positions, 3x3 pooling and a
   standardization. Not built here.
5. **Speed.** The tables above. No arm is faster than the current path anywhere. The whole
   requests of `benchmarks/README.md` were not rerun: no arm earned it.
6. **wgpu.** Below.

## What a full port needs

| Piece | Today | As Burn extensions | Effort |
| --- | --- | --- | --- |
| Dense and expert-grouped products over Q4_K, Q6_K, Q5_0, Q8_0 and FP16 | `jevons-kernels::gemm` | Done in the spike | none |
| Token embedding with the first norm | `embed_q6k` | Done for token rows; image rows are one more | hours |
| Q/K/V preparation, attention, the two norm kernels, the activation, routing, grouping | 7 kernels | Done in the spike | none |
| Self-conditioning: softmax over the vocabulary, the transposed embedding, a scaled norm | 3 kernels and 3 products | 3 more extensions | a day |
| Logits: the tied head and the soft cap, candidate picks | 2 kernels and a product | 2 more extensions | half a day |
| Launch plans and the tuner | `gpu::tune` | Unchanged, a plan per call | none |
| Weights and the loader | `Prefetch`, `quant::pack`, `QMatrix` | Unchanged; they are not Burn tensors | none |
| Vision tower | 7 kernels and FP16 products | 7 extensions, or `jevons-burn`'s layers | 2 to 4 days |
| Parity tests against the current path, the `DiffusionModel` adapter | | | 2 days |

A wrapped port is about a week. A port to Burn's own operations has no route for the quantized
and grouped products and loses the bits, so it was not sized.

## wgpu

The fifteen kernel tests of `gpu::tests`, each in its own process, on CubeCL's wgpu runtime
with SPIR-V on Vulkan, on the same Radeon 8060S. The device reports waves of 32 to 64 lanes;
the kernels assume exactly 32.

| Kernels | Result |
| --- | --- |
| Expert-grouped products, attention (four tests), the matrix-instruction probe | The shader fails validation: "Compilation error: verification failed" |
| Dense products | The small-row kernel runs and returns zeros where values belong; the test stops there, before the matrix-instruction kernel it shares with the grouped products |
| `post_attention`, `post_ffn`, Q/K/V preparation, the Q6_K embedding with its norm, the vision norms and pooling | Run; wrong numbers (errors up to 17% of the largest value) |
| Routing and grouping | Run; the test's checks fail |
| Vision Q/K/V preparation, vision attention | Pass |

A wgpu build needs products and attention without matrix instructions, and sums over a row
that do not depend on the wave size. That is new kernel work with its own speed question, not
a port of what exists.

## How to run it again

On the branch `spike/gemma-burn`, on Windows (WSL has no ROCm), with one GPU process at a
time:

```
set DIFFUSION_MODEL=...\diffusiongemma-26B-A4B-it-Q4_K_M.gguf
set SPIKE_LAYERS=6
cargo test --release -p jevons-burn --lib -- --ignored --nocapture spike::
cargo test --release -p jevons-gemma4-diffusion --features spike-wgpu --lib -- --ignored --exact --nocapture model::burn_spike::the_wrapped_layers_against_the_shipped_forward
cargo test --release -p jevons-gemma4-diffusion --features spike-wgpu --lib -- --ignored --exact --nocapture model::burn_spike::burn_attention_against_the_visibility_kernel
set SPIKE_RUNTIME=wgpu
cargo test --release -p jevons-gemma4-diffusion --features spike-wgpu --lib -- --ignored --exact gpu::tests::<name>
```

- `spike.rs` in `jevons-burn` holds the per-call comparison.
- `model/burn_spike.rs` in `jevons-gemma4-diffusion` holds the extensions, the wrapped forward
  and the two layer tests.
- `Model::load_prefix` loads the first layers only: one layer needs 1.4 GB, six need 4.3 GB.
- Point `DIFFUSION_CUBECL_CACHE` at a copy of the tuned table to leave the app's own untouched.
- Quit the tray app first. A layer prefix fits in memory beside it, but a benchmark that keeps
  the GPU busy can make the app's window lose its device, and the app then quits.
- `just windows-cargo` (`scripts/windows-cargo.sh`) runs a cargo command in the Windows clone
  from WSL, detached, and waits for its log.
