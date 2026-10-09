# DiffusionGemma on Burn: design of the spike

[Proposal](proposal.md) · [Tasks](tasks.md)

## What a forward is today

- **16 kernel launches per layer** (15 in the five layers without a V projection), 476 for a
  prefill pass over the 30 layers. A canvas forward takes 57 ms, so a launch averages 120 µs
  with its work.
- **Nothing is read back** inside a forward. Routing, top-k and grouping stay on the device.
- **Buffers are allocated once** for the largest chunk and written over by every layer.
- **Glue is fused into the kernels:** the norms, the residual adds, the layer scale and the next
  layer's input norm are part of `post_attention` and `post_ffn`; RoPE, the per-head norms and
  the cache write are part of `qkv_prepare`.
- **Prefill kernels are row invariant:** a row's result does not depend on the rows beside it,
  which is what makes a reused prompt prefix bitwise equal to a fresh one.

## Where Burn can cost

- **Per operation:** Burn registers each operation in a lazy stream and runs it later. At 476
  launches, 20 µs each is 17% of a canvas forward.
- **Per output:** each operation allocates its output; today's path writes over scratch.
- **Between kernels:** casts between FP16 and f32, and the glue as separate operations that
  fusion may or may not merge.
- **Order of accumulation:** Burn's own products and attention are not row invariant.

## Three arms

1. **Current.** The shipped path, as the reference for bits and for time.
2. **Wrapped.** The same kernels as Burn backend extensions over Burn tensors, with the tuned
   launch plans and the quantized weights passed as they are. Expected: the same bits. It bounds
   what "one framework" costs.
3. **Burn's own operations** wherever they exist: norms, residuals, RoPE, the gated products'
   activation, routing, attention with a mask. Expected: tolerance-level parity. It is the only
   arm that can run on wgpu, and it gives up bitwise prefix reuse unless attention and the
   products stay wrapped.

## Measurements

- **Step 1, per call:** the FP16 product through the existing extension against the same kernel
  on raw buffers, at DiffusionGemma's shapes, and Burn's norm plus residual add. Synthetic data.
- **Step 2, per layer:** layer 0 alone (sliding window, 8 KV heads), then layers 0 to 5 (the
  last has full attention, 2 KV heads and no V projection) of the real checkpoint, at 32 and
  512 rows, each arm against the current one. The model is loaded as a prefix of its layers,
  so the reference is the shipped code over the same weights in the same process, and no
  recorded traces are needed.
- **Attention alone:** Burn's attention with a mask against the visibility kernel, on
  synthetic rows in both cache layouts.
- **Timing:** warm kernels and pools, batches interleaved between arms, the device waited for
  at the end of each batch, medians over the batches, one output read back and compared.
- **wgpu:** the existing kernel tests, each in its own process, on CubeCL's wgpu runtime
  (SPIR-V on Vulkan) instead of HIP.

## Where it runs

WSL has no ROCm: every GPU run goes through the native Windows clone the desktop build uses,
with the HIP SDK and the checkpoint that are there. One GPU process at a time, started detached
and never killed. A prefix of layers fits in memory beside the tray app, but the app should be
quit first all the same: during the spike its window lost its GPU device under a benchmark's
load, and it quit.
