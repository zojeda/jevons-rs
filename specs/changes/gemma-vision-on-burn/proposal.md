# The Gemma 4 vision encoder on Burn: proposal

[Design](design.md) · [Tasks](tasks.md) · [Report](report.md) · follows [gemma-on-burn](../gemma-on-burn/report.md)

## Why

The spike of issue #5 kept DiffusionGemma's decoder on its own kernels and named one part worth
moving: the vision encoder. Its weights are FP16, its attention needs no mask and it has no
routing, which is what `jevons-burn`'s layers already serve for Nemotron's Pixtral tower. Today
it has seven kernels of its own (`gpu/vision.rs`) and their glue (`vision.rs`), about 800 lines
that only DiffusionGemma uses.

## What changes

`Vision` is rebuilt on `jevons-burn`: the products on the tuned FP16 GEMM, attention on Burn's,
the norms, rotary turns, gate and pooling as Burn operations. Image preprocessing, the token
counts, the errors and the rows handed to the text model stay as they are.

If the rule below holds, `gpu/vision.rs` and the kernel `vision.rs` are deleted, not kept beside
the new encoder. If it does not hold, nothing lands and the measurements say why.

## The rule: no step back

Measured on the benchmark machine with both encoders in one process, interleaved, on the real
projector:

- **Time.** A warm encode is no slower than the kernel encoder's at each size from 70 to 280
  tokens, beyond the spread of its own runs.
- **First use.** The first encode of an image size the process has not seen costs no more than
  it does now. The kernel encoder compiles one variant per weight shape and serves every size
  with it; Burn's attention is tuned per shape.
- **Rows.** Each token's row keeps a cosine of at least 0.999 with the kernel encoder's, the
  bar the llama.cpp comparison used (`docs/cubecl.md`).
- **Answers.** The seven image requests of that comparison give the same labels. This needs
  the whole model.

Image blocks are reused by the image's content, not by the rows' bits, so this change needs no
bitwise promise.

## Specs it touches

- `jevons-gemma4-diffusion`: R19 (the encoder and the tests that check it), its scope (the
  vision kernels go), and a dependency on `jevons-burn`.
- `jevons-burn`: attention with options (a unit scale), and resident tensors from GGUF bytes.
