# DiffusionGemma latency: proposal

[Design](design.md) · [Tasks](tasks.md) · [Measurements](../../../../benchmarks/gemma-latency-2026-10-10/README.md)

## Why

The goal of 2026-10-10: cut DiffusionGemma's mean latency by 20% in generation and in decision.
`latency_bench` on the benchmark machine (natively on Windows) showed where the time went:

- **Generation.** Each denoising step read rows × vocabulary logits back to the host (67 MB for
  a 64-token block) and sampled them on one CPU thread in f64. That cost more than the canvas
  forward it followed.
- **Decision.** A fresh read ran three prefill forwards, one per prompt part (the text, then two
  framing parts of four or five tokens), and a canvas forward of about ten rows. A forward of few
  rows costs what its weights cost to stream, so each small forward cost about as much as the
  canvas forward: 76 ms.

## What changes

1. **Sampling stays on the device.** A model reports each canvas row's best token, sampled
   token and entropy (`DiffusionModel::predictions`). DiffusionGemma computes them in one kernel
   per row and reads four values back. Self-conditioning uses the logits the model still holds,
   so no caller passes them.
2. **Prefill chunks token rows across prompt parts.** Only an image block starts a forward of
   its own.
3. **A one-step read forwards its canvas with its prompt.** The canvas rows ride in the forward
   of the last prompt rows still to compute (`DiffusionModel::prefill_and_forward_canvas`).

## What stays

- Prompt rows keep their bits: reuse of a cached prefix stays exact.
- A read gives the same probabilities, bit for bit, whether its prompt was resident or not.
- Generation's canvas forwards, and the canvases of reads with more steps, run as before.
- Nemotron-Labs-Diffusion takes the trait defaults and does not change.

## What it costs

- A read whose whole prompt is resident, and each extra sample of a read, takes 10 to 25% longer
  (70 to 77 ms one evening, 56 to 70 ms the next day): its canvas now runs on the kernels it
  would share with prompt rows.
- One-step reads change precision in their canvas rows, so probabilities move a little and a
  near-tie can move more. The corpus keeps its 75 of 84.

[The design](design.md) has the numbers and the fallback that pays neither.

## Burn

The goal names Burn. The spike of issue #5 measured DiffusionGemma's forward through Burn
extensions at 1.4 to 3.5% slower per forward than direct launches, with the same bits
([report](../gemma-on-burn/report.md)), and the recommendation then was to keep direct
launches. Asked on 2026-10-11, with the three changes above measured on direct launches, the user
chose the forward through Burn. With two operations per layer it measures within 1% of direct
launches.

4. **The forward runs as Burn operations.** `Model::forward` is a chain of Burn backend
   extensions over Burn tensors (`model::burn`): the embedding, an attention block and a
   feed-forward block per layer, then the logit head, self-conditioning, candidate picks and row
   predictions. The direct-launch forward is gone. The kernels, their plans and the weights are
   the same, so the bits are too.

Burn's own operations in place of the kernels stay out: the spike measured them 5 to 14% slower
with drifting bits. The vision encoder keeps its direct launches; its port failed its rule in
[gemma-vision-on-burn](../../gemma-vision-on-burn/report.md).

## Specs it touches

- `jevons-core`: R11 (`Conditioning::Previous`), R20 (`predictions`), R21
  (`prefill_and_forward_canvas`).
- `jevons-diffusion`: R18 (`uniform::refine` asks the model).
- `jevons-decision`: R9 and R15 (the first canvas goes with the prompt and counts as prefill).
- `jevons-gemma4-diffusion`: R4 (chunking), R8 (row predictions), R9 (self-conditioning), R14
  (warmup), R20 (a canvas read with its prompt), R21 (the forward as Burn operations), and a
  dependency on `jevons-burn`.
- `jevons-kernels`: R4 (a `QMatrix` clone shares its buffers).
- `jevons-api`: the `latency_bench` example.
