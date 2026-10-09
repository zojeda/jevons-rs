# DiffusionGemma on Burn: tasks

[Proposal](proposal.md) · [Design](design.md)

The acceptance list of issue #5, in the order the spike takes it.

- [x] Step 1: a wrapped product against a direct launch, per call.
- [x] Step 2: decoder layers, the wrapped arm against the current path: bits and time.
- [x] Step 2: Burn's own operations for the glue and for attention: parity and time.
- [x] The Burn extensions and kernels a full port needs, with rough effort.
- [x] The vision tower: effort against Nemotron's Pixtral (sized, not built).
- [x] wgpu: what runs and what fails.
- [x] [`report.md`](report.md): the measurements and the recommendation.

## Closed

- [x] The decision: the recommendation was accepted on 2026-10-09, and DiffusionGemma's decoder
      stays on its kernels. This folder is archived with it.
- [x] The vision tower on `jevons-burn`'s layers went on as a change of its own,
      [gemma-vision-on-burn](../../gemma-vision-on-burn/report.md).
