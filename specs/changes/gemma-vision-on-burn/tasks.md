# The Gemma 4 vision encoder on Burn: tasks

[Proposal](proposal.md) · [Design](design.md) · [Report](report.md)

- [x] The encoder on `jevons-burn`'s layers, beside the kernel one (`vision_burn.rs`).
- [x] Its building blocks checked on the CPU against the kernel encoder's rule.
- [x] Both encoders on the real projector: rows, warm time per size, first use of a size.
- [x] The decision under the proposal's rule, with the numbers in [`report.md`](report.md): it
      does not hold, so nothing lands.

Not done, because the rule failed before them:

- The seven image requests.
- Deleting `gpu/vision.rs` and the kernel `vision.rs`.

The encoder and its measurement stay on the branch `spike/gemma-vision-burn`.
