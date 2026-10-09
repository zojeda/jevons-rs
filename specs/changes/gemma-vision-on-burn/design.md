# The Gemma 4 vision encoder on Burn: design

[Proposal](proposal.md) · [Tasks](tasks.md)

## The encoder, step by step

| Step | Kernel encoder | On Burn |
| --- | --- | --- |
| Patch embedding | FP16 product | `linear` (the same GEMM) |
| Positions | `add_positions` | two `select`s by patch column and row |
| Layer input norms | `norm_rows` | `rms_norm` |
| Q, K, V | three products | one stacked product, sliced |
| Per-head norms, 2D rotary | `prepare_qkv` | `head_norm`, then `rotate`: a `select` to each column's partner, a sign, and `cos`/`sin` tables built per image size |
| Attention | a plain loop per query, keys in shared-memory tiles | Burn's attention with unit scale and no mask |
| Post-norm residuals | `add_normed` | `rms_norm` and an add |
| Gate | one product for gate and up, `geglu_quick` | two products, `x * sigmoid(1.702 x)` |
| Pooling | `pool` | a reshape to cells, two sums, the standardization, a weightless norm |
| Projection | FP16 product | `linear` |
| Rows for the text model | a device buffer | read back and uploaded (3 MB at 280 tokens) |

## Choices to measure

- **Head width.** Heads are 72 wide. Burn's matrix-instruction attention tiles the head width
  by 16 and uses another routine otherwise. A variant that widened Q, K and V with zero
  columns to 80 was measured and dropped: slower when warm, and no cheaper on first use.
- **Gate and up.** Separate products, because slicing one fused product into a single
  elementwise kernel reads wrongly under Burn 0.22-pre's fusion. That is one more launch per
  layer than the kernel encoder.
- **The hand-off.** The rows go to the text model through the host. It is counted in the encode
  time; a hand-off that stays on the device is worth writing only if it shows.

## What is checked where

- **On the CPU, in every test run:** the per-head norm and rotary turn, attention with and
  without head padding, the gate and the pooling, each against the kernel encoder's rule
  written out value by value. These replace R19's three kernel tests when the change lands.
- **On the GPU, by hand:** `the_burn_encoder_against_the_kernel_encoder` loads both encoders
  from `DIFFUSION_MMPROJ`, or one alone, and for seven picture sizes prints the first encode's
  time, the warm median of twelve interleaved rounds, and each row's distance from the kernel
  encoder's. It is the proposal's measurement.

## Where it runs

Natively on Windows through `just windows-cargo`, with the tray app quit first, each run in a
process of its own.
