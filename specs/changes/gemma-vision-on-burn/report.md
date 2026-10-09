# The Gemma 4 vision encoder on Burn: report

[Proposal](proposal.md) · [Design](design.md) · [Tasks](tasks.md)

Measured on 2026-10-09 on the benchmark machine (Ryzen AI MAX+ 395, Radeon 8060S), natively on
Windows with the HIP SDK 7.2, release builds, the `gemma4v` F16 projector (27 layers, width
1152, 16 heads of 72). The tray app was closed: every run had the GPU to itself.

## Result

**The rule does not hold, so nothing lands.** The vision encoder stays on its kernels.

| The proposal's rule | Outcome |
| --- | --- |
| Rows: each token's cosine with the kernel encoder's at least 0.999 | Holds: 0.99926 at the lowest |
| Time: a warm encode no slower at each size | Does not hold: from 11% faster to 40% slower, and it changes from one process to the next |
| First use: a new image size costs no more than now | Does not hold: up to 8 s on the first encode of a patch count, where the kernel encoder pays nothing |
| Answers: the seven image requests | Not run: the rule had already failed |

## What was measured

Seven pictures. Each encoder ran in three processes: alone, and in two runs with the other
encoder (the first of those also ran a head-padding variant that was then dropped). Times in
ms. "Warm" is the median of twelve encodes.

### Warm encode

| Picture | Tokens | Patches | Kernel encoder, three processes | Burn encoder, three processes |
| --- | ---: | ---: | --- | --- |
| 64x48 | 80 | 720 | 130.7, 121.7, 127.7 | 122.8, 136.4, 120.8 |
| 224x224 | 81 | 729 | 131.1, 133.2, 129.2 | 136.6, 150.4, 153.4 |
| 330x220 | 77 | 693 | 124.4, 123.3, 122.5 | 128.5, 149.6, 141.3 |
| 640x360 | 104 | 936 | 193.6, 194.8, 199.9 | 196.6, 203.4, 220.8 |
| 300x700 | 90 | 810 | 167.9, 168.6, 169.4 | 149.5, 157.8, 162.7 |
| 960x672 | 280 | 2520 | 840.6, 836.9, 848.9 | 830.0, 882.0, 1190.8 |
| 528x336 | 77 | 693 | 119.0, 117.6, 117.7 | 121.2, 122.3, 135.9 |

The kernel encoder repeats within a few percent. The Burn encoder was level with it in the
first process and up to 40% slower at 280 tokens in the third.

### First encode

| | Kernel encoder | Burn encoder |
| --- | --- | --- |
| The first encode of a process | 25.8 to 28.1 s | 114.3 s alone; 57.8 s when the kernel encoder had run first in the process |
| The first encode of a patch count the process has not seen | as a warm one (116 to 196 ms, 845 to 861 ms at 280 tokens) | 0.1 to 8.2 s, and not the same for a count each time: 693 patches cost 1.7 s alone and 138 ms beside the kernel encoder |
| A patch count seen before, in a new picture size | as a warm one | as a warm one |

### Rows

Against the kernel encoder's rows for the same picture, the Burn encoder's differ by up to 2.8%
of the largest value, and the lowest cosine of any token is 0.99926 (0.99946 at 280 tokens).
That is closer than the kernel encoder is to llama.cpp's (`docs/cubecl.md`).

## Why

- **The kernel encoder compiles one variant per weight shape** and takes the patch count as a
  number at launch, so a new image size costs nothing (`vision.rs`). Its attention is a plain
  loop that needs no tuning.
- **Burn's operations are tuned and compiled by shape.** A new patch count is a new shape for
  the attention, the reductions inside the norms and the fused elementwise kernels. Which of
  them costs the seconds was not separated. One observation: in the run with the padding
  variant, the 80-wide encoder paid at each new count and the 72-wide one, running after it,
  paid nothing. So the cost sits in what the two share, or in a tuning key that does not tell
  72 from 80.
- **The tuning is measured, not fixed**, which fits the warm time moving from process to
  process. Which choice differed was not checked.
- **Widening the heads to 80**, so Burn's matrix-instruction attention would apply, was slower
  when warm (4% to 8%) and paid seconds on first use too. It was dropped.
- **The products are the same GEMM** in both encoders, so there was no speed to gain there.

## What would change the answer

- Tuning and compilation that do not depend on the patch count, in Burn or by padding the
  patches to a few fixed counts with a mask. The decoder spike measured what a mask costs.
- A kernel cache that persists (below) would turn the cost of a patch count into one paid once
  per machine, not once per start. It would still be seconds on the first picture of a size.

## Found on the way: kernels are not kept between processes on Windows

The cache folder the runs pointed at ended with one empty file. The app's log has the reason
at every start: `Unable to open Turso cache ... using memory: Parse error: no such table:
meta`, and its own cache folder holds an empty `default.db`. So on this Windows setup every
process compiles its kernels again: 26 to 28 s for the kernel encoder's first encode here. The
benchmark figures were taken on WSL, where the cache worked. This is separate from this change
and worth its own look.

## How to run it again

On the branch `spike/gemma-vision-burn`, with the tray app quit:

```
just windows-cargo vision 'DIFFUSION_MMPROJ=...\mmproj-diffusiongemma-26b-a4b-f16.gguf' \
  'VISION_ENCODERS=both' -- test --release --locked -p jevons-gemma4-diffusion --lib -- \
  --ignored --exact --nocapture vision_burn::tests::the_burn_encoder_against_the_kernel_encoder
```

`VISION_ENCODERS=burn` or `kernels` runs one encoder alone. A run that stops at start-up with
HIP status 719 is the driver refusing a new process just after another released its memory:
wait half a minute and run it again.
