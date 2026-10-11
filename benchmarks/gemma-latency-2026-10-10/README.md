# DiffusionGemma latency: sampling on the device, one forward per read, the forward through Burn (2026-10-10 and 11)

[Back to benchmarks](../README.md) · [Proposal](../../specs/changes/archive/gemma-latency/proposal.md) · [Design](../../specs/changes/archive/gemma-latency/design.md)

Ryzen AI MAX+ 395 / Radeon 8060S, natively on Windows with the HIP SDK 7.2, release builds, the
Q4_K_M checkpoint, an 8,192-token context, batches of 512, seed 42. `latency_bench` loads the
model and times requests in the process, without HTTP. The tray app was closed.

## Result

Three builds, run one after the other on 2026-10-11 after a reboot:

- **Before:** `dev` at `f2a7260` with the benchmark added.
- **Direct launches:** the first three changes below, the forward launching its kernels
  directly.
- **Through Burn:** all four changes, the same forward as Burn operations.

| Decision read, System One corpus (72 requests), mean | Before | Direct launches | Through Burn | Change |
| --- | ---: | ---: | ---: | ---: |
| After another request: the prompt is prefilled | 406 ms | 263 ms | 265 ms | **-35%** |
| The same, p50 | 353 ms | 208 ms | 212 ms | -40% |
| A sentence added to a resident prompt | 226 ms | 107 ms | 107 ms | **-53%** |
| The same request again: the whole prompt is resident | 56 ms | 71 ms | 70 ms | **+24%** |

| Generative answer, six chat prompts, mean | Before | Direct launches | Through Burn | Change |
| --- | ---: | ---: | ---: | ---: |
| Both budgets, twelve requests | 13.57 s | 4.01 s | 3.87 s | **-71%** |
| `max_tokens=64` (one block, 56 tokens on average) | 8.46 s | 2.21 s | 2.39 s | -72% |
| `max_tokens=192` (2.3 blocks, 139 tokens on average) | 18.67 s | 5.82 s | 5.35 s | -71% |

"Change" compares Through Burn with Before. The figures are the runs taken while nothing else
loaded the machine ([the runs](#the-runs) has the rule and every run).

Four changes, in [the design](../../specs/changes/archive/gemma-latency/design.md):

1. Generation samples each canvas row on the device and reads back four values per row. Before,
   each step copied 64 × 262,144 logits to the host (67 MB) and sampled them on one thread.
2. Prefill chunks token rows across prompt parts. Before, a fresh read ran three prefill
   forwards: the text, then two framing parts of four or five tokens.
3. A one-step read forwards its canvas with the prompt rows still to compute.
4. The forward is a chain of Burn operations over the same kernels: one for the embedding, an
   attention block and a feed-forward block per layer, then the logit head.

## What it costs

- **Reads with a resident prompt take longer:** 56 to 70 ms here, 70 to 77 ms on 2026-10-10. Each
  extra sample of a read pays the same. Their canvas runs alone and now uses the tiled kernels
  it would share with prompt rows. The small-row kernel it used before is faster at ten rows.
  This comes from change 3.
- **One-step reads changed precision in their canvas rows.** The tiled expert kernel rounds
  weights to FP16, as prefill does; the small-row kernel multiplied f32 weights. Against the
  build before, the probabilities of the corpus's 84 slots moved by a median of 1e-4. Three
  slots moved more than 0.01 and one, the near-tie of `relational-join_missing`, by 0.21. No
  slot changed its best candidate. This also comes from change 3.
- **Burn costs nothing measurable.** Through Burn and direct launches differ by 1% on reads in
  the same round (265 against 264 ms, 107 against 106, 70 against 69) and give the same bits.

Change 2 without change 3 pays neither of the first two costs. A third build measured it on
2026-10-10 (parts-only): fresh reads 335 ms against 479 (-30%), added sentence 160 ms against
276 (-42%), resident prompt 63 ms against 70. Its 84 slots are bitwise those of the build before.

## Answers

- **Corpus:** 75 of 84 questions correct in every decision run of every build.
- **Through Burn against direct launches:** the same probabilities, bit for bit, on all 72
  requests in every run.
- **Repeated reads:** a read after another request and the same read with its prompt resident
  gave the same probabilities, bit for bit, in all 216 pairs of each run.
- **Generated text:** the twelve answers are the same, character for character, in all six
  generation runs of the three builds. The device samples in f32 from the draws the f64 sampler
  used.
- **JevBench was not rerun.** It needs the HTTP service and the pinned harness on this machine.
  The corpus is the evidence that answers hold.

## The runs

Each build's binary was built once and kept; one script ran them in turn with 45 s between
processes. A decision run (`--no-generation --rounds 2`) and a generation run
(`--no-decision --rounds 1`) time their requests after one untimed pass. Reports are in
[`runs-burn/`](runs-burn) and [`runs/`](runs).

Other projects' builds and tests ran in WSL on the same machine at times, and a run taken under
them is 30 to 100% slower. The WSL load average was logged each minute; a run counts when its
highest one-minute load stayed under 8.

### 2026-10-11, three builds

Means in ms.

| Run | Build | Fresh | Sentence added | Resident | Correct | Highest load |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | through Burn | 218 | 88 | 60 | 75/84 | 1.1 |
| 2 | direct launches | 261 | 108 | 72 | 75/84 | 3.8 |
| 3 | before | 409 | 228 | 57 | 75/84 | 3.4 |
| 4 | through Burn | 265 | 107 | 70 | 75/84 | 1.2 |
| 5 | direct launches | 264 | 106 | 69 | 75/84 | 1.4 |
| 6 | before | 402 | 223 | 56 | 75/84 | 1.3 |

Run 1 was the first model process after the reboot and ran about 18% faster than run 4 of the
same binary. The result table takes run 4 for Through Burn and the mean of both runs for the
other two builds.

| Run | Build | Both budgets | `max_tokens=64` | `max_tokens=192` | Highest load | Counts |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 7 | through Burn | 3.87 s | 2.39 s | 5.35 s | 1.6 | yes |
| 8 | direct launches | 4.01 s | 2.21 s | 5.82 s | 2.5 | yes |
| 9 | before | 15.37 s | 11.59 s | 19.16 s | 19.3 | no |
| 10 | through Burn | 4.51 s | 2.57 s | 6.44 s | 12.1 | no |
| 11 | direct launches | 5.23 s | 2.53 s | 7.93 s | 10.2 | no |
| 12 | before | 13.57 s | 8.46 s | 18.67 s | 2.6 | yes |

### 2026-10-10, before, direct launches and parts-only

Run late in the evening, before the Burn forward existed and without a load log; the two runs
of each build agree. After, parts-only, before, twice (`--rounds 2`), then after, before, twice
for generation at `max_tokens=64`.

| Run | Build | Fresh | Sentence added | Resident | Correct |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | direct launches | 301 | 131 | 78 | 75/84 |
| 2 | parts-only | 343 | 164 | 64 | 75/84 |
| 3 | before | 470 | 271 | 68 | 75/84 |
| 4 | direct launches | 281 | 118 | 76 | 75/84 |
| 5 | parts-only | 328 | 156 | 62 | 75/84 |
| 6 | before | 488 | 282 | 72 | 75/84 |

| Run | Build | Generation, mean | p50 |
| ---: | --- | ---: | ---: |
| 7 | direct launches | 3.02 s | 3.08 s |
| 8 | before | 9.57 s | 9.56 s |
| 9 | direct launches | 3.40 s | 3.22 s |
| 10 | before | 9.59 s | 9.28 s |

That evening: fresh reads 479 to 291 ms (-39%), added sentence 276 to 124 ms (-55%), resident
prompt 70 to 77 ms (+10%), generation 9.58 to 3.21 s (-66%). The files in `runs/` name the
direct-launch build "after".

The same binary differs between sessions: the build before read a fresh prompt in 479 ms that
evening and in 406 ms the next day. Compare builds within one session.

## One Burn operation per kernel

The first forward through Burn made every kernel its own operation, as the spike of issue #5
did for six layers. It gave the same bits. A 30-layer forward that way queues more than a
thousand operations and dropped tensors, against about sixty with two per layer. Its three runs
read a resident prompt in 90, 103 and 172 ms against 72 to 80 ms for direct launches. No load
log covered the first and the other two ran under other load, so they do not isolate its cost.
The block version replaced it.

## Tests

On the branch, natively on Windows, each model test in its own process:

- The 18 GPU kernel tests of `jevons-gemma4-diffusion`, with the new
  `row_predictions_match_the_reference_sampler`.
- `model_reads_preserve_reproducibility_across_requests`,
  `model_extensions_average_refine_think_and_chunk` and
  `model_images_prefill_and_preserve_text_reproducibility`.
- The `prefix_check` example: reused prompt rows and the logits read after them differ from a
  fresh prefill by 0.

## Run it again

```bash
cargo run --release --locked -p jevons-api --example latency_bench -- --rounds 3 > latency.json
```

`DIFFUSION_MODEL` names the checkpoint. To compare two builds, keep both binaries
(`target/release/examples/latency_bench`) and alternate them. On Windows wait 45 s between
processes, or the next one stops at start-up with HIP status 719. Log the machine's load beside
each run.
