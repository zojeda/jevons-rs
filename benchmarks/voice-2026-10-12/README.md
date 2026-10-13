# Voice: Qwen3-TTS on the fused kernels (2026-10-12)

[Back to benchmarks](../README.md) · [Proposal](../../specs/changes/voice-output/proposal.md) · [Design](../../specs/changes/voice-output/design.md)

Ryzen AI MAX+ 395 / Radeon 8060S, natively on Windows with the HIP SDK 7.2, release builds,
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` in FP16, the voice `lector`. The figures come from the model's
own log (`RUST_LOG=jevons_qwen3_tts=debug`), which times each chunk of an utterance. The tray
app was running beside every run before `bench`, idle, with its models unloaded. `bench` and
the runs after it had the app quit and gave the same figures.

## Result

One frame is 80 ms of audio: a talker step and fifteen code-predictor steps. The same binaries
gave two sets of figures, hours apart:

| Warm, FP16 | The screen on | The screen off, in Modern Standby |
| --- | ---: | ---: |
| A frame's sixteen codes | **22 to 24 ms** | **48 to 51 ms** |
| The codec decoder, per frame, in chunks of twelve | 2.5 ms | 6.4 ms |
| A second of audio | 310 ms: 3.2 times faster than real time | 680 ms: 1.4 times faster |
| Reading a prompt | 13 to 27 ms | 29 to 47 ms |
| First audio of a sentence: the prompt, four frames, their decoding | 120 to 165 ms | 260 ms |
| Through the app, 8.6 s of speech over HTTP: first audio, then all of it | 120 ms, 2.8 s | 279 ms, 6.7 s |

The bar set before the port was 40 ms a frame and 400 ms to the first audio. With the screen
on both are met. The second column meets the first audio and stays faster than real time.

**Why two columns.** The second column is Windows' Modern Standby. The System event log has the
machine entering it at 23:42:15 on an idle timeout, with nobody at it, and leaving it at
03:49:58 on a mouse (Kernel-Power events 506 and 507). Every slow run falls inside those hours and every
fast one outside: 23:04 and 23:25 before, 03:55 on after. With the screen off Windows held the
whole chip back. `% Processor Performance`, sampled during a run, had the busy cores at 51 to
55% of their nominal frequency, against 165 to 169% with the screen on, and the codec decoder,
a few large launches, slowed by the same factor as the frames, a thousand small ones. At 01:24
the standby deepened and processes stopped: a build begun then took 149 minutes. Whoever speaks
to the app has the screen on, so the first column is the app in use, and the second is what an
unattended run measures on this machine. The power scheme was `silent` in both.

**Through the app.** `jevons-desktop --say … --out say.wav` loads the voice model alone, serves
the embedded API and asks `/v1/audio/speech` over HTTP, two sentences. Its frames took what the
model's tests took in each state, so the worker, the route and the client add nothing a frame.

**With the app quit.** On 2026-10-13 the installed app spoke ten sentences, five in Spanish and
five in English, through `--say`, three times, with no other process on the GPU and the screen
on. The second and third runs, warm:

| The app quit, warm, ten sentences | Run 2 | Run 3 |
| --- | ---: | ---: |
| A frame's codes in chunks of twelve: least, middle, most | 23.6, 24.4, 32.0 ms | 23.6, 24.2, 25.5 ms |
| The codec decoder, per frame, in chunks of twelve: least, middle, most | 2.6, 3.3, 3.8 ms | 2.8, 3.2, 3.4 ms |
| Reading a prompt | 16 to 36 ms | 16 to 29 ms |
| First audio of a sentence | 142 to 182 ms | 143 to 157 ms |
| The request over HTTP: speech, then the time it took | 41.4 s, 15.3 s | 39.4 s, 14.1 s |

The idle app had cost nothing: these are the figures taken beside it, at the upper end of
their range. A ten-sentence answer comes out 2.7 to 2.8 times faster than real time. That is
below the 3.2 of the table above, which counts a sentence's full chunks alone: each sentence
also reads a prompt, decodes four frames in a chunk of their own (20 to 33 ms) and pads its
last chunk to twelve (34 to 48 ms), so the decoder costs 3.9 ms a frame over a whole sentence.
The busiest core ran at 76% of its nominal frequency in the middle of these runs (65 to 167
over 29 samples), against 165% in the `ab` run, and a frame took 24 ms for 22 to 24: a frame
hardly follows the CPU's clock.

**A sentence of a new length compiled.** The first `bench` run was that binary's first: 10.4 s
of weights and 152 s of warm-up, compiling. After the warm-up, six of its ten sentences had a
count of text tokens the binary had not read yet. Each took 510 to 680 ms to read its prompt
and 630 to 800 ms to first audio. The four of a count already read took 145 to 147 ms, and
runs 2 and 3 read all ten in 16 to 36 ms. The log has no tuning in between: kernels compiled
for each new count of text tokens, once for a binary, and the cache kept them. Until an
installed build had met a length, a sentence of that length started half a second late, above
the bar of 400 ms. The warm-up spoke one sentence, so it covered one length.

The text of a prompt now reaches the device in 32, 128 or 512 rows, filled with padding, and
the warm-up reads a prompt of each count before it speaks. Two runs of that build:

| The fix | What ran | Reading a prompt | First audio of a sentence |
| --- | --- | ---: | ---: |
| `lengths`, an empty cache | After 139 s of warm-up, eleven sentences of 2 to 52 tokens, ten lengths new to the binary | 15 to 26 ms | 136 to 190 ms |
| `lengths`, the second process | The same sentences, warm-up 2.0 s | 18 to 26 ms | 144 to 213 ms |
| `ab2`, the build before it | The ten sentences of `bench`, three times, in turn with the fix | 16 to 40 ms | 147 to 214 ms |
| `ab2`, the fix | The same, in turn | 15 to 24 ms | 138 to 192 ms |

In `ab2` a frame took 25.6, 26.9 and 26.5 ms in the middle for the build before the fix and
26.1, 31.8 and 25.9 for the fix, and a second of audio 383 to 413 ms against 380 to 448: the
same, on a machine a tenth slower than three hours earlier, minutes after a 23-minute build.
The warm-up of a later process takes 1.5 to 1.9 s for 1.4 to 1.7.

## What bounds a frame

A micro-benchmark of the small-row product on this model's shapes, one row, the screen on:

| What | Time |
| --- | ---: |
| Issuing a launch, on the CPU | 4 to 12 µs |
| The product for queries, keys and values, 1,024 to 4,096 | 49 µs |
| The attention's output product, 2,048 to 1,024 | 15 µs |
| The product for the gate and the up, 1,024 to 6,144 | 70 µs |
| The down product, 3,072 to 1,024 | 36 µs |

A layer takes about 215 µs, 22 ms over the frame's 103 layer steps, and its four products are
170 µs of it. A frame streams 3.3 GB of FP16 weights, the talker's once and the predictor's
fifteen times, at 150 to 170 GB/s. Issuing a frame's thousand launches is 7 ms of CPU while the
GPU works. A frame is the GPU reading weights. A second row costs a product nothing (45 µs for
two rows of queries, keys and values against 49 for one), so two sentences made together would
pay once for the products, four fifths of a frame. That is not built or measured: the kernels
read one sequence.

Three ways to shorten it were measured. None was kept:

| Tried | What it gave |
| --- | --- |
| Five launches a layer for nine: norms, residual sums and the gate inside the products, a single row's attention in one launch | The networks followed the reference as before (175 of 175 first codes, 2,612 of 2,625 others). Frames took 24 to 25.5 ms against 22 to 24, the runs interleaved: a product that normalizes FP32 rows costs more than the launches it saves. Reverted. |
| 8-bit weights | A layer's products in 140 µs against 175, the widest one no faster: about 16% a frame, for the fidelity below. Left out. |
| Four rounds a wave for two, in the small-row kernel | The two products over 1,024 inputs 15% and 7% faster: about 6% a frame. Not applied: DiffusionGemma shares the kernel, and its shapes were not measured. |

| Start of a process | Weights | Warm-up |
| --- | ---: | ---: |
| A binary's first run, kernels compiled | 8.4 s | 162 s |
| Any later run of that binary, kernels from the cache | 8.4 s | **1.3 s** |
| Before the cache opened on Windows, every run | 8.1 s | 201 s |
| The same with Burn's attention in the codec's transformer | 9.5 s | 1,136 s |

The cache serves the binary that filled it: the tray app's first run compiled again (166 s)
after the test binary had filled the same folder, and its second run started warm.

## What the measurements decided

- **Kernels of the crate, a frame per operation.** With Burn's operators layer by layer the
  talker and the predictor tuned a matrix product for minutes at each new shape, and a frame was
  some 3,600 launches. The fused path is nine launches a layer and one Burn operation a frame.
- **FP16, not 8-bit.** With Q8_0 weights the predictor chose the reference's code 94 times in
  100, against 99.7 in FP16. The product kernel alone, timed later, runs a layer's products
  20% faster on 8-bit weights. The option was removed for its fidelity.
- **Two chunk shapes.** The decoder runs four frames, then twelve at a time. Any other count
  compiled kernels for seconds in the middle of an utterance.
- **The kernel cache on Windows.** CubeCL asked Turso for a multi-process WAL, which Windows'
  I/O backend lacks, so the cache never opened and every process compiled everything again
  (issue #17). With `third_party/cubecl-environment`, a second process read its kernels from
  the cache: no kernel was compiled, and the warm-up fell from 162 s to 1.3 s. The same cache
  serves DiffusionGemma and Parakeet.

## Fidelity on the GPU

Against the dump of the `qwen-tts` package (CPU, f32), five utterances:

| Check | FP16 on the GPU | f32 on the CPU device |
| --- | ---: | ---: |
| The talker's output for the prompt, relative RMS | 3.8e-4 | 1.1e-6 |
| The talker's choice of the reference's next first code, reading the reference's frames | 175 of 175 | 175 of 175 |
| The predictor's choice of the reference's other codes | 2,617 of 2,625 | 2,625 of 2,625 |
| Audio from the reference's codes, relative RMS | 8e-4 | 2e-6 |
| Frames of a greedy run of its own before the first that differs | 72 of 175 | 175 of 175 |

A greedy run parts from the reference within a few frames in FP16: one near tie among the
predictor's flat logits falls the other way and the two runs are different utterances from
there. Reading the reference's frames, code by code, is the check that means something.

## The runs

Logs in the Windows clone, `target\windows-cargo\qwen-gpu{1..8}.log` and `kern-bench{1..3}.log`:

| Run | Build | What it showed |
| --- | --- | --- |
| 1 | Burn's operators | The codec and the talker's states match in FP16. 36 minutes, most of it tuning. |
| 2 | Fused talker and predictor | The networks follow the reference. Warm-up 1,136 s: Burn's matrix-product tuning, reached through the codec transformer's attention. |
| 3 | The codec's transformer on the kernels too | Warm-up 201 s, 182 kernels compiled. A warm utterance of 30 frames in 0.84 s. |
| 4 | Q8_0 weights | 94 in 100 codes, 26.7 ms a frame. The cache's own log named the WAL. |
| 5 | The cache patch, two processes | The second process compiled nothing: warm-up 1.3 s. FP16 frames at 22.1 ms. |
| `say` | The tray app's `--say`, three times, in standby | End to end through HTTP. Frames at 48 to 51 ms. |
| direct | The run 5 binary again, four times, in standby | Frames at 48 ms, with the CPU at half frequency during the run. |
| 6 | Five launches a layer, in standby | The kernels' own tests pass, the one-launch attention gives the two kernels' bits. 2,612 of 2,625. |
| `kern-bench` 1 to 3 | The product kernel alone | A launch's cost, FP32 and normalized rows, 8-bit weights, lane shapes. |
| `ab` | The app (nine launches) and run 6's binary (five), in turn, twice | Nine: 22 to 24 ms a frame. Five: 24 to 25.5 ms. Cores at 165% during the run. |
| 7 | Nine launches a layer again | Run 5's counts to the code: 2,617 of 2,625. Frames at 22.2 to 22.9 ms, first audio after 123 to 164 ms. |
| `bench` | The installed app's `--say`, ten sentences, three times, the tray app quit | Frames at 24 ms, first audio after 142 to 182 ms, 39 to 41 s of speech in 14 to 15 s. In the first run, a sentence of a new length took 630 to 800 ms to first audio. |
| `lengths` | The text in fixed row counts, an empty cache, two processes | No sentence compiles after the warm-up: prompts read in 15 to 26 ms whatever their length. |
| `ab2` | The build before the fix and the fix, in turn, three times each | The same frames and the same first audio. |
| 8 | The text in fixed row counts | Run 5's counts to the code again: 175 of 175 first codes, 2,617 of 2,625 others. |

The logs of `say`, `direct`, `ab`, `bench`, `lengths` and `ab2` are in
`%USERPROFILE%\jevons\voice-bench\`, with the speech of the `say` run in `say.wav`, of the
third `bench` run in `bench.wav` and of the second `lengths` run in `lengths.wav`.

To repeat a run:

```bash
scripts/windows-cargo.sh voice 'QWEN3_TTS_MODEL=C:\models\qwen3-tts-12hz-0.6b-base' \
  'RUST_LOG=info,jevons_qwen3_tts=debug' -- \
  test --release -p jevons-qwen3-tts --lib -- --ignored --nocapture --test-threads=1 \
  the_networks_follow codec_audio speaking_stops
```
