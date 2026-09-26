# Benchmarks

[Back to README](../README.md)

Measurements of the System One API (typed answers from the diffusion language models) and of speech to text. Local results use an AMD Ryzen AI MAX+ 395 / Radeon 8060S (ROCm 7.2.1, WSL2) and a release build; they come from that one machine.

| Directory | Contents |
| --- | --- |
| [jevbench](jevbench/README.md) | Public JevBench cases: results, published comparisons, reproduction commands |
| [system-one](system-one/README.md) | Synthetic System One corpus: local and hosted results, scoring, runner settings |
| [two-models-2026-09-24](two-models-2026-09-24/README.md) | DiffusionGemma and Nemotron-Labs-Diffusion 8B back to back |
| [nemotron-2026-09-25](nemotron-2026-09-25/README.md) | Nemotron-Labs-Diffusion 3B and self-speculative decoding |
| [cubecl](cubecl/README.md) | CubeCL kernel and backend comparisons |

# System One

System One results use the default request settings: `steps=1`, `samples=1`, `think=0`, seed 42, and an 8,192-token context.

## Both models (2026-09-24)

Measured back to back on the current stack (CubeCL 0.11 / Burn 0.22), one model process at a time. See the [two-model report](two-models-2026-09-24/README.md) for per-case files and notes.

| Benchmark | DiffusionGemma Q4_K_M | Nemotron-Labs-Diffusion 8B |
| --- | ---: | ---: |
| JevBench public cases, correct | **189/231 (81.8%)** | 146/231 (63.2%) |
| JevBench p50 / p95 latency | **0.400 / 3.696 s** | 0.644 / 7.934 s |
| System One corpus, questions correct | **75/84 (89.3%)** | 68/84 (81.0%) |
| System One p50 / p95 latency | **365.4 / 1,440.1 ms** | 572.9 / 1,990.2 ms |
| Snake prompt prefill p50 (~470 tokens) | **688 ms** | 1,030 ms |
| Snake canvas forward p50 | **57 ms** | 239 ms |

Against the CubeCL 0.10 build of 2026-09-23 (the table below), DiffusionGemma's JevBench and corpus scores each moved by one near-tie case, while JevBench latency and calibration improved. In an interleaved A/B on one day, the new build prefilled 15–25% faster than the old one. The corpus p95 includes two requests that paid one-time kernel compilation on a freshly started service. Nemotron activates all 8B parameters per forward, while DiffusionGemma's MoE activates about 4B; Nemotron's lower accuracy matches the official Python implementation's answers.

The text-only [Nemotron-Labs-Diffusion 3B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B) answered 143/231 JevBench cases at 0.269 s median latency and 56/84 corpus questions at 240 ms, with a 380 ms Snake prefill: 2.4–2.7 times faster than the 8B, for a small JevBench loss and a larger corpus loss. With `--decoding self-speculation`, Nemotron generates thoughts by diffusion drafting and causal verification, with the same tokens as greedy autoregressive decoding. The 8B's thoughts ran at 11.1 tokens/s, against 5.7 with diffusion and 3.0 autoregressive. Its JevBench thoughts are too short to gain, and thinking does not raise its JevBench accuracy. See the [3B and self-speculation report](nemotron-2026-09-25/README.md).

## DiffusionGemma: CubeCL and the former llama.cpp backend (2026-09-23)

| Benchmark | llama.cpp HIP | CubeCL |
| --- | ---: | ---: |
| JevBench public cases, correct | 189/231 (81.8%) | **190/231 (82.3%)** |
| JevBench p50 / p95 latency | 0.912 / 8.124 s | **0.424 / 4.401 s** |
| System One corpus, questions correct | 75/84 (89.3%) | **76/84 (90.5%)** |
| System One p50 / p95 latency | 956.7 / 2,065.8 ms | **338.2 / 862.3 ms** |
| Snake prompt prefill, p50 (466 tokens) | 1,121 ms | **601 ms** |
| Snake canvas forward, p50 | 113 ms | **80 ms** |
| Seven image requests, same answer labels | reference | **7/7** (largest probability difference 0.19) |

CubeCL service results are from 2026-09-23 and llama.cpp service results from 2026-09-21. Both used the same runners, settings, and protocol. The Snake and image rows compare the two backends in one build on 2026-09-23, before llama.cpp was removed, with prompt reuse off for Snake. Accuracy differences of one question are within the effect of CubeCL's FP16 activations, since llama.cpp quantizes activations to 8 bits. See the [backend comparison report](cubecl/results-default-2026-09-23/README.md) and the [image comparison](../docs/cubecl.md#images).

## JevBench public cases

On 2026-09-24, DiffusionGemma answered **189/231** and Nemotron-Labs-Diffusion **146/231** public cases correctly, both with 231/231 valid responses. On 2026-09-23, the service on CubeCL 0.10 answered **190/231 public JevBench cases correctly (82.3%)**, with **231/231 valid responses**. The median latency was 0.424 s. On 2026-09-21, llama.cpp scored 189/231 (81.8%) with a median of 0.912 s. See the [per-case CubeCL results](jevbench/results-2026-09-23-cubecl.json).

The earlier llama.cpp `think=1024` run scored **189/231 (81.8%)**, with **231/231 valid responses**; it was not rerun on CubeCL. See the [thinking comparison](jevbench/think1024-defaults-2026-09-21.md) and [per-case results](jevbench/results-2026-09-21-defaults-think1024.json).

All local runs sent one HTTP request at a time over loopback, after one excluded warmup, with no retries. Timeouts were 120 seconds for `think=0` and 900 seconds for `think=1024`. The table compares the same 231 public case IDs. Published reference figures come from [Benchmark Heaven's pinned per-case results](https://github.com/fstandhartinger/jevbench/blob/fd51755eb0c0b546ca206d764faf3302feca913e/results/v1.2/jevbench-v1.2-per-task.json); we did not rerun those deployments.

| Model / configuration | Correct | Accuracy | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| Jev 1.13.0 (TypeSafe AI) | 200/231 | 86.6% | 0.665 s | 0.803 s |
| djev (Maisa, diffusion-gemma) | 194/231 | 84.0% | 0.239 s | 0.354 s |
| OpenJev (DiffusionGemma 26B-A4B NVFP4, razorback16) | 189/231 | 81.8% | 0.246 s | 0.459 s |
| SemIf, formerly OpenJev (Qwen3.5-4B, TheoLeeCJ) | 187/231 | 81.0% | 0.194 s | 0.538 s |
| **Local DiffusionGemma Q4_K_M, CubeCL 0.11 (Sep 24)** | **189/231** | **81.8%** | **0.400 s** | **3.696 s** |
| Local DiffusionGemma Q4_K_M, CubeCL 0.10 (Sep 23) | 190/231 | 82.3% | 0.424 s | 4.401 s |
| Local Nemotron-Labs-Diffusion 8B (Sep 24) | 146/231 | 63.2% | 0.644 s | 7.934 s |
| Local Q4_K_M, llama.cpp | 189/231 | 81.8% | 0.912 s | 8.124 s |
| Local Q4_K_M, llama.cpp, `think=1024` | 189/231 | 81.8% | 18.325 s | 37.483 s |

p50 is the median and p95 the 95th percentile, using linear interpolation over caller wall times for all attempts. Local timings include HTTP and inference but exclude model loading and the warmup. Published timings use millisecond-rounded data from deployments measured from Germany. Hardware, network paths, quantization, and inference settings differ, so this table does not isolate model speed.

These 231 public cases omit 303 decisions from the full benchmark. They do not establish an official leaderboard score or unseen-task accuracy. See the [benchmark report and reproduction commands](jevbench/README.md) and [per-case results and provenance](jevbench/results-2026-09-21-defaults.json). The [earlier baseline and thinking runs](jevbench/baseline-2026-09-21.md) used a 4,096-token context and remain available for historical comparison.

## System One comparison corpus

On 2026-09-23, the service on CubeCL scored **76/84 (90.5%)**, with **72/72 valid responses** and a 338.2 ms median latency. Each run used the same 72 requests and 84 questions, one round, concurrency one, and no benchmark warmups or retries. Hosted TypeSafe was last measured on September 19.

| Metric | Gemma CubeCL 0.11, Sep 24 | Nemotron, Sep 24 | Gemma CubeCL 0.10, Sep 23 | Local llama.cpp, Sep 21 | Local llama.cpp, Sep 19 | Hosted `jev-1.13.0`, Sep 19 |
| --- | --- | --- | --- | --- | --- | --- |
| Question accuracy | 75/84 (89.3%) | 68/84 (81.0%) | **76/84 (90.5%)** | 75/84 (89.3%) | 65/84 (77.4%) | 80/84 (95.2%) |
| Valid responses | 72/72 | 72/72 | 72/72 | 72/72 | 72/72 | 71/72 |
| p50 latency, valid responses | 365.4 ms | 572.9 ms | **338.2 ms** | 956.7 ms | 588.4 ms | 339.9 ms |
| p95 latency, valid responses | 1,440.1 ms | 1,990.2 ms | **862.3 ms** | 2,065.8 ms | 1,314.6 ms | 881.0 ms |
| Mean latency, all attempts | 648.4 ms | 812.3 ms | **484.4 ms** | 1,112.6 ms | 683.6 ms | 838.4 ms |

The September 24 runs each started from a freshly started service; DiffusionGemma's p95 and mean include two requests (7.7 s and 6.9 s) that paid one-time kernel compilation, and its one lost question (`relational-join_missing`) was a near-tie already at 0.48 on September 23. Local p50 latency matches hosted TypeSafe's September 19 figure, though hardware and network paths differ. The CubeCL mean includes one 7.3 s request that paid a one-time kernel compilation; see the [CubeCL corpus run](system-one/README.md#recorded-cubecl-results-2026-09-23). Between September 19 and 21, the llama.cpp score improved by 10 questions (11.9 percentage points). These are exploratory measurements on a small synthetic corpus. The September 21 run used a newly started service without inference warmups; another local service remained loaded and machine activity was not controlled. September 19 lacks hardware and quantization provenance, so the timings do not isolate the effect of the inference changes. The historical hosted timeout counts as incorrect and contributes to its all-attempt mean. See [results and methodology](system-one/README.md) and the per-request snapshots for [CubeCL](system-one/results-2026-09-23-cubecl.json) and [llama.cpp](system-one/results-2026-09-21-defaults.json).

# Speech to text

Parakeet TDT 0.6B v3 on the same Radeon 8060S (2026-09-25), measured after kernels were compiled and tuned (the first request of a new audio length pays that once; the results are cached in `~/.cache/jevons-burn`). Times are whole HTTP requests over loopback, including decoding and resampling.

| Input | Audio | Time | Real-time factor |
| --- | ---: | ---: | ---: |
| `examples/speech-en.flac` | 5.9 s | 0.21 s | 28× |
| `examples/speech-es.flac` | 16.0 s | 0.55 s | 29× |
| `examples/speech-es-browser.webm` (Chrome `MediaRecorder`, Opus) | 16.0 s | 0.58 s | 28× |
| 186 s LibriVox chapter, MP3 (two 120 s windows) | 186.5 s | 5.7 s | 33× |
| `examples/speech-es.flac` while Nemotron 3B generates a chat answer | 16.0 s | 1.8 s | 9× |

Accuracy is checked against the Hugging Face implementation rather than a word-error benchmark: identical greedy tokens on the English and Spanish fixtures, and a 2.6% word error rate between the windowed 186 s transcript and one reference pass over the whole recording. In Realtime sessions, a turn's final transcript arrives as the turn detector ends the turn (after 500–600 ms of silence), with live deltas while it is spoken.
