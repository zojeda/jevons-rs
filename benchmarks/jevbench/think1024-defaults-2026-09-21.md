# JevBench with think=1024 and the current defaults

The rerun completed all **231 public cases** on 2026-09-21: **189/231 correct (81.8%)**, with **231/231 valid responses**. The current `think=0` baseline scored **189/231 (81.8%)** on the same binary and datasets.

| Public tier | think=0 | think=1024 | Thinking p50 | Thinking p95 |
| --- | ---: | ---: | ---: | ---: |
| Easy | 48/48 (100.0%) | 48/48 (100.0%) | 12.714 s | 45.085 s |
| Standard | 69/72 (95.8%) | 69/72 (95.8%) | 18.310 s | 33.876 s |
| Hard | 72/111 (64.9%) | 72/111 (64.9%) | 21.202 s | 37.765 s |
| All | **189/231 (81.8%)** | **189/231 (81.8%)** | **18.325 s** | **37.483 s** |

Thinking fixed one baseline error and lost one previously correct answer; 229 cases kept the same correctness. There was **no net accuracy gain** in this run. Failures count as incorrect.

| Metric | think=0 | think=1024 |
| --- | ---: | ---: |
| Valid responses | 231/231 | 231/231 |
| Median latency, all attempts | 0.912 s | 18.325 s |
| p95 latency, all attempts | 8.124 s | 37.483 s |
| Mean latency, all attempts | 2.255 s | 19.450 s |
| ECE, valid distributions | 0.0898 | 0.0834 |
| Multiclass Brier, valid distributions | 0.2687 | 0.2700 |

Median latency increased 20.1×. All responses were valid, so the valid-response median/p95 were also 18.325/37.483 seconds. Calibration metrics are lower-is-better.

## Configuration and method

- Same HIP/native release binary, Q4_K_M model, AMD Ryzen AI MAX+ 395 / Radeon 8060S, ROCm 7.2.1, and WSL2 environment as the [current baseline](README.md).
- Context 8,192; batch 512; seed 42; eight CPU threads; all layers offloaded; flash attention off.
- `think=1024`, `steps=1`, `samples=1`, `sequential=false`. The model can close its thought before the cap.
- Serial requests in easy, standard, then hard order; one excluded warmup with the tested thought budget; no retries. The client timeout was 900 seconds, compared with 120 seconds for the baseline.
- Same upstream harness and scorer at `fd51755eb0c0b546ca206d764faf3302feca913e`. Every saved wire request matches the baseline except for `think=1024`.
- Raw caller wall times include HTTP and inference, exclude loading and warmup, and use linear-interpolated percentiles. No other local build or inference workload was observed during timing.
- 5,992 output tokens reported; mean 25.9 and median 4 per valid response. Five requests reached the 1,024-token cap.

These are 231 public cases, not the official 534-decision leaderboard. Local compute cost remains unknown. Reference deployments differ in hardware, quantization, settings, and network paths.

Run: `jevbench-defaults-think1024-2026-09-21`; started `2026-09-21T20:28:12.516649+00:00`; finished `2026-09-21T21:37:23.460673+00:00`.

## Historical comparison

The [earlier thinking run](think1024-2026-09-21.md) scored 184/231 (79.7%) with 223 valid responses on a 4,096-token context and no benchmark warmup. This rerun scored 189/231 with 231 valid responses using the current 8,192-token context. Across all cases, 5 changed from incorrect to correct and 0 changed from correct to incorrect. Of the eight previously rejected cases, 8 now returned valid responses and 5 were correct. The configuration differences prevent attributing the overall change to one variable.

## Evidence and reproduction

The [results snapshot](results-2026-09-21-defaults-think1024.json) retains per-case outcomes, probabilities, expected labels, latency, usage, errors, paired comparisons, and provenance hashes. All 231 raw-response hashes, task IDs, and upstream scores were verified. The binary and source hashes match the already validated baseline; HTTP smoke checks passed again after timing. Raw evidence remains under gitignored `benchmarks/results/jevbench-defaults-think1024-2026-09-21/`.

Follow the [baseline setup](README.md#reproduce), then run:

```bash
python3 -B scripts/jevbench-local.py \
  --binary "${CARGO_TARGET_DIR:-target}/release/llama-cpp-system-one" \
  --model "$DIFFUSION_MODEL" \
  --harness "$JEVBENCH_SOURCE" \
  --think 1024 \
  --output "benchmarks/results/jevbench-think1024-$(date -u +%Y-%m-%dT%H-%M-%SZ)"
```

The runner starts and stops its own loopback service and uses the [thinking adapter wrapper](../../scripts/jevbench-think.py) to put the thought budget in the HTTP request.

## Request failures

There were no invalid responses, transport/adapter failures, or context-limit errors.
