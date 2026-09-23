# System One comparison benchmark

[cases.json](cases.json) contains **72 synthetic requests with 84 scored questions** for comparing hosted TypeSafe with local jevons-rs. Both endpoints receive the same state and questions through `@typesafe-ai/sdk` 0.6.0 and the existing [JavaScript client factory](../../examples/javascript/client.mjs). Only the requested model ID differs. Expected answers, rationales, category names, and case IDs stay outside the wire request.

## Recorded CubeCL results (2026-09-23)

With the [CubeCL backend](../../docs/cubecl.md) as the default, the local service scored **76/84 (90.5%)** with **72/72 valid responses**. It answered every question in 64 of the 72 requests. That is one more correct answer than the llama.cpp run of September 21, in the context category. Median latency fell from 956.7 ms to 338.2 ms. The corpus SHA-256 is unchanged.

| Metric | CubeCL, Sep 23 | llama.cpp, Sep 21 |
| --- | --- | --- |
| Valid responses | 72/72 | 72/72 |
| Question accuracy | 76/84 (90.5%) | 75/84 (89.3%) |
| Requests with every answer correct | 64/72 (88.9%) | 63/72 (87.5%) |
| Mean latency, valid responses | 484.4 ms | 1,112.6 ms |
| p50 latency, valid responses | 338.2 ms | 956.7 ms |
| p95 latency, valid responses | 862.3 ms | 2,065.8 ms |
| Failed requests | 0 | 0 |

Every category matched September 21 except context, which scored 20/20 instead of 19/20.

The service was the default build with `hip,native`, started without `--backend`, so CubeCL served every request; the other server settings match the September 21 run. The measurement protocol was also the same: one round, concurrency 1, shuffle seed 42, no warmups or retries, and a 30-second timeout. Model loading, kernel warmup, and the stored tuned launch plans were ready before measurement.

One long-context request took 7.3 s because the new binary compiled a kernel variant on first use. This did not recur: a later fresh service took 1.45 s for that request, then 1.31 s on a second pass. Both of those later runs scored 76/84 again, with p50 latencies of 332.6 and 308.9 ms.

The [snapshot](results-2026-09-23-cubecl.json) includes all 72 response records, per-question evaluations, summary metrics, run settings, and binary/model hashes. Every response was rescored against the corpus, and the summary was recomputed and matched. The binary was built from the uncommitted `experiment/cubecl-inference` worktree.

## Recorded local results (2026-09-21)

The local rerun evaluated all **72 requests and 84 questions**, scoring **75/84 (89.3%)** with **72/72 valid responses**. This is 10 more correct answers than the September 19 local baseline, an increase of 11.9 percentage points. The corpus SHA-256 is unchanged: `9440d653972988809d32ca96d1978cfaf880778bff09c18b0894b41383b70534`. Hosted TypeSafe was not rerun; its September 19 results remain below as historical context.

| Metric | Local, Sep 21 | Local, Sep 19 | Hosted TypeSafe, Sep 19 |
| --- | --- | --- | --- |
| Valid responses | 72/72 | 72/72 | 71/72 |
| Question accuracy | 75/84 (89.3%) | 65/84 (77.4%) | 80/84 (95.2%) |
| Requests with every answer correct | 63/72 (87.5%) | 53/72 (73.6%) | 68/72 (94.4%) |
| Mean latency, valid responses | 1,112.6 ms | 683.6 ms | 427.7 ms |
| p50 latency, valid responses | 956.7 ms | 588.4 ms | 339.9 ms |
| p95 latency, valid responses | 2,065.8 ms | 1,314.6 ms | 881.0 ms |
| Mean latency, all attempts | 1,112.6 ms | 683.6 ms | 838.4 ms |
| Failed requests | 0 | 0 | 1 timeout at 30.0 s |

Question accuracy by category (correct/total):

| Category | Local, Sep 21 | Local, Sep 19 | Hosted TypeSafe, Sep 19 |
| --- | --- | --- | --- |
| Policy | 7/8 | 7/8 | 8/8 |
| Quantitative | 6/8 | 4/8 | 7/8 |
| Temporal | 7/8 | 4/8 | 6/8 |
| Evidence | 8/8 | 8/8 | 7/8 (one timeout) |
| Multilingual | 8/8 | 8/8 | 8/8 |
| Prompt injection | 7/8 | 7/8 | 8/8 |
| Relational | 7/8 | 3/8 | 8/8 |
| Rubric | 6/8 | 5/8 | 8/8 |
| Context | 19/20 | 19/20 | 20/20 |

The [saved snapshot](results-2026-09-21-defaults.json) includes all 72 response records, per-question evaluations, summary metrics, run settings, and binary/model hashes. Every saved response was re-scored against the corpus, and the summary was recomputed and checked for equality. The source was a clean tracked checkout at `11e826d9b32d9a4b5448485c1086ffb3e7c9a2d8`, built immediately before launching a dedicated service on port 8081.

| Configuration | Value |
| --- | --- |
| Hardware / platform | AMD Ryzen AI MAX+ 395 / Radeon 8060S, WSL2 Linux |
| Backend / build | ROCm 7.2.1, `gfx1151`, release, `hip,native`, `GGML_HIP_NO_VMM=ON` |
| Native revision | `12e0a9627d02c6395fd4bbf2aadff93d0d46a0e4` |
| Model | `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`; SHA-256 `24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5` |
| Server | Context 8192, batch 512, 8 threads, seed 42, all GPU layers, flash attention off, no vision projector |
| Request defaults | `steps=1`, `samples=1`, `think=0`, `sequential=false` |
| Client | SDK 0.6.0, Node.js 24.19.0, returned model `gemmadiffusion-0.1` |
| Measurement | One round, concurrency 1, shuffle seed 42, no warmups or retries, 30-second timeout, score tolerance ±0.5 |

The service had no prior inference requests. Model loading, compilation, and model hashing finished before measurement; first-request inference overhead remains included. Another local service remained loaded on port 8080, and other machine activity was not controlled. The September 19 snapshot lacks hardware and quantization provenance. These timings measure SDK latency and cannot isolate a speed change caused by the inference implementation.

To reproduce, first configure the [ROCm build/runtime environment](../../docs/build.md#rocmhip) and `DIFFUSION_MODEL`, then start a dedicated service. (The llama.cpp runs above used `--features hip,native` and `--threads 8`; that backend has been removed.)

```bash
env -u TYPESAFE_API_KEY -u DIFFUSION_MMPROJ \
  cargo run --release --locked -p jevons-rs -- \
  -m "$DIFFUSION_MODEL" --bind 127.0.0.1:8081 \
  --context-size 8192 --batch-size 512 --seed 42
```

After the service reports ready, run from the repository root in another terminal:

```bash
node examples/javascript/benchmark.mjs --endpoint local \
  --local-url http://127.0.0.1:8081 --timeout-ms 30000
```

## Recorded results (2026-09-19)

The full run starting at 14:30 UTC evaluated **72 requests and 84 questions per endpoint**, using SDK 0.6.0 and Node.js 24.19.0. It used one measured round, concurrency one, shuffle seed 42, no benchmark warmups or retries, and a 30-second per-call timeout. The returned model IDs were `gemmadiffusion-0.1` locally and `jev-1.13.0` on hosted TypeSafe. The [saved results snapshot](results-2026-09-19.json) preserves the run settings, corpus hash, returned model IDs, and summary metrics.

| Metric | Local jevons-rs | Hosted TypeSafe |
| --- | --- | --- |
| Valid responses | 72/72 | 71/72 |
| Question accuracy | 65/84 (77.4%) | 80/84 (95.2%) |
| Requests with every answer correct | 53/72 (73.6%) | 68/72 (94.4%) |
| Mean latency, valid responses | 683.6 ms | 427.7 ms |
| p50 latency, valid responses | 588.4 ms | 339.9 ms |
| p95 latency, valid responses | 1,314.6 ms | 881.0 ms |
| Mean latency, all attempts | 683.6 ms | 838.4 ms |
| Failed requests | 0 | 1 timeout at 30.0 s |

The runner measures SDK latency, including network travel and response parsing. The runner counts the hosted timeout as incorrect and includes it in the all-attempt mean. Valid-response latency excludes the timeout. Score questions pass within ±0.5 rubric levels of the expected value.

Question accuracy by category (correct/total):

| Category | Local jevons-rs | Hosted TypeSafe |
| --- | --- | --- |
| Policy | 7/8 | 8/8 |
| Quantitative | 4/8 | 7/8 |
| Temporal | 4/8 | 6/8 |
| Evidence | 8/8 | 7/8 (one timeout) |
| Multilingual | 8/8 | 8/8 |
| Prompt injection | 7/8 | 8/8 |
| Relational | 3/8 | 8/8 |
| Rubric | 5/8 | 8/8 |
| Context | 19/20 | 20/20 |

Hosted TypeSafe answered more questions correctly and had lower median latency in this run. This is one exploratory comparison on a small synthetic corpus with repeated context facts. The saved run did not capture local GPU, ROCm version, build flags, model file/quantization, server settings, or competing machine load, so these timings do not establish a reproducible ROCm hardware baseline or a general model ranking.

## Run

Requires Node.js 20.12+ and the running local service. From the repository root:

```bash
npm --prefix examples/javascript ci

# Validate the corpus and display the call budget without contacting either API.
npm --prefix examples/javascript run benchmark -- --dry-run

# Compare all 72 cases: 72 requests per endpoint, 144 HTTP calls total.
npm --prefix examples/javascript run benchmark

# Small trial: 5 cases per endpoint, 10 HTTP calls total.
npm --prefix examples/javascript run benchmark -- --limit 5

# One category or one endpoint.
npm --prefix examples/javascript run benchmark -- --category injection
npm --prefix examples/javascript run benchmark -- --endpoint local

# Optional repeated measurements with excluded warmups (more calls).
npm --prefix examples/javascript run benchmark -- --repeat 3 --warmup 2

# Validate the harness using the real SDK against a mock HTTP service.
npm --prefix examples/javascript run test:benchmark
```

The runner loads the repository `.env` automatically; exported environment variables take precedence. Use `--env-file PATH` to load a different file. It never prints API keys. `TYPESAFE_API_KEY` authenticates the **remote** service. Set `LOCAL_TYPESAFE_API_KEY` separately if the local server requires authentication; otherwise the SDK sends its `local-no-auth` placeholder. The local server itself still reads `TYPESAFE_API_KEY`, as described in the main README, so configure its shell separately from the hosted client key.

| Setting | Default / override |
| --- | --- |
| Local root | `http://127.0.0.1:8080`; `--local-url URL` |
| Hosted root | `https://api.typesafe.ai`; `--remote-url URL` |
| Local model | `gemmadiffusion-latest`; `--local-model ID` |
| Hosted model | `jev-latest`; `--remote-model ID` |
| Per-call timeout | 180,000 ms; `--timeout-ms N` |
| Shuffle seed | 42; `--seed N` |
| Measured rounds | 1; `--repeat N` |
| Warmups per endpoint | 0; `--warmup N` |
| Score accuracy tolerance | ±0.5 rubric levels; `--score-tolerance N` |
| Results directory | `benchmarks/results/<timestamp>/`; `--output DIRECTORY` |

The benchmark explicitly selects its URLs and models; it does not inherit the example client's `TYPESAFE_BASE_URL` or `TYPESAFE_DEFAULT_MODEL`. This prevents a comparison accidentally sending both sides to the same environment through shared defaults. URLs should be API roots, without `/v1/systemone`. Use immutable model IDs when available, and retain the returned model IDs recorded in results. A shuffle seed controls scheduling only; configure the local model's inference seed when starting the server.

## Coverage

Every category has eight cases. Policies, rubrics, and relevant facts are supplied in the requests, so the answer key does not depend on current events or obscure factual recall.

| Category | What it tests |
| --- | --- |
| `policy` | Inclusive boundaries, exception precedence, missing facts, conjunction and disjunction |
| `quantitative` | Weighted rates, sequential discounts, units, signed balances, deduplication and rounding up |
| `temporal` | UTC offsets, strict expiration, interval boundaries, midnight, leap day, superseded events |
| `evidence` | Entailment versus contradiction versus unknown; quantifiers, scope, claims versus verified facts |
| `multilingual` | Spanish, Portuguese, French, German, Japanese, Arabic, and mixed-language routing with negation |
| `injection` | Fake system messages, role spoofing, proposed JSON answers, quoted payloads, forged answer keys |
| `relational` | Joins, missing references, null versus zero, compound filters, deduplication, tie-breaking |
| `rubric` | Eight objectively anchored severity and report-completeness scores, spanning all four levels |
| `context` | Matched short/long states and one/four-question requests, with relevant evidence among distractors |

There are 50 choice questions, 22 noul questions, and 12 score questions. Context cases deliberately repeat facts while changing context length and question count. They test consistency and latency scaling, so the 72 cases are not 72 statistically independent samples. “Long” means longer than the matched baseline (roughly 3 KB of text), not a maximum-context stress test. The four-question canvas is intended to work with the default local 512-token batch and 8192-token context; tokenization and capacity errors are recorded as failures.

Each entry has a directly usable `request` object, an `expected` map keyed by question ID, and a human-readable `rationale`. To try one manually, send only its `request` and substitute the model ID as needed. Score targets are zero-based rubric levels. Keep ground truth out of `state` and question IDs when adding cases. The loader refuses corpora with more than 100 cases or inconsistent answer keys.

## Measurement and scoring

Calls are sequential, with concurrency one and retries disabled. Cases are shuffled reproducibly, paired across endpoints, and which endpoint runs first alternates. The default makes exactly one attempt per case per endpoint, with no hidden warmups or model-list calls. Warmups, if requested, are logged but excluded from all summaries. Repeats and warmups increase the number of actual HTTP calls; the runner prints that budget before starting.

Latency runs from immediately before `client.systemOne()` until the complete parsed response is available. It includes SDK overhead, connection establishment when needed, network travel, server queuing/inference, body transfer, and JSON parsing. SDK/fetch connection reuse is allowed. It excludes local answer scoring and writing results. This measures the client experience, not GPU forward time. Remote caching and deployment behavior are outside the runner's control. With no warmups, first-connection overhead is included.

The report includes:

- **Question accuracy:** noul uses `p >= 0.5`; choice uses the returned label; score passes when its fractional value is within the configured tolerance of the target. HTTP failures and missing/malformed answers count as incorrect. The accuracy among valid answers is also reported separately.
- **Request exact match:** every question in a request must pass, using the same score tolerance. An incorrect answer is a valid model result, not a transport failure.
- **Probability diagnostics:** binary Brier score for noul and natural-log loss for noul/choice; log loss clips the true-class probability at `1e-15`. These include only valid answers and are diagnostics on this small synthetic set, not evidence of production calibration.
- **Score errors:** mean absolute error in rubric levels and normalized by `levels - 1`, using the returned fractional score without rounding it to an integer.
- **Latency:** mean, p50, p95, minimum and maximum, separately for valid responses, all attempts, and failed attempts. Percentiles use linear interpolation. Failed calls cannot make successful-response latency appear faster.
- **Breakdowns:** category, question type, question count, and equal-weight mean of category accuracies. Multi-question requests have more weight in overall question accuracy.
- **Paired comparison:** latency differences, median remote/local latency ratio, and request-level accuracy wins/ties, restricted to pairs with valid responses on both sides. A ratio above 1 means remote was slower. The overall accuracy totals still include failures.

Distributions must contain exactly the expected labels, finite probabilities in `[0,1]`, and sum to one within `0.001`. Choices must agree with the maximum probability within `0.001`. Score legends must match the requested rubric and weighted means must agree within `0.01` levels. These small tolerances allow response rounding. Invalid responses remain visible as failures. Confidence is checked for a valid range but is not used as an accuracy weight or treated as a calibrated probability of correctness.

## Output and interpretation

Each run creates a new directory, refusing to overwrite an existing one:

- `run.json`: corpus SHA-256, selected IDs, SDK/Node versions, endpoint/model configuration, scheduling and scoring settings.
- `results.jsonl`: one record per attempt, including status, elapsed time, returned model/answers/usage, and per-question evaluation. Records are appended as calls complete so partial runs remain inspectable.
- `summary.json`: overall and grouped results, including failures and paired comparisons.

These files live under the gitignored `benchmarks/results/` by default. Error messages and bodies are excluded because servers can echo requests. Exit status is nonzero for transport or response-validation failures; low model accuracy alone does not fail the command. SDK retries are disabled, including on 429/529 responses.

Treat one run as an exploratory comparison. Eight cases per category and one timing sample per request cannot establish a reliable general ranking or stable tail latency. Freeze the corpus before comparing models; changing prompts after observing errors makes the result a development benchmark. Supplement it with held-out, labeled requests from the intended application. Keep native configuration, quantization, hardware, model IDs, and competing machine load with any published results. The two services use their own inference implementations/defaults; identical wire requests do not imply identical compute or probability calibration.

Protocol references: [official Choice documentation](https://docs.typesafe.ai/primitives/choice), [official Score documentation](https://docs.typesafe.ai/primitives/score), and the installed SDK 0.6.0 types/source. Checked 2026-09-19.
