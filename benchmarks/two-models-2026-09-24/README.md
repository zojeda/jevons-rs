# DiffusionGemma and Nemotron-Labs-Diffusion, 2026-09-24

Both models on the current single GPU stack (CubeCL 0.11 / Burn 0.22, commit `2178b61` plus the
benchmark tolerance option), measured back to back on one machine: AMD Ryzen AI MAX+ 395 / Radeon
8060S, ROCm 7.2.1, WSL2, release build. One model process ran at a time. Service settings:
`--context-size 8192 --batch-size 512 --seed 42`, default request options (`steps=1`,
`samples=1`, `think=0`), no authentication, no `--mmproj`.

- DiffusionGemma 26B-A4B, Q4_K_M GGUF (`gemmadiffusion-0.1`)
- Nemotron-Labs-Diffusion-VLM-8B, BF16 safetensors (`nemotron-diffusion-8b`); projections run as
  FP16 on the tuned GEMM

## Results

| Benchmark | DiffusionGemma | Nemotron |
| --- | ---: | ---: |
| JevBench public cases, correct | **189/231 (81.8%)** | 146/231 (63.2%) |
| · easy / standard / hard | 48/48 · 69/72 · 72/111 | 48/48 · 47/72 · 51/111 |
| JevBench p50 / p95 latency | **0.400 / 3.696 s** | 0.644 / 7.934 s |
| JevBench Brier (lower is better) | **0.272** | 0.504 |
| System One corpus, questions correct | **75/84 (89.3%)** | 68/84 (81.0%) |
| System One requests with every answer correct | **63/72** | 56/72 |
| System One p50 / p95 latency | **365.4 / 1,440.1 ms** | 572.9 / 1,990.2 ms |
| Snake prompt prefill p50 / p95 (464–478 tokens, uncached) | **687.7 / 835.6 ms** | 1,029.9 / 1,176.7 ms |
| Snake canvas forward p50 | **56.6 ms** | 239.3 ms |
| Snake request p50 / p95 | **746.6 / 912.1 ms** | 1,266.6 / 1,433.6 ms |
| Synthetic prompts with reuse, request p50 / p95 | **359.9 / 866.1 ms** | 510.5 / 1,697.5 ms |
| Model load (warm kernel cache) | 28–59 s | 35–45 s |

Every run returned only valid responses (231/231 and 72/72 for both models).

## Notes

- **DiffusionGemma vs. the CubeCL 0.10 build (2026-09-23).** JevBench 189 vs 190 and the corpus
  75 vs 76 questions each differ by one near-tie: `hard-opus-c-probability-03` (0.452/0.436
  before, 0.453/0.473 now) and `relational-join_missing` (`unknown` at 0.48 before, 0.30 now).
  JevBench latency and calibration improved (p50 0.400 vs 0.424 s, p95 3.70 vs 4.40 s, Brier
  0.2721 vs 0.2741). In interleaved A/B runs of the two builds earlier the same day, the new build
  prefilled 15–25% faster; absolute timings vary with machine state between days.
- **Corpus tail latency.** The corpus runner starts from a freshly started service with no
  warmup, as in earlier runs. For DiffusionGemma, two requests (7.7 s and 6.9 s) paid one-time
  kernel compilation for new shapes, which sets its p95 and mean (648 ms); the 2026-09-23 run had
  one such request.
- **Nemotron's errors are the model's.** Easy-tier and single-skill groups are all correct;
  losses concentrate in routing, ordinal and hard cases. On a failing routing case the official
  Python implementation gives the same answer and the same masked-token distribution.
- **Nemotron speed.** A canvas forward (32 positions over the prompt) takes about 240 ms here:
  every forward runs the full 8B model, and Burn's many small elementwise kernels dominate once
  the tuned GEMM handles the products. DiffusionGemma's MoE activates about 4B parameters.
- **Nemotron prompt reuse is not bitwise.** DiffusionGemma's prefill kernels are row invariant,
  so a reused prefix gives bitwise-identical reads. Nemotron's attention runs on Burn's flash
  kernel, whose tiling depends on shapes; after partial reuse the largest probability difference
  from the first read was 0.0004 (`prefill_bench --prompt-cache --tolerance`). Repeating an
  identical prompt reproduces the read.

## Files

Per-case JevBench results and summaries, per-request corpus results, and `prefill_bench` JSON for
each model are in this directory. Raw HTTP evidence and server logs stay in the gitignored
`benchmarks/results/run-2026-09-24T13-52-54Z`.

## Reproduce

```bash
cargo build --release --locked -p jevons-rs --bins --example prefill_bench
BIN="${CARGO_TARGET_DIR:-target}/release"
# Prefill (per model; use --model "$NEMOTRON_MODEL" for Nemotron, a checkpoint directory):
$BIN/examples/prefill_bench --model "$DIFFUSION_MODEL" --requests benchmarks/cubecl/snake-requests.json --rounds 5
$BIN/examples/prefill_bench --model "$DIFFUSION_MODEL" --rounds 5 --prompt-cache           # Nemotron: add --tolerance 0.001
# Corpus: start the service, then
node examples/javascript/benchmark.mjs --endpoint local --local-url http://127.0.0.1:8081 \
  --local-model nemotron-diffusion-8b --timeout-ms 30000
# JevBench (starts its own service):
python3 -B scripts/jevbench-local.py --port 8093 --binary $BIN/jevons-rs --model "$NEMOTRON_MODEL" \
  --model-id nemotron-diffusion-8b --harness "$JEVBENCH_SOURCE" --output benchmarks/results/jevbench-nemotron
```
