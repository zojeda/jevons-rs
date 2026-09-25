# Nemotron-Labs-Diffusion: the text-only 3B, and self-speculative thoughts (2026-09-25)

Radeon 8060S (ROCm 7.2.1, WSL2), release build, CubeCL 0.11 / Burn 0.22, one model process at a
time. JevBench used the pinned harness at `fd51755`, one excluded warmup and default settings
(8,192-token context, seed 42). DiffusionGemma and the 8B `think=0` figures from the
[two-model report](../two-models-2026-09-24/README.md) serve as references; the 8B `think=0` run
was repeated with this build and reproduced 146/231.

## Smaller model: `nvidia/Nemotron-Labs-Diffusion-3B` (text only)

| Benchmark | DiffusionGemma Q4_K_M | Nemotron 3B | Nemotron VLM 8B |
| --- | ---: | ---: | ---: |
| JevBench public cases, correct | **189/231** | 143/231 | 146/231 |
| JevBench p50 / p95 latency | 0.400 / 3.696 s | **0.269 / 3.706 s** | 0.637 / 8.060 s |
| System One corpus, questions correct | **75/84** | 56/84 | 68/84 |
| System One p50 / p95 latency | 365.4 / 1,440.1 ms | **240.2 / 704.1 ms** | 572.9 / 1,990.2 ms |
| Snake prompt prefill p50 (~470 tokens) | 688 ms | **380 ms** | 1,028 ms |
| Snake canvas forward p50 | **57 ms** | 88 ms | 240 ms |

The 3B loads in 15 s and uses about 8 GB. It is 2.4–2.7 times faster than the 8B. On JevBench it
loses 3 cases, but on the System One corpus it loses 12 questions. Its canvas logits match the
official implementation (BF16, CPU) at the top token on 11/11 answer-canvas rows and, within
0.25 logits, on all 32 rows of a mask block. Its causal predictions match `ar_generate` on 32/32
teacher-forced tokens.

The 3B's chat template disables thinking. Given an opened thought, the model closes it with its
first token, so `think` changes nothing for this checkpoint.

## Self-speculation (`--decoding self-speculation`), VLM 8B

Linear self-speculation drafts a 32-token block with one bidirectional forward and verifies it
with one causal forward. It keeps the drafts that match the causal predictions, plus one causal
token. The result equals greedy autoregressive decoding. The engine test
`nemotron_self_speculation_reproduces_autoregressive_thoughts` checked this token for token on
three prompts (128-token budget):

| Thought decoding | Tokens per forward | Thought tokens/s |
| --- | ---: | ---: |
| diffusion (default) | 1.84 | 5.7 |
| self-speculation | **3.58** | **11.1** |
| autoregressive | 1.00 | 3.0 |

On the longest thought (88 tokens of arithmetic) self-speculation verified 12.7 tokens per round
and ran at 18.4 tokens/s, 6.2 times autoregressive decoding and 3.1 times diffusion. Very short
thoughts gain nothing, because every round costs two forwards. A causal forward costs about as
much as a 32-row canvas forward here (about 300 ms on the 8B), so the gain tracks tokens per
forward. On the 3B, NVIDIA's own `linear_spec_generate` produced the same 96 tokens as
`ar_generate` with 41 forwards instead of 96.

On JevBench (`think=256`) the model's thoughts are short: 4 tokens at the median and 15 at most.
Each fits one block, so the decodings take the same time:

| JevBench, VLM 8B | Correct | p50 / p95 latency |
| --- | ---: | ---: |
| `think=0` | **146/231** | **0.637 / 8.060 s** |
| `think=256`, diffusion | 145/231 | 2.051 / 11.287 s |
| `think=256`, self-speculation | 142/231 | 2.024 / 9.476 s |

Thinking does not improve JevBench accuracy for this model: diffusion thoughts fixed 8 cases and
lost 9, and self-speculative (greedy) thoughts fixed 6 and lost 10. The two thought decodings
agreed on 212 of 231 predictions. Self-speculation pays off for long generated text, not for
these classification prompts.

Not covered: the optional LoRA draft adapter of the text checkpoints (`linear_spec_lora`, rank
128 on `o_proj`), which NVIDIA reports raises acceptance, and the quadratic variant in the VLM
code. The quadratic variant verifies and redrafts in one forward over block × (block + 1) rows,
272 rows for its default 16-token block. NVIDIA reports 6.4 tokens per forward for it against 6.0
for the linear variant. Here a 272-row forward costs more than two 32-row forwards (prefill runs
at about 2.2 ms per row on the 8B), so it would be slower.

## Files and reproduction

Per-case JevBench results and summaries, per-request corpus results and `prefill_bench` JSON for
the 3B are in this directory. Raw HTTP evidence stays under the gitignored `benchmarks/results/`.

```bash
cargo build --release --locked -p jevons-rs --bins --example prefill_bench
BIN="${CARGO_TARGET_DIR:-target}/release"
python3 -B scripts/jevbench-local.py --port 8093 --binary $BIN/jevons-rs \
  --model ~/models/nemotron-labs-diffusion-3b --model-id nemotron-diffusion-3b \
  --harness "$JEVBENCH_SOURCE" --output benchmarks/results/jevbench-3b
python3 -B scripts/jevbench-local.py --port 8093 --binary $BIN/jevons-rs \
  --model ~/models/nemotron-labs-diffusion-vlm-8b --model-id nemotron-diffusion-8b \
  --think 256 --decoding self-speculation \
  --harness "$JEVBENCH_SOURCE" --output benchmarks/results/jevbench-8b-think256-selfspec
NEMOTRON_MODEL=~/models/nemotron-labs-diffusion-vlm-8b cargo test --release --locked \
  -p jevons-engine --lib -- --ignored --exact --nocapture \
  engine::tests::nemotron_self_speculation_reproduces_autoregressive_thoughts
```
