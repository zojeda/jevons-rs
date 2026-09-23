# CubeCL as the default backend, 2026-09-23

These measurements were taken after CubeCL became the default backend, before the llama.cpp
backend was removed (the commands below need a checkout from that time). The build compiled both
backends (`--features hip,native`), and the same binary ran each backend in turn. CubeCL used
the launch plans it had tuned for this device ([device tuning](../../../docs/cubecl.md#device-tuning)).

## Snake prefill benchmark

The workload was the eight synthetic Snake requests in [snake-requests.json](../snake-requests.json):
464-468 prompt tokens, 8-12 canvas tokens, `steps=1`, `samples=1`, `think=0`. Each request
ran five rounds after one warmup read, giving 40 samples per backend. Settings: batch 512,
context 8192, flash attention off. Percentiles use nearest rank over all 40 samples.

| Phase, p50 / p95 | llama.cpp HIP | CubeCL | CubeCL, prompt cache on |
| --- | ---: | ---: | ---: |
| Synchronized prefill | 1121 / 1219 ms | 601 / 681 ms | 613 / 635 ms |
| Canvas forward | 113 / 126 ms | 80 / 89 ms | 88 / 94 ms |
| Direct engine request | 1239 / 1340 ms | 687 / 768 ms | 704 / 729 ms |

At the median, CubeCL takes 0.54x llama.cpp's prefill time and 0.55x its request time (about
1.8x faster). Every measured read matched its warmup exactly on both backends.

With the prompt cache on, consecutive requests reused only the shared prompt header, 49.5
tokens on average, because each Snake request describes a different board early in the prompt.
Prompt reuse pays off when one request needs several passes over the same prompt: question
chunks, repeated samples, sequential chunks and thoughts. The difference between the two CubeCL
columns is within run-to-run variation.

The runs were sequential rather than interleaved, and clocks were not pinned. On this GPU,
separate runs of the same configuration vary by about ±6%.

Files: [llama-snake.json](llama-snake.json), [cubecl-snake.json](cubecl-snake.json),
[cubecl-snake-prompt-cache.json](cubecl-snake-prompt-cache.json).

## Service benchmarks

The same binary served both corpora with the default backend. See the
[JevBench report](../../jevbench/README.md) and the
[System One corpus report](../../system-one/README.md).

| Benchmark | llama.cpp HIP (2026-09-21) | CubeCL (2026-09-23) |
| --- | ---: | ---: |
| JevBench public cases, correct | 189/231 (81.8%) | 190/231 (82.3%) |
| JevBench p50 / p95 latency | 0.912 / 8.124 s | 0.424 / 4.401 s |
| System One corpus, questions correct | 75/84 (89.3%) | 76/84 (90.5%) |
| System One p50 / p95 latency | 956.7 / 2,065.8 ms | 338.2 / 862.3 ms |

## Commands

```bash
cargo build --release --locked -p jevons-rs --features hip,native --examples
for backend in cubecl llama; do
  target/release/examples/prefill_bench --backend "$backend" --model "$DIFFUSION_MODEL" \
    --requests benchmarks/cubecl/snake-requests.json --rounds 5 > "$backend-snake.json"
done
target/release/examples/prefill_bench --backend cubecl --prompt-cache --model "$DIFFUSION_MODEL" \
  --requests benchmarks/cubecl/snake-requests.json --rounds 5 > cubecl-snake-prompt-cache.json
```

The first CubeCL run's `load_ms` (71 s) includes compiling kernel variants for the 8,192-token
context; later starts took 41 s.
