# Full CubeCL model, 2026-09-23

First end-to-end measurement of the [CubeCL backend](../../../docs/cubecl.md): the complete
DiffusionGemma forward (prompt prefill and canvas) on CubeCL/HIP, served through the unchanged
engine and System One request compiler.

## Result

Eight synthetic Snake requests (464-468 prompt tokens, 12-token canvas, steps=1, samples=1,
think=0), five rounds each after one warmup read per request. The prompt cache was disabled
(`--prompt-cache` not passed), so every request recomputes its whole prompt, matching the
native baseline's policy.

| Phase | llama.cpp p50 / p95 | CubeCL p50 / p95 | CubeCL latency |
| --- | ---: | ---: | ---: |
| Synchronized prefill | 1270.5 / 1388.6 ms | 628.5 / 712.0 ms | 0.49x |
| Canvas forward | 125.7 / 139.8 ms | 91.3 / 101.4 ms | 0.73x |
| Direct engine request | 1401.5 / 1532.4 ms | 720.8 / 811.1 ms | 0.51x |

The native numbers are the committed-baseline run in
[results-q4k-2026-09-21](../results-q4k-2026-09-21/native-full-prefill.json) (160 requests,
batch 512, flash attention off). CubeCL: [cubecl-snake.json](cubecl-snake.json) (40 requests).
The runs were not interleaved and clocks were not pinned; the RDNA3.5 GPU raises its clock under
sustained load, so short measurements vary.

## Accuracy

Every measured read matched its warmup exactly (the benchmark fails otherwise). Against the
native reference outputs:

- prompt token counts, canvas tokens, candidate tokens, slot positions and initial noise match
  exactly on all eight requests (tokenizer parity);
- the most probable answer matches on all eight slots;
- the largest candidate logit difference is 0.77 and the largest probability difference 0.19.

For scale, llama.cpp's own flash-attention and non-flash-attention paths differ by up to 0.93 in
logits and 0.17 in probability on the same requests. llama.cpp quantizes activations to 8 bits
in its matrix products; CubeCL multiplies FP16 activations with dequantized FP16 weights.

## Reproducibility and caching

- The engine's reproducibility and extension tests pass on the CubeCL backend (A/B/A requests,
  exact context boundary, averaged samples, 3-step self-conditioned refinement, thoughts,
  chunked and sequential questions, no state leakage).
- A prompt served from a reused prefix is bitwise identical to a fresh prefill of the same
  prompt (`examples/prefix_check.rs`, two prompt-pair scenarios). This required row-invariant
  prefill kernels; see the design notes in the backend guide.

## Commands

```bash
cargo build --release --locked -p jevons-rs --no-default-features --features cubecl \
  --example prefill_bench
prefill_bench --backend cubecl --model "$DIFFUSION_MODEL" \
  --requests benchmarks/cubecl/snake-requests.json --rounds 5
```

`load_ms` in the JSON (75.7 s) predates the pipelined loader and includes kernel warmup; the
loader now takes about 34 s on this machine.
