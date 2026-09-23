# Initial implementation results, 2026-09-21

These results establish a starting point for the CubeCL experiment. No CubeCL
DiffusionGemma model execution or model-level speedup is implemented yet.

## Assets and environment

- Base: `0e3a62fdececf5ccb69b705ae8d1bde38716d3f2`, plus the experiment's shared-type
  extraction and prefill counters. Native source: `12e0a9627d02c6395fd4bbf2aadff93d0d46a0e4`.
- Rust 1.98.1; ROCm 7.2.1; AMD Radeon 8060S (`gfx1151`), WSL.
- Burn 0.21.0, CubeCL 0.10.0, CubeK 0.2.0; see the workspace lockfile.
- [Model inventory](model-inventory.json): checkpoint SHA-256
  `24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5`.
- 692 tensors: F32 (423), Q4_K (194), Q5_0 (33), Q6_K (14), Q8_0 (28).
  Stored tensor bytes: 16,790,983,920. FP16 expansion of all stored tensor elements:
  50,501,974,136 bytes, excluding KV, activations and workspace.

The inventory confirms 2816 embedding dimensions, 128 experts with eight selected,
704 expert intermediate dimensions, and a 262144-token vocabulary. It supplies
actual matrix dimensions for subsequent prefill kernel experiments.

## Committed native baseline

Command after the documented HIP release build:

```bash
prefill_bench --model "$DIFFUSION_MODEL" \
  --requests benchmarks/cubecl/snake-requests.json --rounds 5
```

[Raw results](native-prefill-snake.json): eight warmup requests, then 40 measured
requests. Every measured read exactly matched its warmup logits, probabilities,
noise, token positions and usage. The model load took about 28.5 seconds and is
excluded from per-request latency. There were 464–468 prompt tokens and four
executed prefill batches per request, with no cross-request cache reuse.

| Phase | Median across all 40 requests |
| --- | ---: |
| Total direct engine read | 1258.8 ms |
| Synchronized prefill | 1137.4 ms |
| Canvas forward | 114.3 ms |

Phase medians do not necessarily sum to total. The global median combines eight
different inputs; per-case summaries and raw samples are included in the JSON.
Five rounds per input do not establish a stable p95. Host compilation overlapped
part of this exploratory run; no other GPU inference was deliberately launched.
Use a longer isolated run for a performance acceptance baseline. These numbers
support prioritizing prefill but are not a controlled comparison to the sibling
checkout's 900 ms prefill result, which used additional native integration and
cache changes.

Proposed initial performance gate: at least 20% lower fresh-prompt median prefill
on the matched Snake workload and representative longer prompts, with improved
end-to-end median, no more than 5% p95 regression, and the agreed memory/numerical
limits. Re-establish the baseline with at least 20 rounds per case and repeat runs
before judging that gate. Cache-hit gains are reported separately. This is an
experimental target, not a forecast of CubeCL's performance.

## HIP feasibility

The HIP probe checks elementwise arithmetic, reduction, and float matmul, including
actual projection/expert dimensions from the inventory. Debug-build timing is
diagnostic only; the test does not execute quantized weights, routing, attention,
or a full model. Initial execution/JIT and synchronized repeated executions are
reported separately in [the raw probe JSON](hip-probe-debug.json). The run passed
elementwise/reduction checks and five repetitions of each shape below. Checked
matmul entries had zero error against CPU f64 accumulation for these inputs;
larger matrices were sampled, not checked exhaustively.

| M × K × N | Checked entries per repetition | Repeated matmul median, debug build |
| --- | ---: | ---: |
| 3 × 7 × 5 | 15 (all) | 0.75 ms |
| 128 × 2816 × 4096 | 64 | 3.44 ms |
| 512 × 2816 × 4096 | 64 | 10.68 ms |
| 128 × 2816 × 704 | 64 | 1.74 ms |

These synchronized host timings exclude upload/readback, include dispatch, and
are not GPU kernel durations. First execution ranged from 0.56 to 4.11 seconds;
the process did not clear persistent compiler caches. Use a release build and
actual quantized data paths before drawing throughput conclusions.

## Verification

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check`, `git diff --check` | Passed. |
| `cargo test --workspace --locked` | 20 regular tests passed; four model-dependent tests ignored by this command. |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | Passed. |
| HIP probe build and HIP-feature Clippy | Passed with the pinned optional dependencies. |
| Native reproducibility test, release HIP/native | Passed with the actual model, including counter reset after validation failure. |
| Native extension test, release HIP/native | Passed: repeated samples, refinement, thoughts, chunking and sequential context; physical prefill count remains distinct from logical usage. |
| Native Snake baseline | 40 measured reads matched their warmup references exactly. |
| HIP numerical probe | Elementwise/reduction and all sampled matmul checks passed. |
| Inventory | File hash captured; tensor storage totals and model dimensions checked. |
| Dependency boundary | `jevons-system-one` tests build without native code; the HIP probe has no llama/ggml dependency. |

The two image-specific ignored tests were not run in this slice. The service and
SCM CLI still use their existing native engine; there is no CubeCL engine selector
yet. HTTP request/response types and usage behavior were preserved.

## Remaining work

Complete the low-level backend adapter; evaluate actual quantized prefill kernels
and their memory behavior; implement GGUF/tokenizer parity and the DiffusionGemma
graph. A successful dense float matmul does not settle the Burn versus direct
CubeCL choice. The service still uses llama.cpp.
