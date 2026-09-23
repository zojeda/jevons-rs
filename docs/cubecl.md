# CubeCL backend

[Back to README](../README.md)

The service runs DiffusionGemma with Rust [CubeCL](https://github.com/tracel-ai/cubecl) kernels
on AMD GPUs through HIP. It reads the same GGUF checkpoint and vision projector as llama.cpp,
tokenizes with a Rust port of the model's Gemma 4 tokenizer, and encodes images with a CubeCL
port of the Gemma 4 vision tower. Earlier versions of this project ran llama.cpp; that backend
was removed after CubeCL matched its answers (see [Performance](#performance)).

## Status

| Area | Status |
| --- | --- |
| Text requests, answer codes, framing, usage | Same behavior as the former llama.cpp backend; tokenizer matched llama.cpp exactly (all 262,144 token pieces, 269,290 test strings). |
| `steps`, `samples`, `think`, `sequential` | Supported, including the self-conditioning network used by refinement and thoughts. |
| Images (`--mmproj`) | Supported: `gemma4v` projector on the GPU, 70–280 tokens per image, bidirectional image blocks; cached image blocks are reused exactly. |
| Prompt KV reuse | Longest common token prefix is reused across calls; results are bitwise identical to recomputing the prompt. Disable with `--no-prompt-cache`. |
| Hardware | AMD RDNA3/RDNA3.5 (32-lane waves, WMMA). Tested on the Radeon 8060S (`gfx1151`) under ROCm 7.2.1/WSL. |

## Build and run

Install ROCm/HIP and set the [ROCm/WSL environment](build.md#rocmhip). Building needs
Rust 1.92+.

```bash
export DIFFUSION_MODEL="$HOME/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
cargo run --release --locked -p jevons-rs -- --bind 127.0.0.1:8080
```

| Option | Behavior |
| --- | --- |
| `--context-size` | Positions reserved in the KV cache for prompt, thought and canvas. |
| `--batch-size` | Maximum tokens per prefill chunk and canvas (at most 1024). |
| `--main-gpu` | HIP device index. |
| `--no-prompt-cache` | Recompute every prompt; useful for fresh-prefill measurements. |
| `--mmproj` | Load the vision projector (about 1.1 GB) and accept image input. |

Startup loads about 17 GB of weights onto the GPU (34 s on the test machine) and compiles
kernels. CubeCL caches compiled kernels in `~/.cache/diffusion-cubecl` (override with
`DIFFUSION_CUBECL_CACHE`), and the service warms common kernel variants before it reports
ready, so the first request does not pay compilation time.

### Device tuning

Tile shapes and split-K factors are not hardcoded for one GPU. On startup the backend times the
candidate launch plans for every weight shape in the model and every row-count bucket (powers
of two up to `--batch-size`):

- prompt products: row and column tile, without split-K;
- canvas products: tile, split-K factor, or the small-row kernel;
- expert products: row and column tile.

Each timed run cycles through the same-shape matrices of several layers, so candidates read
weights from memory as a forward pass does instead of from cache. A measured plan replaces the
built-in heuristic only when it is at least 5% faster.

Results are stored in `autotune.txt` in the kernel cache directory, keyed by the device
properties CubeCL reports and a kernel version, and are reused on later starts. The first
start on a new device also compiles every candidate kernel variant, which took about five
minutes on the test machine; with a warm kernel cache, tuning takes 15-45 s, and with a stored
table it is skipped.

| Variable | Effect |
| --- | --- |
| `DIFFUSION_CUBECL_AUTOTUNE=0` | Use the built-in heuristics; do not measure or read stored plans. |
| `DIFFUSION_CUBECL_AUTOTUNE=retune` | Measure again, ignoring the stored table. |
| `DIFFUSION_CUBECL_CUS` | Compute unit count (CubeCL's HIP runtime does not report it; default 40). It sets the workgroup target of the split-K heuristic and is part of the table key. |

Tuning cannot change a prompt's results: prompt plans only vary tile shapes, and each output
row is accumulated in the same order for every tile shape. The kernel tests check this bitwise.
On the 8060S, whose heuristics were hand-tuned, tuned plans perform the same as the heuristics
on 466-token prompts and are 3-4% faster on 1000-token prompts with a 256-token canvas
(`examples/tune_ab.rs`, interleaved in one process).

On APUs such as Strix Halo, GPU memory is system memory: the model, caches and a concurrent
build share the same RAM. Keep build directories on disk rather than in `/dev/shm` and avoid
running several model processes at once.

## Performance

Measured on 2026-09-23 on the Radeon 8060S, with both backends in one binary. See the
[backend comparison report](../benchmarks/cubecl/results-default-2026-09-23/README.md).

| | llama.cpp HIP | CubeCL |
| --- | ---: | ---: |
| Snake prompt prefill p50 (466 tokens, recomputed) | 1121 ms | 601 ms |
| Snake canvas forward p50 | 113 ms | 80 ms |
| JevBench public cases (p50 latency) | 189/231 (0.912 s) | 190/231 (0.424 s) |
| System One corpus (p50 latency) | 75/84 (956.7 ms) | 76/84 (338.2 ms) |

The llama.cpp service results are from 2026-09-21, with the same runners and settings. Answer
labels match llama.cpp on the Snake slots. Probability differences come from CubeCL's FP16
activations, where llama.cpp quantizes activations to 8 bits. They are comparable to
llama.cpp's own spread between its flash-attention and non-flash-attention paths.

Prompt reuse helps when one request needs several passes over the same prompt: multiple question
chunks, repeated samples, sequential chunks and thoughts reuse the resident prompt instead of
recomputing it.

### Images

Before the llama.cpp backend was removed, seven image requests (the hot dog example, flat and
patterned synthetic images from 64x48 to 640x360, two images in one request, and the hot dog with
`steps=3, samples=2`) were answered by both implementations on 2026-09-23:

- every answer label matched, and prompt token counts were identical (image token counts 77-104);
- the largest probability difference was 0.19, in the `steps=3` request; with the default
  `steps=1` it was at most 0.17 (the "circle" choice for a tall green circle, 0.55 vs 0.72);
- the hot dog example scored 0.971 hot dog and 0.925 mustard (llama.cpp: 0.953 and 0.894), with a
  922 ms request versus 1,298 ms. The released encoder uses fixed matrix tiles without split-K,
  which rounds slightly differently: 0.963 and 0.913. A repeated image request reuses the cached
  image block and prompt (0.36 s instead of 1.08 s).

Projected image embeddings had cosine similarity of at least 0.999 with llama.cpp's for most
tokens. A few high-norm tokens per image differed more (worst 0.69), which is consistent with
llama.cpp running parts of its vision graph in FP16. Encoding a 77-token image takes about 150 ms.

## Design notes

- **Weights stay quantized.** Q4_K, Q5_0, Q6_K and Q8_0 blocks are repacked into aligned
  structure-of-arrays buffers and dequantized to FP16 tiles inside the matrix kernels, which use
  RDNA3 16x16x16 WMMA instructions with FP32 accumulation.
- **Prefill** processes each chunk in one forward and writes the prompt KV cache; it never
  computes vocabulary logits. **Canvas** forwards attend bidirectionally over the canvas and the
  cached prompt. With `steps=1`, only the candidate answer logits are computed.
- **Small row counts** (canvas projections, experts with one or two tokens) use split-K tiles or a
  barrier-free half-wave matrix-vector kernel.
- **Exact prefix reuse.** Prefill kernels are row invariant: a token's result never depends on
  which other tokens share its chunk. RDNA3 WMMA rounding depends on values paired with zero
  probabilities, so attention aligns query tiles to 16 positions, accumulates each tile's
  diagonal key chunk with scalar FMAs and zeroes cache contents beyond the visible keys.
- **Router and grouping** run on the GPU; experts are grouped deterministically.
- **Vision encoder.** Images are resized with llama.cpp's Pillow-compatible fixed-point bicubic
  filter and padding, then run through the 27-layer Gemma 4 ViT (per-head Q/K RMS norms, 2D NeoX
  rotary embeddings, weightless V norm, quick-GELU gate as llama.cpp uses when the GGUF names no
  activation), 3x3 average pooling, standardization and the projection into the text width.
  F16 weights use the same matrix kernels; the 4304-wide FFN is zero-padded to 4352.
- **Image prefill.** Each image is one forward between `<|image>` and `<image|>`; its rows skip the
  embedding scale and attend to every key in their block, while earlier text stays causal (and
  per-query windowed in sliding layers). Image rows have content-derived cache keys, so a repeated
  image is reused exactly.

- **No int8 matrix path.** llama.cpp quantizes activations to 8 bits and uses integer dot
  products. CubeCL 0.10's HIP backend exposes only FP16/BF16 WMMA on RDNA3, and its `dot`
  compiles to scalar multiplies: LLVM does not form `v_dot4` from them
  (`examples/dot4_probe.rs`). An int8 path would need changes to CubeCL itself.

The practical ceilings measured on the 8060S under WSL are about 20 TFLOP/s for FP16 WMMA,
6 TFLOP/s for FP32 vector math and 190 GB/s of memory bandwidth. Prefill is now dominated by
the routed-expert products, which are limited by dequantization throughput.

## Tests

```bash
# Kernel tests against CPU references (GPU required)
cargo test --locked --release -p jevons-cubecl --features hip --lib -- --include-ignored
# Engine model tests: one process per test (each loads the 17.7 GB model)
for t in model_reads_preserve_reproducibility_across_requests \
         model_extensions_average_refine_think_and_chunk \
         model_images_prefill_and_preserve_text_reproducibility; do
  cargo test --locked --release -p jevons-engine --lib -- --ignored --exact "engine::tests::$t"
done
```

The image test also needs `DIFFUSION_MMPROJ`. `tokenizer_matches_llama_reference` compares against
a llama.cpp tokenizer dump (`TOKENIZER_REFERENCE`); see its doc comment.

`crates/jevons-cubecl/examples/prefix_check.rs` compares reused-prefix results with fresh
prefills bitwise; `gemm_bench`, `moe_bench` and `bandwidth` measure kernels in isolation;
`tune_ab` compares tuned and heuristic plans on the full model; `vision_check` compares image
embeddings with reference dumps.
