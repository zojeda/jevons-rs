# jevons-gemma4-diffusion

## Purpose

`jevons-gemma4-diffusion` runs DiffusionGemma, Google's text diffusion model on the Gemma 4 26B A4B
mixture-of-experts architecture, on tuned CubeCL kernels over HIP. It loads the GGUF checkpoint that
llama.cpp reads, prefills prompts with causal attention into a KV cache that it reuses across calls,
evaluates canvases with bidirectional attention and optional self-conditioning, and returns
candidate or full logits. With a `gemma4v` projector it encodes images into prompt rows.

## Scope

It owns:

- `model`: the checkpoint's hyperparameters and their limits, weights on the device, prompt prefill
  with prefix reuse, canvas forwards, self-conditioning and logits.
- `gpu::attention`, `gpu::ops`, `gpu::vision`: attention with DiffusionGemma's visibility rules,
  expert routing and grouping, fused norms, embedding and logit kernels, and the vision encoder's
  kernels.
- `gpu::tune`: launch plans measured per device and their stored table.
- `vision_input` and `vision`: image preprocessing and the Gemma 4 vision encoder.

It leaves to other crates:

- Device buffers and the matrix products to `jevons-kernels`.
- GGUF reading and block packing to `jevons-formats`, and the Gemma 4 tokenizer to
  `jevons-tokenizer` (both re-exported).
- The `DiffusionModel` adapter to `jevons-models`: chat markers, answer-code pieces, the 8-image
  cap, `--mmproj` and `--batch-size` errors for images, and the prompt-cache switch.
- Sampling, thoughts and answer reads to `jevons-diffusion` and `jevons-decision`.

The examples `gguf_info`, `prefix_check`, `golden_dump` and `vision_check` are diagnostics.
Principle P1 applies: the crate forbids unsafe code and its kernels launch in checked mode.

## Requirements

### R1 Supported checkpoints

The GGUF's `general.architecture` must be `diffusion-gemma`. Loading fails as unsupported when the
model width is not a multiple of 256, the expert count exceeds 128 or is not a multiple of 32, top-k
exceeds 32, a head size is neither 256 nor 512, rotary or value lengths differ from the head sizes,
`rope_freqs` does not have half as many entries as the full head size, a per-layer array does not
have one entry per layer, or a layer's KV head count is zero or does not divide the head count.

Tests: none yet

### R2 Weights stay quantized

Matrix weights stay on the device in their GGUF block encodings: Q4_K, Q5_0, Q6_K or Q8_0. The token
embedding, which doubles as the output projection, must be Q6_K. A tensor with an unexpected shape
or type, or a fused gate and up pair whose halves differ, fails as unsupported. Embedding rows and
candidate logits read from the Q6_K table match the dequantized table (embedding within 1e-5,
candidate logits within 1e-4).

Tests: `q6k_embedding_rows_and_candidate_logits_match_dequantized_table`

### R3 Context and batch limits

The context rounds up to a multiple of 64 positions. The batch size must lie between 1 and the
smaller of the context and 1024, or loading fails with an input error. A prompt must be shorter than
the context, a canvas must fit in one batch and in the context after the prompt, and every token id
must lie in the vocabulary; each violation fails with an input error.

Tests: none yet

### R4 Prefill writes the prompt cache

Prefill runs causal attention over the prompt in chunks of at most the batch size, writes keys and
values for every layer and computes no vocabulary logits. Prompt queries in sliding-window layers
see keys less than `window` positions back. Attention stays within 2e-2 and Q/K/V preparation within
2e-3 of f64 references.

Tests: `attention_matches_reference_for_causal_sliding_and_canvas_queries`, `qkv_preparation_normalizes_rotates_and_writes_caches`

### R5 Exact prefix reuse

Each prefill keeps the longest prefix of resident positions whose keys (token ids, or content keys
for image rows) match the new prompt, and recomputes the rest. A prompt row's keys, values and
outputs do not depend on which rows share its chunk, on later keys, or on cache contents past the
visible keys, so reused rows are bitwise identical to a fresh prefill. A prefill that fails forgets
the whole cache.

Tests: `every_tunable_dense_plan_matches_reference_and_tiles_are_row_invariant`, `attention_rows_do_not_depend_on_query_chunking`, `causal_rows_are_bitwise_independent_of_later_keys`, `attention_ignores_cache_contents_beyond_the_visible_keys`, `model_reads_preserve_reproducibility_across_requests`

### R6 Image blocks

An image segment is prefilled in one forward of at most the batch size; a segment with no rows or
more rows than the batch fails with an input error. Image rows skip the embedding scale and attend
to every key in their block, while earlier text stays causal and windowed. Reuse does not resume
inside an image block, and a repeated image reproduces the read bit for bit.

Tests: `attention_matches_reference_for_causal_sliding_and_canvas_queries`, `model_images_prefill_and_preserve_text_reproducibility`

### R7 Canvas forwards

A canvas must follow the resident prompt: a prompt length other than the resident one fails with an
input error. Canvas queries see every canvas key and every prompt key, or in sliding-window layers
the last `window - 1` prompt keys. A canvas forward leaves the resident prompt unchanged.

Tests: `attention_matches_reference_for_causal_sliding_and_canvas_queries`, `model_images_prefill_and_preserve_text_reproducibility`

### R8 Logits

Logits are soft-capped by the checkpoint's `final_logit_softcapping`. A canvas forward computes all
rows × vocabulary logits when asked for full logits. Without them, callers read candidate logits at
one row. An out-of-range row or candidate, or a request for full logits after a forward that
computed none, fails with an input error.

Tests: `q6k_embedding_rows_and_candidate_logits_match_dequantized_table`

### R9 Self-conditioning

A canvas forward can take the previous step's full logits and an inverse temperature, which
condition its embeddings through the checkpoint's self-conditioning network. Logits of the wrong
shape fail with an input error.

Tests: none yet

### R10 Norm and residual kernels

The fused norm kernels match f64 references: residual streams within 1e-5, router inputs within
1e-5, and FP16 attention, FFN and expert inputs within 2e-3.

Tests: `post_attention_norms_update_residual_and_emit_ffn_inputs`, `post_ffn_norms_combine_experts_residual_and_scale`

### R11 Expert routing

Routing picks each token's top-k experts with normalized weights (within 1e-4 relative of the
reference). Grouping places every assignment once, in its expert's range, and the grouped products
use each assignment's expert weights.

Tests: `routing_selects_normalized_top_k_and_grouping_covers_every_assignment`, `grouped_products_use_each_assignments_expert_weights`

### R12 Launch plans are measured per device

Tuning times candidate plans for every weight shape in the model and every power-of-two row bucket
from 4 up to the batch size (at most 1024); a bucket `b` serves row counts in `(b/2, b]`. A measured
plan replaces the heuristic when it is at least 5% faster. Prefill products choose among
row-invariant plans, so tuning cannot change a prompt's results.

Tests: `heuristic_plan_is_kept_unless_clearly_slower`, `row_buckets_cover_each_power_of_two_range`

### R13 The tuning table

Measured plans are stored in `autotune.txt` in the kernel cache directory, under a key of the tuning
version and the device profile (device index, compute units, wave size, shared memory, page size,
load width). A table with another key is ignored. `DIFFUSION_CUBECL_AUTOTUNE=0` or `off` uses the
heuristics and neither reads nor measures plans; `retune` ignores the stored table and measures
every plan. `DIFFUSION_CUBECL_CUS` sets the compute unit count, which defaults to the runtime's
value or 40.

Tests: none yet

### R14 Warmup

`warmup` tunes and compiles the common kernel variants: prompt chunks of both tile heights, a canvas
with candidate and with full logits, and one self-conditioned step. It clears the prompt cache
before it returns.

Tests: none yet

### R15 A dropped model frees its memory

Dropping the model returns its pooled device memory to the device, through a release guard
(`jevons-kernels` R13) that runs after every buffer of the model is dropped: the layers, the
embeddings, the caches, the scratch buffers and the logits. The vision tower's buffers are
dropped before the model, so the same release covers them.

Tests: `the_release_guard_returns_what_was_dropped_before_it`

### R16 Vision projectors

The projector GGUF must have `clip.vision.projector_type = gemma4v` and a projection width equal to
the text model's width. Its width must be a multiple of 128, its head size a multiple of 4, and it
must name no GELU or SiLU activation (the encoder uses the quick-GELU gate). Matrices must be F16
and norm weights F32 with the expected element counts. Any other projector fails as unsupported.

Tests: none yet

### R17 Image sizes

An image is resized, keeping its aspect ratio, to multiples of `patch × merge` (48 pixels for
Gemma 4) so that its token count `(w / 48) × (h / 48)` falls in 70..=280 where possible. Token
counts match llama.cpp's `mtmd`: 330×220 gives 77, 224×224 gives 81, 640×360 gives 104, 300×700
gives 90 and 64×48 gives 80.

Tests: `target_sizes_match_llama_cpp_token_counts`

### R18 Image preprocessing

Resizing uses Pillow's fixed-point bicubic filter and centers the result on black. Patches are taken
in raster order, each laid out by channel, row and column, with pixel values mapped to
`2v / 255 - 1`.

Tests: `bicubic_resize_preserves_flat_colors_and_pads_with_black`, `patches_are_channel_major_and_scaled_to_unit_range`

### R19 Vision encoding

`encode` returns one projected row per image token, in the text model's width. An image with zero
size or a byte count other than `width × height × 3` fails with an input error, as does an image
whose patch grid exceeds the position table. The encoder kernels (per-head Q/K norms with 2D
rotation by patch column and row, bidirectional attention over partial key tiles, the quick-GELU
gate, position embeddings, norms and 3×3 pooling) match f64 references within 1e-2, and the
post-norm residual within 1e-4.

Tests: `vision_qkv_normalizes_heads_and_rotates_by_patch_column_and_row`, `vision_attention_is_bidirectional_over_partial_key_tiles`, `vision_ffn_pooling_positions_and_norms_match_reference`
