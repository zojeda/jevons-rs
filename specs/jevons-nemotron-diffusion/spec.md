# jevons-nemotron-diffusion

## Purpose

`jevons-nemotron-diffusion` runs NVIDIA's Nemotron-Labs-Diffusion on Burn: a Ministral-3 decoder
trained for LLaDA-style masked diffusion over 32-token blocks, from a Hugging Face checkpoint
directory. It serves the text-only checkpoints (3B, 8B, 14B) and the VLM, whose Pixtral tower
encodes images, and implements the `DiffusionModel` contract. Its logits, image features and greedy
predictions are checked against dumps of the official Python implementation.

## Scope

It owns:

- `config`: the checkpoint's `config.json` and the variants this runtime accepts.
- `rope`: default and YaRN rotary frequencies.
- `image` and `vision`: Pixtral preprocessing, the vision tower and the multimodal projector.
- `model`: weight loading, the chat format, the masked-diffusion scheme, prompt prefill with prefix
  reuse, canvas forwards, logits and causal predictions.

It leaves to other crates:

- Device, weight streaming, layers and the tuned GEMM to `jevons-burn`.
- Safetensors reading to `jevons-formats` and the tokenizer to `jevons-tokenizer`.
- Detection and the default model ID to `jevons-models`.
- The masked denoiser, self-speculation and autoregressive thoughts to `jevons-diffusion`.

The reference dumps come from `scripts/reference/nemotron_dump.py --dtype bfloat16` under
`$JEVONS_GOLDEN_DIR`, in the directory `NEMOTRON_GOLDEN` names (default `nemotron-diffusion-bf16`,
the VLM). `prefill_layer_outputs_match_the_reference_implementation` prints per-layer errors as a
diagnostic and checks nothing. Principle P1 applies: the crate forbids unsafe code.

## Requirements

### R1 Supported configurations

`config.json` must have `model_type` `nemotron_labs_diffusion` or `nemotron_labs_diffusion_vlm`,
`dlm_paradigm` `bidirectional`, `rope_type` `yarn` or `default`, no sliding window, `silu`
activations, no attention or MLP biases, untied embeddings, a head count that is a multiple of the
KV head count, an even head size, a `mask_token_id` inside the vocabulary and a nonzero
`block_size`. Any other value fails with `UnsupportedModel`; a missing `config.json` fails with
`ModelLoad`. The published 3B and VLM 8B configurations parse.

Tests: `the_text_only_3b_config_parses`, `the_8b_config_parses_and_unsupported_variants_are_rejected`

### R2 Load checks

Loading fails with `InvalidInput` when the configuration names `mmproj` (the VLM keeps its tower in
the checkpoint) or when the context size exceeds `original_max_position_embeddings`. It fails with
`UnsupportedModel` when the tokenizer's vocabulary size differs from the model's, when a VLM's
vision config has a patch size other than 14 or an activation other than `silu`, or when its
tokenizer does not map `<|image_start|>`, `<|image_pad|>`, `<|image_break|>` and `<|image_end|>` to
ids 18 to 21.

Tests: none yet

### R3 Weights

Weights load from safetensors one tensor at a time onto HIP device `main_gpu`. Projections and the
diffusion head are converted from BF16 to FP16 for the tuned GEMM, with rows padded to multiples of
64, and logits are cut back to the vocabulary. A missing tensor, or one with another shape, fails
with `UnsupportedModel`.

Tests: `fp16_conversion_preserves_real_projection_weights`

### R4 Rotary positions

Rotary frequencies follow `transformers` 4.57 in f32. YaRN keeps the fast frequencies, divides the
slow ones by `factor` and ramps between them across the correction range, whose bounds are truncated
to whole pairs by default. The cos and sin tables repeat each frequency across both halves of a
head. Within the allowed context, the Llama-4 query scale is 1.

Tests: `yarn_keeps_fast_frequencies_and_interpolates_slow_ones`, `tables_repeat_frequencies_across_halves_and_query_scale_starts_at_one`

### R5 Chat format

Prompts use ChatML without BOS: `<|im_start|>user\n` opens the user turn and
`<|im_end|>\n<|im_start|>assistant\n` opens the model turn. A thought opens with `<think>\n`. The
empty thought, `<think></think>`, starts each earlier model turn of a conversation. Answers stop at
`<|im_end|>` or `</s>`, and thoughts at `</think>` or `<|im_end|>`. The tokenizer joins a leading
space to the next word, so a slot prefix's trailing space moves onto the answer codes:
`"Answer: " + "A"` reads as `"Answer:"` and `" A"`.

Tests: none yet

### R6 Masked diffusion

The model reports masked diffusion with the checkpoint's mask token and block size, a confidence
threshold of 0.9 and at most `block_size` steps. A canvas holds at most one block. Asking for
self-conditioning fails with `InvalidInput`.

Tests: `nemotron_reads_are_calibrated_reproducible_and_support_extensions`

### R7 Prompt prefill and reuse

Prefill runs causal attention in batches of at most `batch_size` and reuses the longest prefix of
resident positions whose keys (token ids, or content keys for image rows) match; with `prompt_cache`
off, it recomputes the whole prompt. A prompt longer than the context, or a token id outside the
vocabulary, fails with `InvalidInput`. Canvas forwards leave the prompt cache unchanged, so a
repeated prompt processes no tokens. After partial reuse a read agrees with the first within 1e-3.

Tests: `canvas_logits_match_the_reference_implementation`, `nemotron_reads_are_calibrated_reproducible_and_support_extensions`

### R8 Canvas forwards

A canvas must follow the resident prompt; another prompt length fails with `InvalidInput`, and so
does an empty canvas or one longer than `batch_size`. Canvas queries attend to the whole prompt and
the whole canvas. Callers read candidate logits at one row, or every row's logits and greedy
proposals (argmax and its probability) after a forward that computed full logits; reading logits
that were not computed fails with `MissingLogits`.

Tests: `canvas_logits_match_the_reference_implementation`

### R9 Canvas parity with the reference

On the reference prompt, for a text canvas and for a block of 32 masks, every row's top token is the
reference's top token or within 0.25 logits of it. Candidate logits at the last canvas row stay
within 0.25 of the reference.

Tests: `canvas_logits_match_the_reference_implementation`

### R10 Causal predictions

`prefill_predict` returns greedy next-token predictions for the last `rows` prompt positions with
the same head (at least one row, at most the prompt and `batch_size`). Against the reference's
greedy `ar_generate` on a text checkpoint, teacher forcing over one 32-token block misses at most
one prediction, and free-running generation matches at least the first 32 tokens.

Tests: `causal_predictions_follow_the_reference_greedy_thought`

### R11 Image preprocessing

The longest edge is capped at 1400 pixels (halves round to even, as in Python), and the size rounds
up to 28-pixel cells. Resizing follows OpenCV's `INTER_CUBIC` (A = -0.75, centers aligned, borders
replicated, no antialiasing), pixels are normalized with the CLIP mean and standard deviation, and
14-pixel patches are laid out by channel, row and column. Normalized pixels stay within 1e-4 of the
reference's.

Tests: `token_grids_cap_the_longest_edge_and_round_up_to_cells`, `cubic_resize_keeps_constants_and_matches_opencv_weights`, `patches_follow_the_convolution_weight_layout`, `image_preprocessing_matches_the_reference_pixels`

### R12 Image prompts

Text-only checkpoints reject images with `InvalidInput`. A request holds at most 8 images. An image
of `w × h` cells becomes `<|image_start|>` and each row's `w` embeddings followed by
`<|image_break|>`, with the last break replaced by `<|image_end|>`: `1 + h × (w + 1)` positions,
which must fit in the context. The block is prefilled with causal attention like text, and its rows
are keyed by the image's content, so a repeated image is served from the cache.

Tests: `image_features_and_reads_match_the_reference_implementation`, `nemotron_images_are_read_and_their_prompt_blocks_reused`

### R13 Vision parity with the reference

The Pixtral tower's output and the projector's output each stay within 3% relative RMS error of the
FP32 reference. A canvas read after the image prompt meets the top-token rule of R9.

Tests: `image_features_and_reads_match_the_reference_implementation`
