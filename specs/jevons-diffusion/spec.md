# jevons-diffusion

## Purpose

`jevons-diffusion` is the layer the Generative and Decision services share. `DiffusionEngine` owns
one loaded diffusion language model with its chat framing, its verified answer codes and its context
limits, and generates bounded runs of tokens (thoughts and answers) in each `Decoding` mode. The
crate holds the two denoising samplers and a scripted fake model that lets the services test their
policy without a GPU or model files.

## Scope

It owns:

- `DiffusionEngine`: loading, answer codes, prompt framing, canvas sizes, thoughts and token
  generation, and the prefill diagnostics of the last request.
- `Decoding`: `diffusion`, `self-speculation` and `autoregressive`.
- The samplers: `masked::commits` (confidence-ordered unmasking) and `uniform::refine` and
  `uniform::noise` (entropy-bound refinement over uniform noise).
- `fake`, behind the `testing` feature: a scripted model, tokenizer and log for service tests.

It leaves to other crates:

- The model contracts (`DiffusionModel`, `ChatFormat`, `DiffusionScheme`, `TextTokenizer`) to
  `jevons-core`.
- Detecting and loading an architecture to `jevons-models`, which the default `models` feature
  pulls in and re-exports (`Architecture`, `default_model_id`, `resolve_architecture`).
- The models themselves to `jevons-gemma4-diffusion` (uniform noise, CubeCL) and
  `jevons-nemotron-diffusion` (masked, Burn).
- Free-form answers, stop sequences and tool calls to `jevons-generative`; restricted-canvas reads
  to `jevons-decision`; the settings file and the worker thread to `jevons-api`.

## Requirements

### R1 A model with an unusable chat format does not load

`DiffusionEngine::new` fails with `UnsupportedModel` when the chat format's model turn is not
`turn_close` followed by `assistant_open`, or when a thought or answer stop marker is not one token.
`single_token` and `marker_tokens` fail the same way for any marker that is not one token.

Tests: none yet

### R2 Every engine has 128 verified answer codes

At load the engine picks 128 answer codes, starting with `A` to `Z`, `a` to `z` and `0` to `9` and
continuing with other vocabulary pieces. Each code encodes to one token in an answer slot, no two
codes share a token, and none is the mask token. A vocabulary with fewer than 128 such codes fails
to load with `InvalidInput`.

Tests: `framing_and_the_empty_thought_come_from_the_model_chat_format`, `masked_canvases_pad_to_the_block_and_answers_carry_the_prefix_space`

### R3 Decoding names and their models

`Decoding` parses from `diffusion`, `self-speculation` and `autoregressive`, prints the same ids,
and rejects any other name with `InvalidInput`. A new engine decodes with `diffusion`.
`set_decoding` accepts `self-speculation` and `autoregressive` on a masked model and fails with
`InvalidInput` on a uniform-noise one.

Tests: `causal_decoding_needs_a_masked_model`

### R4 The prompt frames user text as literal text

`prompt_parts` returns the user-turn opener (with BOS when the chat format asks for it), the encoded
images, the trimmed user text and the model-turn opener, in that order. The user text is tokenized
without special tokens, so a chat marker written in it stays plain text.

Tests: `framing_and_the_empty_thought_come_from_the_model_chat_format`, `model_reads_preserve_reproducibility_across_requests`

### R5 Answer tokens carry the joining space

When the model's chat format sets `space_joins_answers`, `answer_text` puts a space before a
candidate and `prefix_tokens` drops one trailing space from a slot prefix. Without it, both pass the
text through.

Tests: `masked_canvases_pad_to_the_block_and_answers_carry_the_prefix_space`, `nemotron_reads_are_calibrated_reproducible_and_support_extensions`

### R6 Canvas sizes

`canvas_capacity` is the smaller of the model's largest canvas and its batch size. On a masked
model, `padded(n)` rounds `n` up to whole blocks, caps the result at the canvas capacity, and
returns no less than `n`. On a uniform-noise model it returns `n`.

Tests: `masked_canvases_pad_to_the_block_and_answers_carry_the_prefix_space`, `nemotron_reads_are_calibrated_reproducible_and_support_extensions`

### R7 Generation is bounded and ends before a stop marker

`generate_tokens` returns at most `budget` tokens after `prompt + start`. When a token in `stops` is
decided, generation ends: the returned tokens stop before it and `stopped` is set.

Tests: `thought_generation_stops_at_the_first_stop_marker_and_closes_the_thought`, `chat_generation_frames_every_turn_and_ends_at_the_turn_marker`, `answers_end_at_max_tokens_and_text_prompts_are_not_framed`, `autoregressive_and_speculative_thoughts_agree_and_respect_the_budget`

### R8 The sink sees each decided run and can end generation

The sink receives each run of tokens as it is decided, in order. When it returns false, generation
ends with `cancelled` set and the model runs no further forward.

Tests: `a_client_that_stops_listening_ends_generation`, `stop_sequences_cut_the_answer_and_are_never_streamed_in_part`

### R9 A thought is opened, bounded and closed

`think_with` generates up to `budget` tokens after the thought opener and ends at the first thought
stop marker. The returned suffix is the opener, the thought tokens and the thought closer, whether a
marker or the budget ended it. `output_tokens` counts the thought tokens, plus one when a marker
ended the thought.

Tests: `thought_generation_stops_at_the_first_stop_marker_and_closes_the_thought`, `masked_thought_blocks_start_from_the_causal_prediction_and_stop_at_a_marker`, `nemotron_self_speculation_reproduces_autoregressive_thoughts`

### R10 Uniform-noise generation

On a uniform-noise model every decoding mode uses the denoiser. Each block of up to the canvas
capacity starts as seeded noise that excludes the mask token. Its first iteration runs without
self-conditioning, and each later one is conditioned on the previous logits. Refinement stops at 48
iterations, or at the first iteration whose best tokens repeat the previous ones with a mean entropy
below 0.005. The block's best tokens join the prompt before the next block.

Tests: `thought_generation_stops_at_the_first_stop_marker_and_closes_the_thought`

### R11 Masked diffusion generation

With `diffusion` on a masked model, each block of up to the model's block size starts with the
causal prediction after the committed text, and its other positions start as masks. Each step
commits the most confident open position plus every other at or above the model's threshold, judged
over the full vocabulary; the last allowed step commits the rest. A block ends once a stop marker
and everything before it are committed.

Tests: `masked_thought_blocks_start_from_the_causal_prediction_and_stop_at_a_marker`

### R12 Self-speculation gives the autoregressive tokens

With `self-speculation`, each round drafts a block in one bidirectional forward and verifies it in
one causal forward. It keeps the drafts that match the causal predictions and the prediction after
the last match, so each round decides from one token to a whole block. The tokens equal those of
`autoregressive` decoding.

Tests: `self_speculation_keeps_matching_drafts_and_the_next_causal_token`, `autoregressive_and_speculative_thoughts_agree_and_respect_the_budget`, `nemotron_self_speculation_reproduces_autoregressive_thoughts`

### R13 Autoregressive decoding

With `autoregressive`, each token comes from one causal forward, and no canvas is evaluated.

Tests: `autoregressive_and_speculative_thoughts_agree_and_respect_the_budget`

### R14 The seed affects uniform noise and nothing else

Equal prompts, starts, budgets, decodings and seeds give equal tokens. The seed feeds the
uniform-noise denoiser; masked diffusion, self-speculation and autoregressive decoding are greedy
and ignore it.

Tests: none yet

### R15 Generation accounting

`Generated` and `Thought` count as input the resident prompt length of every prefill that starts a
block (diffusion), a round (self-speculation) or a token (autoregressive). `forward_ms` adds the
time of canvas forwards and leaves prefill out.

Tests: none yet

### R16 Prefill diagnostics cover one request

`reset_profile` clears the prefill profile, and `prefill_profile` returns the calls, batches,
processed tokens and reused tokens of the prefills since the last reset.

Tests: `context_reservation_is_checked_before_prefill_at_the_exact_boundary`, `model_extensions_average_refine_think_and_chunk`, `model_reads_preserve_reproducibility_across_requests`

### R17 Masked commits

`masked::commits` returns the index of the most confident proposal, plus every other proposal at or
above the threshold. Equal confidences keep the earlier position first. No proposals give no
commits.

Tests: `the_most_confident_position_always_commits_and_others_need_the_threshold`

### R18 Entropy-bound refinement

`uniform::refine` changes the listed positions and leaves the rest of the canvas intact. It orders
them by entropy at the given temperature; positions reached while the summed entropy is at most 0.1
take their sampled token, and the others get fresh noise. Probabilities are computed relative to
each row's largest logit, so extreme logits stay finite. Non-finite logits, an empty row, a logit
count that does not match the canvas, or a position past its end fail with `InvalidLogits`.
`uniform::noise` does not return the mask token.

Tests: `refinement_preserves_fixed_tokens_and_accepts_certain_predictions`, `entropy_and_sampling_use_stable_probabilities`

### R19 The fake model's tokenizer

With the `testing` feature, `fake::FakeTokenizer` parses the markers `<user>`, `<model>`,
`<system>`, `<think>`, `</think>`, `<end>` and `<nothought>` when special tokens are on; with them
off, the markers are plain text. `w0` to `w99` are one token each. With `space_joins`, each of ` w0`
to ` w99`, and each space with the ASCII letter or digit after it, is one token. Every other
character is one token, and decoding returns the text without markers. `fake::tokens` tokenizes with
it.

Tests: `framing_and_the_empty_thought_come_from_the_model_chat_format`, `masked_canvases_pad_to_the_block_and_answers_carry_the_prefix_space`, `chat_generation_frames_every_turn_and_ends_at_the_turn_marker`

### R20 The fake model's scripted predictions

`fake::FakeModel::new()` is a uniform-noise model with a 512-token vocabulary, a 256-token context
and a 64-token canvas. Full logits favor `favored[row]` at each canvas row, or `20 + row` past the
script. Candidate `i` at row `r` gets the logit `0.01 r - i`. The causal prediction for the `k`th
position after the last `<think>` or `<nothought>` (or, without either, after the prompt of the
first prediction) is `causal[k]`, and `<end>` past the script. `fake::masked_model()` is the same
model with masked diffusion: blocks of 4, a 0.9 threshold and at most 4 steps. The fake rejects
images.

Tests: `masked_reads_commit_the_most_confident_slot_each_step`, `masked_thought_blocks_start_from_the_causal_prediction_and_stop_at_a_marker`, `self_speculation_keeps_matching_drafts_and_the_next_causal_token`, `causal_decoding_needs_a_masked_model`

### R21 The fake model logs what the engine asks

`fake::fake_engine` returns an engine on the fake model and a shared `fake::Log`. The log records
each prefill's full prompt, each canvas, whether each canvas forward was self-conditioned, and the
rows each causal prediction asked for, in call order. The fake's prefill profile counts the tokens
it shares with the previous prompt as reused.

Tests: `slots_beyond_the_canvas_capacity_are_read_in_seeded_chunks`, `refinement_conditions_later_steps_on_the_previous_logits`, `context_reservation_is_checked_before_prefill_at_the_exact_boundary`, `a_client_that_stops_listening_ends_generation`
