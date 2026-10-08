# jevons-decision

## Purpose

`jevons-decision` is the Decision service: typed, probabilistic answers read from a diffusion
model's canvas, as the System One API serves them. A `ReadRequest` names a prompt and answer slots,
each with single-token candidates. `Decide` reads every slot's distribution over its candidates from
one canvas forward, or a few refinement steps, after an optional bounded thought. It averages
samples and splits the slots into chunks when they exceed one canvas. The crate ships the
`jevons-scm` CLI and the `golden` example, which captures reads as a reference and compares against
one.

## Scope

It owns:

- `ReadRequest`, `Slot`, `ReadOptions`, `ReadResult` and `SlotRead`.
- `Decide` on `DiffusionEngine`: framing, the context check, chunks, uniform and masked canvas
  reads, samples, thoughts, sequential chunks and images.
- `restricted_softmax`.
- `serde` support: `ReadRequest` and `Slot` serialize and deserialize, and `ReadResult` and
  `SlotRead` serialize. The CLI and `golden` print and store reads as JSON.
- The `jevons-scm` binary and the `golden` example, both behind the default `models` feature.

It leaves to other crates:

- Thoughts, answer codes, prompt framing and the samplers to `jevons-diffusion`.
- Compiling System One questions into slots, answer codes and labels, and mapping probabilities
  to `noul`, `choice` and `score` answers, to `jevons-api`.
- Image decoding to `jevons-core`, and image encoding to the models.

## Requirements

### R1 Read options have limits

`ReadOptions` defaults to one step, one sample, no thought and no sequential chunks. A read fails
with `InvalidInput` unless `steps` is 1 to 8, `samples` is 1 to 32 and `think` is 0 to 4096.

Tests: none yet

### R2 Malformed requests fail before prefill

A read fails with `InvalidInput` when the prompt is blank, when there are no slots, when a slot has
no candidates, or when a candidate, written as in an answer slot, does not encode to one token
distinct from the slot's other candidates. Images combined with a thought or with sequential chunks fail the same
way.

Tests: none yet

### R3 The prompt is framed by the model's chat format

A read's prompt is the engine's framed user turn with the prompt text as literal text, followed by
the empty thought when no thought is requested. Chat markers written in the prompt stay plain text.

Tests: `framing_and_the_empty_thought_come_from_the_model_chat_format`, `model_reads_preserve_reproducibility_across_requests`

### R4 Each slot is its prefix and one answer position

The canvas holds each slot's prefix tokens followed by its answer position, in request order. Each
`SlotRead` reports the answer's position in the canvas, its position after the prompt, its initial
token and its candidate tokens. On a model whose answers carry a joining space, a prefix's trailing
space moves onto the candidates.

Tests: `framing_and_the_empty_thought_come_from_the_model_chat_format`, `masked_canvases_pad_to_the_block_and_answers_carry_the_prefix_space`, `model_reads_preserve_reproducibility_across_requests`

### R5 Probabilities are a softmax over the candidates

Each slot's probabilities are a softmax over its candidates' logits and no other tokens, in
candidate order, and sum to 1. Extreme logits stay finite. `restricted_softmax` fails with
`InvalidLogits` on no logits or a non-finite one.

Tests: `softmax_handles_extremes_and_rejects_nonfinite_values`, `masked_reads_commit_the_most_confident_slot_each_step`

### R6 The context is checked before prefill

The prompt, the thought reserve and the canvas reserve must fit the model's context, or the read
fails with `InvalidInput` ("context allows") with no prefill run. A request that fills the context
to the last token is read. The canvas reserve is the largest padded chunk, or, with sequential
chunks, every slot plus the padding.

Tests: `context_reservation_is_checked_before_prefill_at_the_exact_boundary`, `model_reads_preserve_reproducibility_across_requests`

### R7 Slots beyond one canvas are read in chunks

Slots are grouped in order into chunks that fit the canvas capacity, and no slot is split across
chunks. A slot whose prefix and answer exceed the capacity fails with `InvalidInput`. Each chunk is
read after its own prefill of the full prompt, and the results keep request order.

Tests: `chunking_preserves_order_and_keeps_whole_questions`, `slots_beyond_the_canvas_capacity_are_read_in_seeded_chunks`, `model_extensions_average_refine_think_and_chunk`

### R8 Seeds are derived per chunk and sample

Chunk `c` reads with the seed plus `104729 c`, and sample `s` of a chunk with the chunk seed plus
`7919 s`, with wrapping arithmetic. Equal chunks start from different noise. `ReadResult` echoes the
request's seed.

Tests: `slots_beyond_the_canvas_capacity_are_read_in_seeded_chunks`, `model_extensions_average_refine_think_and_chunk`

### R9 Uniform-noise reads

On a uniform-noise model, each answer position starts as a seeded random token that is not the mask
token. Each step before the last refines the answer positions and keeps the prefixes. The first step
is not self-conditioned, and each later step is conditioned on the previous step's logits. The last
step reads the candidate logits at temperature 1.

Tests: `framing_and_the_empty_thought_come_from_the_model_chat_format`, `refinement_conditions_later_steps_on_the_previous_logits`, `model_extensions_average_refine_think_and_chunk`

### R10 Masked reads

On a masked model, each answer position starts as the mask token, and the canvas is padded with
masks to whole blocks. Each step commits the slot whose best candidate is most probable, plus every
slot whose best candidate reaches the model's threshold, to that candidate. A committed slot keeps
the logits and probabilities of the step that committed it and is not read in later steps. The last
step reads every open slot. Masked reads are not self-conditioned and do not depend on the seed.

Tests: `masked_reads_commit_the_most_confident_slot_each_step`, `masked_canvases_pad_to_the_block_and_answers_carry_the_prefix_space`, `nemotron_reads_are_calibrated_reproducible_and_support_extensions`

### R11 Samples average probabilities

With more than one sample, each slot's probabilities are the mean of the samples' probabilities, not
of their logits or winning labels, and its logits are left empty. All samples of a chunk share one
prefill.

Tests: `samples_average_probabilities_instead_of_logits_or_winning_labels`, `model_extensions_average_refine_think_and_chunk`

### R12 A thought comes before the reads

With `think`, the engine generates a thought of at most that many tokens with its decoding, and
every chunk is read after the closed thought. The thought is not returned. `output_tokens` counts
its tokens, and is 0 without a thought.

Tests: `thought_generation_stops_at_the_first_stop_marker_and_closes_the_thought`, `masked_thought_blocks_start_from_the_causal_prediction_and_stop_at_a_marker`, `model_extensions_average_refine_think_and_chunk`, `nemotron_reads_are_calibrated_reproducible_and_support_extensions`

### R13 Sequential chunks see earlier answers

With `sequential`, each chunk's slot prefixes and most probable candidates join the context of the
chunks after it. Without it, chunks share no answers.

Tests: none yet

### R14 Images precede the prompt text

Images go into the prompt before the text. A model that cannot encode images fails the read; for
DiffusionGemma without a vision projector the error names `--mmproj`. Repeating an image reuses its
cached prompt block, and text reads before and after image reads agree.

Tests: `model_images_prefill_and_preserve_text_reproducibility`, `nemotron_images_are_read_and_their_prompt_blocks_reused`, `model_extensions_average_refine_think_and_chunk`

### R15 Usage counts logical reads

`prompt_tokens` adds the prompt length of every chunk and sample, plus the thought's prompt tokens.
`canvas_tokens` adds the canvas length (padding included) of every chunk and sample. Steps do not
multiply either count. `forward_ms` adds canvas and thought forwards and leaves prefill out.

Tests: `framing_and_the_empty_thought_come_from_the_model_chat_format`, `masked_canvases_pad_to_the_block_and_answers_carry_the_prefix_space`, `model_extensions_average_refine_think_and_chunk`

### R16 Reads are reproducible across requests

A read repeated with the same request and seed reproduces its initial tokens and its probabilities
within 1e-5, whatever reads ran in between (1e-3 on Nemotron-Labs-Diffusion after partial prompt
reuse). Each read resets the prefill profile, and its processed and reused tokens add up to the
prompt.

Tests: `model_reads_preserve_reproducibility_across_requests`, `model_extensions_average_refine_think_and_chunk`, `nemotron_reads_are_calibrated_reproducible_and_support_extensions`, `context_reservation_is_checked_before_prefill_at_the_exact_boundary`

### R17 Model reads answer the fixtures

On Nemotron-Labs-Diffusion, the SCM read gives slag a yes probability above 0.7 and steel
reinforcement bars one at least 0.2 lower. On both models with images, a red image reads as red, and
on Nemotron-Labs-Diffusion a blue image reads as blue.

Tests: `nemotron_reads_are_calibrated_reproducible_and_support_extensions`, `model_images_prefill_and_preserve_text_reproducibility`, `nemotron_images_are_read_and_their_prompt_blocks_reused`

### R18 The SCM fixture

`ReadRequest::scm(material)` asks whether the material is a supplementary cementitious material.
The prompt is the material and a legend (`A = yes`, `B = no`). Its one slot has the prefix
`Is this material an SCM?\nAnswer: ` and the candidates `A` and `B`.

Tests: `model_reads_preserve_reproducibility_across_requests`

### R19 The `jevons-scm` CLI

`jevons-scm` loads the model at `--model` (or `DIFFUSION_MODEL`) and reads the SCM fixture for
`--prompt` once with the default options. `--arch` (or `JEVONS_ARCH`, default `auto`), `--seed`
(42), `--main-gpu` (0), `--context-size` (8192), `--batch-size` (512) and `--no-prompt-cache`
configure the model. It prints the token, logit and probability of `A (yes)` and `B (no)`, the
prompt and canvas tokens, the slot's positions, the seed, the slot's initial token and the forward
time. With `--json` it prints the `ReadResult` as JSON.

Tests: none yet

### R20 The `golden` example

`golden dump` reads eight fixtures at seed 42 (the SCM read with defaults, with 3 steps, with 2
samples and with an 8-token thought; 12 slots in chunks, plain and sequential; a two-question choice
with 2 steps; and a red image when `DIFFUSION_MMPROJ` is set) and writes the answer codes and every
read to `reads.json`, with logits and probabilities as exact bit patterns. The folder is `--dir`, or
`<architecture>-engine` under `$JEVONS_GOLDEN_DIR` or `~/.cache/jevons/golden`. `golden compare`
repeats the reads, prints each read's largest differences and argmax agreement, and exits with
status 1 at the first difference: any bit with no `--tolerance`, or an absolute difference above it.
Other usage exits with status 2.

Tests: none yet
