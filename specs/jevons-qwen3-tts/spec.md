# jevons-qwen3-tts

## Purpose

`jevons-qwen3-tts` runs Alibaba's Qwen3-TTS (12 Hz, Base checkpoints) on Burn. A talker writes the
first code of each audio frame from the text, a code predictor fills the frame's other fifteen
codes, and the codec decoder turns frames into 24 kHz audio. It implements the `VoiceModel`
contract for one utterance, and its codes and audio are checked against dumps of the `qwen-tts`
reference.

## Scope

It owns:

- `config`: `config.json` and the speech tokenizer's `config.json`, and the variants this runtime
  accepts.
- `prompt`: the positions the talker reads: the role, the language, the voice and the text.
- `stack`: the causal transformer the talker, the code predictor and the codec decoder share, on
  Burn's operators.
- `talker`: the talker's text projection and codec embedding, and its layers and the code
  predictor on Burn's operators.
- `gpu`, `fused`: the same networks on the GPU as kernels of the crate, a frame per operation.
- `sample`: choosing a code from logits.
- `codec`: the codec decoder, streamed.
- `model`: loading, voices, `speak` and its timings.

It leaves to other crates:

- Sentences, number rules and the default voice to `jevons-voice`.
- Device, weight streaming, layers and the tuned GEMM to `jevons-burn`.
- The tokenizer to `jevons-tokenizer`.
- Detection and the default model ID to `jevons-models`.
- Making a voice from a recording to `scripts/reference/qwen3_tts_voice.py`: the runtime ports
  neither the speaker encoder nor the codec encoder.

The reference dumps come from `scripts/reference/qwen3_tts_dump.py` (the `qwen-tts` package on CPU
in F32) under `$JEVONS_GOLDEN_DIR`, in the directory `QWEN3_TTS_GOLDEN` names (default
`qwen3-tts-12hz-0.6b-base`). Principle P1 applies: the crate forbids unsafe code.

## Requirements

### R1 Supported configurations

`config.json` must describe a 12Hz Base checkpoint (`tts_model_type` `base`, `tokenizer_type`
`qwen3_tts_tokenizer_12hz`) with `silu` networks, no attention biases, a code predictor as wide as
the talker, head counts that divide into whole groups, sizes that are multiples of 64, the same
codebook count in the talker, the predictor and the decoder with one semantic codebook, a speaker
vector as wide as the talker, control and language ids inside the codec vocabulary, a name for each
of the ten languages, and a decoder whose four vocoder strides and upsampling ratios multiply to
`decode_upsample_rate`. Anything else fails with `UnsupportedModel` naming what differs. A missing
`config.json` or `speech_tokenizer/config.json` fails with `ModelLoad`.

Tests: `the_base_checkpoint_is_accepted_and_names_its_languages`, `other_variants_are_refused_by_name`

### R2 Languages

The model speaks `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr` and `ru`. A prompt names a
language by the codec id `codec_language_id` gives its English name. An utterance without a
language leaves it to the talker.

Tests: `the_base_checkpoint_is_accepted_and_names_its_languages`, `a_prompt_without_a_language_skips_the_thought`

### R3 The prompt

The talker reads, in order: the role (`<|im_start|>`, `assistant`, a line break) as text; the codec
prefix on text padding (`think`, `think_bos`, the language id and `think_eos`, or `nothink`,
`think_bos` and `think_eos` without a language); the speaker vector on text padding; the codec pad
on the start-of-text mark; and the text's first token on the codec's start-of-audio id. Each later
position adds the text's next token to the frame the talker reads, then the end-of-text mark, then
text padding. A text token enters through the text embedding and its two-layer projection. Text is
tokenized as plain text, so it never yields a marker. The talker's input for the reference's cases
is within 2e-3 relative RMS of the reference's.

Tests: `a_prompt_names_the_language_then_the_speaker_then_starts_the_audio`, `a_prompt_without_a_language_skips_the_thought`, `prompts_match_the_reference`

### R4 The talker and the code predictor

The talker writes the first code of each frame from its last position. The code predictor then
reads the talker's output for that position and the first code's embedding, and writes the frame's
other fifteen codes one at a time, each through its own head and embedding. The talker reads a
frame as the sum of its sixteen embeddings. Over the reference's prompts the talker's hidden states
stay within 5e-3 relative RMS after its first layer and 2e-2 after its middle layer, its final norm
and its head. Given the reference's greedy frames to read, the talker chooses the reference's next
first code at least 95 times in 100 and the predictor the reference's other codes at least 85
times in 100 (measured on the GPU: 175 of 175, and 2,617 of 2,625). In f32 on the CPU device every
choice is the reference's, and so is every frame of a greedy run of its own; in FP16 on the GPU a
near tie falls the other way within a few frames, and a greedy run then parts from the
reference's for good.

Tests: `talker_hidden_states_match_the_reference`, `the_networks_follow_the_reference_frame_by_frame`, `greedy_codes_match_the_reference`

### R5 Choosing codes

`speak` draws each code from the 50 most likely at a temperature of 0.9, with a new seed per
utterance. Before the talker's draw, the logit of a first code already used in the utterance is
divided by 1.05 when positive and multiplied by it when negative, once however often the code was
used, and the 1,024 control ids at the
top of the vocabulary are out, except the end of the audio. The talker may not end the audio in its
first two frames. A code tied with the fiftieth stays in the draw, and a suppressed code is never
drawn. The same seed gives the same draws.

Tests: `the_penalty_pushes_seen_codes_down_whatever_their_sign`, `greedy_takes_the_first_maximum_and_skips_suppressed_codes`, `sampling_draws_only_from_the_top_codes_in_their_proportions`, `a_seed_reproduces_its_draws`

### R6 The transformer

A layer is an RMS norm, grouped-query attention with rotary positions over the keys and values of
the positions before it, and a SiLU-gated MLP, each added to the residual. The talker and the
predictor RMS-normalize each head's queries and keys. The codec's layers scale each branch per
channel and attend over a window of 72 positions, the row's own included. Rotations repeat each
frequency across both halves of a head and are computed in f32. On the CPU device attention runs in
f32 over the written keys; on a CubeCL device Burn's operators run it in FP16 over a power-of-two
bucket of keys, at least 16.

Tests: `rotations_repeat_each_frequency_across_both_halves`, `key_buckets_are_powers_of_two_within_the_capacity`, `exact_attention_averages_the_values_a_row_may_see`

### R7 The codec decoder

A frame's sixteen codes become the sum of their codebook vectors (a cluster's summed embedding over
its usage, floored at 1e-5, through the quantizer's output projection). The sum goes through a
causal convolution, the transformer, two upsampling stages with a ConvNeXt block each, and a vocoder
of four stages (a transposed convolution, then three residual units with dilations 1, 3 and 9
around SnakeBeta activations), to 1,920 samples a frame at 24 kHz, clamped to [-1, 1]. For the
reference's codes, decoded in the chunks `speak` uses, the audio is within 5e-2 relative RMS of
the reference's (8e-4 on the GPU in FP16). In f32 on the CPU device the summed vectors, the
transformer's output and the audio of one pass are within 1e-4.

Tests: `codec_audio_matches_the_reference`, `codec_chunks_give_the_audio_of_one_pass`, `snake_adds_a_squared_sine_per_channel`

### R8 The decoder streams

Every layer of the decoder is causal and keeps what its next output needs: a convolution its last
inputs, a transposed convolution the tail its next input adds to, the transformer its keys and
values. Frames decoded in any chunking give the audio of one pass. A transposed convolution's
kernel is its stride or twice its stride; another kernel fails to load.

Tests: `a_convolution_sees_the_past_through_its_dilated_taps_in_any_chunking`, `a_transposed_convolution_overlaps_its_neighbours_in_any_chunking`, `codec_chunks_give_the_audio_of_one_pass`

### R9 Voices

A voice is a speaker vector as wide as the talker. Two ship with the crate: `reader` and `lector`.
Every `voices/<name>.safetensors` in the checkpoint directory adds the voice `<name>` from its
`speaker` tensor, and replaces a built-in voice of that name. A file whose vector has another width
or a value that is not finite fails the load with `UnsupportedModel` naming the voice.
`VoiceInfo::voices` lists the built-in voices, `reader` first, then the directory's by name. The
first is the voice the layers above speak in when nobody names one.

Tests: `the_built_in_voices_list_reader_first`

### R10 Speaking an utterance

`speak` hands the sink mono samples at 24 kHz a chunk at a time: three chunks of four frames,
the first as soon as it is decoded, then twelve frames at a time, then what is left. A speaker
may start with the first sample, and twelve frames take longer to make than four play. After
three short chunks the audio handed over outlasts the making of every chunk that follows, while
a frame takes up to 40 ms to write: such a speaker never runs out of audio. The decoder runs in
those two chunk shapes alone: a last chunk short of its shape is filled by repeating its last
frame, and the audio past its own frames is dropped, which leaves the audio before it as it was.
It stops when the talker ends the audio, when the sink returns false, or after 600 frames
(48 seconds). A voice the model lacks, a language it lacks, a text with nothing to speak and a
text of more than 400 tokens fail with `InvalidInput` before anything is spoken.

Tests: `speaking_stops_when_the_sink_says_so_and_refuses_what_it_lacks`, `codec_audio_matches_the_reference`, `the_audio_sent_outlasts_the_making_of_each_chunk_at_forty_ms_a_frame`

### R11 Loading

`Qwen3Tts::load` reads `config.json`, `vocab.json`, `merges.txt`, `model.safetensors` and
`speech_tokenizer/` from the checkpoint directory. Matrices go to the GPU as FP16 for the tuned
GEMM, padded with zeros to its 64-wide tile. The text embedding stays in its file: an utterance
reads the rows of its own tokens. A tensor that is missing or has another shape fails with
`UnsupportedModel` ("Qwen3-TTS weights: …"). The text a prompt's later positions carry reaches the
device in 32, 128 or 512 rows: its tokens, the end-of-text mark, then text padding up to the
first of those counts that leaves a row of it. On the GPU a row count compiles kernels the first
time a binary reads it, so the load ends by reading a prompt of each count and one that names no
language, then speaking a sentence into nothing, long enough for both of the decoder's chunk
shapes. No utterance then pays for compiling kernels, whatever its length. It logs how long the
weights and the warm-up took.

Tests: `prompts_match_the_reference`, `the_text_to_come_fills_one_of_three_row_counts`

### R12 Timings

After an utterance `timings` reports its frames, the time to read the prompt, the time to make
the frames, the time in the codec decoder, the time to the first audio and the whole time. No
log line holds the text or the audio.

Tests: none yet

### R13 The networks on the GPU

On a GPU with 32-lane waves, the talker, the code predictor and the codec's transformer run on
kernels of the crate over FP16 weights, where the shapes allow it (heads of whole waves up to
256 values, hidden and intermediate sizes and vocabularies that are multiples of 256, at most 16
rows a forward). A layer is nine launches. Reading a prompt is one operation, which also chooses
the first frame's first code. A frame is one operation: the predictor's fifteen steps, each
code's choice by the rules of R5 and its embedding, the talker's step and its choice all stay on
the device, and the host reads the frame's codes and the next first code back, once. The talker's
output for a prompt is within 2e-2 relative RMS of the reference's, rows together and a row at a
time. No operator of Burn's that tunes a matrix product runs on this path. `Engine::Layers` runs
the talker and the predictor on Burn's operators, for comparison; the codec's transformer stays
on the kernels wherever they run.

Tests: `talker_hidden_states_match_the_reference`, `the_networks_follow_the_reference_frame_by_frame`, `codes_are_chosen_as_the_host_chooses_them`
