# jevons-core

## Purpose

jevons-core is the foundation every other layer builds on. It defines the contracts a diffusion
language model and a speech model implement, the error type they share, model configuration,
bounded decoding of request images, prefill counters, the transcript types, and `Script`, which
keeps a transcript in the writing system of its language. It holds no model, no GPU code and no
file reader.

## Scope

jevons-core owns:

- `DiffusionModel` with `TextTokenizer`, `ChatFormat`, `ModelInfo`, `DiffusionScheme`,
  `PromptPart`, `Conditioning` and `Logits`.
- `SpeechModel` with `SpeechConfig`, `SpeechInfo`, `SpeechToken`, `Word`, `Segment` and
  `Transcript`.
- `Error` and `Result`, `ModelConfig`, `PrefillProfile`, `decode_image` and `Script`.

It leaves to other crates:

- Detecting and loading a model from its files: `jevons-models`.
- Implementing the contracts: `jevons-gemma4-diffusion`, `jevons-nemotron-diffusion` and
  `jevons-parakeet`, with the Gemma tokenizer adapter in `jevons-models`.
- Sampling policy, context limits and answer codes: `jevons-diffusion`.
- Building words, segments and transcripts from speech tokens: `jevons-speech`.
- Mapping errors to HTTP statuses: `jevons-api`.

Neither model contract requires `Send`. A model stays on the thread that loaded it (P2).

## Requirements

### R1 One error type names six failures

`Error` has six kinds, each with a fixed message: `InvalidInput` ("Invalid input: …"),
`ModelLoad` ("Could not load the model"), `UnsupportedModel` ("Unsupported model: …"),
`MissingLogits` ("The canvas forward returned no logits"), `InvalidLogits` ("Candidate logits must
be finite and nonempty") and `Backend` ("GPU backend failure: …").

Tests: none yet

### R2 Configurations start from fixed defaults

`ModelConfig::new(path)` turns on the prompt cache, sets no architecture override and no vision
projector, and selects GPU 0, a context of 8192 positions and a batch of 512 tokens.
`SpeechConfig::new(path)` selects GPU 0.

Tests: none yet

### R3 A batch must fit in a nonempty context

`ModelConfig::validate` accepts a configuration when `0 < batch_size <= context_size <= i32::MAX`.
Any other configuration fails with `InvalidInput` ("Require 0 < batch_size <= context_size <=
i32::MAX").

Tests: `batch_size_must_fit_in_a_nonempty_context`

### R4 Images decode to RGB

`decode_image` accepts JPEG, PNG, WebP and GIF bytes and returns the image's width, height and
8-bit RGB pixels in row-major order. A truncated or corrupt image fails with `InvalidInput`.

Tests: `supported_formats_decode_to_rgb_and_truncated_images_fail`

### R5 Image decoding is bounded

`decode_image` fails with `InvalidInput` when the input is empty or larger than 5 MiB, when its
format is unknown or not one of the four in R4, when a side exceeds 8192 pixels or decoding needs
more than 64 MiB, and when the image has more than 16,777,216 pixels.

Tests: none yet

### R6 Chat markers must be single tokens

`ChatFormat::validate` fails with `UnsupportedModel` when `model_open` is not `turn_close` followed
by `assistant_open`, or when a marker in `answer_stops` or `thought_stops` does not tokenize, with
special markers parsed, to one token.

Tests: none yet

### R7 A text tokenizer keeps user text plain

A `TextTokenizer` called with `special = false` returns no control or added special token,
and `bos = true` prepends the beginning-of-sequence token. `code_piece` returns a non-control
token's text when it is 1 to 16 ASCII letters and digits after one optional leading space, and
`None` otherwise. `decode` leaves control tokens out and turns incomplete UTF-8 at the end into
U+FFFD.

Tests: `markers_parse_only_in_framing_and_user_text_never_yields_them`, `code_pieces_exclude_added_tokens_and_non_alphanumeric_text`, `code_pieces_ignore_one_leading_space_and_skip_control_tokens`

### R8 A diffusion model reports its facts and scheme

A `DiffusionModel` reports `ModelInfo` (a stable architecture ID, a display name, the vocabulary
size, the most prompt and canvas positions, the most tokens per forward and the largest canvas),
its `ChatFormat` and its `DiffusionScheme`. The scheme is either uniform noise with
self-conditioning, with the mask token kept out of the noise, or masked commits with a block size,
a commit threshold and a step limit.

Tests: none yet

### R9 Prefill makes the resident prompt and reuses its prefix

`prefill(parts, suffix)` makes `parts + suffix` the resident causal prompt and returns its length
in tokens. With `prompt_cache` on, a prompt that shares a prefix with the resident one evaluates
the tokens past that prefix and serves the rest from the cache. A `PromptPart` counts its text
tokens, or the tokens of its image.

Tests: `model_reads_preserve_reproducibility_across_requests`

### R10 Images become prompt parts

`encode_images` decodes each image with `decode_image`, encodes it and returns its prompt parts
with the image delimiters. The parts stay valid until the next call. Prefilling the same image
again reuses its cached block.

Tests: `model_images_prefill_and_preserve_text_reproducibility`

### R11 Canvas forwards leave the prompt alone

`forward_canvas` evaluates a canvas over the resident prompt without changing the prompt cache.
Afterwards `candidate_logits(row, candidates)` returns the logits of those candidates at that row.
`full_logits` returns every row's logits (rows times vocabulary) when the forward asked for
`Logits::Full`. `Conditioning::Previous` passes the prior step's full logits at an inverse
temperature.

Tests: none yet

### R12 Greedy proposals take the first maximum

`greedy_proposals` returns each canvas row's argmax over the vocabulary with its softmax
probability. The default reads `full_logits` and takes the first maximum in a tie. It fails with
`InvalidLogits` when the vocabulary is empty, the logits are empty, their count is not a multiple
of the vocabulary size, or a row holds a value that is not finite.

Tests: `greedy_takes_the_first_argmax_with_its_stable_probability`

### R13 Causal predictions need a causal model

`prefill_predict(parts, suffix, rows)` prefills like `prefill`, evaluates the last `rows`
positions, and returns the resident length with each row's greedy next-token prediction in order.
A model without a causal language-model objective keeps the default, which fails with
`UnsupportedModel`.

Tests: `causal_predictions_follow_the_reference_greedy_thought`

### R14 Prefill counters describe the last read

`profile()` returns the model's `PrefillProfile`: wall time in milliseconds, prefill calls,
batches, tokens evaluated and tokens served from the prompt cache. The engine resets it at the
start of each read.

Tests: `model_reads_preserve_reproducibility_across_requests`

### R15 A speech model transcribes one window

A `SpeechModel` reports `SpeechInfo`: its architecture, display name, mono sample rate, seconds per
output frame, longest window and ISO-639-1 languages. `transcribe` takes mono samples at that rate,
no longer than the longest window, and returns tokens with an ID, a vocabulary piece (a leading
`▁` starts a word), start and end times in seconds from the window start and a natural-log
probability. `detokenize` returns the text of token IDs.

Tests: `greedy_transcripts_match_the_reference`

### R16 Language restriction is opt-in

`set_language(Some(code))` keeps later transcriptions in the script of that language, and `None`
lifts the restriction. A model that does not restrict scripts keeps the default, which accepts any
value and changes nothing.

Tests: none yet

### R17 A transcript lists its tokens through its words

A `Transcript` holds its text, the seconds of audio it covers, its words and its segments. A
`Word` holds its text, times and tokens. A `Segment` holds an ID, times, text, token IDs and the
mean token log probability. `Transcript::tokens` yields every word's tokens in order.

Tests: none yet

### R18 A language names its script

`Script::of_language` returns Greek for `el`, Cyrillic for `be`, `bg`, `kk`, `mk`, `mn`, `ru`,
`sr` and `uk`, Latin for any other code of two lowercase ASCII letters, and `None` for anything
else.

Tests: `scripts_keep_letters_of_other_alphabets_out`

### R19 A script accepts its own letters

`Script::writes(text)` is true when every alphabetic character of `text` belongs to the script.
Latin covers A to Z, a to z, U+00C0 to U+024F and U+1E00 to U+1EFF. Greek covers U+0370 to U+03FF
and U+1F00 to U+1FFF. Cyrillic covers U+0400 to U+052F. Digits, punctuation and `▁` pass in every
script, and a letter outside all three ranges fails in every script.

Tests: `scripts_keep_letters_of_other_alphabets_out`
