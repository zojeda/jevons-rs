# jevons-parakeet

## Purpose

`jevons-parakeet` runs NVIDIA's Parakeet TDT speech recognition on Burn, from Hugging Face
`ParakeetForTDT` checkpoints: log-mel features on the host, the FastConformer encoder on the GPU,
and greedy token-and-duration decoding. It implements the `SpeechModel` contract for one window of
audio, and its encoder outputs and transcripts are checked against dumps of the `transformers`
reference.

## Scope

It owns:

- `config`: `config.json` and `processor_config.json`, and the variants this runtime accepts.
- `encoder`: convolutional subsampling and the conformer blocks with relative-position attention.
- `decoder`: the LSTM prediction network, the joint network and greedy TDT decoding.
- `model`: loading, `transcribe`, `detokenize` and `set_language`.

It leaves to other crates:

- Log-mel features and audio decoding to `jevons-audio`.
- Windowing recordings longer than one pass, words and segments to `jevons-speech`.
- Device, weight streaming, layers and the tuned GEMM to `jevons-burn`.
- The tokenizer to `jevons-tokenizer`, and language scripts (`Script`) to `jevons-core`.
- Detection and the default model ID to `jevons-models`.

The reference dumps come from `scripts/reference/parakeet_dump.py` (`transformers`' `ParakeetForTDT`
on CPU in F32) under `$JEVONS_GOLDEN_DIR`, in the directory `PARAKEET_GOLDEN` names (default
`parakeet-tdt-0.6b-v3`), with an English and a Spanish clip. Principle P1 applies: the crate forbids
unsafe code.

## Requirements

### R1 Supported configurations

`config.json` must have `model_type` `parakeet_tdt`, a `silu` encoder and a `relu` joint, no
attention or convolution biases, no input scaling, as many KV heads as heads with a hidden size
divisible by them, subsampling with kernel 3, stride 2 and a power-of-two factor of at least 2 that
divides the mel bins, an odd convolution kernel, the blank as the last vocabulary id, durations
`0, 1, 2, …`, at least one decoder layer and a positive `max_symbols_per_step`. Any other value, or
an unreadable file, fails with `UnsupportedModel`. The published 0.6B v3 configuration parses.

Tests: `the_published_v3_configs_parse`, `unsupported_variants_are_rejected`

### R2 Feature extractor

`processor_config.json` gives the sample rate, FFT size, window, hop, mel count and preemphasis. Its
feature size must equal the encoder's mel bins, its window must not exceed the FFT size, and its hop
must be positive. A file that breaks these rules, or cannot be read, fails with `UnsupportedModel`.

Tests: `the_published_v3_configs_parse`

### R3 Load checks

A checkpoint directory needs `config.json`, `processor_config.json`, `tokenizer.json` and
safetensors weights. The tokenizer's size, `<blank>` included, must equal `vocab_size`, and every
weight must have its expected shape; each failure is `UnsupportedModel`. The model reports
architecture `parakeet-tdt`, the processor's sample rate, frames of `hop × subsampling factor`
samples (0.08 s for v3), a 120-second window and the 25 languages of v3.

Tests: none yet

### R4 Encoder frames

Each stride-2 subsampling step maps `n` mel frames to `ceil(n / 2)`: with factor 8, 1600 frames give
200 and 585 give 74. Encoder input is padded to a multiple of 64 encoder frames (5.12 s), and padded
frames are hidden from attention and zeroed before every convolution.

Tests: `the_published_v3_configs_parse`

### R5 Relative-position attention

Relative positions run from `frames - 1` down through 0 to `-(frames - 1)`, with sine and cosine
interleaved. The relative shift aligns each query with its distance to each key. Inference batch
norm folds into the depthwise convolution's kernel and bias.

Tests: `relative_positions_count_down_through_zero`, `rel_shift_aligns_each_query_with_its_relative_distance`, `batch_norm_folds_into_the_depthwise_kernel`

### R6 Encoder parity with the reference

For the English and the Spanish clip, on the valid frames, the subsampling output stays within 1e-2
relative RMS error of the reference, and the encoder output and its projection into the joint space
within 3e-2.

Tests: `encoder_matches_the_reference`

### R7 Greedy token-and-duration decoding

A blank advances by its predicted duration, at least one frame. A token is emitted at its frame,
spans `max(duration, 1)` frames and advances by its duration. Tokens with zero duration repeat at
one frame at most `max_symbols_per_step` times before decoding moves on.

Tests: `blanks_skip_ahead_and_tokens_advance_by_their_duration`, `repeated_zero_durations_are_capped_per_frame`

### R8 Transcript parity with the reference

For the English and the Spanish clip, the token ids, their start frames and the decoded text equal
the reference's greedy transcript. Every token's log probability is at most 0 and its end lies after
its start.

Tests: `greedy_transcripts_match_the_reference`

### R9 Token times and scores

A token's start is its frame times the frame duration, its end is its frame plus its span times the
frame duration, and its log probability is the natural log of its softmax probability over the
allowed vocabulary tokens.

Tests: `greedy_transcripts_match_the_reference`

### R10 Window limits

`transcribe` takes at most 120 seconds of samples at the model's rate; longer audio fails with
`InvalidInput`. Audio shorter than one hop returns no tokens.

Tests: none yet

### R11 Language restriction

`set_language` takes an ISO-639-1 code and fails with `InvalidInput` for any other string. A known
code restricts decoding to tokens whose pieces are written in that language's script; `None` lifts
the restriction.

Tests: none yet
