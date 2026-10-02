# jevons-audio

## Purpose

jevons-audio prepares audio for speech models on the CPU. It decodes uploaded files to mono
samples at the model's rate within size and duration bounds, resamples streams, converts raw
16-bit PCM and G.711 bytes, computes the log-mel features Parakeet expects, and detects speech
turns in a live stream. All of it is Rust: no ffmpeg and no libopus.

## Scope

jevons-audio owns `decode_audio` with `AudioLimits`, `Resampler` and `resample`, `pcm16_le`,
`mulaw` and `alaw`, `LogMel` with `MelConfig` and `Features`, and `Vad` with `VadConfig` and
`VadEvent`.

It leaves to other crates:

- Upload and Realtime wire formats, their size limits and their input rates: `jevons-api`.
- Windowing long audio, words and segments: `jevons-speech`.
- Running the model on the features: `jevons-parakeet`.
- Microphone capture: `jevons-desktop`.

## Requirements

### R1 Uploads decode to mono at the requested rate

`decode_audio(bytes, extension, rate, limits)` decodes WAV, FLAC, MP3, OGG Vorbis, Ogg Opus, WebM
with Opus, and M4A or MP4 with AAC to mono `f32` samples at `rate`. It decodes the first track
that has a codec. `extension`, from the upload's file name, helps the container probe; WebM and
Ogg Opus decode without it.

Tests: `stereo_wav_is_averaged_to_mono_and_resampled`, `webm_and_ogg_opus_decode_like_their_lossless_originals`, `example_flac_files_decode_like_the_reference_reader`, `wav_uploads_decode_back_to_the_same_audio`

### R2 Channels average to mono

`decode_audio` averages each frame's channels into one sample, then resamples to `rate`.

Tests: `stereo_wav_is_averaged_to_mono_and_resampled`

### R3 Uploads are bounded

`decode_audio` fails with `InvalidInput` when the input is empty or larger than
`limits.max_bytes` ("Audio must contain 1 byte to N MiB"), and when the decoded audio runs past
`limits.max_seconds` ("Audio exceeds the N second limit"). Bytes in no supported format fail with
a message that lists the supported formats.

Tests: `empty_oversized_long_and_garbage_audio_is_rejected`

### R4 Every decoding failure is invalid input

`decode_audio` fails with `InvalidInput` when the file has no audio track, the track has no sample
rate, the codec is unsupported, the container is corrupt or truncated, or decoding yields no
samples.

Tests: `empty_oversized_long_and_garbage_audio_is_rejected`

### R5 A corrupt packet is skipped

A packet that fails to decode is dropped, and decoding goes on with the next packet.

Tests: none yet

### R6 Opus decodes at 48 kHz like libopus

Opus packets in WebM and Ogg decode at their native 48 kHz before resampling to `rate`. On the
WebM fixture the samples differ from libopus's by less than 1e-6. Decoded Opus files match their
lossless originals in length within 1% and correlate with them above 0.9.

Tests: `webm_opus_decodes_bit_exactly_like_libopus`, `webm_and_ogg_opus_decode_like_their_lossless_originals`

### R7 Opus headers give channels, pre-skip and gain

An `OpusHead` sets the channel count, the 48 kHz samples dropped from the start, and the output
gain in Q7.8 decibels. A head shorter than 19 bytes or without the `OpusHead` magic fails. Mapping
families other than 0 and 1, zero or more than two channels, and multistream audio fail with a
message that asks for mono or stereo. A WebM track without a head decodes with the track's channel count
(one or two), no pre-skip and unit gain.

Tests: `opus_heads_give_channels_pre_skip_and_gain_and_reject_surround`

### R8 16-bit PCM covers the full range

`pcm16_le` reads little-endian 16-bit samples as `value / 32768`, in `[-1, 1)`. A trailing odd
byte is ignored.

Tests: `pcm16_covers_the_full_range_and_ignores_a_trailing_byte`

### R9 G.711 decodes to the reference values

`mulaw` and `alaw` decode each byte to its ITU-T G.711 linear value divided by 32768. µ-law
`0xFF` and `0x7F` give 0, `0x80` gives 32124 and `0x00` gives -32124. A-law `0xD5` gives 8,
`0x55` gives -8, `0xAA` gives 32256 and `0x2A` gives -32256.

Tests: `g711_decodes_reference_codes`

### R10 Resampling keeps the duration

A stream of `len` samples resampled from `from` to `to` yields `round(len · to / from)` samples
once `finish` runs. Output sample `n` sits at input position `n · from / to`. Equal rates return
the input unchanged. Both rates must be positive; `Resampler::new` panics otherwise.

Tests: `common_rates_preserve_length_and_an_in_band_tone`, `equal_rates_pass_samples_through`

### R11 Resampling keeps the band and removes aliases

Away from the edges, a 440 Hz tone resampled from 48, 44.1, 24 or 8 kHz to 16 kHz differs from the
tone generated at 16 kHz by an RMS error below 5e-3. A 12 kHz tone resampled from 48 kHz to 16 kHz
leaves an RMS below 1e-2.

Tests: `common_rates_preserve_length_and_an_in_band_tone`, `tones_above_the_output_nyquist_are_removed`

### R12 Streaming resampling matches one pass

`process` appends every output sample whose input it has seen in full, and `finish` flushes the
rest as if silence followed. Feeding a signal in chunks gives the same samples as `resample` over
the whole signal.

Tests: `streaming_in_chunks_matches_one_pass`

### R13 Mel frames follow a centered STFT

`LogMel::features` over `len` samples returns `1 + len / hop_length` frames of `n_mels` values,
row-major. The first `len / hop_length` frames are valid, and the rows from `valid` on are zero.
`LogMel::new` panics unless `win_length <= n_fft`, `hop_length > 0` and `n_mels > 0`.

Tests: `frame_counts_follow_the_centered_stft`, `a_tone_peaks_in_its_mel_band_and_padding_rows_are_zero`

### R14 Mel features match the Parakeet extractor

`features` computes the features of the NeMo and transformers `ParakeetFeatureExtractor`:
pre-emphasis (0 turns it off), a centered STFT with a symmetric Hann window, a power spectrum, a
Slaney mel filterbank, a guarded natural log, and per-mel normalization over the valid frames. On
the reference clips no value differs by 1e-3 or more. The filterbank matches librosa's
`filters.mel` with `norm="slaney"`, and a tone peaks in its own mel band.

Tests: `features_match_the_reference_extractor`, `slaney_filters_match_librosa_reference_points`, `a_tone_peaks_in_its_mel_band_and_padding_rows_are_zero`

### R15 Voice detection defaults follow server_vad

`VadConfig::default()` sets a threshold of 0.5, 300 ms of prefix padding and 500 ms of silence, the
OpenAI `server_vad` defaults.

Tests: `a_turn_starts_with_prefix_padding_and_stops_after_the_silence`

### R16 Speech starts after 60 ms of loud frames

`Vad` measures 20 ms frames against a noise floor that follows the quiet frames. A frame is loud
when its level is above -55 dB and more than `6 + 12 · threshold` dB above the floor, with the
threshold clamped to `[0, 1]`. Three loud frames in a row start speech: `SpeechStarted` reports
the first loud frame's start minus the prefix padding, floored at 0. A higher threshold
ignores quieter speech.

Tests: `a_turn_starts_with_prefix_padding_and_stops_after_the_silence`, `short_clicks_and_pauses_shorter_than_the_silence_do_not_split_turns`, `a_higher_threshold_ignores_quieter_speech`

### R17 Speech stops after the silence duration

Once speech has started, `silence_duration_ms` of quiet frames in a row stop it, and
`SpeechStopped` reports the end of the last loud frame. A shorter pause keeps the turn going.

Tests: `a_turn_starts_with_prefix_padding_and_stops_after_the_silence`, `short_clicks_and_pauses_shorter_than_the_silence_do_not_split_turns`

### R18 Voice detection streams

`push` takes chunks of any length, keeps samples short of a frame for the next call, and returns
the events the new frames complete. Event positions are sample offsets from the start of the
stream. `is_speaking` reports whether a turn is open.

Tests: `a_turn_starts_with_prefix_padding_and_stops_after_the_silence`

### R19 A manual commit ends the turn

`reset_turn` closes the open turn without an event, as a manual commit does. The next turn starts
after three new loud frames.

Tests: none yet
