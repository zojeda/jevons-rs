# jevons-speech

## Purpose

`jevons-speech` is the Speech service: transcription of recordings and live utterances over any
`SpeechModel` (Parakeet TDT on Burn). `Transcriber` splits recordings longer than one model pass
into overlapping windows, keeps each word once, and groups the words into timed segments.
`Transcriber::pass` runs one pass over a live utterance.

## Scope

It owns:

- `Transcriber`: loading, one-pass transcription, windowed transcription and its text.
- `Transcription`: a windowed job decoded one window at a time.
- `words`, which groups model tokens into words, and the rules that close segments.
- `CONTEXT_SECONDS`, the context each window sees on each side.

It leaves to other crates:

- The `SpeechModel` contract and the `Word`, `Segment` and `Transcript` types to `jevons-core`.
- Detecting and loading a model to `jevons-models` (re-exported with the default `models` feature:
  `SpeechArchitecture`, `default_speech_model_id`, `detect_speech`), and the model to
  `jevons-parakeet`.
- Audio decoding, resampling and voice activity detection to `jevons-audio`.
- The worker thread, its queue, upload limits and the transcription and Realtime wire formats to
  `jevons-api`.

## Requirements

### R1 Pieces become words

`words` starts a word at each token whose piece begins with `▁` and appends the other tokens to the
word before them. Leading tokens without `▁` form one word. Each word spans its first token's start
to its last token's end, its text is trimmed, and words with empty text are dropped.

Tests: `words_join_continuation_pieces_and_segments_split_at_pauses`

### R2 Where segments end

A segment ends after a word that ends a sentence (`?`, `!`, `…`, `。`, or a period not after a listed
title or abbreviation such as `Mr.` or `Sra.`), before a pause of 0.8 seconds or more, before it
would pass 30 seconds, and at the last word.

Tests: `words_join_continuation_pieces_and_segments_split_at_pauses`, `long_audio_is_windowed_and_every_word_is_kept_once_in_order`

### R3 A segment describes its words

Segments are numbered from 0 in order. A segment spans its first word's start to its last word's
end, carries its token ids and their mean log probability, and its text is the model's
detokenization of its tokens, trimmed.

Tests: `long_audio_is_windowed_and_every_word_is_kept_once_in_order`

### R4 Short audio takes one pass

Audio no longer than the model's window is transcribed in one pass.

Tests: `short_audio_takes_one_pass`

### R5 Long audio is windowed and keeps each word once

Longer audio is read in windows as long as the model's window, each starting the window minus
twice `CONTEXT_SECONDS` (5 seconds) after the one before. Each window keeps the words that start in its central span; the first
window keeps from the start of the audio and the last to its end. Timestamps are absolute, and every
word appears once, in order.

Tests: `long_audio_is_windowed_and_every_word_is_kept_once_in_order`, `long_recordings_are_windowed_like_one_reference_pass`

### R6 A window must exceed twice the context

Windowed transcription panics when the model's window is not longer than twice `CONTEXT_SECONDS`.

Tests: none yet

### R7 Segments stream once they are settled

`transcribe` passes each segment to `on_segment` with its words once no later window can change it,
so each segment arrives once and in order. The open segment at the end of a window waits for the
next window.

Tests: `long_audio_is_windowed_and_every_word_is_kept_once_in_order`, `recordings_stream_segments_then_the_transcript`

### R8 A false callback stops the transcription

When `on_segment` returns false, transcription stops after that segment, and the transcript ends
with it: later segments and words are dropped.

Tests: `a_false_callback_stops_after_that_segment`

### R9 The transcript

The transcript holds the text of every kept word (the model's detokenization, trimmed), the words,
the segments, and the duration of the audio in seconds.

Tests: `long_audio_is_windowed_and_every_word_is_kept_once_in_order`

### R10 A job decodes one window per step

`Transcriber::step` decodes the next window of a `Transcription` and returns the segments it closed,
so a caller can run other passes between windows. `finish` returns the transcript of a finished or
stopped job. `step` on a finished job panics.

Tests: `live_passes_run_between_the_windows_of_a_long_recording`, `recordings_stream_segments_then_the_transcript`

### R11 A live pass

`pass` runs one model pass over at most the model's window and returns its tokens with their
timestamps. `words` and `text` turn them into words and text.

Tests: `live_passes_return_words_and_text`

### R12 The language keeps later passes in its script

`set_language` hands the language to the model, which keeps later passes in that language's script.
`None` lifts the restriction.

Tests: none yet

### R13 Windowed transcripts match one pass

On Parakeet TDT, a long Spanish recording transcribed in windows stays within a 5% word error rate
of one reference pass over the whole recording, and its word starts do not go back in time.

Tests: `long_recordings_are_windowed_like_one_reference_pass`
