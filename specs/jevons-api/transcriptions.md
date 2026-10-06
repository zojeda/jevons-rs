# Transcriptions

[jevons-api](spec.md)

## Purpose

`POST /v1/audio/transcriptions` on the Speech service: the OpenAI multipart form, audio limits,
response formats and streams.

## Scope

Audio decoding belongs to `jevons-audio`, and windowing, words and segments to
`jevons-speech`. The queue the job waits in is in [workers](workers.md).

## Requirements

### R1 Form fields

The request is a multipart form. `model` and `file` are required. The other fields are optional:

| Field | Accepted |
| --- | --- |
| `language` | An ISO-639-1 code the model transcribes, in any case; empty means none |
| `response_format` | `json` (default), `text`, `srt`, `vtt` or `verbose_json` |
| `stream` | `true`, `True`, `1`, `false`, `False` or `0` |
| `timestamp_granularities[]` | `word` or `segment`, repeatable |
| `include[]` | `logprobs`, repeatable |
| `temperature` | A number from 0 to 1; decoding is greedy, so it changes nothing |
| `prompt` | Empty |
| `chunking_strategy` | `auto` |

The array fields may be sent with or without `[]`. A field not named here, a non-empty
`prompt`, `response_format=diarized_json`, an `include[]` other than `logprobs` or another
`chunking_strategy` gets `400` `unsupported_parameter`. An invalid value gets `400` with
`param` naming the field. A field other than the two arrays given twice, `file` included, gets
`400`.

Tests: `invalid_and_unsupported_fields_are_rejected`, `defaults_are_json_without_timestamps`

### R2 Fields that need a format

`timestamp_granularities` needs `verbose_json`, `include[]=logprobs` needs `json`, and `stream`
needs `json` or `text`. Any other combination gets `400`.

Tests: `invalid_and_unsupported_fields_are_rejected`

### R3 Checks come before inference

The server validates the fields, then the model name, then the file, then decodes the audio,
and queues the transcription last. An unknown model, or any model while Speech is off, gets
`404` `model_not_found`. A missing `file` gets `400` with `param` `file`. No failed check
reaches the speech worker.

Tests: `transcription_requests_fail_before_inference_with_openai_errors`

### R4 Audio

The file is decoded by its contents, with the file name's extension as a hint, and resampled to
the model's rate. Audio that is empty, over 25 MiB, longer than `max_audio_seconds`, or in an
unrecognized format gets `400`; the unrecognized-format message lists the supported formats.

Tests: `transcription_requests_fail_before_inference_with_openai_errors`

### R5 The json format

`json` answers `{"text":TEXT,"usage":{"type":"duration","seconds":N}}`, where `N` is the audio
duration rounded up to whole seconds. With `include[]=logprobs`, `logprobs` lists every token as
`{"token","logprob","bytes"}`, the word-start marker written as a space.

Tests: `defaults_are_json_without_timestamps`, `transcriptions_answer_in_every_response_format`

### R6 Text and subtitle formats

- `text` answers the transcript as `text/plain; charset=utf-8`.
- `srt` answers `application/x-subrip; charset=utf-8`: one cue per segment, numbered from 1,
  with `HH:MM:SS,mmm --> HH:MM:SS,mmm` times.
- `vtt` answers `text/vtt; charset=utf-8`: a `WEBVTT` header, then one unnumbered cue per
  segment with `HH:MM:SS.mmm` times.

Tests: `subtitles_number_cues_and_format_hours`, `transcriptions_answer_in_every_response_format`

### R7 The verbose_json format

`verbose_json` answers `task: "transcribe"`, `language` (the requested code, or `null`),
`duration` in seconds, `text` and `usage`. It includes `segments` when no granularity is given
or `segment` is, each with `id`, `seek` (0), `start`, `end`, `text`, `tokens`, `temperature`
(0) and `avg_logprob`. It includes `words`, each `{"word","start","end"}`, when `word` is given.

Tests: `verbose_json_has_segments_by_default_and_words_on_request`, `transcriptions_answer_in_every_response_format`

### R8 Streams

With `stream=true`, the answer is server-sent `data:` events: one `transcript.text.delta` per
closed segment, where each delta after the first starts with a space, then one
`transcript.text.done` with the full `text` and `usage`. With `include[]=logprobs`, each delta
carries its segment's token log probabilities and the last event carries all of them.

Tests: `streamed_transcriptions_send_segment_deltas_then_the_text`, `streams_space_segments_and_finish_with_the_full_text`

### R9 Transcription failures

A failure the speech model reports as invalid input gets `400`. Any other failure gets `500`
(`server_error`, `Transcription failed`) and is logged. After a stream has started, a failure
sends one event with `type: "error"` and the OpenAI error body, then ends the stream.

Tests: none yet
