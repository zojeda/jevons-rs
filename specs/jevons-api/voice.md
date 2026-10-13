# Voice

[jevons-api](spec.md)

## Purpose

`POST /v1/audio/speech` on the Voice service: the OpenAI JSON body, the checks against the voice
model, the two audio formats and the streamed answer.

## Scope

Cleaning the text, sentences, number rules and each sentence's language belong to `jevons-voice`,
and PCM16, the WAV header and resampling to `jevons-audio`. The queue the job waits in is in
[workers](workers.md), and the `/health` entry in [http](http.md). In this workspace `speech`
names speech to text, so the service behind OpenAI's speech route is Voice.

## Requirements

### R1 Body fields

The request is a JSON object. `model` and `input` are required. The other fields are optional,
and `null` counts as absent:

| Field | Accepted |
| --- | --- |
| `model` | The served ID or an alias |
| `input` | The text to speak, with at least one character that is not whitespace |
| `voice` | A string. Absent: the service's default voice |
| `response_format` | `wav` (default) or `pcm` |
| `language` | An extension: an ISO-639-1 code, in any case, for the whole text. Absent or empty: each sentence's language comes from its words |
| `speed` | `1` or `1.0` |
| `stream_format` | `audio` |
| `instructions` | No value: the model takes none |

A field not named here, another `response_format` (`mp3`, `opus`, `aac`, `flac`), `speed` or
`stream_format`, and any `instructions` get `400` `unsupported_parameter` with `param` naming the
field. A missing or blank `model` or `input`, and a `model`, `input`, `voice`, `response_format`
or `language` that is no string, get `400` with `param` naming the field. A body that is no
object gets `400`.

Tests: `a_speech_request_defaults_to_wav_in_the_default_voice`, `unsupported_speech_parameters_are_rejected_not_ignored`, `invalid_speech_fields_name_their_parameter`

### R2 The model's limits

An `input` longer than `max_input_chars` characters gets `400` with `param` `input`. A `voice`
the model does not list gets `400` with `param` `voice` and a message that lists the model's
voices. A `language` the model does not speak gets `400` with `param` `language` and a message
that lists the model's codes.

Tests: `a_request_is_checked_against_the_model_that_serves_it`, `speech_requests_fail_before_the_model_with_openai_errors`

### R3 Checks come before the model

The server validates the fields (R1), then the model name, then the model's limits (R2), and
queues the speech last. An unknown model, or any model while Voice is off, gets `404`
`model_not_found`. A body without a JSON content type gets `415`. No failed check reaches the
voice model.

Tests: `speech_requests_fail_before_the_model_with_openai_errors`, `speech_needs_the_key_and_a_voice_service`

### R4 The wav format

`wav` answers `Content-Type: audio/wav`: the 44-byte header of a mono 16-bit PCM stream at the
model's sample rate, with both size fields `0xFFFFFFFF` because the length is unknown when the
header leaves, then the samples as PCM16 little-endian.

Tests: `a_wav_body_is_a_streamed_header_then_pcm16_at_the_models_rate`, `speech_answers_a_streamed_wav_at_the_models_rate`

### R5 The pcm format

`pcm` answers `Content-Type: audio/pcm`: mono PCM16 little-endian at 24 kHz with no header. Audio
from a model at another rate is resampled to 24 kHz across the whole answer, so the body lasts as
long as the speech.

Tests: `a_pcm_body_is_headerless_pcm16_at_24_khz`, `a_pcm_body_is_resampled_from_another_model_rate`, `speech_answers_pcm16_at_24_khz_whatever_the_models_rate`

### R6 The answer streams

The status and the headers leave once the model has produced its first audio, and the body
carries each stretch of audio as the model produces it. The response has no `Content-Length`.
The text is spoken in the request's `voice` and `language`, or in the default voice with each
sentence's language guessed.

Tests: `a_client_that_hangs_up_stops_the_model`, `speech_answers_pcm16_at_24_khz_whatever_the_models_rate`, `speech_answers_a_streamed_wav_at_the_models_rate`

### R7 A client that hangs up stops the model

When the client closes the connection, the model stops at its next stretch of audio and speaks
none of the sentences left. The next queued speech then starts.

Tests: `a_client_that_hangs_up_stops_the_model`

### R8 Speech failures

Before the first audio, a failure the Voice service reports as invalid input, such as a text
with nothing to say, gets `400`. Any other failure gets `500` (`server_error`, `Speech failed`)
and is logged. After the first audio the status has left: the failure is logged, and the body
breaks off without its end, so a client can tell the answer from a whole one.

Tests: `a_speech_that_fails_is_an_error_before_its_audio_and_a_broken_body_after`, `speech_requests_fail_before_the_model_with_openai_errors`
