# Realtime

[jevons-api](spec.md)

## Purpose

`GET /v1/realtime` on the Speech service: OpenAI Realtime transcription sessions over a
WebSocket. Client events arrive as JSON text frames, and the server answers with JSON text
frames.

## Scope

Voice activity detection, resampling and G.711 decoding belong to `jevons-audio`, and model
passes to `jevons-speech`. Authentication is in [http](http.md).

## Requirements

### R1 Opening a session

The route exists when Speech is on with `realtime = true`; otherwise the path is unknown
([http](http.md) R2). The query may carry `model`, which must be a name the speech model answers
to (`404` `model_not_found` otherwise), and `intent`, which must be `transcription` (`400`
`unsupported_parameter` otherwise). A request that is not a WebSocket upgrade gets an OpenAI
error with the upgrade rejection's status. The server accepts the `realtime` subprotocol. The
key may arrive as a subprotocol entry ([http](http.md) R4).

Tests: `the_api_key_may_arrive_as_a_subprotocol`, `server_vad_turns_are_detected_committed_and_transcribed`

### R2 The first event

The server opens with `session.created`, describing a GA transcription session:
`type: "transcription"`, an `id` `sess_<uuid>`, `object: "realtime.transcription_session"`,
input format `{"type":"audio/pcm","rate":24000}`, transcription
`{"model":ID,"language":null,"prompt":""}` with the served model ID, `server_vad` turn
detection (threshold 0.5, `prefix_padding_ms` 300, `silence_duration_ms` 500),
`noise_reduction: null` and an empty `include`.

Tests: `the_first_event_describes_the_default_ga_session`, `server_vad_turns_are_detected_committed_and_transcribed`

### R3 Server events

Every server event carries `type` and an `event_id` `event_<n>`, counting from 1 within the
session.

Tests: `the_first_event_describes_the_default_ga_session`

### R4 GA session updates

`session.update` patches the configuration and answers `session.updated` with the whole session.
Absent fields keep their values. Its `session` may set:

- `type`, which must be `transcription`;
- `audio.input.format`: `audio/pcm` with `rate` 8000, 16000, 24000 (default) or 48000,
  `audio/pcmu` or `audio/pcma`;
- `audio.input.transcription`: `model` (a name the speech model answers to), `language` (a code
  the model transcribes, in any case; empty clears it), and `prompt`, which must be empty;
- `audio.input.turn_detection`: `null` for client commits, or `server_vad` with `threshold`
  (0 to 1), `prefix_padding_ms` and `silence_duration_ms` (0 to 10000), each defaulting as in R2;
- `include`: `["item.input_audio_transcription.logprobs"]` or empty.

Tests: `ga_updates_patch_the_configuration`, `manual_turns_commit_on_request_and_bad_events_get_errors`

### R5 Beta session updates

`transcription_session.update` takes `input_audio_format` (`pcm16` at 24 kHz, `g711_ulaw` or
`g711_alaw`), `input_audio_transcription`, `turn_detection`, `input_audio_noise_reduction` and
`include`, checked as in R4. From then on the session answers in the beta shape:
`transcription_session.updated` with a beta session object, `conversation.item.created` in place
of `conversation.item.added`, and no `conversation.item.done`.

Tests: `beta_updates_switch_replies_to_the_beta_shape`

### R6 Rejected client events

A rejected event leaves the session configuration as it was and gets an `error` event with
`error.type` `invalid_request_error`, a `code`, a `message`, a `param` and the client's
`event_id` (`null` when it has none). The codes:

| Code | When |
| --- | --- |
| `invalid_json` | The frame is not JSON |
| `invalid_event` | `type` is missing or unknown, or the frame is binary |
| `unsupported_event` | `response.create`, `response.cancel` or `conversation.item.create` |
| `unsupported_parameter` | A session type other than `transcription`, a non-empty `prompt`, `semantic_vad`, noise reduction, or an `include` entry other than logprobs |
| `model_not_found` | A transcription `model` the speech model does not answer to |
| `invalid_value` | Any other bad value, including audio that is not base64 |

Tests: `bad_events_are_rejected_with_their_event_id`, `manual_turns_commit_on_request_and_bad_events_get_errors`

### R7 The input audio buffer

`input_audio_buffer.append` adds base64 audio in the session's format (PCM16 little-endian,
G.711 mu-law or A-law), resampled to the model's rate. `input_audio_buffer.clear` empties the
buffer, drops the live turn and answers `input_audio_buffer.cleared`. A buffer longer than
`max_audio_seconds` is cleared, with an `error` event of code `input_audio_buffer_too_long`.

Tests: `audio_commands_decode_and_items_follow_the_ga_order`, `manual_turns_commit_on_request_and_bad_events_get_errors`

### R8 Server turn detection

With `server_vad`, the server sends `input_audio_buffer.speech_started` (`audio_start_ms`,
`item_id`) when speech begins and `input_audio_buffer.speech_stopped` (`audio_end_ms`, `item_id`)
after the configured silence, then commits the turn (R10). Changing turn detection restarts it
with the audio that follows and drops the live turn.

Tests: `server_vad_turns_are_detected_committed_and_transcribed`

### R9 Client commits

`input_audio_buffer.commit` commits the buffered audio, or the live turn, as one item. With less
than 100 ms of audio, it gets an `error` event with code `input_audio_buffer_commit_empty`.

Tests: `manual_turns_commit_on_request_and_bad_events_get_errors`

### R10 Committed items

A commit sends `input_audio_buffer.committed` (`previous_item_id`, `item_id`), then
`conversation.item.added` with a completed user `message` item whose content is one
`input_audio` part with `transcript: null`. Item IDs are unique within the session, and each
item's `previous_item_id` names the item committed before it.

Tests: `audio_commands_decode_and_items_follow_the_ga_order`, `server_vad_turns_are_detected_committed_and_transcribed`

### R11 Live deltas

While a turn is live and no committed turn is being transcribed, a pass over the turn so far
runs after each 0.7 seconds of new audio, for the first 30 seconds of the turn. Words that two
consecutive passes agree on, ignoring case and surrounding punctuation, go out as
`conversation.item.input_audio_transcription.delta` events (`item_id`, `content_index` 0,
`delta`). The newest pass's last word waits for the next pass. A
word already sent is never sent again, even when a later pass revises it. Without turn
detection, the whole buffer is the live turn, so deltas flow before the client commits.

Tests: `agreement_skips_sent_revisions_and_holds_back_the_last_word`, `manual_turns_stream_live_deltas_before_the_commit`

### R12 Final transcripts

Each committed turn is transcribed in full, one turn at a time in commit order. When no live
deltas were sent for the turn, each closed segment goes out as a delta. When the deltas sent so
far begin the final transcript, a last delta sends the rest. Then
`conversation.item.input_audio_transcription.completed` carries `transcript` and
`usage: {"type":"duration","seconds":S}` with the turn's length in seconds. GA sessions then
send `conversation.item.done` with the transcript in the item. The completed transcript is
authoritative and may revise words sent as deltas.

Tests: `server_vad_turns_are_detected_committed_and_transcribed`, `manual_turns_commit_on_request_and_bad_events_get_errors`, `audio_commands_decode_and_items_follow_the_ga_order`

### R13 Log probabilities

With logprobs included, deltas and completed transcripts carry `logprobs`, each
`{"token","logprob","bytes"}`. Without it, they carry none.

Tests: `audio_commands_decode_and_items_follow_the_ga_order`

### R14 Failed transcriptions

A committed turn that cannot be transcribed gets
`conversation.item.input_audio_transcription.failed` with `error.type` `transcription_error`,
code `transcription_failed` and one of these messages: `The transcription queue is full. Retry
later.`, `The audio could not be transcribed.` or `The speech worker stopped.` The session goes
on.

Tests: none yet

### R15 The session ends with the connection

The session ends when the client closes the socket or a send fails. Ping and pong frames get
no events.

Tests: none yet
