# jevons-desktop-core: client

[Back to jevons-desktop-core](spec.md)

The client speaks to the jevons API, embedded in the app or on a server: typed requests for
Realtime sessions, transcription uploads, System One decisions, Responses and Chat Completions,
with the API log that can record them. Next to it sit the model catalog and its downloads from
Hugging Face, and the tray icon's frames. The wire formats themselves belong to jevons-api; the
client mirrors them without sharing its types. Choosing the runtime (embedded or remote) is in
jevons-desktop.

## Requirements

### R1 Requests carry the key when there is one

A client has a server root, with any trailing `/` dropped, and an optional key; an empty key
counts as none. Every request carries the key as a bearer token, and a Realtime session as an
`Authorization` header. A connection gives up after five seconds. `GET /health` names the model
serving each service.

Tests: none yet

### R2 Errors keep the server's message

An error status gives the status and the server's message: `error.message` from an OpenAI-style
body, the `detail[].msg` entries of a System One body joined with `; `, a `detail` string, or else
the first 200 characters of the body. Other failures say the API cannot be reached, that the
runtime does not serve something, that a Realtime session failed, or that a response was not what
was expected.

Tests: `error_messages_come_from_either_error_shape`

### R3 Realtime sessions transcribe what the app sends

A session opens `/v1/realtime?intent=transcription` (over `ws://` or `wss://`) with the `realtime`
subprotocol, then configures transcription of PCM16 at 24 kHz with the model and language. With
the client ending turns, `turn_detection` is null; with server turns, it is `server_vad` with its
silence. The client appends base64 audio, commits a turn, and clears the buffer. It reads deltas,
completed transcripts and errors, and names any other event by its type. A 404 means Realtime is
not served; another error status keeps the server's message.

Tests: `transcription_events_parse`, `realtime_404_falls_back_to_batch_upload`, `live_dictation_shows_each_phrase_and_types_the_whole_text_once_stopped`

### R4 Uploads send the take as WAV

A transcription upload posts a multipart form to `/v1/audio/transcriptions`: the audio as
`take.wav` (mono PCM16 at the take's rate), the model, `response_format = json`, and the language
when one is set. The WAV decodes back to the same samples.

Tests: `wav_uploads_decode_back_to_the_same_audio`

### R5 System One requests mirror the server's schema

A decision request holds the model, the state, the questions, and optional `steps`, `samples` and
`think`. A question is `noul` (yes or no, with optional criteria for each), `choice` (one of 1 to
128 labels, each with its description) or `score` (2 to 10 levels, lowest first). The request
serializes as the server's own fixture does, and every answer type parses.

Tests: `systemone_questions_round_trip_the_server_fixture`, `systemone_answers_parse_every_type`

### R6 Responses stream their text

A generation posts to `/v1/responses` with `stream: true`, passes each text delta on as it
arrives, and returns the whole text. An `error` or `response.failed` event fails it with the
server's message. Server-sent events split across chunks are joined. A thought budget becomes the
smallest `reasoning.effort` that covers it: `minimal` (64 tokens), `low` (256), `medium` (1024) or
`high`.

Tests: `events_split_across_chunks_are_joined`, `the_log_holds_each_call_s_request_and_response_only_while_it_is_on`, `adk_contents_become_chat_turns_with_calls_and_results`

### R7 Chat Completions stream text and join tool calls

A chat turn posts to `/v1/chat/completions` with `stream: true`, the messages, and optional
`tools`, `tool_choice`, a JSON Schema `response_format`, `max_completion_tokens` and
`reasoning_effort`. Text deltas pass on as they arrive. A tool call's pieces are joined into one
call with its id, name and parsed arguments.

Tests: `a_streamed_tool_call_is_joined_and_its_arguments_parsed`, `text_streams_through_the_delta_callback`

### R8 The API log records calls only while it is on

With `privacy.log_api` on, every decision, generation and chat call, the investigator's and
agents' included, appends one record to `~/jevons/logs/api.log`: when it started, the API, how
long it took, the exact request body, and the response. A decision's response is its whole
answer. A streamed call's response is the assembled text, the number of events, and every event
that is not a text delta. A call that fails also records its error, and one dropped before its
answer (a time limit or a cancelled take) records that. With the log off, nothing is written.

Tests: `the_log_holds_each_call_s_request_and_response_only_while_it_is_on`

### R9 The API log never holds keys and stays bounded

Keys travel in headers, never in bodies, so the API log never holds one. Records are
pretty-printed JSON, one after another, which `jq` reads as a stream. Past 32 MB the log starts
over, and the full one is kept as `api.previous.log`.

Tests: `the_log_holds_each_call_s_request_and_response_only_while_it_is_on`

### R10 The catalog lists the models the app can download

The built-in catalog holds, in order: DiffusionGemma 26B-A4B Q4_K_M (generative and decision, from
`unsloth/diffusiongemma-26B-A4B-it-GGUF`, with its vision projector from
`FreedomAISVR/DiffusionGemma-26B-A4B-it-MXFP4-GGUF`), Nemotron-Labs-Diffusion 3B and VLM 8B
(generative and decision), and Parakeet TDT 0.6B v3 (speech). Each entry downloads into a folder
named by its id under the models folder, and loads as its model file or as that folder.

Tests: `user_entries_extend_and_override_the_builtin_catalog`, `downloaded_catalog_models_fill_unset_services_with_gemma_first`

### R11 `models.toml` extends the catalog

Entries in `models.toml` next to the settings file add to the built-in catalog, and an entry with
a built-in id replaces that one. Unknown fields are errors. A file that does not parse leaves the
built-in catalog, with the error.

Tests: `user_entries_extend_and_override_the_builtin_catalog`

### R12 Downloaded models fill the services nobody chose

A service with no model selected uses the first downloaded catalog entry that serves it, in
catalog order: DiffusionGemma first for generative and decision, then Parakeet for speech. A
selected model stays. With `runtime_config` set, nothing is filled.

Tests: `downloaded_catalog_models_fill_unset_services_with_gemma_first`

### R13 Downloads fetch only what the entry names

A download lists the repository's files at the entry's revision (and those of its extra sources)
and keeps the ones its globs match; none matching is an error. `HF_TOKEN`, when set, goes with
every request. Nothing downloads unless the user asks.

Tests: `interrupted_download_resumes_with_range`, `extra_sources_download_into_the_same_folder`

### R14 Downloads resume and verify

Each file downloads into `<file>.part`, resuming an interrupted one with an HTTP range. A file with
a SHA-256 in the repository is checked against it: a mismatch removes the part file, keeps no
final file and fails the download. A finished file is renamed into place, and once every file is
in place a marker makes the entry ready.

Tests: `interrupted_download_resumes_with_range`, `checksum_mismatch_marks_model_corrupt_and_keeps_no_final_file`

### R15 Downloads can be cancelled and never run twice

A cancelled download stops, keeps its part file for a later resume, and leaves the entry not
ready. A model folder that is being downloaded cannot start a second download in the same
process.

Tests: `a_cancelled_download_keeps_its_part_file`, `a_model_being_downloaded_cannot_be_downloaded_twice`

### R16 Downloads stay in their folder

A repository path that is absolute or climbs out with `..` is refused, so no file lands outside
the model's folder.

Tests: `repository_paths_cannot_escape_the_model_folder`

### R17 Every tray state has its own frame

The tray icon is the jevons alien at 32 pixels: cyan when ready, blue while the models load, grey
with no model loaded, amber while GPU kernels are tuned, red after a failed take, and magenta
while a demonstration is recorded. Listening shows a waveform with one frame per meter level,
louder levels drawing taller bars. Transcribing (green) and deciding and writing (violet) show
dots that cycle through three frames, 180 ms apart. Each state indexes its own frame and has its
own tooltip.

Tests: `every_state_indexes_its_own_frame`, `tuning_has_its_own_amber_frame_and_explains_itself`, `louder_audio_draws_taller_bars`, `processing_states_cycle_through_three_frames`

### R18 The app icon glows at the eyes

The app icon, drawn at any size, is a dark head with glowing eyes and clear corners.

Tests: `the_app_icon_glows_at_the_eyes_on_a_dark_head_with_clear_corners`
