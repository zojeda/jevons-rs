# jevons-desktop-server: client

[Back to jevons-desktop-server](spec.md)

The client speaks to the jevons API, embedded in the app or on a server, and to other providers of
the same APIs: typed requests for Realtime sessions, transcription uploads, System One decisions,
Responses, Chat Completions and speech, with the API log that can record the ones that hold no
audio. Each capability has a route to its provider, and a decision route knows what its provider
takes. The forwarder serves the same APIs to other clients. The wire formats themselves belong to
jevons-api; the client mirrors them without sharing its types.
Turning the settings' providers and routes into live routes is in jevons-desktop
([runtime](../jevons-desktop/runtime.md)).

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
serializes as the server's own fixture does, and every answer type parses. A response's `id`,
`provider` and `usage.cost` are ignored. A choice or a score may leave out its probabilities and
its confidence: the probability of a choice is its own among the probabilities, else the
confidence, and an answer that gives neither is taken as given.

Tests: `systemone_questions_round_trip_the_server_fixture`, `systemone_answers_parse_every_type`, `an_external_answer_parses_with_what_it_leaves_out`

### R6 Responses stream their text

A generation posts to `/v1/responses` with `stream: true`, passes each text delta on as it
arrives, and returns the whole text. An `error` or `response.failed` event fails it with the
server's message. Server-sent events split across chunks are joined. A thought budget becomes the
smallest `reasoning.effort` that covers it: `minimal` (64 tokens), `low` (256), `medium` (1024) or
`high`.

Tests: `events_split_across_chunks_are_joined`, `the_log_holds_each_call_s_request_and_response_only_while_it_is_on`

### R7 Chat Completions stream text and join tool calls

A chat turn posts to `/v1/chat/completions` with `stream: true`, the messages, and optional
`tools`, `tool_choice`, a JSON Schema `response_format`, `max_completion_tokens` and
`reasoning_effort`. Text deltas pass on as they arrive. A tool call's pieces are joined into one
call with its id, name and parsed arguments.

Tests: `a_streamed_tool_call_is_joined_and_its_arguments_parsed`, `text_streams_through_the_delta_callback`

### R8 The API log records calls only while it is on

With `privacy.log_api` on, every decision, generation and chat call, the investigator's
included, appends one record to `~/jevons/logs/api.log`: when it started, the API, how
long it took, the exact request body, and the response. A decision's response is its whole
answer. A streamed call's response is the assembled text, the number of events, and every event
that is not a text delta. A call that fails also records its error, and one dropped before its
answer (a time limit or a cancelled take) records that. With the log off, nothing is written.
A request for speech is never recorded, on or off: neither its text nor its audio.

Tests: `the_log_holds_each_call_s_request_and_response_only_while_it_is_on`, `the_api_log_never_holds_a_speech_request_or_its_audio`

### R9 The API log never holds keys and stays bounded

Keys travel in headers, never in bodies, so the API log never holds one. Records are
pretty-printed JSON, one after another, which `jq` reads as a stream. Past 32 MB the log starts
over, and the full one is kept as `api.previous.log`.

Tests: `the_log_holds_each_call_s_request_and_response_only_while_it_is_on`

### R10 The catalog lists the models the app can download

Removed: now [models](../jevons-desktop-core/models.md) R1.

### R11 `models.toml` extends the catalog

Removed: now [models](../jevons-desktop-core/models.md) R2.

### R12 Downloaded models fill the services nobody chose

Removed: now [models](../jevons-desktop-core/models.md) R3.

### R13 Downloads fetch only what the entry names

Removed: now [models](../jevons-desktop-core/models.md) R4.

### R14 Downloads resume and verify

Removed: now [models](../jevons-desktop-core/models.md) R5.

### R15 Downloads can be cancelled and never run twice

Removed: now [models](../jevons-desktop-core/models.md) R6.

### R16 Downloads stay in their folder

Removed: now [models](../jevons-desktop-core/models.md) R7.

### R17 Every tray state has its own frame

Removed: now [models](../jevons-desktop-core/models.md) R8.

### R18 The app icon glows at the eyes

Removed: now [models](../jevons-desktop-core/models.md) R9.

### R19 Each capability has a route

A take reaches inference through five routes, each a provider's client, the model to ask it for,
the provider's name and its decision profile: speech (uploads), Realtime, decisions, generation
and the voice. A capability with no route is not served.

- A take streams to the Realtime route when there is one, and uploads to the speech route when
  there is none, when the session cannot open or when it fails. Live dictation needs the Realtime
  route.
- Decisions, the root's and machines' included, go to the decision route.
- Generation, tool arguments the model writes, the investigator and the automation author go to
  the generation route.
- An answer read aloud goes to the voice route, which also holds the voice to speak in.

A route is named `<provider>/<model>` in traces, such as `openrouter/typesafe/jev-1.13`.

Tests: `each_capability_goes_to_its_own_provider`, `realtime_404_falls_back_to_batch_upload`, `an_answer_is_read_aloud_a_stretch_at_a_time_and_shown_too`

### R20 A decision provider has a profile

A profile says which of our System One extensions (`steps`, `samples`, `think`) the provider
takes, the most questions one request may ask, whether a noul question's criteria need both
`true` and `false`, and the probability from which its model's choice counts as sure.

| Kind of provider | Extensions | Questions a request | Noul criteria | Sure from |
| --- | --- | --- | --- | --- |
| `embedded`, `jevons` | all three | no limit | either alone | 0.7 |
| `openrouter`, `openai-compatible` | none | 8 | both | 0.7 |

A provider's `extensions`, `max_questions` and `min_probability` in the settings replace that part
of its kind's profile. The 8 and the 0.7 of the second row are not documented by OpenRouter: they
are cautious defaults until measured.

Tests: `a_provider_s_kind_gives_its_profile_and_its_settings_replace_parts`

### R21 What a provider lacks is left out, and the trace says so

A decision request goes to its route without the extensions the profile does not take, and each
one dropped adds a note to the take's trace: "steps dropped: openrouter/typesafe/jev-1.13 does not
support it". The trace keeps the request as it went. For a provider whose noul criteria need
both keys, the one a question leaves out goes as `null`. A request to our own System One goes
unchanged.

Tests: `extensions_a_provider_lacks_are_dropped_with_a_note`, `a_provider_without_our_extensions_gets_none_and_the_trace_says_so`

### R22 Questions over a provider's limit go in several requests

A request with more questions than the profile takes is asked in as many requests as needed, one
after the other, in the questions' order, each with the same state. The answers come back as one
response with the usage summed, and the trace notes it: "12 questions asked in 3 requests:
openrouter/typesafe/jev-1.13 takes 5 in one". A request that fails fails the decision.

Tests: `questions_over_a_provider_s_limit_go_in_several_requests`, `questions_over_the_provider_s_limit_are_asked_in_several_requests`

### R23 Other clients' requests go to the provider of their model

The forwarder is the API the app serves to other clients. A `/v1/*` request goes to the provider
of the route that asks for the request's model: the `model` of a JSON body, the `model` field of a
multipart form, or `model` in the query. A model no route names goes to the route of the path's
capability: `/v1/audio/transcriptions` to speech, `/v1/audio/speech` to the voice,
`/v1/realtime` to Realtime, `/v1/systemone` to decisions, and `/v1/responses`,
`/v1/chat/completions` and `/v1/completions` to generation. So a provider's other names for its
models (`jev-latest`) still reach it. With neither, the answer is 404 "No route serves <path>
for the model <model>", and no provider is asked.

Tests: `a_request_goes_to_the_provider_of_its_model_with_that_provider_s_key`, `an_upload_goes_to_the_speech_provider_as_it_was_sent`, `the_routes_are_the_forwarder_s_targets`, `a_request_for_speech_goes_to_the_voice_provider_and_its_audio_comes_back_whole`

### R24 Forwarded requests carry the provider's key, never the client's

The forwarder asks for the app's own key as jevons-api does: `Authorization: Bearer <key>`, or for
a WebSocket the subprotocol `openai-insecure-api-key.<key>`. Without one the answer is 403 "No API
key provided", with a wrong one 401 "Invalid API key", and with no key configured the API is
open. `/health` needs no key. The request goes on with the provider's key in place of the
client's, so other clients never hold a provider's key, and a provider's key does not open the
app's API.

Tests: `the_app_s_key_is_asked_for_and_the_providers_keys_stay_with_the_app`

### R25 A forwarded request and its answer pass unchanged

The method, path, query, body and headers go to the provider as they came, but for the headers of
one connection (`Host`, `Content-Length`, `Connection`, `Transfer-Encoding`, `Upgrade`, `TE`,
`Trailer`), `Accept-Encoding` (the answer is asked for uncompressed, so the API log reads it) and
`Authorization`. The answer's status, headers and body come back the same way, the body as it
arrives, so a stream of events streams and audio passes byte for byte. A request body over 64 MiB is
refused with 413, and a provider that cannot be reached is a 502 "Cannot reach <provider>: …".

Tests: `an_answer_comes_back_as_the_provider_gave_it`, `a_request_goes_to_the_provider_of_its_model_with_that_provider_s_key`, `an_upload_goes_to_the_speech_provider_as_it_was_sent`, `a_request_for_speech_goes_to_the_voice_provider_and_its_audio_comes_back_whole`

### R26 Health and the model listing are the forwarder's own

`GET /health` answers `{"status": "ok", "services": {...}}` with the model of the generation,
decision, speech and voice routes (`generative`, `decision`, `speech`, `voice`; null where there
is no route), in jevons-api's shape, so another jevons app can use this one as a `jevons`
provider. `GET /v1/models` lists the models the routes ask for, each once, in System
One's shape (`models`) and OpenAI's (`object`, `data`), with the provider's name as the owner. No
provider is asked for either, and neither holds a key.

Tests: `the_app_s_key_is_asked_for_and_the_providers_keys_stay_with_the_app`, `a_request_for_speech_goes_to_the_voice_provider_and_its_audio_comes_back_whole`

### R27 A Realtime session passes both ways

`GET /v1/realtime` opens the session with the Realtime route's provider first, with its key and
the `realtime` subprotocol, so a provider that refuses it answers the client with its own status
(a 404 still reads as "not served"). Then the client's upgrade is accepted and text, binary, ping,
pong and close messages pass both ways until either side closes.

Tests: `a_realtime_session_passes_both_ways_through_the_forwarder`

### R28 The API log covers forwarded decisions and generations

With `privacy.log_api` on, a forwarded request with a JSON body appends one record: the API as
"POST /v1/systemone → <provider>", the request body, and the answer's status with its body (as
JSON when it is, as text for a stream of events, cut at 1 MiB and marked `truncated`). A provider
that cannot be reached records its error, and an answer the client stopped reading records that.
Uploads and Realtime sessions are not logged, and no key is. A request to `/v1/audio/speech` is
not logged either, though its body is JSON: it holds the text to speak, and its answer is audio.

Tests: `the_api_log_holds_forwarded_calls_and_never_a_key_or_audio`

### R29 Speech is asked for as PCM, and streams back

A speech request posts JSON to `/v1/audio/speech`: the `model`, the `input` text,
`response_format = "pcm"`, and a `voice` and a `language` only when they are set. The answer is
mono PCM16 little-endian at 24 kHz, and its bytes are passed on as they arrive, in pieces of any
size; a sample that a piece's end cuts in two is joined with its other half, and an odd byte at
the very end is dropped. Dropping the stream hangs up, which is how the provider learns to stop.
An error status keeps the provider's message (R2). An answer in another audio format, such as
`audio/mpeg`, is refused before a byte of it is played.

Tests: `speech_is_asked_for_as_pcm_and_its_bytes_come_as_they_arrive`, `a_speech_request_that_fails_keeps_the_provider_s_message`, `samples_cut_by_a_piece_s_end_are_joined`

### R30 Health may name a voice model

`GET /health` of a jevons provider names its voice model under `services.voice`, as it names the
speech model under `speech`. A server with no voice model, or from before the voice, leaves the
key out or null: it serves none.

Tests: `health_names_the_voice_model_when_the_server_has_one`

### R31 The voice route names its provider, its model and a voice

`[routes] voice = { provider, model, voice }` in `jevons-server.toml` sends speech requests to a
provider. Left out, it goes to `embedded` like the others. `model` may be left out where the
provider names its own, and is needed elsewhere. `voice` is the voice to speak in, as the
provider names it; left out, the provider's default. It belongs to this route alone: on another
route it is refused with "[routes] generation takes no `voice`: only the voice route speaks in
one". `[models.voice]` chooses the model the embedded provider loads for it, as the other
services choose theirs.

Tests: `the_voice_route_names_its_provider_its_model_and_the_voice_to_speak_in`, `each_capability_goes_to_the_provider_its_route_names`
