# Control API: design

[Proposal](proposal.md) · [Tasks](tasks.md)

## Shape

```
Claude Code ──HTTP──► desktop server listener ([server] bind:port)
                        /v1/*            inference, forwarded           (exists)
                        /desktop         desk clients, WebSocket        (exists)
                        /mcp             MCP, streamable HTTP  ┐ control API: key required,
                        /api/v1/<op>     REST                  │ requests with an Origin refused
                        /api/openapi.json                      ┘
control::ops ── server operations: the session (status; later takes, machines, flows)
             └─ desk operations: session.desk().control(caller, call)
                  in process: LocalDesk → core control (ids, Hands, approvals, the app's bubble)
                  over --serve: RemoteDesk → Effect::Control → the client's LocalDesk
```

The server never touches the platform: screen reads, actions and the bubble go through the desk,
so they work the same with the server in the app and with `--serve`.

## Settings

`jevons-server.toml`:

```toml
[control]
enabled = true   # serve /mcp and /api on [server] bind:port
```

It is served only with a key (`TYPESAFE_API_KEY`, else `[server] api_key`); without one the log
and the status say why. The listener runs while anything is served: `/v1` (`expose`), `/desktop`
(`--serve`) or the control API.

`jevons-desktop.toml`, what an outside agent may do at this desk:

```toml
[control]
apps = ["slack.exe", "chrome.exe"]  # process-name globs it may read and act in; empty: none
confirm = "once"                    # once | every | never: asking before it acts in an app
bubble = true                       # may show messages, ask, and read the bubble
```

## Operations

One table in `jevons-desktop-server::control::ops`. Each operation has a name, a summary, an
input and an output type (schemars), whether it only reads, and a handler. The two doors are
generated from it:

- MCP: one tool per operation, the input's schema as its `inputSchema`; the output as text
  (an output's `text` field as plain text, the rest as JSON). Errors are tool results with
  `isError`. The server's `instructions` (under 4 KB) say what the agent may do and how ids work.
- REST: `GET /api/v1/<op>` for operations without input, `POST /api/v1/<op>` with the input as a
  JSON body otherwise. Errors are `{ "error": { "kind", "message" } }` with a status per kind.
- `/api/openapi.json`: OpenAPI 3.1, schemas from the same types (`components/schemas`), bearer
  security.

Phase 1:

| Operation | Side | What it does |
|---|---|---|
| `status` | server + desk | What is served and what the agent may do here: apps, confirm, bubble. |
| `windows` | desk | The windows of the listed apps, the front one first, as `w<n>`. |
| `focused` | desk | The focused element and its window, when its app is listed. |
| `outline` | desk | A compact outline below a window or element, with ids. |
| `find` | desk | Elements below a node by role and text. |
| `xpath` | desk | An XPath expression over a window or element: elements with ids, or a value. |
| `read` | desk | All the text of an element and its descendants. |
| `act` | desk | `invoke`, `click`, `focus`, `set_value`, `type_text`, `toggle`, `select`, `expand`, `collapse`, `scroll_into_view` on an element. |
| `press` | desk | Keys pressed together, such as `ctrl+k`, in the front window. |
| `type_text` | desk | Text typed into the front window. |
| `activate` | desk | Brings a window to the front. |
| `show` | desk | A message in the bubble, as the agent's. |
| `ask` | desk | A yes/no question in the bubble; the answer. |
| `bubble` | desk | What the bubble shows now. |

## Ids

The desk keeps one table for the control API: windows get `w<n>`, elements `e<n>`, each mapped to
its platform id and its window. A platform element seen again gets its old id. The table holds the
latest few thousand; an id that fell out, or whose element is gone, answers `not_found` ("list the
windows again"). The window an id belongs to is how the desk checks the app before reading or
acting.

## Safety (P15)

- **Where.** Reads and actions happen only in windows of the apps `[control] apps` lists. Nothing
  else is listed, outlined, searched or read; the focused element of another app is not returned.
- **Asking.** With `confirm = "once"`, an agent's first action in an app asks in the bubble ("Let
  Claude Code act in slack.exe?", with the action). A yes holds for that agent and app until it has
  been quiet for 10 minutes. The MCP revision Claude Code speaks has no sessions, so quiet time
  stands for disconnecting. `every` asks each time; `never` does not ask. An unanswered question is
  a no.
- **Never over you.** Actions are refused (`busy`) while a take runs or a key is held: an agent's
  keys never mix with your dictation or typing.
- **Passwords and disabled elements**, as for automations: no text into a password field, a
  password field's text is never read, a disabled element is not acted on.
- **Keys go only to a listed app.** `press` and `type_text` go to the front window only when its
  app is listed (`window` brings one forward first).
- **You see it.** Each action shows as a line in the agent's bubble.
- **Logs** name the operation and its outcome, never its arguments or results (P6).

Actions reuse `automation::hands::Hands` with the listed apps.

## The bubble

The app draws it; the desk reaches it through a `Bubble` trait the app implements.

- `show`: the agent's message, under its name, Markdown. During a take it waits until the take's
  bubble closes: an agent's text never sits beside a take's words.
- `ask`: the question with Yes/No (the yes button's word may be given), answered by click, Enter or
  Esc; unanswered after its timeout (default 60 s, at most 5 min) it is a no. Phase 2 adds a spoken
  reply.
- `bubble`: what it shows now: nothing, a take (status, heard, output, whether it is an answer, the
  stages), a task's conversation (its turns), a question, or a message (and from whom).

## The wire

`Desk::control(caller, call) -> Result<Value, ControlError>`; `Effect::Control` and
`Reply::Controlled`. `VERSION` goes to 3. `Nobody` answers `unavailable`.

`Caller` is the agent's name: MCP's `clientInfo.name`, or REST's `X-Jevons-Agent` header, else
"agent". Approvals are kept per name and app.

## Errors

| Kind | REST | When |
|---|---|---|
| `invalid` | 400 | Bad input: an expression that does not parse, an unknown action. |
| `unauthorized` | 401 | No key or a wrong one. |
| `denied` | 403 | Not a listed app, a password field, you said no, or a request from a browser. |
| `not_found` | 404 | No such id, or nothing matched. |
| `busy` | 409 | A take runs or a key is held. |
| `failed` | 500 | The platform could not do it. |
| `unavailable` | 503 | No desk, no inspector, the bubble off. |
| `timeout` | 504 | A wait ran out. |

## Later phases

Their operations follow the same table; each phase gets its details when it starts.

- **Voice and takes.** `say` (text through the voice route, P14 stays: never while listening),
  `ask` with a spoken reply, `transcript` (words run through the flows as if said), recent takes,
  the machines (`machines`, `cancel_task`, `answer`), a tray item that withdraws every approval.
- **Configuration.** The settings folder's paths; `check_flows` and `reload_flows` (committed to
  the settings repository); the tool catalog; automations: list, check, dry run, run (approved
  versions only, P12), ask for approval; approved automations as MCP tools; a `selector` for an
  element; trying an `[extract]`; recording a tree. The guides (`flows/AGENTS.md`,
  `automations/AGENTS.md`, `API.md`) as MCP resources and prompts. Settings files with tools and
  permissions are never written through the API: tools could then run without asking.
- **Dictating to the agent.** A flow route ("Claude, …") hands a take's words to an inbox;
  `next_said` waits for them; `/api/v1/events` streams takes, the machines and the inbox; a stdio
  bridge (`jevons-desktop --channel`) turns the inbox into a Claude Code channel (research
  preview).
