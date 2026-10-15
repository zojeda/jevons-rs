# Control API

The control API lets an outside agent, such as Claude Code, use your desktop through jevons. It
can read the interfaces of the applications you list, act on them (click, type, press keys), and
talk to you in jevons's bubble: show a message, ask yes or no, read what the bubble shows. The
desktop server serves it on its listener as **MCP** (`/mcp`, streamable HTTP) and as **REST**
(`/api/v1/<operation>`, described by `/api/openapi.json`). Both come from one table of
operations, so they offer the same things.

It is off until you turn it on, it needs a key, and it reads and acts only where you allow it.

## Turning it on

1. Pick a key and make it the server's: set `TYPESAFE_API_KEY`, or `api_key` under `[server]` in
   `jevons-server.toml`. Without a key the control API is not served, and the log says so.
2. Turn the API on, in `jevons-server.toml`:

   ```toml
   [control]
   enabled = true
   ```

   It listens on `[server] bind:port` (`127.0.0.1:5380` by default), whether or not `expose` is on.
3. Say what an agent may do at your desk, in `jevons-desktop.toml`:

   ```toml
   [control]
   apps = ["slack.exe", "notepad.exe"]   # the applications it may read and act in
   confirm = "once"                      # once | every | never
   bubble = true                         # it may show messages, ask, and read the bubble
   ```

With no `apps`, an agent can use the bubble but sees no application. Reading the bubble includes
what you dictated while a take shows there: set `bubble = false` to keep your words, and the
bubble, out of its reach.

## Connecting Claude Code

Add the server to a project's `.mcp.json` (or with `claude mcp add`), keeping the key in an
environment variable:

```json
{
  "mcpServers": {
    "jevons": {
      "type": "http",
      "url": "http://127.0.0.1:5380/mcp",
      "headers": { "Authorization": "Bearer ${JEVONS_KEY}" }
    }
  }
}
```

```bash
claude mcp add --transport http jevons http://127.0.0.1:5380/mcp \
  --header "Authorization: Bearer $JEVONS_KEY"
```

Claude Code then has the tools `mcp__jevons__status`, `mcp__jevons__outline` and so on, and reads
the server's instructions when it connects. From WSL with mirrored networking
(`networkingMode=mirrored` in `.wslconfig`), `127.0.0.1:5380` reaches the app on Windows.

## Operations

| Operation | What it does |
| --- | --- |
| `status` | What you may do: the listed applications, `confirm`, whether the bubble is available. Call it first. |
| `windows` | The windows of the listed applications, the one in front first, as `w1`, `w2`… |
| `focused` | The element with the keyboard focus, when its application is listed. |
| `outline` | A compact outline below a window or element (`node`, `depth` 1 to 6), one line per element with its id: `e12 Button "Send"`. |
| `find` | Elements below a node by `role` and `text`. |
| `xpath` | An XPath expression in a window or below an element: elements, texts or a value. `$name` takes `variables`. |
| `read` | All the text below an element. |
| `act` | `invoke`, `click`, `focus`, `set_value`, `type_text`, `toggle`, `select`, `expand`, `collapse` or `scroll_into_view` on an element; `text` for the two that enter text. |
| `press` | Keys pressed together in the window in front, such as `ctrl+k` or `enter`; `window` brings one forward first. |
| `type_text` | Text typed into the window in front; `window` brings one forward first. |
| `activate` | Brings a window to the front. |
| `show` | A Markdown message in the bubble, under the agent's name. |
| `ask` | A yes/no question in the bubble, with its own word for yes (`yes`) and a timeout (`timeout_seconds`, 60 by default). Unanswered is no. |
| `bubble` | What the bubble shows now: a take (what you said, what jevons did), a task's conversation, a question or a message. |

XPath names elements by role (`//ListItem`, `//Button[@name = 'Send']`); attributes are `@name`,
`@value`, `@class`, `@automation_id`, `@role`, `@enabled`, `@selected`, `@toggled`, `@expanded`
and `@offscreen`. The flows folder's `AGENTS.md` has the whole vocabulary.

Ids stay valid while jevons keeps them (the latest 5,000 elements). When one is not found, list the
windows or outline again.

## What keeps it safe

- **Only the listed applications.** Other windows are not listed, outlined, searched or read, and
  keys go to the window in front only when its application is listed.
- **It asks first.** With `confirm = "once"`, an agent's first action in an application asks you
  in the bubble: **Let claude-code act in slack.exe?**, with the action. A yes holds for that agent
  and application until it has been quiet for 10 minutes or the settings change. `every` asks
  before each action, `never` does not ask.
- **Never over you.** While a take runs or you hold a key, an action answers `busy` and does not
  happen.
- **No passwords.** A password field is never read, and no text goes into it.
- **You see it.** Each action shows as a line in the bubble under the agent's name. Its messages
  never sit beside a take's words: during a take they wait.
- **A key, and no browsers.** Every request needs the key; a request a browser sends (with an
  `Origin` header) is refused, so a web page cannot drive your desk.
- **Your words only through the bubble.** An agent reads what you dictated only through `bubble`,
  while the take shows there, and only with `[control] bubble` on.
- **Logs** name the operation and how it went, never what was read or typed.

## REST

Operations without input take `GET` or `POST`; the others take `POST` with a JSON body. An agent
names itself in `X-Jevons-Agent`.

```bash
curl -s -H "Authorization: Bearer $JEVONS_KEY" http://127.0.0.1:5380/api/v1/windows
curl -s -H "Authorization: Bearer $JEVONS_KEY" -H "X-Jevons-Agent: script" \
  -H "Content-Type: application/json" \
  -d '{"node": "w1", "depth": 3}' http://127.0.0.1:5380/api/v1/outline
```

A refusal is `{"error": {"kind", "message"}}`:

| Kind | Status | When |
| --- | --- | --- |
| `invalid` | 400 | The input does not fit the operation. |
| `unauthorized` | 401 | No key, or a wrong one. |
| `denied` | 403 | An application that is not listed, a password field, you said no, or a browser. |
| `not_found` | 404 | No such id, or nothing matched. |
| `busy` | 409 | You are dictating or holding a key. |
| `failed` | 500 | The platform could not do it. |
| `unavailable` | 503 | Nobody is at the desk, or it cannot do this. |
| `timeout` | 504 | A wait ran out. |

Over MCP a refusal is a tool error whose text starts with its kind (`busy: …`).

## With `--serve`

`jevons-desktop --serve` serves the control API too, over the desk of the client connected: each
call goes to that client, which checks it against its own `[control]`. With no client connected,
`status` says nobody is at the desk and every other operation is `unavailable`.

## Coming next

[specs/changes/control-api](../specs/changes/control-api/tasks.md) lists what follows: speaking
and spoken replies, takes and the machines, the configuration loop (checking and reloading flows,
automations as tools), and dictating to the agent.
