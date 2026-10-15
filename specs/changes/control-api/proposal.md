# Control API: proposal

[Design](design.md) · [Tasks](tasks.md)

## Why

You want Claude Code, or another agent, to use the desktop the way jevons does: read the
interfaces of your applications, act on them, talk to you through the bubble, and improve
jevons itself by editing its settings folder. Today the screen is read only by the context
investigator and by automations, nothing outside a take reaches the bubble, and the server's
only door is the inference API (`/v1`) and the desktop protocol (`/desktop`).

## What changes

- The desktop server serves a **control API** to outside agents on its listener, behind its key:
  `/mcp` (MCP over streamable HTTP) and `/api/v1/*` (REST, described by `/api/openapi.json`).
  Both are built from one table of operations, so they cannot drift.
- The desk gains one call, `control`, which carries an agent's screen reads, actions and bubble
  calls to the client. The client enforces its own `[control]` settings, as it does for its tools.
- An outside agent reads and acts only in the applications `[control] apps` lists; its first
  action in each one asks you in the bubble, and a yes holds while it keeps working.
- The bubble shows an agent's messages and questions as its own, never inside a take's.

What else is exposed, by phase: voice and takes, the machines, the configuration loop (flows,
automations, tools, the guides) and dictating to the agent. [tasks.md](tasks.md) has the list.

## Decisions (2026-10-15)

- Acting: listed applications, ask once per application.
- A key is always needed; without one `/mcp` and `/api` are not served.
- "Open API": REST with a generated OpenAPI 3.1 document, beside MCP.
- The bubble: show messages, read what it shows, ask (yes/no, or a spoken reply), and dictate
  to the agent.

## Specs it touches

- `jevons-desktop-protocol`: the control call, its types, the wire (`VERSION` 3).
- `jevons-desktop-core`: the desk's side of the control API.
- `jevons-desktop-server`: the operations, the two doors, the guard.
- `jevons-desktop`: the bubble's agent messages and questions, the settings, the listener.
- `principles.md`: a rule for outside agents (P15).
- `jevons-api`: none.
