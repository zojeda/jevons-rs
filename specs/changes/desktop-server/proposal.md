# Desktop server: proposal

[Design](design.md) · [Tasks](tasks.md)

## Why

You want stateful tasks you can watch, so that risky tools (web search, opening apps, running
commands) run only inside predefined states, with the decision model choosing among drawn
transitions. You also want inference providers to be swappable: the embedded engine, a remote
jevons, OpenRouter.

## What changes

- The flows root becomes a machine that dispatches each take to an agent. Agents run several tasks
  at once; tasks are instances of task machines. Dictation is an agent.
- A sans-IO engine, `jevons-machine`, holds the machine semantics.
- A desktop server hosts the flows, the machines, the tools and an inference router that forwards
  `/v1/*`. The desktop app becomes its client, over the desktop protocol.
- jevons-api stays a pure inference provider.

The spike on `spike/desktop-machines` already turns the root into a machine and runs nested
tasks. [tasks.md](tasks.md) lists what it holds and the phases after it.

## Specs it touches

- `jevons-desktop-core`: flows, machines, pipeline, tools, client.
- `jevons-desktop`: the agent, the Machines tab, the bubble, the CLI.
- New crates, each with a spec when it lands: `jevons-machine`, `jevons-desktop-server`,
  `jevons-desktop-protocol`.
- `jevons-api`: none. It stays as it is.
