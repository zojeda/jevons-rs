# Control API: tasks

[Proposal](proposal.md) · [Design](design.md)

Work happens on `spike/control-api`, cut from `dev`; it merges only when you ask.

## Phase 1: the doors, the screen, actions and the bubble

- [x] Protocol: `control` types (calls, inputs, outputs, `Caller`, `ControlError`), `Desk::control`,
      `Nobody` and `Seat`, `Effect::Control` / `Reply::Controlled`, `VERSION` 3.
- [x] Core: the desk's side: the id table, the screen operations, actions through `Hands` with
      approvals, the busy rule, the `Bubble` trait; `LocalDesk::with_control`.
- [x] Server: `[control]` in `jevons-server.toml`; `control::ops` (the table), `rest` (REST and
      the OpenAPI document), `mcp` (rmcp, streamable HTTP), the key and Origin guard; mounted by
      `serve::listen`.
- [x] App: `[control]` in `jevons-desktop.toml`; the bubble's agent messages and questions and its
      view; the listener serves the control API in the tray app and in `--serve`.
- [x] Specs: protocol, core, server and app requirements with their tests; P15 in
      `principles.md`.
- [x] Docs: `docs/control-api.md` (turning it on, the key, Claude Code's `.mcp.json`, the
      operations); README.
- [x] Checks: fmt, clippy, tests, `just check-specs`; an MCP client and `curl` against `--serve`
      with a recorded tree (headless, two processes): the headless client's desk answered
      `windows` with the recorded Slack window while connected.
- [ ] On Windows: Claude Code in WSL reaches `127.0.0.1:5380` (mirrored networking), reads Slack
      or Notepad, acts once approved, shows and asks in the bubble. Installed 2026-10-15 (a6ff5b4):
      from WSL with the key, `status`, `windows`, `outline` and `xpath` answered over UI
      Automation (Notepad: 30 elements); acting, showing and asking wait for a try at the desk.

Known gaps of phase 1:

- The agent's bubble logic in the app (an agent's message waiting for a take, its action lines,
  one question at a time) has no test: only the bubble's render and its reading are tested.
- An element's `enabled` is checked as it was when its id was given, not as it is now.
- Hands' refusal for a password field now reads "text is never entered into a password field",
  for automations too.

## Phase 2: voice, takes and the machines

- [ ] `say`, `ask` with a spoken reply, `transcript`, recent takes, `machines`, `cancel_task`,
      `answer`; a tray item that withdraws approvals; `[control]` in the Settings tab.

## Phase 3: the configuration loop

- [ ] Paths, `check_flows`, `reload_flows`, the tool catalog, automations (list, check, dry run,
      run, ask for approval), approved automations as MCP tools, `selector`, `extract`,
      `record_tree`; the guides as MCP resources and prompts.

## Phase 4: dictating to the agent

- [ ] A flow route to the agent's inbox, `next_said`, `/api/v1/events`, `jevons-desktop --channel`.
