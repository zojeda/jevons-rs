# Desktop server: tasks

[Proposal](proposal.md) · [Design](design.md)

Goal: get from the state machine spike to the [design](design.md).
That means a sans-IO machine engine, agents that run several tasks, a desktop server with an
inference router, a desktop client, and the protocol between them, without slowing dictation down
or weakening a safety guarantee.

## Constraints

- **Dictation costs one decision call per take,** as the decision root did, through every routing
  level (root, agent, work). The pipeline tests
  that assert it (`words_needing_no_edits_are_typed_after_one_merged_decision`,
  `every_decision_and_the_generation_report_their_stages_in_order`) stay unchanged in every phase.
- **The safety guarantees are tested where they live, in tiers 1 and 2:**
  - an unsure take never moves a task on;
  - a machine calls only the tools its `tools` lists;
  - a tool asks first;
  - text goes only to the window the take started in.
- **No unsafe code,** and platform code goes only through safe wrapper crates (AGENTS.md).
- **No backward compatibility** for config files, flow formats or settings. An unedited built-in
  flows folder is brought up to date, as `flow/earlier.rs` does now.
- **The all-in-one app pays nothing for the split:** no sockets and no serialization inside one
  process.

## Where the spike stands

Branch `spike/desktop-machines`, cut from `dev`: the spike (a903986, 2dd7e37), the machine files'
names (c476be6) and the search example's Spanish trigger (9c135f2).

- **The root is a machine.**
  - `examples/desktop/flows/root.toml` and `root.fsm` replace the root `decide.toml`.
  - The root waits in `idle`; `said` leads to `ask`, `dictate` or `run`, whose folders are now
    the states' work.
  - `root.toml` holds the old question, `min_probability = 0.7` and `tools = ["script:*"]`.
  - The Slack extracts moved to `ask/decide.toml`.
  - `flow/earlier.rs` records the previous tree, and `defaults::init` removes the files an
    unedited folder's earlier tree had and the current one lacks. They are reported in
    `InitReport::removed` and committed through `InitReport::changed`.
- **The diagram:** `crates/jevons-desktop-core/src/flow/machine/mod.rs`.
  - The diagrams (`root.fsm`, `task.fsm`) are read with `oxidate-fsm` 0.1.0 (default features off: parser and model only)
    and converted into our own `Machine`.
  - Our checks:
    - states are lowercase, since they name folders;
    - events are `said`, `done`, `failed`, `denied` and timers;
    - no actions in the diagram;
    - at most one `[else]` per state and event, and `[else]` wherever a task must leave;
    - choice points need `[else]` and no cycles of their own;
    - every state is reached and has a way out;
    - timers are waited for.
  - `Loaded` adds the compiled `[guards.<name>]` (`when`, `prefer`, `criterion`).
- **Loading:** `flow/tree.rs`.
  - `Kind::Machine` and `MachineSpec` are new.
  - A machine's subfolders must be its states.
  - Named guards must be declared and used.
  - A nested machine must end.
  - A machine's `tools` must cover everything below it.
  - Later states may read earlier states' results as `{state}`.
- **The runtime:** `flow/machine/runtime.rs`, about 1,400 lines.
  - **What it does:**
    - It keeps the root and its nested tasks across takes.
    - A take is `said` for the innermost machine waiting for it.
    - Transitions are weighed like a decision's branches, with the candidates' first decisions in
      the same request (`walk::lookahead`, `walk::check_branch`, and `ahead` answers passed to
      `walk::run`).
    - Entering a state walks its folder and delivers the leaf (`pipeline::deliver_leaf`).
    - An unsure `said` stays put.
    - Timers are armed per state entry and dropped when stale.
  - `Runtime::cancel` ends every task.
  - It mixes the semantics with the host: tokio timers, the walker, delivery and System One
    requests all live in the same file.
- **Layout:** `flow/machine/layout.rs`, a layered (Sugiyama) layout in plain geometry.
  - Back edges run up the right side.
  - The end has a row of its own.
  - Labels are spread so they don't overlap.
- **App:**
  - `Env.machines`, `Trace.machine` (the steps) and `Update::State` (the path) are new.
  - The agent has `MachineTimer` and `CancelTask` commands, and the bubble shows the path while a
    task runs.
  - `ui/machines.rs` is the Machines tab: the path, Cancel task, the diagram with the current state
    and latest transition lit, a state's details, and the history.
  - `--transcript` can be repeated, for several takes against the same machines.
- **Example:** `examples/desktop/machines/search` is a web search you follow up on. It waits in
  `results`, opening a result asks first, and declining goes back to `results`. Its README has the
  tools to register and the root's two new lines.
- **Tests:**
  - 7 diagram tests and 3 layout tests;
  - 4 loader tests (`the_built_in_root_is_a_machine_whose_states_are_the_old_branches`,
    `machine_folders_are_checked_against_their_diagram`,
    `a_nested_machine_must_end_and_list_every_tool_its_states_call`,
    `states_read_what_earlier_states_wrote`);
  - 5 pipeline tests: a task across takes, an unsure take that stays, a timer and stale timers, a
    declined tool taking `denied`, and the search example end to end with an approved open;
  - 2 UI tests of the Machines tab.

  The whole earlier pipeline suite passes unchanged on the root machine. The desktop and core
  suites (35 and 199 tests) and workspace clippy pass.
- **oxidate-fsm findings:**
  - A second `entry /` action silently replaces the first.
  - Descriptions keep their quotes.
  - Its `validate()` rejects `<<choice>>` targets.
  - Choice branches cannot lead to `[*]`.
  - Action parameters can only be identifiers.
  - Semantic errors carry no positions.
  - It pulls in `thiserror` 1, `env_logger`, `quote` and `proc-macro2` (7 crates in the lock file).

  Our dialect bans actions and checks everything itself, so none of these bite yet.

## Phase 0: land the spike

Scope: the spike as it is, reviewed and running on Windows.

- [x] Commit the spike on `spike/desktop-machines`.
- [x] Give the machine files their phase 2 names first, so a settings folder is upgraded once:
      the root is `root.toml` and `root.fsm`, a task `task.toml` and `task.fsm`, each only at its
      level.
- [x] Build and install it on Windows (`windows-build`: `just desktop-windows` mirrors the WSL
      working tree, so nothing needs pushing or merging first).
- [x] Check the upgrade on the real settings folder: an unedited flows folder loses `decide.toml`
      and gains `root.toml` and `root.fsm` in one commit of the settings repository; an edited
      one is left alone. The real folder was edited, so it was left alone; its edits were carried
      onto the new tree by a three-way merge by hand, and a headless take on Windows went
      `idle → dictate → idle` with one decision call.
- [ ] Push the branch.
- [ ] Dictate for a day: routing as before (Pregunta, terminals, Slack ask), one decision call per
      take in the API log.
- [ ] Register `web_search` and `open_url`, copy `search/` in, and run the task live: follow-ups,
      the confirmation, `denied`, the quiet timer, Cancel task. Done so far, with English
      Wikipedia's article search as `web_search`: "Buscar …" starts it with no model call, it
      answers and waits in `results`, "abre el primero" asks and opens the result, and the quiet
      timer ends it. What the live run changed:
      - "Buscar …" said in `results` stayed (end 0.58, searching 0.33): the example now searches
        again by rule (`[guards.again]`).
      - The answer went when a follow-up's bubble hid: a waiting task's bubble is now its
        conversation, and the tray brings it back.
- [ ] Squash-merge into `dev` and then `main` (`gh`, as for the earlier desktop PRs).

Crates: none new. Tested by the existing suites, plus the manual checks above.
Done when: the spike is on `main` and live use shows no routing regression.

## Phase 1: extract the engine

Scope: create `jevons-machine` and move the semantics behind the sans-IO interface. The runtime
in desktop-core becomes the host.

- [x] New crate `crates/jevons-machine` with no tokio, HTTP or `serde_json` dependency. Serde
      derives are allowed, for instances.
- [x] Move the definition (today's `Machine`, `Event`, `Condition`, `Target`, `Choice`, `Timer`),
      the parser front end (`oxidate-fsm`) and the checks.
- [x] Move `layout`: it depends only on the definition.
- [x] Write `Instance::handle(def, input, facts) -> Vec<Effect>` with
      `Input { Event, Decided, Finished }` and
      `Effect { Decide, Run, Arm, Entered, Ended }` (and `Chose`, `Step` and `Stopped`, which say
      what happened). It covers:
      - candidates;
      - rules, then `[prefer]`, then the oracle, then `[else]` or staying put;
      - choice points;
      - `done` chains, bounded at 32;
      - ending into the parent;
      - unhandled failures;
      - generations for timers.
- [x] Add a `Facts` trait for named guards and targets' `[when]` and `[prefer]`. The host
      implements it over the frame and the target folders, as `Turn::check` did.
- [x] Keep `Decide` as data (the candidates with their labels and criteria) and `Decided` as a
      label and probability, or nothing.
- [x] Value checks in guards: `when = { value = "{state.field}", … }` with `empty`, `equals`,
      `matches` and number comparisons, answered by `Facts` from the instance's values. The loader
      checks each path against the state results and the shapes of typed results, as it checks
      placeholders now.
- [x] Typed results: a tool's result (JSON) and a structured generation keep their fields, so value
      checks and `{state.field}` placeholders can read them.
- [x] Decided-by: the loader reports, for each state and event, whether the event alone, rules or
      the model decides its transitions; `--check-flows` prints it. Every decision made also says
      what settled it (`By`), so the bubble and the route preview no longer read the trace's
      words.
- [x] Turn `flow/machine/runtime.rs` into a host that carries out the effects: walks, delivery,
      tokio timers, and System One with the lookahead batching.
- [x] Engine tests with a scripted oracle and scripted facts: every rule of the selection order,
      stays, choice points, timers and stale generations, the step bound, and an event that
      interrupts a state's work. Nesting is the host's and is tested there.

Crates: `jevons-machine` (new), `jevons-desktop-core`, `jevons-desktop` (imports).
Tested by: the new engine tests (value checks included: a search with no results taking
`no_results` with no model call), and every pipeline and UI test unchanged.
Done when: `runtime.rs` holds no selection logic, and the one-call tests still pass.

## Phase 2: agents and tasks

Scope: the three routing levels of the [design](design.md#agents-and-tasks):
the root dispatches to agents, an agent runs several tasks at once, and a take for an agent may go
to one of its running tasks.

- [x] Rename the tool-loop node `agent.toml` to `loop.toml` (`Kind::Loop`), so "agent" names only
      the new level. Update the schemas, `AGENTS.md`, the docs and the tests. The built-in tree
      has no such node, so `flow/earlier.rs` needs nothing for it.
- [x] Node files per level: an agent is `agent.toml` and `agent.fsm` (the root's `root.toml` and
      `root.fsm` and a task's `task.toml` and `task.fsm` landed in phase 0). The loader checks
      that agents sit under the root and tasks under agents, and that each agent's scope covers its
      tasks' `tools`. Every state folder of the root is an agent, and a task cannot hold a task.
- [x] The instance forest in the host: the root, one instance per agent (created when the tree
      loads, kept across takes, rebuilt when the tree changes and everything is at rest), and
      each agent's tasks. Every step, effect, timer and trace record carries its instance.
- [x] Spawning: entering an agent's state whose folder is a task machine starts the task and is
      `done` at once. Entering the root's state for an agent hands it the take and returns the root
      to `idle`.
- [x] Dispatch to running tasks: when an agent gets `said`, add each of its running tasks that
      waits for `said` to the candidates, described by the task's current state. A chosen task
      gets the take as its own `said`. In the engine these are candidates from outside
      (`handle_among`), weighed with the transitions.
- [x] Task results: `task_done` and `task_failed` events for the agent, with the task's name and
      last result as values (`{task.name}`, `{task.result}`).
- [x] One request per take: batch the root's question, the agent's (with its running tasks) and
      the chosen work's first decision, within the question limit. The task's question rides
      too, so a follow-up through all three levels is one call.
- [x] The built-in tree becomes three agents with inline work: `dictation` (today's `dictate`),
      `assistant` (today's `ask`) and `automations` (today's `run`). The search example becomes a
      task of a `research` agent. Record the spike's tree in `flow/earlier.rs` so an unedited
      folder is brought up to date.
- [x] The Machines tab: the agents, each with its running tasks; any instance's diagram; Cancel
      per task and for all.

What phase 2 decided on the way:

- **An agent is a choice only when it has something to do with the take:** a transition its rules
  allow, or a task that waits. So `research`, whose one transition needs "Search …", takes no
  ordinary dictation, and gets follow-ups only while a search waits. The root also prefers an
  agent whose own rules choose what it would do.
- **A task runs beside its agent within the take that moves it.** It is "in the background" for
  the agent, which is free at once; its work still runs before the take ends. Work that outlives
  a take is not built.
- **A take always starts at the root.** A follow-up reaches a waiting task through the root and
  its agent, in the same request. Dictation while a search waits is therefore typed, where the
  spike gave every take to the waiting task.
- **The bubble follows one task:** the one the latest take reached. A take that goes elsewhere
  (dictation while a search waits) is no part of its conversation, which stays for the next take
  that reaches it. The app's side of this has no test harness; the runtime's side has.
- **A follow-up can now be mistyped.** It goes through the root, so when the model is unsure
  between the agent whose task waits and dictation, the root's `[else]` types the words. In the
  spike an unsure follow-up stayed in the task. Dictation while a task waits is typed again,
  which the spike could not do.
- **Before the next Windows install, the user's flows folder needs moving by hand again:** it is
  an edited tree with inline root states and `search/` under the root, which no longer loads
  (the app then runs the built-in tree). Its states go under agents, and `search/` under a
  `research` agent, as `examples/desktop/machines/research` shows.

Crates: `jevons-machine` (candidates separate from taking a transition), `jevons-desktop-core`
(loader, host), `jevons-desktop` (Machines tab, bubble).
Tested by:
- two searches running at once, a follow-up routed to the right one;
- a take for the agent itself while a task waits;
- a task's end reaching its agent as `task_done`;
- the one-call tests through all three levels;
- the loader's placement and scope checks.

Done when: dictation behaves as before, and two tasks of one agent run side by side, each getting
its own follow-ups.

## Phase 3: the inference router and provider profiles

Scope: route each capability to its own provider, and describe each decision provider.

Before installing it: the single runtime mode goes away, and the settings file rejects unknown
fields, so an existing `jevons-desktop.toml` stops loading. The user's needs moving by hand, as
their flows folder did; keep `[providers]` and `[routes]` a short diff from `[server]`.

- [x] Settings:
      - `[providers.<name>]`: the kind (embedded, jevons, openrouter, openai-compatible), base
        URL and key (`${env:NAME}`);
      - `[routes]`: speech, realtime, decision and generation, each a provider and a model.
      The single runtime mode (embedded or remote) goes away. `embedded` is always there, and a
      route left out goes to it, so settings that set neither behave as before.
- [x] A client per route in place of one `Client`. Realtime falls back to uploads per route, as
      now. Only the models of the capabilities routed to `embedded` are loaded.
- [x] Provider profiles:
      - extensions supported (`steps`, `samples`, `think`);
      - the most questions per request;
      - label rules: whether noul criteria need both `true` and `false`;
      - the decision model's `min_probability`.

      Built-in profiles for the embedded engine and remote jevons (ours) and for OpenRouter's Jev
      and other servers (TypeSafe's contract alone). A provider's settings replace parts of its
      kind's.
- [x] With an external System One:
      - drop unsupported extensions and add a trace note ("steps dropped: … does not support
        it");
      - split requests that ask more questions than the provider takes, noting the extra calls.
- [x] Let the decision model's profile set `min_probability`. Remove `min_probability = 0.7` from
      the built-in root; a node or machine that sets it still overrides the profile.
- [x] Forward `/v1/*`: a pass-through by model name, with the provider's key added and `log_api`
      applied. It listens where the exposed API listens now. The embedded API is always private
      to the app, and `/v1/realtime` is relayed too.

Crates: `jevons-desktop-core` (client, config, pipeline), `jevons-desktop` (settings UI, runtime).
Tested by:
- fake providers (axum, as the pipeline tests do) asserting the dropped fields, the notes, the
  split requests and the routing;
- one manual headless run against OpenRouter (`--transcript` with a decision route to Jev).

Done when: dictation runs with decisions on Jev and generation elsewhere, and its traces say what
was dropped.

What phase 3 decided on the way:

- **What is known of Jev comes from OpenRouter's reference, not from a run.** Its schema requires
  `instructions` on every question (the desktop always sends them) and both keys of a noul's
  criteria (the profile fills the missing one with `null`), and makes a choice's probabilities
  and confidence optional (the client takes either, or the answer as given). It states no limit
  on questions, labels or state size, and nothing about calibration: 8 questions a request and
  0.7 are cautious defaults a provider's settings replace, and the manual run is what settles
  them.
- **`sequential` and images are not in the profile:** the desktop never sends either to System
  One, so there is nothing to drop.
- **The profile sets the floor for machines only.** A decision node (`decide.toml`) without
  `min_probability` still takes the model's choice at any probability, as before: its floor is
  what starts `enrich` and the fallback, a different contract from a machine's "unsure".
- **Split requests go one after the other,** so a split take is slower by a request each; they
  are rare at 8 a request (a take asks 2 to 6).
- **A model no route names goes by the path's capability** (`/v1/systemone` to the decision
  route), so `jev-latest` and a provider's other models still work through the forwarder.
  `GET /health` and `GET /v1/models` are the forwarder's own, from the routes.
- **The forwarder lives in jevons-desktop-core** (`forward.rs`) with the routes, ready to move to
  the server crate in phase 4; jevons-desktop's runtime thread only binds it.
- **Not run: the manual headless take against OpenRouter.** It needs an OpenRouter key, which
  this work did not have. Before relying on Jev: run `--transcript` with `[routes] decision` on
  OpenRouter and read the trace's notes and the API log.
- **A provider that fails leaves the others running:** the status says what failed, and the
  capabilities on providers that answer are served.

## Phase 4: split desktop-core into server and client

Scope: move the desktop server's parts out, behind a Rust trait boundary, and keep everything in
one process.

- [x] New crate `jevons-desktop-server`, holding:
      - `flow/` (tree, walker, extract and investigate orchestration, the investigator, agents,
        `JevonsLlm`);
      - the machine host;
      - the server tools in the tool host;
      - the router and providers;
      - the server's settings.
- [x] `jevons-desktop-core` keeps the client:
      - platform traits, context, XPath, delivery and paste safety;
      - gestures, recording, the automation engine and library;
      - client tools, icons, the catalog and downloads.
- [x] Define the boundary as a trait the server calls: deliver, confirm, read the screen
      (an extract, an investigator step), run a client tool. The client core implements it
      in-process (`LocalDesk`). "Show" and "open" are noted below.
- [x] Split the pipeline:
      - the client captures (audio, context) and delivers;
      - the server transcribes through its speech route, moves the machines, walks and decides.
- [x] Split the settings file in two (client: hotkeys, microphone, privacy, automations; server:
      providers, routes, models, flows folder, tools, the API log), in the same settings folder
      and git repository in the all-in-one mode.

How phase 4 is cut (decided 2026-10-03, before carving):

- **`jevons-desktop-protocol` starts now, not in phase 5,** with the types that cross and the
  boundary trait, and no serde shapes or transports yet. With it the server and the client core
  each depend on the protocol and neither on the other. Without it the client core would have
  to depend on the server to implement its trait.
- **What crosses:** the context snapshot and privacy limits, the delivery vocabulary (action,
  method, request, outcome, audio events), `[extract]` specs with their checks and the XPath
  grammar (the server checks an extract when the tree loads, the client when it reads), and
  shapes. Traces and the take's stream stay in the server crate until phase 5 puts them on a
  wire.
- **The boundary is the `Desk` trait:** `deliver`, `confirm`, `read` (an extract), `look`,
  `look_step` and `look_end` (an investigation: the server keeps the model loop, the client the
  elements it has seen and the remembered paths), `tools` and `run_tool` (the client's tools).
  "Show" is the take's stream today and becomes a message with the streams in phase 5. "Open"
  is a client tool from phase 6 on; until then `open` and `command` tools run with the server,
  on the same machine.
- **Only automations (`script:<name>`) cross as client tools in this phase,** because the server
  cannot link the automation engine. Where the other tools run is phase 6.
- **Steps, each one green:** (A) the protocol crate with the moved types, the client core
  re-exporting them at their old paths; (B) the trait, and the walker, the pipeline, the
  investigator and the tool host going through it inside the core; (C) the server crate, as a
  move of `client/`, `forward`, `flow/`, the pipeline's server half and the server's settings,
  with the pipeline suite run against the client core's desk as a dev-dependency; (D) the
  settings file in two.

What phase 4 found on the way:

- **Neither crate depends on the other** (`cargo tree` shows it both ways). The server's tests
  use the client core as a dev-dependency: the pipeline suite runs against `LocalDesk` over
  fakes.
- **The settings folder as a whole is the app's** (`jevons-desktop/src/settings.rs` and
  `config.rs`): it sees both halves. The app keeps one value for the Settings tab and writes
  each half to its own file.
- **`privacy.log_api` is the server's,** under `[privacy]` in `jevons-server.toml`. `[models]` is
  the server's too (the embedded provider's), and filling unselected services from the
  downloaded catalog is the app's.
- **The automation author gets its model as a `Planner`** the app implements over the generation
  route, since the client core does not know the API client.
- **The guarded writer exists twice,** in the server for the flows folder's guides and in the
  client core for the automations library's: forty lines, and no crate both may depend on is
  the place for file writing.
- **Before the next Windows install, the settings file needs splitting by hand:** `[server]`,
  `[providers.*]`, `[routes]`, `[models]`, `[tools.*]`, `[mcp.*]`, `flows_dir` and `log_api`
  (under `[privacy]`) move to a new `jevons-server.toml` next to `jevons-desktop.toml`. Left in
  the client's file they are unknown fields, and the app starts on the defaults.

Crates: `jevons-desktop-server` (new), `jevons-desktop-core`, `jevons-desktop`.
Tested by: the pipeline suite moved to the server crate and run against a client-core
implementation of the boundary; the client-side safety tests (delivery, confirmation) in the
client core.
Done when: the server crate does not depend on the client core, and the app behaves as before.

## Phase 5: the desktop protocol

Scope: put a wire under the phase 4 boundary, so the desktop server can move from the app's
process to a local background process (the expected main setup) as a deployment change.

How phase 5 is cut (decided 2026-10-03):

- **A `Session` in the server first, with no transport:** one desk, the server's parts, the
  machines and their timers. The app and the headless runs go through it. It is the one surface
  a wire goes in front of. (Done.)
- **Then the messages, the two ends and the transports,** in the protocol crate: what the client
  says to the server and back, the server's desk over a stream, the client attending to it, an
  in-process transport and a WebSocket one.
- **Only takes, effects, updates, the machines' view and traces travel.** The inspector's Flows
  tab, the extract workbench, the Machines tab's diagrams and `--check-flows` keep reading the
  flows folder through the server crate: on loopback it is the same settings folder, and the app
  links the server for its fallback anyway. The workbench needs the reader and the tree at once,
  so it cannot be put on the wire in this phase.
- **Window handles are this machine's.** A delivery names the window a take started in by its
  handle, which means something only on the client's machine: fine over loopback, and for a
  client elsewhere as long as the client is the one that reads it.

Left by phase 4 for this one:

- **One desk per session.** Today the tool host holds a desk of its own, built once at startup
  (the automations library), and each take builds a second one. They share the automation host,
  so it works in one process; over a wire the host's client tools must come from the session's
  desk.
- **The server builds its own parts.** `Env.tools` and `Env.investigator` are server constructs
  the app assembles today; the server should build them from its settings and the session's
  desk.
- **Traces and the take's stream** (`Trace`, `Update`) are still the server crate's types. They
  move to the protocol when they travel.

- [x] New crate `jevons-desktop-protocol`, holding:
      - events (takes, effect results, control) and effects (deliver, confirm, read, an
        investigation's steps, run a client tool);
      - the streams (the machines' view, stages, output, traces);
      - versioned serde shapes, and an effect id that each result answers.
- [x] An in-process transport (channels carrying the Rust values).
- [x] A WebSocket transport (axum on the server, `tokio-tungstenite` on the client), with a key.
- [x] Disconnection: pending confirmations are denied, a take in flight fails, the machines keep
      their state on the server, and the client gets the path again when it reconnects.
- [x] A headless client mode (`--server <url>` with `--transcript` or `--replay`) for scripted
      checks against a running server.
- [ ] A server-only mode of the binary (no tray, no window) that the app connects to over
      loopback, started at login, with the app falling back to its in-process server when none
      answers. Done: the server-only mode (`--serve`). Not done: the tray app as a client of it.

What phase 5 found on the way:

- **The version is one number in the hello,** not on every message, and a message one end does
  not know is skipped.
- **The client's tools travel in the hello,** so the server's desk answers `tools()` without an
  effect, as the design said ("open a session with what the client can do").
- **Traces and the machines' view travel as JSON,** which the client prints or parses with the
  server crate's types; only what a take is doing while it runs (`Update`, `Stage`) moved to
  the protocol.
- **The desktop endpoint is `/desktop` on the settings' `bind:port`,** the address the exposed
  API has. `--serve` always listens there, and serves `/v1/*` on it only when the settings
  expose the API. One address and one key, in place of a second port to configure.
- **Checked across two processes on loopback,** with a fake provider: `--serve` in one, `--server`
  with `--transcript` in another, and the trace equal to the one process's.
- **The tray app is not a client of a separate server yet.** It runs its session in its own
  process, behind the same `Session` the host serves. Making it a client needs the trace and the
  machines' view parsed back into the app's types, the Machines tab drawn from a view that
  arrives, and the app starting no runtime when a server answers. Left unticked.

Crates: `jevons-desktop-protocol` (new), the server, the client, `jevons-desktop`.
Tested by: the pipeline suite run over both transports; serde round trips of every message; a
disconnect test (pending confirmation denied, state kept).
Done when: the app runs either in one process or against a server on another port, with the same
traces.

## Phase 6: where tools run

Scope: each tool runs on its side, and confirmation is enforced by the side that runs it.

- [x] Each tool runs on its side. Adapted: where a tool is written is where it runs (the
      server's settings file or the client's), in place of a `runs` key; see below.
- [x] Client tools become effects. The client checks its own settings' `confirm` and `allow`
      before running one, whatever the server asked.
- [x] A server tool that needs confirmation sends a confirm effect and waits for its answer.
- [x] `TOOLS.md` and the catalog list each tool with where it runs. A machine's `tools` list keeps
      covering both kinds.

What phase 6 decided on the way:

- **The file is the partition, not a field.** The client must check "its own settings' `confirm`
  and `allow`", so a client tool's definition has to be in the client's file, and the server
  learns of it from the hello. A `runs` key could only repeat, or contradict, the file it is in.
  Any kind may be in either file: an `http` tool in the client's file sends its request from the
  client's machine.
- **`jevons-desktop-tools` holds what both sides run:** the settings' tool types, the three
  built-in runners, MCP servers and `ToolSet`. The client core gets `adk-core` and `adk-tool`
  back through it, for MCP.
- **`run_tool` carries the calling node and whether the server asks too.** The client asks once:
  when its own settings say so, or when the node's `confirm` does. The server does not ask for a
  client tool, so nobody is asked twice. The protocol's version stays 1: nothing was released
  between.
- **A name in both files is an error,** reported by the app, `--check-flows` and at the call.
- **`--serve` reads the flow tree again when a client says hello,** since a client's tools are
  unknown before that.
- **Before the next Windows install,** the fourth step of the hand migration: `open` and
  `command` tools (the example's `open_url`) stay in, or move back to, `jevons-desktop.toml`;
  `http` tools (`web_search`) go to `jevons-server.toml`.

Crates: server, client core, protocol.
Tested by: a server `http` tool with confirmation asking through the client; a client tool still
asking when the server does not; a declined client tool taking `denied`.
Done when: the search example runs with `web_search` on the server and `open_url` on the client.

## Phase 7: persistence

Scope: tasks survive an app restart.

- [x] Serialize instances after each transition, into the data folder (not the settings
      repository). **Clear history** covers them.
- [x] Restore them on start: a waiting state waits again; a state whose work was running takes
      `failed` and is never re-run.
- [x] Record a hash of the machine's files with the instance. When they changed, end the instance
      with a note instead of resuming it.

What phase 7 decided on the way:

- **Timers start over.** A restored state's timers run their whole time again. Keeping the time
  left would end a search the moment the app starts after a long stop, which is the opposite of
  "still there after a restart".
- **A decision that was out is dropped,** and its machine waits in its state: a decision read
  acts on nothing. Only work that ran is a failure.
- **The engine keeps and restores an instance; the host answers the failure.** `restore` leaves
  a machine whose work ran waiting for how it ended, and the host feeds `failed` through the
  path every failure takes, so the rules and the trace are those of any take.
- **The hash is over what the machine runs,** as the loader read it: its node file, diagram and
  instructions, and those of its states' folders, not the machines below. An agent's change does
  not end its tasks.
- **A changed task ends unseen by its agent,** like a cancelled one. A changed root or agent
  starts over in its first state and keeps its tasks.
- **The file is written at every publish,** so before a state's work runs, and removed when
  nothing is worth keeping. It is `~/jevons/machines/machines.json`; the app gives the server
  the path, as it does for the API log.
- **When they come back:** in the app, once a provider answers; under `--serve`, once a client
  has said hello too, since the flow tree is checked against the client's tools. The first take
  does it otherwise.
- **Clearing the tasks' history ends them** in the running app. `--clear machines` from a
  terminal only removes the file.
- **Checked across two processes** with a fake provider: a search started through
  `--server … --transcript`, the server killed and started again, and the follow-up opened a
  result of that search, with one search in all. It found what the tests had not: a headless
  client of a server listed no tools, so the server's flows that name a client's tool never
  loaded. It lists them now, and runs none.
- **A write is skipped when nothing changed,** so after `--clear machines` beside a running app
  the file is back at the tasks' next change, not before.
- **Not kept:** the bubble's conversation. Not done: showing in the bubble what a `failed`
  transition taken at start delivers; it goes to the desk as a timer's work does, with no trace
  of its own.

Crates: `jevons-machine` (serde on instances), the server.
Tested by: save and restore round trips; a running state restored as failed; a changed machine
ended.
Done when: a search waiting in `results` is still there after a restart.

## Phase 8: UI follow-ups

- [ ] **Decided-by on the diagram.** Colour each edge by what decides it (event, rules, model),
      from the loader's report.
- [ ] **Answer a decision from the Machines tab.** When the model is unsure, show the candidates
      and let a click answer the `Decide`. Keep each answer, with the state and the
      probabilities, as a labelled example for tuning descriptions.
- [ ] **Cancel task** in the tray menu. Cancelling also ends a take that waits on a confirmation.
      Today cancel waits for the running take to finish.
- [ ] **Draft states under a machine** in the Flows tab: write the state's folder and append its
      transitions (`idle --> x : said`, `x --> idle`) to the parent's `.fsm`. Today only decision
      parents are offered.
- [ ] **Edge routing:** fewer crossings (more ordering sweeps, transposition), and orthogonal or
      spline edges instead of straight segments.
- [ ] **Hand unsure takes back up.** The live case: "Buscar …" said while a search waited matched
      the root's `[search]` rule, but the task had it first, was unsure and stayed. Phase 2
      already lets an agent choose between its running tasks and its own transitions. What remains: a take the chosen task is then unsure about,
      and a take its agent is unsure about, may go up a level (an opt-in per machine,
      `unsure = "parent"`; the default stays `"stay"`), so dictation keeps working while a task
      waits.

## Risks

- **Dictation latency.**
  - Every phase keeps the one-call tests. Splitting a request for a provider's question limit,
    or a remote server, adds round trips.
  - Measure take latency (API log timings) before and after phases 2 to 5, and keep the
    all-in-one mode free of serialization.
- **Calibration differs between providers.** A threshold tuned on DiffusionGemma does not carry
  over to Jev. Profiles hold per-model thresholds. Moving the decision route needs a day of
  checking routing in the traces, and a probability is never compared across models.
- **Privacy.**
  - With a remote server or provider, the context snapshot (window titles, field text, the
    selection) and extract answers leave the machine.
  - The router config makes each route's destination visible in Settings, the trace says which
    provider answered, and `log_api` stays off by default.
  - Persisted instances hold the same kind of text and stay in the data folder.
- **oxidate-fsm.**
  - Version 0.1.0 has one author and one release.
  - Vendor its grammar and model (MIT, about 900 lines) under `jevons-machine` if any of these
    happens:
    - we need a fix in its grammar (choice branches to `[*]`, quoted parameters, positions for
      semantic errors);
    - it breaks on a Rust release;
    - its dependencies get in the way.
  - Since phase 1 keeps our own model, vendoring touches only the front end.
- **Three routing levels.** Root, agent and task each choose. Keep their questions in one
  request, prefer rules where the answer never changes, and watch the traces for takes routed to
  the wrong agent or task: the descriptions of agents and of tasks' waiting states are new text
  for the model.
- **Two moving parts in one process.** Phases 4 and 5 change how the app is assembled. Keep the
  trait boundary first and the wire second, so each phase has one kind of change.

## Deferred

- **Committing headless writes:** `--transcript` and `--check-flows` write the generated flow
  files (`AGENTS.md`, `_schemas/`, `.taplo.toml`) into the settings folder without committing them;
  only the app commits its writes.
- **Upgrading an edited flows folder:** today only an unedited earlier tree is brought up to date.
  A three-way merge against the recorded earlier tree (its files are in `flow/earlier.rs` by hash,
  so the texts would need recording too) would carry the user's edits onto the new tree, and stop
  on a conflict.
- **Rules learned from decisions:** from the traces, propose the `[prefer]` rule that would decide
  a transition the model always takes when a rule-visible fact holds; you accept or reject it.
- **Agent memory:** an agent's log of everything it handled, structured recall with no model,
  full-text recall as a tool (a pure-Rust index such as tantivy), embeddings later. Out of scope
  until decided: it is a new store of your words and screen text, and needs a per-agent setting,
  a modest default for dictation and a kind under **Clear history**. See the
  [design](design.md#deferred).
