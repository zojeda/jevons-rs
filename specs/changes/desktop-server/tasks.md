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

- [ ] New crate `crates/jevons-machine` with no tokio, HTTP or `serde_json` dependency. Serde
      derives are allowed, for instances.
- [ ] Move the definition (today's `Machine`, `Event`, `Condition`, `Target`, `Choice`, `Timer`),
      the parser front end (`oxidate-fsm`) and the checks.
- [ ] Move `layout`: it depends only on the definition.
- [ ] Write `Instance::handle(def, input, facts) -> Vec<Effect>` with
      `Input { Event, Decided, Finished }` and
      `Effect { Decide, Run, Arm, Entered, Ended }`. It covers:
      - candidates;
      - rules, then `[prefer]`, then the oracle, then `[else]` or staying put;
      - choice points;
      - `done` chains, bounded at 32;
      - ending into the parent;
      - unhandled failures;
      - generations for timers.
- [ ] Add a `Facts` trait for named guards and targets' `[when]` and `[prefer]`. The host
      implements it over the frame and the target folders, as `Turn::check` does now.
- [ ] Keep `Decide` as data (the candidates with their labels and criteria) and `Decided` as a
      label and probability, or nothing.
- [ ] Value checks in guards: `when = { value = "{state.field}", … }` with `empty`, `equals`,
      `matches` and number comparisons, answered by `Facts` from the instance's values. The loader
      checks each path against the state results and the shapes of typed results, as it checks
      placeholders now.
- [ ] Typed results: a tool's result (JSON) and a structured generation keep their fields, so value
      checks and `{state.field}` placeholders can read them.
- [ ] Decided-by: the loader reports, for each state and event, whether the event alone, rules or
      the model decides its transitions; `--check-flows` prints it.
- [ ] Turn `flow/machine/runtime.rs` into a host that carries out the effects: walks, delivery,
      tokio timers, and System One with the lookahead batching.
- [ ] Engine tests with a scripted oracle and scripted facts: every rule of the selection order,
      stays, choice points, nesting, timers and stale generations, the step bound.

Crates: `jevons-machine` (new), `jevons-desktop-core`, `jevons-desktop` (imports).
Tested by: the new engine tests (value checks included: a search with no results taking
`no_results` with no model call), and every pipeline and UI test unchanged.
Done when: `runtime.rs` holds no selection logic, and the one-call tests still pass.

## Phase 2: agents and tasks

Scope: the three routing levels of the [design](design.md#agents-and-tasks):
the root dispatches to agents, an agent runs several tasks at once, and a take for an agent may go
to one of its running tasks.

- [ ] Rename the tool-loop node `agent.toml` to `loop.toml` (`Kind::Loop`), so "agent" names only
      the new level. Update the schemas, `AGENTS.md`, the docs and the tests; an unedited built-in
      folder is brought up to date through `flow/earlier.rs`.
- [ ] Node files per level: an agent is `agent.toml` and `agent.fsm` (the root's `root.toml` and
      `root.fsm` and a task's `task.toml` and `task.fsm` landed in phase 0). The loader checks
      that agents sit under the root and tasks under agents, and that each agent's scope covers its
      tasks' `tools`.
- [ ] The instance forest in the host: the root, one instance per agent (created when the tree
      loads, kept across takes, rebuilt when the tree changes and the agent is at rest), and each
      agent's tasks. Every step, effect, timer and trace record carries its instance.
- [ ] Spawning: entering an agent's state whose folder is a task machine starts the task and is
      `done` at once. Entering the root's state for an agent hands it the take and returns the root
      to `idle`.
- [ ] Dispatch to running tasks: when an agent gets `said`, add each of its running tasks that
      waits for `said` to the candidates, described by the task's current state. A chosen task
      gets the take as its own `said`.
- [ ] Task results: `task_done` and `task_failed` events for the agent, with the task's name and
      last result as values (`{task.name}`, `{task.result}`).
- [ ] One request per take: batch the root's question, the agent's (with its running tasks) and
      the chosen work's first decision, within the question limit.
- [ ] The built-in tree becomes three agents with inline work: `dictation` (today's `dictate`),
      `assistant` (today's `ask`) and `automations` (today's `run`). The search example becomes a
      task of a `research` agent. Record the spike's tree in `flow/earlier.rs` so an unedited
      folder is brought up to date.
- [ ] The Machines tab: the agents, each with its running tasks; any instance's diagram; Cancel
      per task and for all.

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

- [ ] Settings:
      - `[providers.<name>]`: the kind (embedded, jevons, openrouter, openai-compatible), base
        URL and key (`${env:NAME}`);
      - `[routes]`: speech, realtime, decision and generation, each a provider and a model.
      The single runtime mode (embedded or remote) goes away.
- [ ] A client per route in place of one `Client`. Realtime falls back to uploads per route, as
      now.
- [ ] Provider profiles:
      - extensions supported (`steps`, `samples`, `think`, `sequential`, images);
      - the most questions per request;
      - label rules;
      - the decision model's `min_probability`.

      Built-in profiles for the embedded engine, remote jevons, and OpenRouter's Jev
      (`typesafe/jev-latest`).
- [ ] With an external System One:
      - drop unsupported extensions and add a trace note ("steps dropped: … does not support
        it");
      - split requests that ask more questions than the provider takes, noting the extra calls.
- [ ] Let the decision model's profile set `min_probability`. Remove `min_probability = 0.7` from
      the built-in root; a node or machine that sets it still overrides the profile.
- [ ] Forward `/v1/*`: a pass-through by model name, with the provider's key added and `log_api`
      applied. It listens where the exposed API listens now.

Crates: `jevons-desktop-core` (client, config, pipeline), `jevons-desktop` (settings UI, runtime).
Tested by:
- fake providers (axum, as the pipeline tests do) asserting the dropped fields, the notes, the
  split requests and the routing;
- one manual headless run against OpenRouter (`--transcript` with a decision route to Jev).

Done when: dictation runs with decisions on Jev and generation elsewhere, and its traces say what
was dropped.

## Phase 4: split desktop-core into server and client

Scope: move the desktop server's parts out, behind a Rust trait boundary, and keep everything in
one process.

- [ ] New crate `jevons-desktop-server`, holding:
      - `flow/` (tree, walker, extract and investigate orchestration, the investigator, agents,
        `JevonsLlm`);
      - the machine host;
      - the server tools in the tool host;
      - the router and providers;
      - the server's settings.
- [ ] `jevons-desktop-core` keeps the client:
      - platform traits, context, XPath, delivery and paste safety;
      - gestures, recording, the automation engine and library;
      - client tools, icons, the catalog and downloads.
- [ ] Define the boundary as a trait the server calls: deliver, show, confirm, read the screen
      (a batch of extracts, an investigator step), open, run a client tool. The client core
      implements it in-process.
- [ ] Split the pipeline:
      - the client captures (audio, context) and delivers;
      - the server transcribes through its speech route, moves the machines, walks and decides.
- [ ] Split the settings file in two (client: hotkeys, microphone, privacy, client tools,
      automations; server: providers, routes, flows folder, server tools), in the same settings
      folder and git repository in the all-in-one mode.

Crates: `jevons-desktop-server` (new), `jevons-desktop-core`, `jevons-desktop`.
Tested by: the pipeline suite moved to the server crate and run against a client-core
implementation of the boundary; the client-side safety tests (delivery, confirmation) in the
client core.
Done when: the server crate does not depend on the client core, and the app behaves as before.

## Phase 5: the desktop protocol

Scope: put a wire under the phase 4 boundary, so the desktop server can move from the app's
process to a local background process (the expected main setup) as a deployment change.

- [ ] New crate `jevons-desktop-protocol`, holding:
      - events (takes, effect results, control) and effects (deliver, show, confirm, read, open,
        run a client tool);
      - the streams (path and history, stages, output, traces);
      - versioned serde shapes, and an effect id that each result answers.
- [ ] An in-process transport (channels carrying the Rust values) for the all-in-one app.
- [ ] A WebSocket transport (axum on the server, `tokio-tungstenite` on the client), with a key.
- [ ] Disconnection: pending confirmations are denied, a take in flight fails, the machines keep
      their state on the server, and the client gets the path again when it reconnects.
- [ ] A headless client mode (`--server <url>` with `--transcript`) for scripted checks against a
      running server.
- [ ] A server-only mode of the binary (no tray, no window) that the app connects to over
      loopback, started at login, with the app falling back to its in-process server when none
      answers.

Crates: `jevons-desktop-protocol` (new), the server, the client, `jevons-desktop`.
Tested by: the pipeline suite run over both transports; serde round trips of every message; a
disconnect test (pending confirmation denied, state kept).
Done when: the app runs either in one process or against a server on another port, with the same
traces.

## Phase 6: where tools run

Scope: each tool runs on its side, and confirmation is enforced by the side that runs it.

- [ ] Add `runs = "server" | "client"` to tools and MCP servers. Defaults: `http` on the server;
      `open` and `command` on the client; MCP where it is configured; automations always on the
      client.
- [ ] Client tools become effects. The client checks its own settings' `confirm` and `allow`
      before running one, whatever the server asked.
- [ ] A server tool that needs confirmation sends a confirm effect and waits for its answer.
- [ ] `TOOLS.md` and the catalog list each tool with where it runs. A machine's `tools` list keeps
      covering both kinds.

Crates: server, client core, protocol.
Tested by: a server `http` tool with confirmation asking through the client; a client tool still
asking when the server does not; a declined client tool taking `denied`.
Done when: the search example runs with `web_search` on the server and `open_url` on the client.

## Phase 7: persistence

Scope: tasks survive an app restart.

- [ ] Serialize instances after each transition, into the data folder (not the settings
      repository). **Clear history** covers them.
- [ ] Restore them on start: a waiting state waits again; a state whose work was running takes
      `failed` and is never re-run.
- [ ] Record a hash of the machine's files with the instance. When they changed, end the instance
      with a note instead of resuming it.

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
