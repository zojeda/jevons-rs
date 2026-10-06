# jevons-desktop: app

[Back to jevons-desktop](spec.md)

The app lives in the tray. The agent thread turns hotkeys and menu items into takes, recordings
and automation runs, asks the user in the bubble before anything risky, and keeps the settings
folder, the flow tree and the automations library in step with the files on disk. What a take does
once it starts is in jevons-desktop-core's [pipeline](../jevons-desktop-server/pipeline.md); this
file covers what the app does around it. The window and the bubble are in [ui](ui.md).

## Requirements

### R1 The tray icon shows what the app is doing

The icon starts as loading and then follows the runtime: ready (or using a server), loading, no
models, or failed. While a take listens it shows the microphone's level, then transcribing, then
deciding and writing, and red after a failed take. Between takes during a recording it is the
recording icon. Tuning GPU kernels outranks every state but listening, from a tuning event until
four seconds after the last one. The first tuning event after a quiet period wakes the tray once,
so the icon turns amber without waiting for its next frame.

Tests: `a_tuning_event_makes_tuning_active_and_notifies_once`

### R2 The tray menu

The menu holds: **Start dictation** or **Stop dictation**, **Start live dictation** or **Stop
live dictation**, **Cancel the current take** (while one runs), **Cancel the tasks that run**
(while one does), **Show the task's conversation** (while a task waits with one), **Machines**
(R28), **Start takes at** (the root or
each top-level branch), **Show context inspector**, **Live feedback**, **Pause context capture**,
**Record an automation…** or **Stop recording**, **Automations** (per automation: its
description, **Run** and **Run step by step** when approved or **Review and approve…** when not,
and **Record it again…**; **None yet: record one** when empty; **Open the automations folder**),
**Reload the flow tree**, **Open settings folder**, **Reset settings to the defaults…**, **Open
logs and traces**, **Clear history** (**Logs…**, **Take traces…**, **Recorded interfaces…**,
**Recordings…**, **Tasks that run…**, **All of it…**) and **Quit**. Reset and Clear history are disabled while a take,
an automation or a recording runs.

Tests: `menu_ids_map_to_commands`, `the_machines_menu_names_each_machine_s_state_and_counts_the_tasks`

### R3 A left click on the tray icon toggles dictation

A left click starts a push-to-talk take, or stops the one the menu started.

Tests: none yet

### R4 Without a tray, the window opens

When the tray icon cannot be created, the app runs without it and shows the inspector window at
start.

Tests: none yet

### R5 Hotkeys register from the settings

The app registers every hotkey the settings bind ([pipeline](../jevons-desktop-server/pipeline.md)
R18), again whenever they change. A hotkey assigned twice, one that cannot be registered, and one
that does not parse are reported in the window's status bar, and the others work.

Tests: none yet

### R6 Hold and toggle gestures

In `hold` mode, pressing a dictation hotkey starts a take and releasing it ends the listening. In
`toggle` mode, a press starts the take and the next press of the same hotkey ends it.
Push-to-talk and the branch hotkeys follow `hotkey_mode`, live dictation `live_hotkey_mode`, and
an automation's hotkey `hotkey_mode`. While a take runs, key repeats and other dictation hotkeys
change nothing.

Tests: none yet

### R7 Takes from the menu

**Start dictation** starts a push-to-talk take that the menu (or a left click) stops. **Start live
dictation** starts live dictation that the menu stops. Each says so instead when the other kind
runs, or when the last take is still being processed.

Tests: none yet

### R8 A take reads the context before anything else

A take reads the context at the press, before the microphone opens, so the text goes to the field
the user started in. It starts at its hotkey's branch, else at the branch **Start takes at**
names, else at the root. With **Pause context capture** on, the context is empty and says capture
is paused. Without a runtime to reach, the take does not start, the window says why and the icon
turns red; a microphone that cannot open does the same.

Tests: none yet

### R9 The bubble follows each take

The bubble's first line says how to finish (release the hotkey, press it again, or stop from the
tray). It then shows the words as heard, each stage as it runs and how it ended, and the text
written. At the end it says "Inserted", "On the clipboard: <reason>", "Answer", "Too short to hold
speech", "Done", or the error; a take that left a task where it was, unsure of the words, says so
("<Task> stayed at <state>: …"). Text that only repeats the transcript is not shown twice, an
answer always is, and stages still open close with the take's outcome. The bubble hides four
seconds after a take ends, eight after an error, and an answer stays until it is closed or the
next take starts.

Tests: `feedback_shows_phrases_as_heard_and_the_outcome_at_the_end`, `an_answer_stays_in_the_bubble_even_when_it_repeats_the_words`, `push_to_talk_feedback_keeps_the_streamed_words_as_the_transcript`, `a_waiting_task_s_turns_stay_in_the_bubble_for_the_next_take`

### R10 Questions wait in the bubble

A tool call that asks shows "Run <tool>?" with its arguments, and the app's own questions (reset,
clear, approve) say what they do. Enter answers yes and Esc no; the app registers both as global
hotkeys only while a question waits. A new question answers the one before it no. While a
question waits, the app asks no other question of its own and says to answer the first. Cancelling
or finishing a take answers its question no.

Tests: none yet

### R11 Cancel the current take

**Cancel the current take** stops the microphone, abandons the take, answers a waiting question
no, stops a running automation, and delivers nothing.

Tests: none yet

### R12 Finished takes are kept

Each finished take's trace is written to `~/jevons/traces/<start ms>-take<n>.json` (with
`-turn<t>` for a live turn), and the folder keeps the newest 200. The Takes tab keeps the newest
50. A take that leaves its text on the clipboard says why in the window.

Tests: none yet

### R13 Recording from the tray and the record hotkey

**Record an automation…** or the record hotkey starts a recording, unless a take runs, and the
icon turns to recording. While the record hotkey is held for half a second or more, the app
listens: the first thing said is the task's description, and later ones are notes. A shorter tap
during a recording stops it.
Text a push-to-talk take types during the recording becomes a step. A recording with no steps is
not saved. A saved one is written into an automation by the author, with `author_model` or the
generative model when the runtime has one, and then goes to review.

Tests: none yet

### R14 Approval happens in the bubble

Reviewing an automation runs its checks. One with problems cannot be approved, and the bubble
names its first problems. Otherwise the bubble asks "Approve <name>?" with its applications, what
it does, how many recorded steps it replays, and its version. Enter pins that version in the
settings; Esc keeps it a draft, which the tray offers as **Review and approve…**. While a take
waits on a question, the app says to approve later from the tray.

Tests: none yet

### R15 Running an automation from the tray or its hotkey

An automation not approved goes to review instead. One that takes required arguments listens
first: the user says them, and a one-node `run.toml` tree fills them. One without runs at once.
A run asks "Run script:<name>?" first unless the settings list it as unconfirmed, and a no stops
it as not confirmed. Its steps show in the bubble, and the bubble then says "<name> done." with
its answer, or "<name> failed:" with the message and `script.rhai:<line>:<column>`. The run's
trace goes to `~/jevons/traces/<ms>-automation-<name>.json`. **Run step by step** asks before
every action. Nothing runs while a take or another automation does.

Tests: none yet

### R16 Record it again

**Record it again…** records the task anew and replaces the automation with the new version,
which needs approving again.

Tests: none yet

### R17 Saved settings apply without a restart

Saving from the Settings tab writes the file and applies it: hotkeys are registered again when
they changed, a moved flows folder is watched and loaded, the tools and MCP servers start again
when the tools, the flows folder or the library changed, the runtime takes the new settings, and
the API log starts or stops. **Live feedback** in the menu saves the setting as well.

Tests: none yet

### R18 jevons commits its own writes

In a settings folder jevons versions, the app commits each write it makes there, and only the
files written: saving the Settings tab, the **Live feedback** toggle, approving an automation,
writing an automation from a recording, saving an extract from the Context tab, `TOOLS.md`, and
the flows folder's and library's guides when they are refreshed.

Tests: none yet

### R19 Reset and clear ask first

**Reset settings to the defaults…** asks in the bubble, naming the folder and what it holds. Yes
cancels the take, resets the folder, runs with the defaults without a restart, and names the
commit that keeps the earlier settings. **Clear history** asks, naming each folder; yes clears
them and reports what was removed. Clearing the traces also empties the Takes tab, and clearing
the tasks that run ends them, as **Cancel all tasks** does. No changes nothing.

Tests: none yet

### R20 The flows folder and the library reload on change

The app watches the flows folder and the automations library. A burst of changes reloads once,
a quarter second after the first. A flow tree with problems is reported, and the last tree that
loaded without problems keeps running, or the built-in tree when none has since the start. A
library change reloads the automations, rewrites `TOOLS.md`, and checks the flow tree against the
new tools.

Tests: none yet

### R21 MCP servers start in the background

At start, and whenever the tools change, the app starts the MCP servers without holding up takes,
writes `TOOLS.md` when it changed, reports each server that failed in the Flows tab, and checks
the flow tree against the tools they listed.

Tests: none yet

### R22 Machine timers run as takes of their own

When a task's timer runs out, its event moves the machines as a take of its own, whose work
delivers into the window the task started in. With no take running, the bubble shows the timer.
A timer whose state the machine has left does nothing and shows nothing. An answer to an unsure
decision, given in the Machines tab, runs the same way: a take of its own, whose bubble says
which candidate was chosen. In the Machines tab,
**Cancel all tasks** ends every task and a task's own **Cancel** ends that one; when it is the
task the bubble follows, its conversation goes with it.

Tests: `a_timer_ends_a_task_that_waits_and_stale_timers_do_nothing`

### R23 The app logs to a file

The tray app has no console on Windows. It logs to `~/jevons/logs/jevons-desktop.log`, appending,
and keeps the run before as `jevons-desktop.previous.log`. A panic on any thread is logged with
its backtrace. `RUST_LOG` sets the detail.

Tests: none yet

### R24 Quit ends everything

**Quit** cancels the take, removes the tray icon, unloads the models and closes the window.
Closing the window only hides it.

Tests: none yet

### R25 A waiting task's bubble is its conversation

While the task the latest take reached waits for what the user says next, its bubble stays after
each take and timer, with no time limit. The next take's bubble starts with that take alone:
what is being said may be dictation, and earlier turns above it would read as part of it. Once
the words turn out to be for the task (the server's `task` update), the earlier turns go above
the take, each with what was said and what the task answered or how the turn ended. **Copy** and **Insert** take
the latest text the conversation holds. **Close** hides it, and **Show the task's conversation**
in the tray menu brings it back while the task waits and no take runs. When the task ends, the
conversation ends with it: the take that ended it shows its own outcome and no earlier turn, the
bubble follows R9 again, and a conversation left resting in it goes. A take that goes elsewhere
while the task waits (dictation, another agent) is no part of the conversation: its bubble
follows R9 and never shows the task's turns, and the conversation stays as it was for the next
take that reaches the task. A take that reaches another
task starts that task's conversation.

Tests: `a_waiting_task_s_turns_stay_in_the_bubble_for_the_next_take`, `menu_ids_map_to_commands`

### R26 Tasks are there after a restart

The app keeps the tasks that run in `machines/machines.json` of the data folder (`~/jevons`), and
brings them back once a provider answers after it starts
([machines](../jevons-desktop-server/machines.md) R53 to R57): a search that waited for what
the user says next waits again, and the Machines tab shows it. What does not go on (work that
ran when the app stopped, a machine whose files changed) is logged, by machine and state alone,
and shows in the Machines tab's transitions. The conversation in the bubble is not kept: the
next take that reaches the task starts one.

Tests: none yet

### R27 Cancel the tasks that run

**Cancel the tasks that run** in the tray menu ends every task and puts the root and the agents
back in their first states, as **Cancel all tasks** in the Machines tab does, and ends the
conversation in the bubble. A take that waits on a question holds the machines until it is
answered: the cancel answers it no, so that take ends as its machine handles a declined call,
and the tasks end after it. A task's own **Cancel** does the same for a question of a take that
reached that task. A take that does anything else is waited for. The menu item is enabled while
a task runs or a machine is away from its first state.

Tests: `menu_ids_map_to_commands`

### R28 The tray menu lists the machines that run

The **Machines** submenu lists what runs: the root, then each agent followed by the tasks it
started, each line with its name (`/`, the agent's folder, `search-2`) and the state it is in.
Its title says how many tasks run ("Machines · 2 tasks run"), and before anything has run it
holds **Nothing runs yet**. A click on the root or an agent opens the window on the Machines tab
with that machine's diagram ([ui](ui.md) R18). A task opens to what it waits for ("Waits for
said, timeout", or "Working" in a state that waits for nothing), **Show in the Machines tab**,
which does the same for the task and, when it is the task the bubble follows and no take runs,
brings its conversation back (R25), and **Cancel**, which ends that task as its **Cancel** in
the Machines tab does (R22, R27). The menu follows the machines as they move: when a take starts
or reaches a task, at each state a task enters, when a take or a timer ends, on a cancel, and
when the tasks come back after a restart (R26). Showing a machine changes nothing about where
the next take goes.

Tests: `the_tray_menu_lists_each_agent_with_its_tasks_under_it`, `the_machines_menu_names_each_machine_s_state_and_counts_the_tasks`, `menu_ids_map_to_commands`, `a_machine_picked_in_the_tray_shows_in_the_machines_tab`
