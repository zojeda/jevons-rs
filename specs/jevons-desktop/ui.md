# jevons-desktop: ui

[Back to jevons-desktop](spec.md)

The inspector is one window on dioxus-native (Blitz), styled after Dioxus Components, with six
tabs: Context, Takes, Flows, Machines, Settings and Models. The feedback bubble is a small window
by the tray icon that follows each take. Both render from the view the agent writes. The flow
tree, extracts, the interface browser and the machines they show are specified in
jevons-desktop-core.

## Requirements

### R1 The window has six tabs and a status bar

The top bar shows the runtime's state as a badge (one that leads to the Models tab when there are
no models or they failed), a "Tuning GPU kernels" badge while tuning runs, and the tabs. The
status bar shows the runtime's description, the tray's state, the push-to-talk and live hotkeys
with "hold" or "press", hotkey errors and the latest notice. While a take runs, the words heard
and the text written show above it. Every tab rebuilds when another replaces it, and the tabs take
clicks however far the page is scrolled.

Tests: `switching_between_every_page_rebuilds_in_blitz`, `the_tabs_take_clicks_while_the_page_is_scrolled`

### R2 Closing the window hides it

Closing the inspector hides it, and the app goes on in the tray. **Show context inspector** in the
tray menu, or the inspector hotkey, shows it again.

Tests: none yet

### R3 The Context tab shows the window in front

The Context tab shows what the accessibility layer reports for the window in front: application,
window, address, the focused element with its role, name, value, selection and text around the
caret, as fields, as a tree, or as raw JSON. The agent reads it twice a second while the tab shows
and the window is on screen, and ignores the inspector's own window. **Capture in 3 s** reads the
window in front three seconds later, and **Freeze** keeps the tab on the captured window.

Tests: `routes_changing_as_the_focused_app_changes_rebuild_in_blitz`

### R4 The Route card walks the tree by rules alone

The Route card shows where a take from that window goes by guards and rules alone, stopping at the
first decision the model would make. Each decision's branches sit on a guide line, marked passed or
failed, with the chosen one highlighted and the next decision nested under it. A branch's chevron
unfolds its rule checks: each pattern, the value compared, and whether it passed.

Tests: `a_route_nests_each_decision_under_the_branch_it_took`, `routes_changing_as_the_focused_app_changes_rebuild_in_blitz`

### R5 The tab reads the flow tree's extracts in its window

**Read by the flow tree** reads every `[extract]` that applies to the tab's window, lazy ones
included, and shows each answer with its expression, how many elements it matched and how long it
took. It reads again when the window or its title changes, or on **Read again**, and **Edit**
opens an extract in the workbench.

Tests: none yet

### R6 The extract workbench tries an expression as a take reads it

The **Extract workbench** card edits any `[extract]` of the tree, or a **New expression**: its
expression, its answer type and a table's columns (`column = expression`, one per line). A trial
checks it as the node file would, then reads it in the tab's window with a take's permissions,
`app` filter and `$variables`, and shows the answer, the elements matched and the time. With
**Live** on, each edit is tried after a pause in typing and again when the window or its title
changes. **Copy as TOML** copies the expression as an `[extract.<name>]` table.

Tests: `the_extract_workbench_loads_the_chosen_extract_and_shows_a_trial_in_blitz`, `table_columns_round_trip_through_the_editor`, `a_trial_reads_an_edited_expression_as_a_take_would`

### R7 Saving an extract writes its node file

**Save to <file>** writes the expression, answer type and columns back into the extract's node
file, keeping the file's comments, only when the tree still loads with the change. The tree then
reloads, and the save is committed.

Tests: `saving_an_extract_keeps_the_file_s_comments_and_refuses_a_broken_tree`

### R8 The Interface card browses the tab's window

The Interface card shows the accessibility tree of the tab's window and of no other. Each row
shows the element's role, name, value (never a password field's), first classes, automation id,
and whether it is offscreen or disabled. Its chevron opens and closes it, and its label selects
it. A level lists its first 200 children, and a **+** reads the next 200. Levels already read are
not read again.

Tests: `the_browser_opens_the_snapshot_s_window_a_level_at_a_time`, `levels_already_read_are_not_read_again_when_opening_below`, `levels_cap_their_children_and_keep_no_password_text`, `interface_rows_toggle_by_their_chevron_and_select_by_their_label_with_real_clicks`

### R9 Opening many levels at once

**Expand to level** 1 to 5 opens the tree from the window down to that level and closes what lies
deeper. **Collapse all** closes everything. **Open all below** reads everything a few levels under
the selected element, or the window. Each reads at most 1,500 elements.

Tests: none yet

### R10 The search box searches the whole window

The search box looks through the whole window, not only what is open, for a text in any element's
name, value, class, automation id or role, read as jevons-desktop-core's
[extract](../jevons-desktop-core/extract.md) describes. It reads at most 5,000 elements and says
when it stopped early or which parts it could not read. It lists the first 100 matches, highlights
them in the tree, and shows where the first ten sit.

Tests: `a_search_finds_text_anywhere_in_the_window_with_the_way_down_to_it`, `a_part_of_the_window_that_cannot_be_read_is_reported_not_skipped`

### R11 Revealing an element opens the way down to it

Choosing a match opens the tree down to it, selects it and scrolls it into view. **Show focused**
does the same for the element that had the focus when the context was captured, and says so when
the snapshot has none.

Tests: `an_accented_query_finds_a_direct_message_and_reveals_it_by_walking_up`, `revealing_reads_only_the_levels_the_tree_lacks_and_the_way_up_matches`

### R12 Selecting an element lists the expressions that select it alone

A selected element shows its properties and the expressions that select it alone, most robust
first: by automation id, by class, below a stable ancestor, by its text, by name, by position.
Each is checked against the window as it is now. **Try in workbench** loads one into the workbench
as a new expression and tries it, and **Copy** copies it. An element that went away asks for
**Reload**.

Tests: `selectors_find_the_element_alone_and_a_gone_one_asks_to_reload`, `the_interface_browser_shows_the_opened_tree_and_hands_a_selector_to_the_workbench`

### R13 Record tree saves the window's interface

**Record tree** saves the interface of the tab's window to `~/jevons/trees/<ms>-<app>.json`, 40
levels deep and at most 5,000 elements, and says where.

Tests: none yet

### R14 The Takes tab shows recent takes step by step

The Takes tab lists the newest takes first, at most 50. Each shows its transcript, its route drawn
as the Route card draws it with the model's probability for each branch it asked about, the leaf,
the output, the delivery, the tool calls and the timings, and **Copy trace as JSON** copies the
whole trace. New takes arriving while the tab shows are added on top.

Tests: `new_takes_arriving_while_the_takes_page_shows_rebuild_in_blitz`

### R15 The Flows tab draws the tree

The Flows tab draws the flow tree with each branch indented under the decision that chooses it:
each node's kind, what it does (who chooses and the fallback, or where its output goes), its guard
and `[prefer]` rules and its priority. Shared branches say so under each decision that uses them,
and the route of the Context tab's window is marked. Branches fold and unfold, one by one or all
at once. Selecting a node shows its rules in full and its file, with **Open file**.

Tests: `the_flow_tree_nests_branches_folds_them_and_shows_a_node_in_full`

### R16 The Flows tab reports the folder's problems

Problems of the flows folder show with their file and line, with a line saying which tree runs
meanwhile: the last one that loaded without problems, or the built-in one. The notes from
preparing the folder and the MCP servers that failed show too. **Reload** reloads the tree and
**Open folder** opens it.

Tests: none yet

### R17 A new branch drafts its guard from the context

**New branch from the current context** writes a folder under the decision chosen, named in
lowercase letters, digits, `-` and `_`. Its node file has the window's title as its description
and a guard that matches the application, page and field the Context tab shows, with the exact
window title as a commented-out rule. Over a tree with shared actions it is a `decide.toml` that
uses them; otherwise a `generate.toml` for the application. An existing folder is refused. The
tree then reloads, and the folder opens.

Tests: `a_drafted_guard_matches_the_context_it_came_from`

### R18 The Machines tab draws each machine and where it is

The Machines tab shows where the machines are (`task › state`, or "not started") and **Cancel
task**, disabled while the root rests in its first state. It draws a machine's diagram, the
innermost running one unless another is picked: states as boxes where the layout puts them,
start and end dots, choice diamonds, and an arrowhead and label per edge. The current state is
marked and the latest transition's label lit. Each state shows its work in a few words (such as
`tool · web_search`, `agent`, `run`, `decide`) or `waits`. A tree whose root is a decision says
it has no machines.

Tests: `the_machines_page_draws_the_built_in_root_machine`, `a_nested_task_shows_its_current_state_and_a_state_in_full`

### R19 A state in full and the latest transitions

Selecting a state shows it in full: its description, its folder (or that it has none and waits
for an event), its transitions with their guards, and whether the machine is there now. The
Transitions card lists the latest 40 moves first, each with what moved it.

Tests: `a_nested_task_shows_its_current_state_and_a_state_in_full`

### R20 Settings are edited, then applied together

The Settings tab edits the runtime (embedded or a server, exposing the API with its address, port
and key, the server's address and key, and a button that copies the API's base URL), dictation (the hotkeys and their hold or toggle mode, live
feedback, the inspector hotkey, the microphone, the language, whether to ask the decision model,
the most tokens a generation writes), a push-to-talk hotkey per top-level branch, the record
hotkey and one per automation, and privacy (characters per field, the clipboard, the API log). A
hotkey field records the combination pressed, as an accelerator the global hotkeys accept.

Tests: `recorded_combinations_parse_as_global_hotkeys`, `edited_settings_mark_the_page_and_save_from_a_bar_that_shows_only_then`

### R21 Edits mark the page and save from a bar

A change marks the page with a bar on its left and the tab with a dot, and shows a save bar under
the page wherever it is scrolled: **Apply and save** writes every change at once, and **Revert**
drops them. Edits survive visits to other tabs. A typed field out of range blocks the save and
says which: the port (1 to 65535), max output tokens (16 to 8192), characters per field (100 to
20,000), or an address that is not an IP address.

Tests: `edited_settings_mark_the_page_and_save_from_a_bar_that_shows_only_then`

### R22 Saving applies only the edits

A save applies the fields edited onto the settings as they are saved now, so what changed
meanwhile elsewhere (the tray's **Live feedback**, an approval, a branch hotkey) stays.

Tests: `saving_applies_only_the_edits_onto_settings_changed_meanwhile`

### R23 The Models tab manages the models

The Models tab shows the models folder (**Choose folder…** moves it), the model each service uses
(**GGUF file…** or **Checkpoint folder…** points at one already on disk without copying it, and
**Automatic** takes the catalog's), whether speech serves Realtime, and the approximate memory of
the selected models. A `runtime_config` in use says the selections are ignored.

Tests: none yet

### R24 The catalog downloads on request

Each catalog entry offers **Download**, or **Resume download** for a partial one, with its
progress and **Cancel** while it runs. A downloaded entry offers **Use for <service>** and
**Delete**, which asks first and removes its folder. Downloads go on while another tab shows, and
a finished one makes the runtime load the new model without a restart. **Add a Hugging Face
model** writes an entry into `models.toml`: a repository, a revision, file globs, a model file,
and whether it is a speech model.

Tests: none yet

### R25 The bubble sits by the tray icon and keeps out of the way

The bubble opens above the tray icon (below it for a taskbar at the top, at the bottom right of
the screen when the icon's place is unknown), stays on screen, and stays on top. It never takes
the focus and has no taskbar entry. It lets clicks through to the application underneath, except
while it asks a question or shows an answer or a task's conversation. It shows while a take runs
when live feedback is on, and always for a question, an answer, a task's conversation or a message
of the app's own.

Tests: none yet

### R26 The bubble shows the take as it happens

The bubble shows a hint to speak, then the words as heard and the phrase being spoken, the text
written, and the last three stages: a running one with moving dots, a finished one with a check
or a cross, and a decision's branches with the chosen one and its probability. While a task runs,
it shows where the task is. A question shows its text, its details, and "<action> (Enter)" and
"Cancel (Esc)".

Tests: `the_feedback_bubble_follows_a_live_take_to_its_end_in_blitz`

### R27 Answers render as Markdown

An answer opens a larger bubble that renders its Markdown: headings, bold and italics, lists and
task lists, quotes, inline code, code blocks, rules and tables. Raw HTML shows as text, and links
as styled text that goes nowhere. A half-streamed answer renders too, with an unclosed `**` left
as written until its end arrives. The answer scrolls with the wheel while it streams and after
(R31).

Tests: `answers_parse_into_headings_emphasis_lists_code_and_tables`, `a_half_streamed_answer_and_raw_html_stay_text`, `answers_keep_the_spaces_around_bold_text_and_list_items_flow_as_one_line`

### R28 An answer can be copied, selected or inserted

Once complete, an answer offers **Copy** (plain text, the markup dropped and the layout kept),
**Copy raw** (the Markdown as written), **Select text** (the Markdown in a text field to select
and copy from, with **Done selecting** to go back), **Insert** (into the window the take started
in, with the usual delivery checks, once the bubble has closed) and **Close**. Copying leaves the
bubble open.

Tests: `an_answer_turns_into_selectable_text_and_copies_as_plain_text_or_markdown`, `plain_text_drops_the_markup_and_keeps_the_layout`

### R29 Text fields show their caret on the dark theme

A focused text field paints its caret in the field's text colour, in the middle of the field.

Tests: `the_text_caret_is_painted_in_the_field_s_text_colour_on_the_dark_theme`

### R30 A task's conversation shows its turns

A task's conversation opens the larger bubble. Each earlier turn shows what was said and, under
it, the answer as Markdown, or else the text written or how the turn ended. The latest turn
follows: the words as heard, its stages while it runs, then its answer or outcome. Once the turn
is done the bubble offers **Close**, and an answer's other buttons (R28) when the conversation
holds text; **Select text** shows the latest text.

Tests: `a_waiting_task_s_bubble_shows_its_turns_and_goes_back_to_the_newest`

### R31 The larger bubble follows its newest text

An answer or a conversation stays scrolled to its end as text arrives, and each new take, each
new bubble and the button below start there. Scrolling up with the wheel stops that, and a button
over the text's corner then goes back to the end at once. When more text has arrived below, the
button says "New": it blinks while the take still runs and stays lit after. Scrolling back to the
end hides the button and follows the newest text again.

Tests: `a_waiting_task_s_bubble_shows_its_turns_and_goes_back_to_the_newest`
