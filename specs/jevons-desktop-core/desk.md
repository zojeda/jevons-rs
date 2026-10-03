# jevons-desktop-core: desk

[Back to jevons-desktop-core](spec.md)

The desk is the client's side of the boundary with the server
([jevons-desktop-protocol](../jevons-desktop-protocol/spec.md) R2): `LocalDesk` carries out what the
server asks with this machine's platform layers, in the app's own process. The client's safety
rules live here, whatever the server asks for: text goes only into the window a take started in,
a tool runs only once the user said yes, the screen is read within the privacy settings, and an
automation runs only in the version the user approved.

## Requirements

### R1 Each part of the desk is optional

A desk is built from the parts at hand: a text sink, an extract reader, the investigations'
`Looks`, a confirmer and the automations library. Without a sink nothing is delivered. Without a
reader an extract is empty in its shape, with "No interface reader is available here". Without
`Looks` no investigation opens ("No context investigator is available here"). Without a confirmer
every confirmation is denied. Without the library it offers no tools and serves no `script`.

Tests: `a_declined_tool_call_takes_the_denied_transition`, `each_capability_goes_to_its_own_provider`, `a_desk_that_cannot_look_answers_empty_with_its_note`

### R2 Text goes only into the window the take started in

`deliver` types or pastes once every key is released, and only while the window the take started
in is the foreground one. When the window changed or keys stayed down, the text is left on the
clipboard with the reason; a request for the clipboard only copies. A sink that fails leaves the
text on the clipboard too. The rules are those of
[pipeline](../jevons-desktop-server/pipeline.md), applied here.

Tests: `paste_is_skipped_when_foreground_window_changed`, `delivery_waits_for_held_keys_until_the_deadline`, `a_newer_take_cancels_the_delivery`, `a_changed_window_leaves_the_text_on_the_clipboard`

### R3 Confirmation waits for the user, and no answer is no

`confirm` sends the call to whoever shows it (the bubble) and waits 60 seconds for the answer. An
unanswered call, or one with nobody to show it, is denied.

Tests: `a_call_runs_only_when_approved_in_time`

### R4 An extract is checked again and read off the async workers

`read` compiles the extract as the flow loader did, then evaluates it on a blocking thread, since
accessibility calls block. One that does not compile is empty with its errors as the note.

Tests: `each_kind_of_answer_fits_its_shape`, `variables_bind_values_and_other_apps_need_permission`

### R5 An investigation's elements and remembered paths stay with the client

`look` opens an investigation over the windows the take may read
([investigator](../jevons-desktop-server/investigator.md)), and reads at once what a path
remembered for the look's key still leads to. `look_step` runs one navigation tool (`outline`,
`find`, `xpath`, `read`, `list_windows`) against the elements this investigation has seen, and
answers with its text, the ids and roles that may be named next, and what it did since the last
step. A step for an investigation that is over, or with a tool there is not, says so in its text.
`look_end` forgets the investigation and, when asked to remember, keeps the path to the last
element read under the look's key and returns it.

Tests: `an_investigation_reports_each_step_and_remembers_the_path_to_its_answer`, `other_windows_need_the_settings_and_their_app_allowed`, `outlines_skip_empty_wrappers_and_tools_offer_only_seen_ids`

### R6 The client's tools are its settings' and the automations library

`tools` lists what the client runs: every built-in tool and listed MCP tool of its own settings
file, and every automation as `script:<name>`, each with its description, its arguments' schema,
whether it asks first and the flow nodes that may call it; an automation also says whether this
version is approved. It serves the kind `script` and each of its MCP servers by name, whether or
not their tools are listed yet.

`run_tool` runs one for a flow node, whatever the server asked:

- a node its own `allow` does not name is refused with "<tool> does not allow the node <node>",
  and the user is not asked;
- when its own settings say it asks, or the server adds a confirmation, the user is asked once,
  and anything but a yes fails with "it was not confirmed";
- then it runs. An automation answers with its result alone (`{"done": true}` when it has none),
  and a script's error keeps its line and column.

Anything it does not register is "no tool … is registered".

Tests: `a_client_tool_runs_only_for_a_node_it_allows_and_after_a_yes`, `only_approved_versions_run_and_tools_carry_their_arguments`, `automations_are_script_tools_that_ask_and_respect_allow`, `a_server_tool_asks_through_the_client_and_a_client_tool_asks_by_itself`

### R7 A dry desk runs nothing

A desk in dry-run mode checks `allow` and asks as always, and then answers
`{"dry_run": true, "would_call": <reference>, "arguments": <arguments>}` without running the
tool. Headless takes use it.

Tests: `a_dry_run_checks_and_asks_as_always_and_runs_nothing`
