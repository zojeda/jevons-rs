# jevons-desktop-core: control

[Back to jevons-desktop-core](spec.md)

The desk's side of the control API ([server](../jevons-desktop-server/control.md)): what an outside
agent (Claude Code, a script) may read and do at this desk. `Control` checks every call against the
client's own `[control]` settings in `jevons-desktop.toml`, whatever the server passes on, and talks
to the user through the bubble the app implements (`Bubble`). `LocalDesk::with_control` gives a
desk this part.

## Requirements

### R1 Only the listed applications are read

`[control] apps` lists process-name globs (any case). Windows of other applications are not
listed, outlined, searched or read, and the focused element is returned only when its window is
of a listed application. With no applications listed, every read is `denied`; with none of their
windows open, a read without a node is `not_found`.

Tests: `only_listed_applications_are_read`

### R2 Short ids, each tied to its window

Windows get ids `w<n>` and elements `e<n>`. The same window or element keeps its id from one call
to the next. Each id is tied to the window it was seen in, which is how an id's application is
checked again before anything is read or done with it. The table keeps the latest 5,000 elements;
an id it no longer holds is `not_found`. `outline` shows the elements below a node to a depth of
1 to 6 (2 by default), skipping wrappers without text, at most 200 lines; `find` matches role and
text ignoring case; `xpath` evaluates in the node's window only and answers elements, texts or a
value; `read` returns the text below an element, at most `[control] max_chars` (20,000).

Tests: `the_listed_application_is_outlined_searched_selected_and_read_by_short_ids`

### R3 The first action in an application asks once

With `confirm = "once"` (the default), an agent's first action in an application asks the user in
the bubble ("Let <agent> act in <app>?", with the action); a yes holds for that agent and that
application until the agent has been quiet for ten minutes, or the settings change. A no, or no
answer, is `denied` and nothing happens. Each agent asks for itself. `every` asks before every
action; `never` does not ask. Without a bubble an action that must ask is `denied`.

Tests: `the_first_action_in_an_application_asks_once_and_a_no_stops_it`, `every_asks_each_time_and_never_does_not_ask`

### R4 Nothing is done over the user

While a take runs, or the user holds a key, an action is `busy` and does not happen, and a
question is `busy` too. Keys and typed text go to the window in front only when its application is
listed; `window` brings a listed window forward first.

Tests: `nothing_is_done_while_the_user_dictates_or_holds_a_key`

### R5 Password fields are neither read nor typed into

A password field's name and value are never returned, its text is never read (`denied`), and no
text goes into it (`denied`). A disabled element is not acted on.

Tests: `text_never_goes_into_a_password_field_and_its_value_is_never_read`

### R6 The bubble shows, asks and tells what it shows

With `[control] bubble` on (the default) and a bubble at the desk, `show` gives the bubble an
agent's Markdown message and says whether it waits for a take's bubble to close; `ask` asks yes or
no and answers `{yes, answered}` (unanswered within its timeout, 5 to 300 s, 60 by default, is no);
`bubble` reads what it shows. Off, each is `unavailable`. `status` says the backends, the listed
applications, `confirm` and whether the bubble is available.

Tests: `the_bubble_shows_asks_and_tells_what_it_shows_unless_it_is_off`

### R7 Every action shows as a line in the agent's bubble

After each action the agent's bubble gets a line saying what was done and where, such as
`type_text into Edit "Message" (14 characters) in slack.exe`. The text typed is not shown.

Tests: `the_first_action_in_an_application_asks_once_and_a_no_stops_it`

### R8 A desk without the control part takes no agent calls

A `LocalDesk` built without `with_control` answers every call `unavailable`.

Tests: `a_desk_without_control_takes_no_calls_from_outside_agents`

### R9 What was read or done is never logged

A call is logged with its operation, the agent's name and how it went (its error kind), never with
its arguments or what it read.

Tests: none yet
