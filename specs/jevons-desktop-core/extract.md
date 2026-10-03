# Extracts and XPath

[jevons-desktop-core](spec.md)

## Purpose

An extract reads part of an application's interface with an XPath expression, with no model. The
XPath here is a subset of XPath 1.0 over accessibility trees: element names are roles and
attributes are an element's properties. The same expressions serve the inspector's workbench, the
selectors that find a recorded element again, and the interface browser that helps write them.

## Scope

This file covers the XPath subset (`xpath/parse.rs`), its evaluation (`xpath/eval.rs`), the windows
an expression may read, `[extract.<name>]` and its reads (`flow/extract.rs`), the workbench's trial
and save, selectors (`xpath/selector.rs`) and the interface browser (`interface.rs`).

Elsewhere:

- When the walk reads an extract, lazy reads, reuse within a take, and the placeholder checks on
  `$variables`: [flows.md](../jevons-desktop-server/flows.md).
- The investigator's `xpath` tool: [investigator.md](../jevons-desktop-server/investigator.md). Automation scripts' use of
  XPath: [automations.md](automations.md).
- Recorded interfaces (`--tree`, **Record tree**), which every read here also reads:
  [recording.md](recording.md).

## The XPath subset

### R1 Element names are roles

An element name is a UI Automation control type, spelled as the platform reports it: `Window`,
`Pane`, `Group`, `Document`, `List`, `ListItem`, `Tree`, `TreeItem`, `Edit`, `Button`, `Text`,
`Hyperlink` and the others. Any other name is an error that suggests the role meant
(`no role "Listitem"; did you mean ListItem?`).

Tests: `mistakes_are_errors_at_their_column`, `operators_and_names_are_told_apart_by_what_precedes_them`

### R2 Attributes are a fixed vocabulary

The attributes are `@name`, `@value`, `@class`, `@automation_id`, `@role`, `@enabled`,
`@offscreen`, `@selected`, `@toggled`, `@expanded` and `@password`, and on top-level windows
`@app`, `@title` and `@front`. Flags read `'true'` or `'false'`. Any other attribute is an error
that lists them.

Tests: `mistakes_are_errors_at_their_column`, `absolute_paths_start_at_the_windows_and_relative_ones_at_the_context`

### R3 Paths, axes and node tests

Paths take `/`, `//`, `.`, `..` and `@`, and the axes `child`, `descendant`,
`descendant-or-self`, `parent`, `ancestor`, `ancestor-or-self`, `self`, `following-sibling`,
`preceding-sibling` and `attribute`. A step tests a role, `*`, `node()` or `text()`. Any other axis
is an error.

Tests: `paths_predicates_and_functions_parse`, `axes_climb_and_walk_siblings_after_a_search`, `mistakes_are_errors_at_their_column`

### R4 Functions and operators

The functions are XPath 1.0's (`last`, `position`, `count`, `string`, `concat`, `starts-with`,
`contains`, `substring`, `substring-before`, `substring-after`, `string-length`,
`normalize-space`, `translate`, `boolean`, `not`, `true`, `false`, `number`, `sum`, `floor`,
`ceiling`, `round`, `name`, `local-name`) plus `ends-with`, `lower-case`, `upper-case`, `matches`
and `has-class`. An unknown function, a wrong number of arguments and a `matches` pattern that
does not compile are errors. The operators are `or`, `and`, `=`, `!=`, `<`, `<=`, `>`, `>=`, `+`,
`-`, `*`, `div`, `mod`, unary `-` and `|`; what precedes a word tells an operator from a name.

Tests: `mistakes_are_errors_at_their_column`, `strings_numbers_and_comparisons_follow_xpath`, `operators_and_names_are_told_apart_by_what_precedes_them`

### R5 Mistakes are errors at their column

A mistake is an error with its column, counted in characters from 1: an empty expression, an
unclosed string or bracket, an empty predicate `[]`, and an unexpected token.

Tests: `mistakes_are_errors_at_their_column`

### R6 Variables come from outside the expression

`$name` and `$name.field` take a value from outside the expression, so a value never changes what
the expression means. An expression lists the variables it uses. A variable with no value at
evaluation is an error.

Tests: `variables_are_listed_and_positional_predicates_known`, `unbound_variables_and_limits_are_errors`

## Evaluation

### R7 Absolute paths start at the windows

The document's root has the readable windows as its children, the take's own first. An absolute
path starts at the root, so `/Window[@app='slack.exe']` selects a window. A relative path starts
at the context node, the take's window for an extract.

Tests: `absolute_paths_start_at_the_windows_and_relative_ones_at_the_context`

### R8 Values follow XPath 1.0

Conversions between strings, numbers and booleans, comparisons involving node sets, arithmetic,
and unions in document order follow XPath 1.0. `//Text[1]` counts among each element's siblings,
and `(//Text)[1]` over the whole result.

Tests: `strings_numbers_and_comparisons_follow_xpath`, `positional_descendant_steps_keep_their_meaning`

### R9 A descendant step is one native search

`//Role[…]` whose predicates do not depend on positions is one native search, with the role, and
the name, class and automation id its predicates compare, as conditions. Every match is checked
against the full predicates again.

Tests: `a_descendant_step_is_one_native_search_with_its_conditions`, `descendant_steps_read_channels_and_messages`

### R10 An element's text holds its descendants'

An element's string value is its name and value followed by its descendants', in reading order,
without a text the one before already holds. It is cut at the document's text limit.

Tests: `element_text_gathers_descendants_and_never_reads_passwords`

### R11 Password fields are never read

A password field's name, value and text read as empty, in its own string value and in its
ancestors'. `@password` tells one apart, but no predicate can probe its value.

Tests: `element_text_gathers_descendants_and_never_reads_passwords`

### R12 Reads are bounded

An evaluation reads at most 20,000 elements by default, and stops at its deadline when it has one.
Going past either is an error, never a shortened answer.

Tests: `unbound_variables_and_limits_are_errors`

### R13 Other windows need the settings

An expression reads the take's own window: the application's window with the take's title, else
the one in front. It reads another window only when it names that application in `scope`,
`privacy.read_other_windows` is on and `privacy.readable_apps` names the application too.
Otherwise a note says why, such as reading other windows being off.

Tests: `variables_bind_values_and_other_apps_need_permission`, `other_windows_need_the_settings_and_their_app_allowed`

## `[extract]`

### R14 An extract is checked by field

`[extract.<name>]` sets `xpath`, and may set `as`, `fields`, `limit`, `scope`, `app` and
`lazy`. The expression and each column parse, with errors that name the field and the column.
`as = "table"` needs `fields`, and only a table takes them. Column names are lowercase
identifiers. `limit` is 1 to 500 (50 by default). `scope` and `app` globs compile.

Tests: `mistakes_are_reported_by_field`, `extracts_are_checked_and_their_answers_are_placeholders_below`

### R15 Each kind of answer has its shape

- `text` (the default): the first match's text, an attribute's value, or the expression's value;
  `null` when empty.
- `list`: the text of each match up to `limit`, empty ones left out.
- `count`: how many elements match, or a number value rounded.
- `exists`: whether anything matches.
- `table`: a row per element matched up to `limit`, with a column per `fields` expression
  evaluated from the row's element, `null` when empty.

Their shapes are a string, a list of strings, an integer, a boolean and a list of rows.

Tests: `each_kind_of_answer_fits_its_shape`

### R16 An extract that cannot read answers empty

When no window may be read, the evaluation fails, or a table's expression gives a value that is not
elements, the answer is empty for its kind (`null`, `[]`, `0` or `false`) with a note saying why.
An extract never fails the take.

Tests: `variables_bind_values_and_other_apps_need_permission`

### R17 `app` limits where an extract reads

With `app` (process-name globs, case ignored), an extract is read only in takes from those
applications. Elsewhere nothing is read, and its answer is empty with a note.

Tests: `the_built_in_slack_branch_reads_the_conversation`, `a_trial_reads_an_edited_expression_as_a_take_would`

### R18 `$variables` bind the take's values

Each `$name` takes the text of the placeholder of that path in the take, such as `$transcript` or
`$chat.name`, as a string. A lazy value a variable names is read first.

Tests: `variables_bind_values_and_other_apps_need_permission`, `a_trial_reads_an_edited_expression_as_a_take_would`

### R19 An extract has five seconds

One extract reads for at most 5 seconds. An element's text is cut at `privacy.max_context_chars`,
and never below 200 characters.

Tests: none yet

## The workbench

### R20 The extracts a context reads

`applicable` lists the extracts of every node whose guard, and every guard above it, passes in a
context, lazy ones included, whose `app` fits, each name once, nearest the root first.
`read_applicable` reads them in that order, each binding its variables from those before it.

Tests: `the_built_in_slack_branch_reads_the_conversation`

### R21 A trial reads as a take would

`trial` checks an edited extract as a node file's, then reads it in the context as a take would:
the same windows, permissions and `app` filter. The tree's applicable extracts its variables name
are read first, and listed. It returns the answer, the match count, a line for each of the first 50
matches and its time. An extract that does not compile comes back with its errors and reads
nothing.

Tests: `a_trial_reads_an_edited_expression_as_a_take_would`

### R22 Saving keeps the file and refuses a broken tree

`save` writes an extract's `xpath`, `as` and `fields` into `[extract.<name>]` of its node file,
keeping the file's comments and order, once the tree with the change loads with no errors.
Otherwise nothing is written and the errors come back. A file that declares no such extract is an
error. `as_toml` gives an `[extract.<name>]` table to paste.

Tests: `saving_an_extract_keeps_the_file_s_comments_and_refuses_a_broken_tree`

## Selectors

### R23 Selectors find a recorded element, most robust first

The selectors for a recorded element are expressions that select it alone in the interface it was
recorded in; each is checked to select that element alone. They come most robust first: its
automation id, its classes (one, then two), a stable ancestor with its role and class, the text it
or a descendant `Text` shows, its name, and last its position from the window. Each says how it
selects, whether it holds when the user's language changes, and the text it matches.

Tests: `a_composer_is_found_by_its_class_first`, `a_channel_is_found_by_its_id_and_by_the_text_an_argument_carries`

### R24 Generated ids and classes are left out

An automation id with a run of 6 or more digits, or of 16 or more hexadecimal characters, is left
out as generated per item. So is a class token a build generates (`name__AGMar`). Text in an
expression is quoted with the quote it lacks, or through `concat()` when it holds both.

Tests: `generated_ids_and_classes_are_left_out`

### R25 Selectors for a live element

For an element of a live window, `selectors` records the window as a recording does and returns
its selectors. An element no longer in the window is an error that asks to reload the tree.

Tests: `selectors_find_the_element_alone_and_a_gone_one_asks_to_reload`

## The interface browser

### R26 The browser shows only the take's window

The browser opens the snapshot's own window and no other; with none, it says why. A level lists
the first 200 children with how many there are, and each page adds up to 200 more. A password
field keeps no value, and a value is cut at 200 characters.

Tests: `the_browser_opens_the_snapshot_s_window_a_level_at_a_time`, `levels_cap_their_children_and_keep_no_password_text`

### R27 Opening below reads within a budget

`open_below` opens every level below an element to a depth, parents first, reading at most its
budget of elements, and says when it stopped. Levels the tree already holds are used, not read
again. A parent that is gone is an error.

Tests: `the_browser_opens_the_snapshot_s_window_a_level_at_a_time`, `levels_already_read_are_not_read_again_when_opening_below`

### R28 Search reads text as a person does

A search matches an element's role, name, value, class and automation id with case, accents and
spacing folded ("andres chort" finds "Andrés Chort"). An element that holds every word, in any
order or across its properties, matches after those that hold the whole text. A password field's
name and value are never compared.

Tests: `text_folds_case_accents_and_spacing_as_a_person_reads_it`, `whole_text_matches_come_before_matches_of_every_word`, `the_filter_matches_role_name_value_class_and_automation_id`, `an_accented_query_finds_a_direct_message_and_reveals_it_by_walking_up`

### R29 Search covers the whole window and says what it missed

A search reads the whole window, not only what is open, within a budget of elements and a
deadline, and says when it stopped early. It keeps the first 100 matches and knows where the first
of them sit. Where the one-call search fails it walks the window instead, and counts the parts it
could not read with the first reason.

Tests: `a_search_finds_text_anywhere_in_the_window_with_the_way_down_to_it`, `a_part_of_the_window_that_cannot_be_read_is_reported_not_skipped`

### R30 Revealing reads only what the tree lacks

`ancestry` gives the way from the window down to an element, and fails for one that is gone.
`reveal` reads the levels on that way the tree does not hold, in whole pages, and stops where the
way leaves the window.

Tests: `revealing_reads_only_the_levels_the_tree_lacks_and_the_way_up_matches`, `levels_cap_their_children_and_keep_no_password_text`
