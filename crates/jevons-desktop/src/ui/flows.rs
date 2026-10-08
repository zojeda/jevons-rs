//! The flow tree: its nodes, the problems of the flows folder, and new branches drafted from the
//! current context: under a decision, or as a state of an agent or a task. The Machines tab
//! draws a state's work inside its box with the same rows ([`branch`]), in short, and shows a
//! row's node with the same card ([`details`]).

use super::Ctx;
use super::components::{Choice, Icon, Select, badge, icon};
use crate::agent::{Command, open_folder};
use dioxus::prelude::*;
use jevons_desktop_server::flow::guard::{When, draft_when};
use jevons_desktop_server::flow::machine::{self, Level};
use jevons_desktop_server::flow::spec::Select as Selecting;
use jevons_desktop_server::flow::tree::{Node, NodeId, NodeSpec};
use jevons_desktop_server::flow::{FlowTree, Kind, defaults};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What the tree view needs to draw a node and its branches.
#[derive(Clone, Copy)]
pub(super) struct TreeState {
    /// Rows folded or unfolded by hand, by their place in the tree (`parent>…>path`), so a
    /// shared branch folds separately under each decision that uses it.
    pub opened: Signal<BTreeMap<String, bool>>,
    /// The selected row's place.
    pub selected: Signal<Option<String>>,
    /// Rows in short: a node's kind and name, and the marks of a walk.
    pub compact: bool,
}

/// What a tree marks on its rows, by their place.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Marks {
    /// The rows a route went through.
    pub route: BTreeSet<String>,
    /// The row a walk is at, while it runs.
    pub now: Option<String>,
    /// The branches a decision could not choose: their guards failed.
    pub out: BTreeSet<String>,
    /// Why a branch was taken, such as `model 0.91`.
    pub why: BTreeMap<String, String>,
    /// What the walk does where it is, such as `writing` or `calling web_search`.
    pub activity: Option<String>,
    /// Branches off the route start folded.
    pub folded: bool,
}

impl Marks {
    /// Whether the row at `here`, under `at`, shows its branches: as folded or unfolded by
    /// hand, else unless it is off a route that folds. The top of a tree starts open.
    fn opens(&self, opened: &BTreeMap<String, bool>, here: &str, at: &str) -> bool {
        let starts_open = !self.folded || self.route.contains(here) || at.is_empty();
        opened.get(here).copied().unwrap_or(starts_open)
    }
}

/// How many rows [`branch`] draws for the node `id` under the place `at`: its own, and those
/// of the branches that show.
pub(super) fn rows(
    tree: &FlowTree,
    id: NodeId,
    at: &str,
    marks: &Marks,
    opened: &BTreeMap<String, bool>,
) -> usize {
    let node = tree.node(id);
    let here = place(at, node);
    if !marks.opens(opened, &here, at) {
        return 1;
    }
    let below: usize = node
        .children
        .iter()
        .map(|child| rows(tree, *child, &here, marks, opened))
        .sum();
    1 + below
}

/// A node's place in the tree, under the place `at`.
pub(super) fn place(at: &str, node: &Node) -> String {
    format!("{at}>{}", node.path)
}

/// The places a route went through from `top` down, with their nodes. The first node is found
/// by its folder, and each one after it under the one before: a node that is no branch of the
/// one before it ends the way.
pub(super) fn way<'a>(
    tree: &FlowTree,
    top: NodeId,
    nodes: impl IntoIterator<Item = &'a str>,
) -> Vec<(NodeId, String)> {
    let mut way: Vec<(NodeId, String)> = Vec::new();
    for path in nodes {
        let path = path.trim_matches('/');
        let next = match way.last() {
            None => below(tree, top, path),
            Some((id, here)) => tree
                .children(*id)
                .find(|child| child.path == path)
                .map(|child| (child.id, place(here, child))),
        };
        let Some(next) = next else { break };
        way.push(next);
    }
    way
}

/// The nodes a place names from the root down: the folders above its first node, then its own.
fn nodes_at<'a>(tree: &'a FlowTree, place: &str) -> Vec<&'a Node> {
    let own: Vec<&str> = place.split('>').skip(1).collect();
    let mut paths: Vec<String> = Vec::new();
    // `a` and `a/b` are above `a/b/c`, under the root.
    if let Some(first) = own.first().filter(|first| !first.is_empty()) {
        paths.push(String::new());
        let mut names: Vec<&str> = first.split('/').collect();
        names.pop();
        for end in 1..=names.len() {
            paths.push(names[..end].join("/"));
        }
    }
    paths.extend(own.iter().map(|path| path.to_string()));
    paths
        .iter()
        .filter_map(|path| tree.find(path))
        .map(|id| tree.node(id))
        .collect()
}

/// The instructions `nodes` add, in their order and as written, each with the file it is in:
/// a folder's `instructions.md`, then its node file's `instructions`.
fn instructions(nodes: &[&Node]) -> Vec<(String, String)> {
    let written = |text: Option<&str>| {
        text.map(str::trim)
            .filter(|text| !text.is_empty())
            .map(String::from)
    };
    let mut out = Vec::new();
    for node in nodes {
        if let Some(text) = written(node.instructions_md.as_deref()) {
            let file = if node.path.is_empty() {
                "instructions.md".to_string()
            } else {
                format!("{}/instructions.md", node.path)
            };
            out.push((file, text));
        }
        if let Some(text) = written(node.spec.common().instructions) {
            out.push((node.file.clone(), text));
        }
    }
    out
}

/// The folder `path` at or under `top`, with its place: down the folders, so never a shared
/// branch.
fn below(tree: &FlowTree, top: NodeId, path: &str) -> Option<(NodeId, String)> {
    let mut node = tree.node(top);
    let mut here = place("", node);
    while node.path != path {
        node = tree.children(node.id).find(|child| {
            path.strip_prefix(child.path.as_str())
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
        })?;
        here = place(&here, node);
    }
    Some((node.id, here))
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Machine => "machine",
        Kind::Decide => "decide",
        Kind::Generate => "generate",
        Kind::Transcript => "transcript",
        Kind::Tool => "tool",
        Kind::Loop => "loop",
        Kind::Run => "run",
    }
}

/// A guard's (or `[prefer]`'s) rules, one short phrase each.
fn rules(when: &When) -> Vec<String> {
    let list = |items: &[String]| {
        let shown: Vec<&str> = items.iter().take(3).map(String::as_str).collect();
        let more = items.len().saturating_sub(3);
        if more > 0 {
            format!("{} +{more}", shown.join(", "))
        } else {
            shown.join(", ")
        }
    };
    let yes_no = |value: Option<bool>, yes: &str, no: &str| {
        value.map(|v| if v { yes } else { no }.to_string())
    };
    let mut out = Vec::new();
    if !when.app.is_empty() {
        out.push(format!("app {}", list(&when.app)));
    }
    if let Some(title) = &when.window_title {
        out.push(format!("title /{title}/"));
    }
    if !when.url.is_empty() {
        out.push(format!("url {}", list(&when.url)));
    }
    if !when.role.is_empty() {
        out.push(format!("role {}", list(&when.role)));
    }
    if let Some(name) = &when.element_name {
        out.push(format!("field /{name}/"));
    }
    out.extend(yes_no(when.selection, "text selected", "nothing selected"));
    out.extend(yes_no(when.text, "field has text", "field empty"));
    out.extend(yes_no(
        when.editable,
        "accepts typing",
        "does not accept typing",
    ));
    if let Some(said) = &when.transcript {
        out.push(format!("said /{said}/"));
    }
    out
}

/// How a node chooses or what it produces, in a few words.
fn what(node: &Node) -> String {
    match &node.spec {
        NodeSpec::Decide(d) => {
            let mut out = match d.select {
                Selecting::Model => "the model chooses".to_string(),
                Selecting::Rules => "rules choose".to_string(),
            };
            if let Some(fallback) = &d.fallback {
                out.push_str(&format!(", else {fallback}"));
            }
            if let Some(floor) = d.min_probability {
                out.push_str(&format!(", from {:.0}%", floor * 100.0));
            }
            out
        }
        other => other
            .output()
            .map(|output| format!("→ {output:?}").to_lowercase())
            .unwrap_or_default(),
    }
}

/// The text of a node's file, from the folder or the built-in tree.
fn file_text(builtin: bool, dir: &Path, file: &str) -> String {
    if builtin {
        defaults::TREE
            .iter()
            .find(|(path, _)| *path == file)
            .map(|(_, text)| text.to_string())
            .unwrap_or_default()
    } else {
        std::fs::read_to_string(dir.join(file)).unwrap_or_default()
    }
}

/// Every place (`parent>…>path`) of the tree, to fold or unfold them all.
fn places(tree: &FlowTree, id: NodeId, at: &str, out: &mut Vec<String>) {
    let node = tree.node(id);
    let here = place(at, node);
    if !node.children.is_empty() {
        out.push(here.clone());
        for child in &node.children {
            places(tree, *child, &here, out);
        }
    }
}

/// A node and, unless folded, its branches beneath it on a guide line, marked as `marks` say.
pub(super) fn branch(
    tree: &Arc<FlowTree>,
    id: NodeId,
    at: &str,
    marks: &Marks,
    state: TreeState,
) -> Element {
    let node = tree.node(id);
    let here = place(at, node);
    let on_route = marks.route.contains(&here);
    let open = marks.opens(&state.opened.read(), &here, at);
    let folded = !open;
    let leaf = node.children.is_empty();
    let kind = kind_name(node.kind());
    let name = if node.name.is_empty() {
        "/".to_string()
    } else {
        node.name.clone()
    };
    // A branch of a shared folder (such as _actions) shows under every decision that uses it.
    let parent_path = at.rsplit('>').next().unwrap_or("");
    let shared = node
        .path
        .split('/')
        .next()
        .filter(|top| top.starts_with('_') && !parent_path.starts_with(*top))
        .filter(|_| !state.compact)
        .map(String::from);
    let mut notes: Vec<String> = Vec::new();
    let when = rules(node.guard.spec());
    if !when.is_empty() {
        notes.push(format!("when {}", when.join(" · ")));
    }
    let prefer = rules(node.prefer.spec());
    if !prefer.is_empty() {
        notes.push(format!("prefer {}", prefer.join(" · ")));
    }
    let priority = node.spec.common().priority;
    if priority != 0 {
        notes.push(format!("priority {priority}"));
    }
    let reads = node.extracts.len() + node.investigations.len();
    if reads > 0 {
        notes.push(format!("reads {reads}"));
    }
    if state.compact {
        notes.clear();
    }
    let now = marks.now.as_deref() == Some(here.as_str());
    let activity = marks.activity.clone().filter(|_| now);
    let out = marks.out.contains(&here);
    let why = marks.why.get(&here).cloned();
    let selected = state.selected.read().as_deref() == Some(here.as_str());
    let (toggled, labelled) = (here.clone(), here.clone());
    let mut opened = state.opened;
    let mut chosen = state.selected;
    let summary = if state.compact {
        String::new()
    } else {
        what(node)
    };
    rsx! {
        div { class: "flow-node", key: "{here}",
            div { class: "flow-row",
                "data-selected": if selected { "true" } else { "false" },
                "data-route": if on_route { "true" } else { "false" },
                "data-now": if now { "true" } else { "false" },
                "data-off": if out { "true" } else { "false" },
                if leaf {
                    span { class: "flow-spacer" }
                } else {
                    button { class: "flow-toggle", title: if folded { "Unfold" } else { "Fold" },
                        onclick: move |_| {
                            opened.write().insert(toggled.clone(), !open);
                        },
                        {icon(if folded { Icon::ChevronRight } else { Icon::ChevronDown })}
                    }
                }
                span { class: "flow-label",
                    // A second click takes the selection back.
                    onclick: move |_| chosen.set((!selected).then(|| labelled.clone())),
                    span { class: "flow-kind", "data-kind": kind, "{kind}" }
                    span { class: "flow-name", "{name}" }
                    if now {
                        span { class: "flow-now", "now" }
                    }
                    if let Some(why) = why {
                        span { class: "flow-why", "{why}" }
                    }
                    if let Some(activity) = activity {
                        span { class: "flow-activity", "{activity}" }
                    }
                    if let Some(top) = shared {
                        span { class: "flow-shared", "shared from {top}" }
                    }
                    if !summary.is_empty() {
                        span { class: "flow-what", "{summary}" }
                    }
                    if !notes.is_empty() {
                        span { class: "flow-notes", "{notes.join(\"  ·  \")}" }
                    }
                }
            }
            if !leaf && !folded {
                div { class: "flow-children",
                    {node.children.iter().map(|child| branch(tree, *child, &here, marks, state))}
                }
            }
        }
    }
}

/// The node of the selected row in full: what it does, its rules, the instructions it works
/// under (those the folders on the way to its row add, from the root down), and its file.
pub(super) fn details(tree: &FlowTree, place: &str, builtin: bool, dir: &Path) -> Element {
    let way = nodes_at(tree, place);
    let Some(node) = way.last().copied() else {
        return rsx! {};
    };
    let adds = instructions(&way);
    let kind = kind_name(node.kind());
    let when = rules(node.guard.spec());
    let prefer = rules(node.prefer.spec());
    let text = file_text(builtin, dir, &node.file);
    let file = dir.join(&node.file);
    let title = if node.path.is_empty() {
        "/".to_string()
    } else {
        node.path.clone()
    };
    let mut reads: Vec<String> = node
        .extracts
        .keys()
        .map(|k| format!("extract {k}"))
        .collect();
    reads.extend(
        node.investigations
            .keys()
            .map(|k| format!("investigation {k}")),
    );
    rsx! {
        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "row", {badge(kind, "secondary")} div { class: "dx-card-title mono", "{title}" } }
                    div { class: "dx-card-description", "{node.description().unwrap_or(\"(no description)\")}" }
                }
                if !builtin {
                    button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                        onclick: move |_| open_folder(&file),
                        "Open file"
                    }
                }
            }
            div { class: "dx-card-content",
                div { class: "kv",
                    span { class: "k", "Does" }
                    span { class: "v", "{what(node)}" }
                    span { class: "k", "Applies when" }
                    span { class: "v", if when.is_empty() { "always (no guard)" } else { "{when.join(\" and \")}" } }
                    if !prefer.is_empty() {
                        span { class: "k", "Preferred when" }
                        span { class: "v", "{prefer.join(\" and \")}" }
                    }
                    if node.spec.common().priority != 0 {
                        span { class: "k", "Priority" }
                        span { class: "v", "{node.spec.common().priority}" }
                    }
                    if !reads.is_empty() {
                        span { class: "k", "Reads" }
                        span { class: "v", "{reads.join(\", \")}" }
                    }
                    span { class: "k", "File" }
                    span { class: "v mono", "{node.file}" }
                }
                if !adds.is_empty() {
                    div { class: "flow-instructions",
                        span { class: "muted", "The instructions it works under, from the root down" }
                        {adds.into_iter().enumerate().map(|(i, (file, text))| rsx! {
                            div { key: "{i}", class: "flow-instruction",
                                span { class: "mono muted", "{file}" }
                                pre { class: "code", "{text}" }
                            }
                        })}
                    }
                    span { class: "muted", "Its file" }
                }
                pre { class: "code", "{text}" }
            }
        }
    }
}

/// Whether a new branch may go under `node`: a decision, or an agent or a task, whose state
/// it becomes. Not a shared folder, whose branches others take, nor the root machine, whose
/// states are agents.
fn takes_branches(node: &Node) -> bool {
    if node.path.starts_with('_') {
        return false;
    }
    match node.kind() {
        Kind::Decide => true,
        Kind::Machine => matches!(node.level(), Some(Level::Agent | Level::Task)),
        _ => false,
    }
}

/// Creates the branch `name` under `parent` in the flows folder `dir`: its folder with the
/// node file `file_name` holding `text`. Under a machine the branch is a state, and the
/// machine's diagram gets its transitions: entered on `said` from the state the machine waits
/// in, and back when its work is done. The node file made.
fn create_branch(
    dir: &Path,
    tree: &FlowTree,
    parent: &str,
    name: &str,
    file_name: &str,
    text: &str,
) -> Result<PathBuf, String> {
    let node = tree
        .find(parent)
        .map(|id| tree.node(id))
        .filter(|node| takes_branches(node))
        .ok_or_else(|| format!("{parent} takes no branches"))?;
    let folder = dir.join(parent).join(name);
    if folder.exists() {
        return Err(format!("{} already exists", folder.display()));
    }
    // The diagram first: one that cannot take the state leaves nothing behind.
    let diagram = match node.kind() {
        Kind::Machine => {
            let file = dir.join(machine::diagram_file(&node.file));
            let before = std::fs::read_to_string(&file)
                .map_err(|e| format!("Cannot read {}: {e}", file.display()))?;
            let after = machine::add_state(&before, name)
                .map_err(|e| format!("{}: {}", file.display(), e.join("; ")))?;
            Some((file, after))
        }
        _ => None,
    };
    let file = folder.join(file_name);
    std::fs::create_dir_all(&folder)
        .and_then(|()| std::fs::write(&file, text))
        .map_err(|e| format!("Cannot create {}: {e}", file.display()))?;
    if let Some((diagram, after)) = diagram {
        std::fs::write(&diagram, after)
            .map_err(|e| format!("Cannot write {}: {e}", diagram.display()))?;
    }
    Ok(file)
}

#[component]
pub fn FlowsPage(rev: u64) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    // The decision or machine a new branch goes under; the first that takes one until picked.
    let mut parent = use_signal(|| None::<String>);
    let mut name = use_signal(String::new);
    // What creating a branch did: Ok with what to do next, or the error.
    let mut message = use_signal(|| None::<Result<String, String>>);
    let selected = use_signal(|| None::<String>);
    let state = TreeState {
        opened: use_signal(BTreeMap::new),
        selected,
        compact: false,
    };
    let view = ctx.view.lock().expect("the view lock");
    // The rows the route went through: a shared branch only under the decision that took it.
    let marks = Marks {
        route: way(
            &view.flows,
            view.flows.root(),
            view.route.iter().map(|step| step.node.as_str()),
        )
        .into_iter()
        .map(|(_, place)| place)
        .collect(),
        ..Marks::default()
    };
    let dir = view.config.flows_dir(&view.config_file);
    let errors: Vec<String> = view.flow_errors.iter().map(ToString::to_string).collect();
    let notes = view.flow_notes.clone();
    let tool_problems = view.tool_problems.clone();
    let tree = view.flows.clone();
    let context = view.context.clone();
    drop(view);

    let builtin = tree.source == defaults::BUILTIN;

    let id = name();
    let valid = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_');
    // Where a new branch can go: the decisions, and the agents and tasks, whose state it is.
    let parents: Vec<Choice> = tree
        .nodes()
        .iter()
        .filter(|n| takes_branches(n))
        .map(|n| Choice {
            value: Some(n.path.clone()),
            label: match (n.path.is_empty(), n.level()) {
                (true, _) => "/ (the root)".into(),
                (_, Some(Level::Agent)) => format!("{} (an agent: a new state)", n.path),
                (_, Some(Level::Task)) => format!("{} (a task: a new state)", n.path),
                _ => n.path.clone(),
            },
        })
        .collect();
    let parent_path = parent()
        .or_else(|| parents.first().and_then(|c| c.value.clone()))
        .unwrap_or_default();
    let parent_node = tree
        .find(&parent_path)
        .map(|p| tree.node(p))
        .filter(|n| takes_branches(n));
    let parent_ok = parent_node.is_some();
    let as_state = parent_node.is_some_and(|n| n.kind() == Kind::Machine);
    let shared = tree.nodes().iter().any(|n| n.path.starts_with("_actions/"));
    let draft = context.as_ref().map(|c| {
        let title = if c.window.title.is_empty() { c.app.process_name.clone() } else { c.window.title.clone() };
        if shared {
            // A decision's branch has a priority among its siblings; a state has none.
            let priority = if as_state { "" } else { "priority = 10\n" };
            format!(
                "description = {title:?}\n{priority}branches = \"_actions\"\ninstructions = \"Describe how text should read here.\"\n\n{}",
                draft_when(c)
            )
        } else {
            format!("description = {title:?}\noutput = \"target\"\n\n{}", draft_when(c))
        }
    });
    let file_name = if shared {
        "decide.toml"
    } else {
        "generate.toml"
    };
    let open_dir = dir.clone();
    let reload = ctx.clone();
    let fold_tree = tree.clone();
    let create_tree = tree.clone();
    let create_parent = parent_path.clone();

    rsx! {
        div { class: "spread",
            div { class: "stack",
                h2 { "Flow tree" }
                span { class: "muted mono", "{dir.display()}" }
            }
            div { class: "row",
                button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                    onclick: move |_| {
                        let _ = std::fs::create_dir_all(&open_dir);
                        open_folder(&open_dir);
                    },
                    "Open folder"
                }
                button { class: "dx-button", "data-style": "secondary", "data-size": "sm",
                    onclick: move |_| reload.send(Command::ReloadFlows),
                    "Reload"
                }
            }
        }
        if !errors.is_empty() {
            p { class: "warn",
                if builtin { "The flows folder has problems: the built-in tree runs until they are fixed." }
                else { "The flows folder has problems: the last tree that loaded cleanly still runs." }
            }
            {errors.iter().map(|e| rsx! { p { class: "error-text mono", "{e}" } })}
        }
        {notes.iter().map(|n| rsx! { p { class: "muted", "{n}" } })}
        {tool_problems.iter().map(|n| rsx! { p { class: "warn", "{n}" } })}
        p { class: "muted",
            "Every folder is a node; AGENTS.md in the folder explains the format to people and agents, \
             and TOOLS.md lists the tools the settings register. Files reload as soon as they are saved."
        }

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "New branch from the current context" }
                    div { class: "dx-card-description", "Its guard matches the application, page and field the Context tab shows" }
                }
            }
            div { class: "dx-card-content",
                match (context, draft) {
                    (Some(context), Some(draft)) => {
                        let create_ctx = ctx.clone();
                        let dir = dir.clone();
                        let text = draft.clone();
                        let (tree, under) = (create_tree.clone(), create_parent.clone());
                        rsx! {
                            p { class: "muted", "Matches {context.app.process_name} · {context.window.title}" }
                            div { class: "row",
                                Select { value: Some(parent_path.clone()), choices: parents.clone(),
                                    onchange: move |v: Option<String>| parent.set(v) }
                                input { class: "dx-input", placeholder: "branch name, such as slack", value: "{id}",
                                    oninput: move |e| name.set(e.value()) }
                                button { class: "dx-button", "data-style": "accent", "data-size": "sm", disabled: !valid || !parent_ok,
                                    onclick: move |_| {
                                        let made = create_branch(&dir, &tree, &under, &name(), file_name, &text);
                                        message.set(Some(made.map(|file| {
                                            create_ctx.send(Command::ReloadFlows);
                                            if let Some(folder) = file.parent() {
                                                open_folder(folder);
                                            }
                                            format!("Created {}: describe it and adjust its guard", file.display())
                                        })));
                                    },
                                    "Create"
                                }
                            }
                            if !id.is_empty() && !valid {
                                p { class: "error-text", "A branch name uses lowercase letters, digits, - and _ only." }
                            } else {
                                p { class: "muted",
                                    if as_state {
                                        "A state of the machine chosen on the left, entered on what the user says from the state it waits in, as a folder named in lowercase letters, digits, - and _. Its diagram gets the two transitions."
                                    } else {
                                        "Under the decision chosen on the left, as a folder named in lowercase letters, digits, - and _."
                                    }
                                }
                            }
                            pre { class: "code", "{draft}" }
                        }
                    }
                    _ => rsx! { p { class: "muted", "Capture an application in the Context tab first." } },
                }
                match message() {
                    Some(Ok(done)) => rsx! { p { class: "ok-text", "{done}" } },
                    Some(Err(failed)) => rsx! { p { class: "error-text", "{failed}" } },
                    None => rsx! {},
                }
            }
        }

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Tree" }
                    div { class: "dx-card-description",
                        "Each branch sits under the decision that chooses it. Marked rows are the route the Context tab's window takes; select a node to see it in full."
                    }
                }
                div { class: "row",
                    button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                        onclick: move |_| { let mut f = state.opened; f.write().clear(); },
                        "Unfold all"
                    }
                    button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                        onclick: move |_| {
                            let mut all = Vec::new();
                            // The root stays open, so its branches show.
                            for child in &fold_tree.node(fold_tree.root()).children {
                                places(&fold_tree, *child, &format!(">{}", fold_tree.node(fold_tree.root()).path), &mut all);
                            }
                            let mut f = state.opened;
                            f.set(all.into_iter().map(|place| (place, false)).collect());
                        },
                        "Fold all"
                    }
                }
            }
            div { class: "dx-card-content",
                div { class: "flow-tree",
                    {(!tree.nodes().is_empty()).then(|| branch(&tree, tree.root(), "", &marks, state))}
                }
            }
        }
        if let Some(place) = selected() {
            {details(&tree, &place, builtin, &dir)}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jevons_desktop_server::flow::{Catalog, Disk};

    const TREE: &[(&str, &str)] = &[
        ("root.toml", ""),
        (
            "root.fsm",
            "fsm App {\n    [*] --> idle\n    idle --> main : said\n    main --> idle\n}\n",
        ),
        ("main/agent.toml", "description = \"Main\""),
        (
            "main/agent.fsm",
            "fsm Main {\n    [*] --> idle\n    idle --> type : said\n    type --> idle\n}\n",
        ),
        ("main/type/decide.toml", "description = \"Types it\""),
        (
            "main/type/plain/transcript.toml",
            "description = \"As said\"",
        ),
    ];

    #[test]
    fn a_route_s_rows_are_each_under_the_decision_that_took_them() {
        let tree = FlowTree::load(&defaults::builtin(), &Catalog::default());
        let places = |nodes: &[&str]| -> Vec<String> {
            way(&tree, tree.root(), nodes.iter().copied())
                .into_iter()
                .map(|(_, place)| place)
                .collect()
        };
        // From the root down: a shared branch has its place under the decision that took it.
        assert_eq!(
            places(&[
                "/",
                "dictation",
                "dictation/dictate",
                "dictation/dictate/terminal",
                "_actions/insert"
            ])
            .last()
            .map(String::as_str),
            Some(">>dictation>dictation/dictate>dictation/dictate/terminal>_actions/insert")
        );
        // A route that starts at a branch is found by its folder.
        assert_eq!(
            places(&["dictation/dictate/notes", "_actions/rewrite"]),
            [
                ">>dictation>dictation/dictate>dictation/dictate/notes",
                ">>dictation>dictation/dictate>dictation/dictate/notes>_actions/rewrite"
            ]
        );
        // A node that is no branch of the one before it ends the way.
        assert_eq!(places(&["dictation/dictate", "assistant/ask"]).len(), 1);
        assert!(places(&["_actions/insert"]).is_empty());
    }

    #[test]
    fn a_branch_is_created_under_a_decision_or_as_a_state_of_a_machine() {
        let dir = std::env::temp_dir().join(format!("jevons-ui-branch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for (file, text) in TREE {
            let file = dir.join(file);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, text).unwrap();
        }
        let load = || FlowTree::load(&Disk::new(&dir), &Catalog::default());
        let tree = load();
        assert!(tree.is_valid(), "{:?}", tree.errors);
        // Where a branch may go: the decision and the agent, never the root machine.
        let takes: Vec<&str> = tree
            .nodes()
            .iter()
            .filter(|n| takes_branches(n))
            .map(|n| n.path.as_str())
            .collect();
        assert_eq!(takes, ["main", "main/type"]);
        let text = "description = \"Replies in Slack\"\noutput = \"target\"\n";
        // Under a decision: a folder with its node file, and nothing else changes.
        let file = create_branch(&dir, &tree, "main/type", "slack", "generate.toml", text).unwrap();
        assert_eq!(file, dir.join("main/type/slack/generate.toml"));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), text);
        // As a state of the agent: its diagram enters it from where the agent waits.
        let file = create_branch(&dir, &tree, "main", "reply", "generate.toml", text).unwrap();
        assert_eq!(file, dir.join("main/reply/generate.toml"));
        assert_eq!(
            std::fs::read_to_string(dir.join("main/agent.fsm")).unwrap(),
            "fsm Main {\n    [*] --> idle\n    idle --> type : said\n    type --> idle\n    idle --> reply : said\n    reply --> idle\n}\n"
        );
        // The folder as it is now loads, with the new state's work.
        let tree = load();
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let main = tree
            .node(tree.find("main").unwrap())
            .machine
            .clone()
            .unwrap();
        assert!(main.diagram.working.contains("reply"));
        // A folder that is there, a state the machine has, and the root are refused, and
        // leave the diagram as it was.
        let before = std::fs::read_to_string(dir.join("main/agent.fsm")).unwrap();
        let refused = |parent: &str, name: &str| {
            create_branch(&dir, &tree, parent, name, "generate.toml", text).unwrap_err()
        };
        assert!(refused("main", "reply").ends_with("already exists"));
        assert!(refused("main", "idle").ends_with("idle is already a state of Main"));
        assert!(!dir.join("main/idle").exists());
        assert_eq!(refused("", "more"), " takes no branches");
        assert_eq!(
            std::fs::read_to_string(dir.join("main/agent.fsm")).unwrap(),
            before
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
