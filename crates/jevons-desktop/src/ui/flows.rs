//! The flow tree: its nodes, the problems of the flows folder, and new branches drafted from the
//! current context.

use super::Ctx;
use super::components::{Choice, Icon, Select, badge, icon};
use crate::agent::{Command, open_folder};
use dioxus::prelude::*;
use jevons_desktop_core::flow::guard::{When, draft_when};
use jevons_desktop_core::flow::spec::Select as Selecting;
use jevons_desktop_core::flow::tree::{Node, NodeId, NodeSpec};
use jevons_desktop_core::flow::{FlowTree, Kind, defaults};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

/// What the tree view needs to draw a node and its branches.
#[derive(Clone, Copy)]
struct TreeState {
    /// Rows folded by their place in the tree (`parent>…>path`), so a shared branch folds
    /// separately under each decision that uses it.
    folded: Signal<BTreeSet<String>>,
    /// The selected node's path.
    selected: Signal<Option<String>>,
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
    let here = format!("{at}>{}", node.path);
    if !node.children.is_empty() {
        out.push(here.clone());
        for child in &node.children {
            places(tree, *child, &here, out);
        }
    }
}

/// A node and, unless folded, its branches beneath it on a guide line.
fn branch(
    tree: &Arc<FlowTree>,
    id: NodeId,
    at: &str,
    route: &BTreeSet<String>,
    state: TreeState,
) -> Element {
    let node = tree.node(id);
    let here = format!("{at}>{}", node.path);
    let folded = state.folded.read().contains(&here);
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
    let on_route = route.contains(&node.path);
    let selected = state.selected.read().as_deref() == Some(node.path.as_str());
    let path = node.path.clone();
    let place = here.clone();
    let mut folded_set = state.folded;
    let mut chosen = state.selected;
    let summary = what(node);
    rsx! {
        div { class: "flow-node", key: "{here}",
            div { class: "flow-row",
                "data-selected": if selected { "true" } else { "false" },
                "data-route": if on_route { "true" } else { "false" },
                if leaf {
                    span { class: "flow-spacer" }
                } else {
                    button { class: "flow-toggle", title: if folded { "Unfold" } else { "Fold" },
                        onclick: move |_| {
                            let mut set = folded_set.write();
                            if !set.remove(&place) {
                                set.insert(place.clone());
                            }
                        },
                        {icon(if folded { Icon::ChevronRight } else { Icon::ChevronDown })}
                    }
                }
                span { class: "flow-label", onclick: move |_| chosen.set(Some(path.clone())),
                    span { class: "flow-kind", "data-kind": kind, "{kind}" }
                    span { class: "flow-name", "{name}" }
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
                    {node.children.iter().map(|child| branch(tree, *child, &here, route, state))}
                }
            }
        }
    }
}

/// The selected node in full: what it does, its rules and its file.
fn details(tree: &FlowTree, path: &str, builtin: bool, dir: &Path) -> Element {
    let Some(id) = tree.find(path) else {
        return rsx! {};
    };
    let node = tree.node(id);
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
                pre { class: "code", "{text}" }
            }
        }
    }
}

#[component]
pub fn FlowsPage(rev: u64) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    let mut parent = use_signal(|| "dictate".to_string());
    let mut name = use_signal(String::new);
    // What creating a branch did: Ok with what to do next, or the error.
    let mut message = use_signal(|| None::<Result<String, String>>);
    let state = TreeState {
        folded: use_signal(BTreeSet::new),
        selected: use_signal(|| None::<String>),
    };
    let view = ctx.view.lock().expect("the view lock");
    let route: BTreeSet<String> = view
        .route
        .iter()
        .map(|step| step.node.trim_matches('/').to_string())
        .collect();
    let dir = view.config.flows_dir(&view.config_file);
    let errors: Vec<String> = view.flow_errors.iter().map(ToString::to_string).collect();
    let notes = view.flow_notes.clone();
    let tool_problems = view.tool_problems.clone();
    let tree = view.flows.clone();
    let context = view.context.clone();
    drop(view);

    let builtin = tree.source == defaults::BUILTIN;

    let parent_path = parent();
    let id = name();
    let valid = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_');
    let parent_ok = tree
        .find(&parent_path)
        .is_some_and(|p| tree.node(p).kind() == Kind::Decide);
    // The decisions a new branch can go under (not shared folders, whose branches others take).
    let decisions: Vec<Choice> = tree
        .nodes()
        .iter()
        .filter(|n| n.kind() == Kind::Decide && !n.path.starts_with('_'))
        .map(|n| Choice {
            value: Some(n.path.clone()),
            label: if n.path.is_empty() {
                "/ (the root)".into()
            } else {
                n.path.clone()
            },
        })
        .collect();
    let shared = tree.nodes().iter().any(|n| n.path.starts_with("_actions/"));
    let draft = context.as_ref().map(|c| {
        let title = if c.window.title.is_empty() { c.app.process_name.clone() } else { c.window.title.clone() };
        if shared {
            format!(
                "description = {title:?}\npriority = 10\nbranches = \"_actions\"\ninstructions = \"Describe how text should read here.\"\n\n{}",
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
                        rsx! {
                            p { class: "muted", "Matches {context.app.process_name} · {context.window.title}" }
                            div { class: "row",
                                Select { value: Some(parent_path.clone()), choices: decisions.clone(),
                                    onchange: move |v: Option<String>| parent.set(v.unwrap_or_default()) }
                                input { class: "dx-input", placeholder: "branch name, such as slack", value: "{id}",
                                    oninput: move |e| name.set(e.value()) }
                                button { class: "dx-button", "data-style": "accent", "data-size": "sm", disabled: !valid || !parent_ok,
                                    onclick: move |_| {
                                        let folder = dir.join(parent()).join(name());
                                        let file = folder.join(file_name);
                                        message.set(Some(if folder.exists() {
                                            Err(format!("{} already exists", folder.display()))
                                        } else {
                                            match std::fs::create_dir_all(&folder).and_then(|()| std::fs::write(&file, &text)) {
                                                Ok(()) => {
                                                    create_ctx.send(Command::ReloadFlows);
                                                    open_folder(&folder);
                                                    Ok(format!("Created {}: describe it and adjust its guard", file.display()))
                                                }
                                                Err(e) => Err(format!("Cannot create {}: {e}", file.display())),
                                            }
                                        }));
                                    },
                                    "Create"
                                }
                            }
                            if !id.is_empty() && !valid {
                                p { class: "error-text", "A branch name uses lowercase letters, digits, - and _ only." }
                            } else {
                                p { class: "muted", "Under the decision chosen on the left, as a folder named in lowercase letters, digits, - and _." }
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
                        onclick: move |_| { let mut f = state.folded; f.write().clear(); },
                        "Unfold all"
                    }
                    button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                        onclick: move |_| {
                            let mut all = Vec::new();
                            // The root stays open, so its branches show.
                            for child in &fold_tree.node(fold_tree.root()).children {
                                places(&fold_tree, *child, &format!(">{}", fold_tree.node(fold_tree.root()).path), &mut all);
                            }
                            let mut f = state.folded;
                            f.set(all.into_iter().collect());
                        },
                        "Fold all"
                    }
                }
            }
            div { class: "dx-card-content",
                div { class: "flow-tree",
                    {(!tree.nodes().is_empty()).then(|| branch(&tree, tree.root(), "", &route, state))}
                }
            }
        }
        if let Some(path) = (state.selected)() {
            {details(&tree, &path, builtin, &dir)}
        }
    }
}
