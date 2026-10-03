//! The machines: what runs now (the root, each agent and the tasks it started), any machine's
//! diagram with its current state and latest transition marked, a state in full, and the latest
//! transitions. The diagram is laid out by
//! `flow::machine::layout`: edges, start and end dots and choice diamonds in one SVG (no SVG
//! text, which Blitz may not draw), states and labels as positioned boxes.

use super::Ctx;
use super::components::{Choice as Option_, Select, badge};
use crate::agent::Command;
use dioxus::prelude::*;
use jevons_desktop_server::flow::machine::layout::{self, Edge, Layout, NodeKind, Placed};
use jevons_desktop_server::flow::machine::runtime::{Running, Step, View as Machines};
use jevons_desktop_server::flow::machine::{
    self, Condition, DecidedBy, Decides, Level, Loaded, Target,
};
use jevons_desktop_server::flow::tree::{FlowTree, Node, NodeSpec};
use jevons_desktop_server::flow::{Kind, defaults};
use std::sync::Arc;

/// The edges' colours: the theme's quiet line, and the accent for the latest transition.
const LINE: &str = "#5d5d5d";
const BACK: &str = "#3e3e3e";
const HOT: &str = "#22e6f2";
/// And by what decides an edge: the event alone is the quiet line.
const BY_RULES: &str = "#6fbf7f";
const BY_RULES_THEN_MODEL: &str = "#d9a441";
const BY_MODEL: &str = "#b98ae6";

/// An edge's colour and its name in `data-by`, by what decides it.
fn by_colour(by: Option<DecidedBy>) -> (&'static str, &'static str) {
    match by {
        None => (LINE, "none"),
        Some(DecidedBy::Event) => (LINE, "event"),
        Some(DecidedBy::Rules) => (BY_RULES, "rules"),
        Some(DecidedBy::RulesThenModel) => (BY_RULES_THEN_MODEL, "rules_then_model"),
        Some(DecidedBy::Model) => (BY_MODEL, "model"),
    }
}
/// The most characters of an edge label drawn on the diagram (the layout leaves room for that
/// many); the rest is in its tooltip.
const LABEL: usize = jevons_desktop_server::flow::machine::layout::LABEL_CHARS;
/// The transitions the history card lists.
const HISTORY: usize = 40;

fn shorten(text: &str, most: usize) -> String {
    if text.chars().count() <= most {
        text.to_string()
    } else {
        let mut cut: String = text.chars().take(most - 1).collect();
        cut.push('…');
        cut
    }
}

/// A state's work, in a few words: its node's kind and what it calls, or `waits`.
fn work_badge(work: Option<&Node>) -> String {
    let Some(node) = work else {
        return "waits".into();
    };
    match &node.spec {
        NodeSpec::Tool(t) => format!("tool · {}", t.tool),
        NodeSpec::Loop(a) => match a.tools.as_slice() {
            [] => "loop".into(),
            [one] => format!("loop · {one}"),
            many => format!("loop · {} tools", many.len()),
        },
        NodeSpec::Run(_) => "run".into(),
        NodeSpec::Generate(_) => "generate".into(),
        NodeSpec::Transcript(_) => "transcript".into(),
        NodeSpec::Decide(_) => "decide".into(),
        NodeSpec::Machine(_) => "machine".into(),
    }
}

/// The folder of a state's work, if it has one.
fn work_of<'a>(tree: &'a FlowTree, machine: &Node, state: &str) -> Option<&'a Node> {
    tree.children(machine.id).find(|c| c.name == state)
}

/// The machine nodes of the tree: the root first, then by folder.
fn machine_nodes(tree: &FlowTree) -> Vec<&Node> {
    let mut nodes: Vec<&Node> = tree
        .nodes()
        .iter()
        .filter(|n| n.kind() == Kind::Machine)
        .collect();
    nodes.sort_by(|a, b| (!a.path.is_empty(), &a.path).cmp(&(!b.path.is_empty(), &b.path)));
    nodes
}

/// An arrowhead at the polyline's end, as polygon points.
/// The edge an edge's transitions include `transition`.
fn holds(edge: &Edge, transition: Option<usize>) -> bool {
    transition.is_some_and(|t| edge.transitions.contains(&t))
}

/// The diagram: edges and dots in one SVG, states and labels as boxes over it. Each edge is
/// coloured by what decides it, from `decisions`.
fn diagram(
    tree: &FlowTree,
    machine: &Node,
    layout: &Layout,
    decisions: &[Decides],
    current: Option<&str>,
    hot: Option<usize>,
    mut selected: Signal<Option<String>>,
) -> Element {
    let (width, height) = (layout.width.ceil(), layout.height.ceil());
    let marks = layout.nodes.iter().filter_map(|p| {
        let (cx, cy) = (p.x + p.width / 2.0, p.y + p.height / 2.0);
        match p.kind {
            NodeKind::Start => Some(rsx! {
                circle { key: "{p.key}", cx: "{cx}", cy: "{cy}", r: "{p.width / 2.0}", fill: "#d4d4d4" }
            }),
            NodeKind::End => Some(rsx! {
                g { key: "{p.key}",
                    circle { cx: "{cx}", cy: "{cy}", r: "{p.width / 2.0}", fill: "none", stroke: "#d4d4d4", stroke_width: "2" }
                    circle { cx: "{cx}", cy: "{cy}", r: "{p.width / 2.0 - 5.0}", fill: "#d4d4d4" }
                }
            }),
            NodeKind::Choice => {
                let (x, y, w, h) = (p.x, p.y, p.width, p.height);
                let points = format!(
                    "{:.1},{:.1} {:.1},{:.1} {:.1},{:.1} {:.1},{:.1}",
                    x + w / 2.0, y, x + w, y + h / 2.0, x + w / 2.0, y + h, x, y + h / 2.0
                );
                Some(rsx! {
                    polygon { key: "{p.key}", points: "{points}", fill: "#1a1a1a", stroke: "#a1a1a1", stroke_width: "1.5" }
                })
            }
            NodeKind::State => None,
        }
    });
    let lines = layout.edges.iter().enumerate().map(|(i, edge)| {
        let hot = holds(edge, hot);
        let (decided, by) = by_colour(edge.decided_by(decisions));
        let color = if hot {
            HOT
        } else if edge.back && decided == LINE {
            BACK
        } else {
            decided
        };
        let d = edge.path();
        let head: String = edge
            .arrow()
            .iter()
            .flatten()
            .map(|(x, y)| format!("{x:.1},{y:.1} "))
            .collect();
        rsx! {
            g { key: "edge-{i}", "data-by": by,
                path { d: "{d}", fill: "none", stroke: color, stroke_width: if hot { "2.5" } else { "1.5" },
                    stroke_dasharray: if edge.back && !hot { "5 4" } else { "none" } }
                polygon { points: "{head}", fill: color }
            }
        }
    });
    let labels = layout
        .edges
        .iter()
        .enumerate()
        .filter(|(_, e)| !e.labels.is_empty())
        .map(|(i, edge)| {
            let full = edge.labels.join(" · ");
            let text = shorten(&full, LABEL);
            let (x, y) = edge.label_at;
            let hot = holds(edge, hot);
            rsx! {
                div { key: "label-{i}", class: "fsm-label", title: "{full}",
                    "data-hot": if hot { "true" } else { "false" },
                    style: "left: {x:.0}px; top: {y:.0}px;",
                    "{text}"
                }
            }
        });
    let choice_names = layout.nodes.iter().filter(|p| p.kind == NodeKind::Choice).map(|p: &Placed| {
        let name = p.key.trim_start_matches("<<").trim_end_matches(">>").to_string();
        let (x, y) = (p.x + p.width + 4.0, p.y + p.height / 2.0 - 8.0);
        rsx! {
            div { key: "choice-{p.key}", class: "fsm-choice-name", style: "left: {x:.0}px; top: {y:.0}px;", "{name}" }
        }
    });
    let boxes = layout.nodes.iter().filter(|p| p.kind == NodeKind::State).map(|p| {
        let work = work_badge(work_of(tree, machine, &p.key));
        let is_current = current == Some(p.key.as_str());
        let is_selected = selected.read().as_deref() == Some(p.key.as_str());
        let key = p.key.clone();
        rsx! {
            div { key: "state-{p.key}", class: "fsm-state",
                "data-current": if is_current { "true" } else { "false" },
                "data-selected": if is_selected { "true" } else { "false" },
                style: "left: {p.x:.0}px; top: {p.y:.0}px; width: {p.width:.0}px; height: {p.height:.0}px;",
                onclick: move |_| selected.set(Some(key.clone())),
                div { class: "fsm-name", "{p.key}" }
                div { class: "fsm-work", "{work}" }
            }
        }
    });
    rsx! {
        div { class: "fsm-scroll",
            div { class: "fsm-canvas", style: "width: {width}px; height: {height}px;",
                svg { class: "fsm-lines", width: "{width}", height: "{height}", view_box: "0 0 {width} {height}",
                    {lines}
                    {marks}
                }
                {labels}
                {choice_names}
                {boxes}
            }
        }
        div { class: "fsm-legend",
            span { class: "muted", "Decided by" }
            {[DecidedBy::Event, DecidedBy::Rules, DecidedBy::RulesThenModel, DecidedBy::Model].into_iter().map(|by| {
                let (colour, name) = by_colour(Some(by));
                rsx! {
                    span { key: "{name}", class: "fsm-legend-item",
                        span { class: "fsm-swatch", style: "background: {colour};" }
                        "{by}"
                    }
                }
            })}
        }
    }
}

/// A state in full: what it is, its work, where it can go and its timers.
fn state_card(
    tree: &FlowTree,
    machine: &Node,
    loaded: &Loaded,
    state: &str,
    current: bool,
) -> Element {
    let diagram = &loaded.diagram;
    let work = work_of(tree, machine, state);
    let description = work
        .and_then(|w| w.description().map(str::to_string))
        .or_else(|| diagram.state(state).and_then(|s| s.description.clone()));
    let out: Vec<(String, String, String)> = diagram
        .transitions
        .iter()
        .filter(|t| t.from == state)
        .map(|t| {
            let guard = match &t.condition {
                Condition::Always => String::new(),
                Condition::Else => "[else]".into(),
                Condition::Criterion(text) => format!("[{text}]"),
                Condition::Named(name) => {
                    match loaded.guards.get(name).and_then(|g| g.criterion.clone()) {
                        Some(criterion) => format!("[{name}]: {criterion}"),
                        None => format!("[{name}] (rules)"),
                    }
                }
            };
            let to = match &t.to {
                Target::State(name) => name.clone(),
                Target::Choice(name) => format!("<<{name}>>"),
                Target::End => "[*] (the task ends)".into(),
            };
            (t.event.to_string(), guard, to)
        })
        .collect();
    let timers: Vec<String> = diagram
        .timers_in(state)
        .iter()
        .map(|t| format!("{}: {:.0} s → {}", t.name, t.after.as_secs_f64(), t.event))
        .collect();
    let work_line = match work {
        Some(node) => format!("{} · {}", work_badge(Some(node)), node.file),
        None => "No folder: the state waits for an event".into(),
    };
    rsx! {
        div { class: "dx-card fsm-detail",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "{state}" }
                    div { class: "dx-card-description",
                        {description.unwrap_or_else(|| "No description".into())}
                    }
                }
                if current {
                    {badge("Here now", "success")}
                }
            }
            div { class: "dx-card-content",
                p { class: "mono", "{work_line}" }
                if out.is_empty() {
                    p { class: "muted", "No transitions leave it." }
                }
                {out.into_iter().enumerate().map(|(i, (event, guard, to))| rsx! {
                    div { key: "out-{i}", class: "fsm-out",
                        span { class: "flow-kind", "data-kind": "event", "{event}" }
                        span { class: "fsm-guard", "{guard}" }
                        span { class: "muted", "→ {to}" }
                    }
                })}
                {timers.into_iter().map(|t| rsx! { p { class: "muted", "Timer {t}" } })}
            }
        }
    }
}

/// One transition of the history.
fn step_row(i: usize, step: &Step) -> Element {
    let mut top: Vec<(&String, &f64)> = step.probabilities.iter().collect();
    top.sort_by(|a, b| b.1.total_cmp(a.1));
    let odds: Vec<String> = top
        .into_iter()
        .take(3)
        .map(|(label, p)| format!("{label} {p:.2}"))
        .collect();
    let through = if step.through.is_empty() {
        String::new()
    } else {
        format!(" via <<{}>>", step.through.join(">>, <<"))
    };
    rsx! {
        div { key: "step-{i}", class: "fsm-step",
            span { class: "mono muted", "{step.machine}" }
            span { class: "fsm-step-move", "{step.from} → {step.to}{through}" }
            span { class: "flow-kind", "data-kind": "event", "{step.event}" }
            span { class: "muted", "{step.how}" }
            if !odds.is_empty() {
                span { class: "mono muted", {odds.join(" · ")} }
            }
            if let Some(take) = step.take {
                span { class: "muted", "#{take}" }
            }
        }
    }
}

#[component]
pub fn MachinesPage(rev: u64) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    // The machine whose diagram shows, by folder; and which of its running ones, when picked.
    let mut picked = use_signal(|| None::<String>);
    let mut instance = use_signal(|| None::<u64>);
    let selected = use_signal(|| None::<String>);
    let view = ctx.view.lock().expect("the view lock");
    let machines: Machines = view.machines.view();
    let tree: Arc<FlowTree> = machines.tree.clone().unwrap_or_else(|| view.flows.clone());
    drop(view);

    let nodes = machine_nodes(&tree);
    let path = machines.path();
    let at_rest = machines.at_rest();
    let header = rsx! {
        div { class: "spread",
            div { class: "stack",
                h2 { "Machines" }
                span { class: "muted",
                    "The root hands each take to an agent. An agent routes it among its own states and the tasks it started, which last several takes."
                }
            }
            div { class: "row",
                if machines.busy {
                    {badge("Moving", "warning")}
                }
                span { class: "mono fsm-path", if path.is_empty() { "not started" } else { "{path}" } }
                button { class: "dx-button", "data-style": "outline", "data-size": "sm", disabled: at_rest,
                    onclick: {
                        let ctx = ctx.clone();
                        move |_| ctx.send(Command::CancelTask)
                    },
                    "Cancel all tasks"
                }
            }
        }
    };
    if nodes.is_empty() {
        return rsx! {
            {header}
            p { class: "muted", "The flows root is a decision, not a machine: the tree has no machines." }
        };
    }
    // The running machine that shows: the one picked while it runs, else the task the latest
    // take reached.
    let shown = instance()
        .filter(|id| machines.running(*id).is_some())
        .or(machines.focus);
    // Its folder, unless another machine was picked; the root when nothing runs but it.
    let current_folder = picked()
        .filter(|p| nodes.iter().any(|n| n.label() == p))
        .or_else(|| {
            shown
                .and_then(|id| machines.running(id))
                .map(|r| r.folder.clone())
        })
        .unwrap_or_else(|| nodes[0].label().to_string());
    let node = nodes
        .iter()
        .copied()
        .find(|n| n.label() == current_folder)
        .unwrap_or(nodes[0]);
    let choices: Vec<Option_> = nodes
        .iter()
        .map(|n| Option_ {
            value: Some(n.label().to_string()),
            label: format!(
                "{} · {}",
                if n.path.is_empty() {
                    "/ (the root)"
                } else {
                    &n.path
                },
                n.machine.as_ref().map_or("", |m| m.diagram.name.as_str())
            ),
        })
        .collect();
    // Of that machine's running instances, the one that shows, else its first.
    let running = shown
        .and_then(|id| machines.running(id))
        .filter(|r| r.folder == node.label())
        .or_else(|| machines.stack.iter().find(|r| r.folder == node.label()));
    let current = running.map(|r| r.state.clone());
    let hot = machines
        .history
        .iter()
        .rev()
        .find(|s| s.machine == node.label() && running.is_none_or(|r| r.id == s.instance))
        .and_then(|s| s.transition);
    // What runs: the root and the agents, each agent with its tasks under it.
    let rows: Vec<&Running> = machines
        .stack
        .iter()
        .filter(|r| r.parent.is_none())
        .flat_map(|r| std::iter::once(r).chain(machines.tasks(r.id)))
        .collect();
    let showing = running.map(|r| r.id);
    let runs = rows.into_iter().map(|r| {
        let (id, folder, task) = (r.id, r.folder.clone(), r.parent.is_some());
        let waiting = if r.waiting.is_empty() {
            String::new()
        } else {
            format!("waits for {}", r.waiting.join(", "))
        };
        let level = match r.level {
            Level::Root => "root",
            Level::Agent => "agent",
            Level::Task => "task",
        };
        let cancel = ctx.clone();
        rsx! {
            div { key: "run-{id}", class: "fsm-run", "data-level": level,
                "data-showing": if showing == Some(id) { "true" } else { "false" },
                onclick: move |_| {
                    picked.set(Some(folder.clone()));
                    instance.set(Some(id));
                },
                {badge(level, "secondary")}
                span { class: "fsm-run-name", "{r.label()}" }
                span { class: "mono fsm-path", "{r.state}" }
                span { class: "muted", "{waiting}" }
                if task {
                    button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                        onclick: move |event: MouseEvent| {
                            event.stop_propagation();
                            cancel.send(Command::CancelOneTask(id));
                        },
                        "Cancel"
                    }
                }
            }
        }
    });
    let content = match &node.machine {
        None => {
            let diagram = machine::diagram_file(&node.file);
            rsx! { p { class: "error-text", "{diagram} did not load" } }
        }
        Some(loaded) => {
            let drawn = layout::layout(&loaded.diagram);
            let decisions = loaded.diagram.decisions();
            let state = selected().filter(|s| loaded.diagram.state(s).is_some());
            let unsure = running.and_then(|r| r.unsure.clone().map(|u| (r.id, u)));
            let answer = ctx.clone();
            rsx! {
                if let Some((id, unsure)) = unsure {
                    div { class: "fsm-unsure",
                        div { class: "stack",
                            span { class: "fsm-unsure-title", "Unsure what you meant" }
                            span { class: "muted", "“{shorten(&unsure.said, 120)}” · {unsure.how}" }
                        }
                        div { class: "row",
                            {unsure.candidates.iter().map(|label| {
                                let (send, chosen) = (answer.clone(), label.clone());
                                // One string: a release build joins a text's parts with `+`.
                                let text = match unsure.probabilities.get(label) {
                                    Some(p) => format!("{label} {p:.2}"),
                                    None => label.clone(),
                                };
                                rsx! {
                                    button { key: "{label}", class: "dx-button fsm-answer", "data-style": "outline", "data-size": "sm",
                                        onclick: move |_| send.send(Command::AnswerDecision { instance: id, label: chosen.clone() }),
                                        "{text}"
                                    }
                                }
                            })}
                        }
                    }
                }
                {diagram(&tree, node, &drawn, &decisions, current.as_deref(), hot, selected)}
                if let Some(state) = state {
                    {state_card(&tree, node, loaded, &state, current.as_deref() == Some(state.as_str()))}
                }
            }
        }
    };
    let builtin = tree.source == defaults::BUILTIN;
    let waiting = running
        .map(|r| r.waiting.join(", "))
        .filter(|w| !w.is_empty());
    rsx! {
        {header}
        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Running" }
                    div { class: "dx-card-description", "The root, the agents and each agent's tasks. Select one to see its diagram." }
                }
            }
            div { class: "dx-card-content fsm-runs",
                {runs}
            }
        }
        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title",
                        {node.machine.as_ref().map_or_else(|| node.label().to_string(), |m| m.diagram.name.clone())}
                        if let Some(running) = running.filter(|r| r.parent.is_some()) {
                            span { class: "muted", " · {running.label()}" }
                        }
                    }
                    div { class: "dx-card-description",
                        match (&current, &waiting) {
                            (Some(state), Some(events)) => rsx! { "At {state}, waiting for {events}. Select a state to see it in full." },
                            (Some(state), None) => rsx! { "At {state}. Select a state to see it in full." },
                            _ => rsx! { "Not running. Select a state to see it in full." },
                        }
                    }
                }
                div { class: "row",
                    if builtin {
                        {badge("built-in", "secondary")}
                    }
                    Select { value: Some(node.label().to_string()), choices,
                        onchange: move |v: Option<String>| {
                            picked.set(v);
                            instance.set(None);
                        } }
                }
            }
            div { class: "dx-card-content",
                {content}
            }
        }
        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Transitions" }
                    div { class: "dx-card-description", "The latest first: what moved each machine, and why." }
                }
            }
            div { class: "dx-card-content",
                if machines.history.is_empty() {
                    p { class: "muted", "Nothing yet: speak, and the root machine moves." }
                }
                {machines.history.iter().rev().take(HISTORY).enumerate().map(|(i, s)| step_row(i, s))}
            }
        }
    }
}
