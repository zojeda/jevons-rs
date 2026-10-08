//! The machines: what runs now (the root, each agent and the tasks it started), any machine's
//! diagram with its current state and latest transition marked, a state in full, and the latest
//! transitions. The diagram is laid out by
//! `flow::machine::layout`: edges, start and end dots and choice diamonds in one SVG (no SVG
//! text, which Blitz may not draw), states and labels as positioned boxes.
//!
//! One state's box is open: the branches of its work, the flow tree under its folder, show
//! inside it in the Flows tab's rows, with the way a take went through them marked. The marks
//! come from the bubble's stages while the take runs ([`live_marks`]), and from the take's
//! trace once it ended ([`traced_marks`]). The layout makes room for the open box.
//!
//! A row selected shows its node in full under the diagram, as the Flows tab does: the
//! instructions it works under and its file.
//!
//! The diagram follows the machine that moved last, unless Follow is off: a take carries it
//! from the root to an agent to a task, and back to the root once the agent rests.

use super::Ctx;
use super::components::{Choice as Option_, Select, Switch, badge};
use super::flows::{self, Marks, TreeState};
use crate::agent::{Command, StageView};
use dioxus::prelude::*;
use jevons_desktop_server::flow::machine::layout::{self, Edge, Layout, NodeKind, Placed};
use jevons_desktop_server::flow::machine::runtime::{Running, Step, View as Machines};
use jevons_desktop_server::flow::machine::{
    self, Condition, DecidedBy, Decides, Level, Loaded, Target,
};
use jevons_desktop_server::flow::tree::{FlowTree, Node, NodeSpec};
use jevons_desktop_server::flow::walk::FlowStep;
use jevons_desktop_server::flow::{Kind, defaults};
use jevons_desktop_server::pipeline::StageKind;
use std::collections::BTreeMap;
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
/// The label of the stage in which the root decides which agent a take is for.
const ROOT_DECIDES: &str = "what to do";
/// An open state's box: its width, a row's height, and the room around its rows.
const OPEN_WIDTH: f32 = 420.0;
const OPEN_ROW: f32 = 26.0;
const OPEN_ROOM: f32 = 8.0;

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

/// The first of `stages` of a kind and a label.
fn stage_at(stages: &[StageView], kind: StageKind, label: &str) -> Option<usize> {
    stages
        .iter()
        .position(|s| s.kind == kind && s.label == label)
}

/// The marks of a walk still under way from `work` down, as far as the bubble's stages say. A
/// decision's stage carries its node's name, the branches it chose among, the one it took and
/// why; a tool's or a loop's that ended well leads on to the node's branch. The walk is at the last
/// node they lead to, doing what the stage still open says.
fn live_marks(tree: &FlowTree, work: &Node, stages: &[StageView]) -> Marks {
    let mut marks = Marks {
        folded: true,
        ..Marks::default()
    };
    let (mut node, mut here) = (work, flows::place("", work));
    marks.route.insert(here.clone());
    let mut rest = stages;
    loop {
        let (at, next) = match &node.spec {
            NodeSpec::Decide(_) => {
                let Some(at) = stage_at(rest, StageKind::Deciding, &node.name) else {
                    break;
                };
                let stage = &rest[at];
                // A decision with no branch that applies lists none, and takes its fallback.
                if !stage.choices.is_empty() {
                    let dropped = tree
                        .children(node.id)
                        .filter(|child| !stage.choices.contains(&child.name))
                        .map(|child| flows::place(&here, child));
                    marks.out.extend(dropped);
                }
                let taken = stage
                    .chosen
                    .as_ref()
                    .and_then(|chosen| tree.children(node.id).find(|child| child.name == *chosen));
                if let Some(child) = taken.filter(|_| !stage.detail.is_empty()) {
                    // The model's answer comes as its probability alone.
                    let why = match stage.detail.parse::<f64>() {
                        Ok(_) => format!("model {}", stage.detail),
                        Err(_) => stage.detail.clone(),
                    };
                    marks.why.insert(flows::place(&here, child), why);
                }
                (at, taken)
            }
            NodeSpec::Tool(_) | NodeSpec::Loop(_) => {
                let stage = match &node.spec {
                    NodeSpec::Tool(t) => stage_at(rest, StageKind::Calling, &t.tool),
                    _ => stage_at(rest, StageKind::Loop, &node.name),
                };
                let Some(at) = stage else { break };
                let branch = tree.children(node.id).next();
                (at, branch.filter(|_| rest[at].ok == Some(true)))
            }
            _ => break,
        };
        let Some(child) = next else { break };
        rest = &rest[at + 1..];
        here = flows::place(&here, child);
        marks.route.insert(here.clone());
        node = child;
    }
    marks.now = Some(here);
    marks.activity = doing(stages);
    marks
}

/// What a take does now, from its stage still open: `deciding`, `writing`,
/// `calling web_search`.
fn doing(stages: &[StageView]) -> Option<String> {
    let open = stages.iter().rev().find(|s| s.ok.is_none())?;
    let what = match open.kind {
        StageKind::Deciding => "deciding".to_string(),
        StageKind::Investigating => format!("reading {}", open.label),
        StageKind::Writing => "writing".into(),
        StageKind::Answering => "answering".into(),
        StageKind::Calling => format!("calling {}", open.label),
        StageKind::Loop => "working with tools".into(),
    };
    Some(if open.detail.is_empty() {
        what
    } else {
        format!("{what} · {}", open.detail)
    })
}

/// The marks of a walk that ended, from its take's route: the nodes from `work` down in the
/// order the walk entered them, the branches each decision could not choose (their guards
/// failed, or others were preferred), and why it took the one it took. `None` when the route
/// never reached `work`.
fn traced_marks(tree: &FlowTree, work: &Node, route: &[FlowStep]) -> Option<Marks> {
    let entered = |step: &FlowStep| step.kind != Kind::Machine && step.node == work.label();
    let walk = &route[route.iter().rposition(entered)?..];
    let mut marks = Marks {
        folded: true,
        ..Marks::default()
    };
    let way = flows::way(tree, work.id, walk.iter().map(|step| step.node.as_str()));
    for ((id, here), step) in way.iter().zip(walk) {
        marks.route.insert(here.clone());
        // With a preferred branch, a decision chooses among the preferred ones only.
        let preferred = step.branches.iter().any(|b| b.preferred);
        for child in tree.children(*id) {
            let below = flows::place(here, child);
            if step
                .branches
                .iter()
                .any(|b| b.name == child.name && !(b.passed && (b.preferred || !preferred)))
            {
                marks.out.insert(below.clone());
            }
            if step.chosen.as_deref() == Some(child.name.as_str())
                && let Some(how) = &step.how
            {
                marks.why.insert(below, how.clone());
            }
        }
    }
    Some(marks)
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

/// Where the machine shown is.
#[derive(Clone, Copy)]
struct Here<'a> {
    /// Its state, while it runs.
    state: Option<&'a str>,
    /// What that state's own node does now, while its work runs.
    working: Option<&'a str>,
    /// Its latest transition.
    hot: Option<usize>,
}

/// The state whose box is open: the branches of its work show inside it, with `marks` on
/// their rows.
#[derive(Clone, Copy)]
struct Open<'a> {
    state: &'a str,
    work: &'a Node,
    marks: &'a Marks,
    rows: TreeState,
}

/// The diagram: edges and dots in one SVG, states and labels as boxes over it. Each edge is
/// coloured by what decides it, from `decisions`. The `open` state's box holds its work's
/// branches under its name, and the current state's says what its work does while it runs.
fn diagram(
    tree: &Arc<FlowTree>,
    machine: &Node,
    layout: &Layout,
    decisions: &[Decides],
    here: Here,
    open: Option<Open>,
    mut selected: Signal<Option<String>>,
) -> Element {
    let Here {
        state: current,
        working,
        hot,
    } = here;
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
        let is_current = current == Some(p.key.as_str());
        let doing = working.filter(|_| is_current);
        let work = doing.map_or_else(|| work_badge(work_of(tree, machine, &p.key)), String::from);
        let is_selected = selected.read().as_deref() == Some(p.key.as_str());
        let key = p.key.clone();
        let inner = open.filter(|open| open.state == p.key).map(|open| {
            let at = flows::place("", open.work);
            let rows: Vec<Element> = open
                .work
                .children
                .iter()
                .map(|child| flows::branch(tree, *child, &at, open.marks, open.rows))
                .collect();
            rsx! {
                // A click on a row folds it: it is not a click on the state.
                div { class: "flow-tree fsm-inner",
                    onclick: move |event: MouseEvent| event.stop_propagation(),
                    {rows.into_iter()}
                }
            }
        });
        let is_open = inner.is_some();
        rsx! {
            div { key: "state-{p.key}", class: "fsm-state",
                "data-current": if is_current { "true" } else { "false" },
                "data-selected": if is_selected { "true" } else { "false" },
                "data-working": if doing.is_some() { "true" } else { "false" },
                "data-open": if is_open { "true" } else { "false" },
                style: "left: {p.x:.0}px; top: {p.y:.0}px; width: {p.width:.0}px; height: {p.height:.0}px;",
                // A second click on a state takes the selection back.
                div { class: "fsm-head",
                    onclick: move |_| selected.set((!is_selected).then(|| key.clone())),
                    div { class: "fsm-name", "{p.key}" }
                    div { class: "fsm-work", "{work}" }
                }
                {inner}
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

/// The line of a state whose folder is a machine of its own (an agent under the root, a task
/// under an agent), with a button that shows that machine through `show`.
fn own_machine(work: Option<&Node>, mut show: impl FnMut(String) + 'static) -> Element {
    let Some(work) = work.filter(|work| work.kind() == Kind::Machine) else {
        return rsx! {};
    };
    let what = match work.level() {
        Some(Level::Task) => "The task",
        _ => "The agent",
    };
    let name = work
        .machine
        .as_ref()
        .map_or(work.name.as_str(), |m| m.diagram.name.as_str());
    let folder = work.label().to_string();
    // One string: a release build joins a text's parts with `+`.
    let own = format!("{what} {name}, with a diagram of its own.");
    rsx! {
        div { class: "row",
            span { class: "muted", "{own}" }
            button { class: "dx-button fsm-show", "data-style": "outline", "data-size": "sm",
                onclick: move |_| show(folder.clone()),
                "Show it"
            }
        }
    }
}

/// A state in full: what it is, its work, where it can go, its timers, and `below` them the
/// machine its folder is.
fn state_card(
    tree: &FlowTree,
    machine: &Node,
    loaded: &Loaded,
    state: &str,
    current: bool,
    below: Element,
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
                {below}
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
    // The machine picked here, by folder, and which of its running ones; with the machines'
    // latest move when it was picked.
    let mut picked = use_signal(|| None::<String>);
    let mut instance = use_signal(|| None::<u64>);
    let mut picked_at = use_signal(|| None::<Step>);
    // Whether the diagram follows the machine that moved last.
    let mut follow = use_signal(|| true);
    let selected = use_signal(|| None::<String>);
    // The rows of a state's work: those folded or unfolded by hand, and the one selected.
    let picked_row = use_signal(|| None::<String>);
    let folds = TreeState {
        opened: use_signal(BTreeMap::new),
        selected: picked_row,
        compact: true,
    };
    let view = ctx.view.lock().expect("the view lock");
    let machines: Machines = view.machines.view();
    let tree: Arc<FlowTree> = machines.tree.clone().unwrap_or_else(|| view.flows.clone());
    let flows_dir = view.config.flows_dir(&view.config_file);
    let latest = machines.history.back().cloned();
    // While the tab follows, a pick holds until the machines move again.
    let following = follow();
    let holds = |at: &Option<Step>| !following || *at == latest;
    // The machine picked in the tray menu, while it runs and its pick holds.
    let asked = view
        .open_machine
        .as_ref()
        .filter(|pick| holds(&pick.moved))
        .and_then(|pick| machines.running(pick.instance))
        .map(|r| (r.id, r.folder.clone()));
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
    // The running machine that shows: the one picked while it runs, else the one that moved
    // last (or its caller, once it rests) while the tab follows, else the task the latest take
    // reached.
    // What was picked here is read either way: a render that does not read it would not
    // follow the next pick.
    let (here, folder, at) = (instance(), picked(), picked_at());
    let (here, folder) = if holds(&at) {
        (here, folder)
    } else {
        (None, None)
    };
    let moving = latest
        .as_ref()
        .filter(|_| following)
        .and_then(|step| machines.running(step.instance))
        .map(|r| {
            // An agent back in its first state, with nothing to ask, gives way to the root
            // that called it: what is said next goes there first.
            let rests = r.level == Level::Agent
                && r.unsure.is_none()
                && tree
                    .node(r.machine)
                    .machine
                    .as_ref()
                    .is_some_and(|m| m.diagram.initial == r.state);
            let root = machines.stack.first().filter(|_| rests);
            root.map_or(r.id, |root| root.id)
        });
    let shown = asked
        .as_ref()
        .map(|(id, _)| *id)
        .or(here.filter(|id| machines.running(*id).is_some()))
        .or(moving)
        .or(machines.focus);
    // Its folder, unless another machine was picked; the root when nothing runs but it.
    let current_folder = asked
        .map(|(_, folder)| folder)
        .or(folder)
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
    // The moves of the machine shown, the latest first.
    let mine = |s: &Step| s.machine == node.label() && running.is_none_or(|r| r.id == s.instance);
    let moves = || machines.history.iter().rev().filter(|s| mine(s));
    let hot = moves().next().and_then(|s| s.transition);
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
        let (pick, cancel, at) = (ctx.clone(), ctx.clone(), latest.clone());
        rsx! {
            div { key: "run-{id}", class: "fsm-run", "data-level": level,
                "data-showing": if showing == Some(id) { "true" } else { "false" },
                onclick: move |_| {
                    pick.view.lock().expect("the view lock").open_machine = None;
                    picked.set(Some(folder.clone()));
                    instance.set(Some(id));
                    picked_at.set(at.clone());
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
            let decisions = loaded.diagram.decisions();
            // The work of a state, when its folder is a flow to walk; and whether that flow
            // has branches to show.
            let work = |state: &str| {
                work_of(&tree, node, state).filter(|work| work.kind() != Kind::Machine)
            };
            let branches = |state: &&str| work(state).is_some_and(|w| !w.children.is_empty());
            let chosen = selected().filter(|s| loaded.diagram.state(s).is_some());
            let (live, opened, marks) = {
                let view = ctx.view.lock().expect("the view lock");
                // The machine works in its state while the take that entered it still runs.
                let live = current.as_deref().and_then(|at| {
                    let feedback = view.feedback.as_ref().filter(|f| !f.done)?;
                    moves()
                        .next()
                        .filter(|s| s.take == Some(feedback.take) && s.to == at)?;
                    Some(live_marks(&tree, work(at)?, &feedback.stages))
                });
                // The state whose box is open, among those whose work has branches: the one
                // the machine works in, else the one selected, else the latest it entered or
                // left, else its first.
                let opened: Option<String> = current
                    .as_deref()
                    .filter(|_| live.is_some())
                    .filter(branches)
                    .or_else(|| chosen.as_deref().filter(branches))
                    .or_else(|| {
                        moves()
                            .flat_map(|s| [s.to.as_str(), s.from.as_str()])
                            .find(branches)
                    })
                    .or_else(|| {
                        let mut states = loaded.diagram.states.iter().map(|s| s.name.as_str());
                        states.find(branches)
                    })
                    .map(String::from);
                let marks = match (&opened, &live) {
                    (Some(state), Some(live)) if current.as_ref() == Some(state) => live.clone(),
                    // The latest take kept that moved this machine and walked the state's work.
                    (Some(state), _) => work(state)
                        .and_then(|work| {
                            view.traces
                                .iter()
                                .filter(|t| t.machine.iter().any(&mine))
                                .find_map(|t| traced_marks(&tree, work, &t.flow))
                        })
                        .unwrap_or(Marks {
                            folded: true,
                            ..Marks::default()
                        }),
                    (None, _) => Marks::default(),
                };
                (live, opened, marks)
            };
            let open = opened.as_deref().and_then(|state| {
                Some(Open {
                    state,
                    work: work(state)?,
                    marks: &marks,
                    rows: folds,
                })
            });
            // The layout makes room for the open box: a row for each branch that shows.
            let mut sizes = BTreeMap::new();
            if let Some(open) = open {
                let at = flows::place("", open.work);
                let folded = folds.opened.read();
                let rows: usize = open
                    .work
                    .children
                    .iter()
                    .map(|child| flows::rows(&tree, *child, &at, &marks, &folded))
                    .sum();
                let height = layout::STATE_HEIGHT + rows as f32 * OPEN_ROW + OPEN_ROOM;
                sizes.insert(open.state.to_string(), (OPEN_WIDTH, height));
            }
            let drawn = layout::layout_with(&loaded.diagram, &sizes);
            // The state's own line says what its work does while the walk is at its node, or
            // has no row to say it on.
            let working = live.as_ref().and_then(|live| {
                let at = current.as_deref().and_then(work)?;
                let head = live.now.as_deref() == Some(flows::place("", at).as_str());
                let shows = open.is_some_and(|open| Some(open.state) == current.as_deref());
                live.activity.clone().filter(|_| head || !shows)
            });
            // Or that the machine itself decides where what was said goes.
            let working = working.or_else(|| {
                let view = ctx.view.lock().expect("the view lock");
                let stages = &view.feedback.as_ref().filter(|f| !f.done)?.stages;
                let open = stages.iter().rev().find(|s| s.ok.is_none())?;
                let own = format!("{} at {}", loaded.diagram.name, current.as_deref()?);
                let decides = open.kind == StageKind::Deciding
                    && (open.label == own || (open.label == ROOT_DECIDES && node.path.is_empty()));
                decides.then(|| "deciding".to_string())
            });
            let show = {
                let (ctx, at) = (ctx.clone(), latest.clone());
                move |folder: String| {
                    ctx.view.lock().expect("the view lock").open_machine = None;
                    picked.set(Some(folder));
                    instance.set(None);
                    picked_at.set(at.clone());
                }
            };
            let here = Here {
                state: current.as_deref(),
                working: working.as_deref(),
                hot,
            };
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
                {diagram(&tree, node, &drawn, &decisions, here, open, selected)}
                if let Some(state) = chosen {
                    {state_card(
                        &tree,
                        node,
                        loaded,
                        &state,
                        current.as_deref() == Some(state.as_str()),
                        own_machine(work_of(&tree, node, &state), show),
                    )}
                }
                if let Some(place) = picked_row() {
                    div { class: "fsm-node",
                        {flows::details(&tree, &place, tree.source == defaults::BUILTIN, &flows_dir)}
                    }
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
                    Switch { checked: following, label: "Follow".to_string(),
                        onchange: {
                            let (ctx, at) = (ctx.clone(), latest.clone());
                            let folder = node.label().to_string();
                            move |on: bool| {
                                // Turned off, the tab stays on the machine it shows.
                                if !on {
                                    ctx.view.lock().expect("the view lock").open_machine = None;
                                    picked.set(Some(folder.clone()));
                                    instance.set(showing);
                                    picked_at.set(at.clone());
                                }
                                follow.set(on);
                            }
                        } }
                    Select { value: Some(node.label().to_string()), choices,
                        onchange: {
                            let (ctx, at) = (ctx.clone(), latest.clone());
                            move |v: Option<String>| {
                                ctx.view.lock().expect("the view lock").open_machine = None;
                                picked.set(v);
                                instance.set(None);
                                picked_at.set(at.clone());
                            }
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
