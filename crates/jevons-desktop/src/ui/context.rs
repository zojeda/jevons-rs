//! The live context and the route it takes through the flow tree.

use super::Ctx;
use super::components::{Collapsible, Icon, JsonTree, Switch, badge, icon};
use super::interface::Interface;
use super::workbench::{self, Workbench};
use crate::agent::{Command, ExtractsProbe};
use dioxus::prelude::*;
use jevons_desktop_core::flow::Check;
use jevons_desktop_core::flow::walk::FlowStep;
use std::collections::BTreeSet;
use std::time::Duration;

#[component]
pub fn ContextPage(rev: u64, frozen: Signal<bool>) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    let mut raw = use_signal(|| false);
    let view = ctx.view.lock().expect("the view lock");
    let backends = format!(
        "context: {} · delivery: {}",
        view.context_backend, view.sink_backend
    );
    let paused = view.context_paused;
    let error = view.context_error.clone();
    let context = view.context.clone();
    let route = view.route.clone();
    let extracts = view.extracts.clone();
    drop(view);
    let reread = ctx.clone();
    let capture = ctx.clone();
    let is_frozen = frozen();
    // The extract the workbench edits; the flow tree's readings can pick one too.
    let chosen = use_signal(|| None::<String>);
    // An expression the interface browser sends to the workbench to try.
    let draft = use_signal(|| None::<String>);

    rsx! {
        div { class: "spread",
            div { class: "row",
                Switch {
                    checked: frozen(),
                    label: "Freeze".to_string(),
                    onchange: move |on| frozen.set(on),
                }
                button {
                    class: "dx-button",
                    "data-style": "outline",
                    "data-size": "sm",
                    onclick: move |_| {
                        frozen.set(true);
                        capture.send(Command::CaptureContextIn(Duration::from_secs(3)));
                    },
                    "Capture in 3 s"
                }
            }
            span { class: "muted", "{backends}" }
        }
        if paused {
            p { class: "warn", "Context capture is paused (tray menu)." }
        }
        if let Some(error) = error {
            p { class: "error-text", "{error}" }
        }
        match context {
            None => rsx! {
                div { class: "dx-card",
                    div { class: "dx-card-content",
                        p { class: "muted",
                            "Switch to another application: the focused window's context appears here. \
                             Use Capture in 3 s to switch to the application you want to inspect."
                        }
                    }
                }
            },
            Some(context) => {
                let value = serde_json::to_value(&context).unwrap_or_default();
                let mut rows: Vec<(String, String)> = vec![
                    ("Application".into(), context.app.process_name.clone()),
                    ("Window".into(), context.window.title.clone()),
                ];
                if let Some(url) = &context.url {
                    rows.push(("Address".into(), url.clone()));
                }
                if let Some(e) = &context.focused {
                    rows.push(("Role".into(), e.role.clone()));
                    rows.push(("Name".into(), e.name.clone()));
                    if let Some(id) = &e.automation_id {
                        rows.push(("Automation id".into(), id.clone()));
                    }
                    rows.push(("Editable".into(), if e.is_editable { "yes" } else { "no" }.into()));
                    if e.is_password {
                        rows.push(("Password".into(), "yes (its text is never read)".into()));
                    }
                    for (key, text) in [
                        ("Selection", &e.selection),
                        ("Before the caret", &e.before_caret),
                        ("After the caret", &e.after_caret),
                        ("Value", &e.value_excerpt),
                    ] {
                        if let Some(text) = text {
                            rows.push((key.into(), excerpt(text)));
                        }
                    }
                }
                for (key, text) in &context.extras {
                    rows.push((key.clone(), excerpt(text)));
                }
                let json = serde_json::to_string_pretty(&value).unwrap_or_default();
                rsx! {
                    div { class: "dx-card",
                        div { class: "dx-card-header",
                            div {
                                div { class: "dx-card-title", "Focused window" }
                                div { class: "dx-card-description",
                                    if is_frozen {
                                        "Frozen: the window captured when you froze the tab or pressed Capture in 3 s"
                                    } else {
                                        "What the accessibility layer reports for the window in front, twice a second"
                                    }
                                }
                            }
                            Switch {
                                checked: raw(),
                                label: "Raw JSON".to_string(),
                                onchange: move |on| raw.set(on),
                            }
                        }
                        div { class: "dx-card-content",
                            if raw() {
                                pre { class: "code", "{json}" }
                            } else {
                                div { class: "kv",
                                    {rows.into_iter().map(|(k, v)| rsx! {
                                        span { class: "k", "{k}" }
                                        span { class: "v", "{v}" }
                                    })}
                                }
                                if !context.errors.is_empty() {
                                    div { class: "stack",
                                        {context.errors.iter().map(|e| rsx! { p { class: "warn", "{e}" } })}
                                    }
                                }
                                Collapsible { title: "Snapshot as a tree".to_string(), subtitle: None, open: false,
                                    JsonTree { value }
                                }
                            }
                        }
                    }
                }
            }
        }
        // From the short answer to the tools: where a take goes, what the tree reads, then the
        // interface and the workbench right under it, which its selectors are tried in.
        if !route.is_empty() {
            {route_card(&route, "Where a take from here goes, by guards and rules alone, with each rule checked")}
        }
        {extracts_card(extracts, chosen, move || reread.send(Command::ReadExtracts))}
        Interface { rev, draft }
        Workbench { rev, chosen, draft }
    }
}

/// The flow tree's `[extract]` values in this window, as a take from here reads them.
fn extracts_card(
    probe: Option<ExtractsProbe>,
    mut chosen: Signal<Option<String>>,
    reread: impl FnMut() + 'static,
) -> Element {
    let mut reread = reread;
    let probe = probe.unwrap_or_default();
    let summary = if probe.reading {
        "Reading…".to_string()
    } else if probe.window.is_empty() {
        String::new()
    } else {
        format!("In {}", probe.window)
    };
    let open = probe.readings.len() <= 4;
    rsx! {
        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Read by the flow tree" }
                    div { class: "dx-card-description",
                        "The [extract] values a take from this window can use, read from the interface \
                         with no model; lazy ones only when a take needs them"
                    }
                }
                button {
                    class: "dx-button",
                    "data-style": "outline",
                    "data-size": "sm",
                    disabled: probe.reading,
                    onclick: move |_| reread(),
                    "Read again"
                }
            }
            div { class: "dx-card-content",
                if !summary.is_empty() {
                    p { class: "muted", "{summary}" }
                }
                if probe.readings.is_empty() && !probe.reading && !probe.window.is_empty() {
                    p { class: "muted", "No extract in the flow tree applies to this window." }
                }
                div { class: "dx-accordion",
                    {probe.readings.iter().map(|r| {
                        let mut about = vec![match r.found.matches {
                            1 => "1 match".to_string(),
                            n => format!("{n} matches"),
                        }];
                        about.push(format!("{} ms", r.ms));
                        if r.lazy {
                            about.push("lazy".into());
                        }
                        about.push(if r.node.is_empty() { "/".into() } else { r.node.clone() });
                        let text = shown(&r.found.value);
                        rsx! {
                            Collapsible { key: "{r.node}/{r.name}", title: r.name.clone(), subtitle: Some(about.join(" · ")), open,
                                div { class: "spread",
                                    p { class: "mono muted", "{r.xpath}" }
                                    button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                                        title: "Edit it in the Extracts card below",
                                        onclick: {
                                            let picked = workbench::key(&r.node, &r.name);
                                            move |_| chosen.set(Some(picked.clone()))
                                        },
                                        "Edit"
                                    }
                                }
                                if let Some(note) = &r.found.note {
                                    p { class: "warn", "{note}" }
                                }
                                if text.is_empty() {
                                    p { class: "muted", "(nothing)" }
                                } else {
                                    pre { class: "code", "{text}" }
                                }
                            }
                        }
                    })}
                }
            }
        }
    }
}

/// An extract's answer as text: a string as it is, a list one item per line.
fn shown(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::Null => String::new(),
        Value::String(s) => excerpt(s),
        Value::Array(items) if items.iter().all(Value::is_string) => items
            .iter()
            .filter_map(Value::as_str)
            .map(excerpt)
            .collect::<Vec<_>>()
            .join("\n"),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    }
}

fn excerpt(text: &str) -> String {
    let short: String = text.chars().take(300).collect();
    if short.len() < text.len() {
        format!("{short}…")
    } else {
        short
    }
}

/// The route the context takes before any model decision, with every guard checked.
/// A route through the flow tree, each decision with its branches, every rule checked and, when
/// the model was asked, each branch's probability. `about` says whose route it is.
pub fn route_card(route: &[FlowStep], about: &str) -> Element {
    rsx! { RouteTree { route: RouteProp(route.to_vec()), about: about.to_string() } }
}

/// A route, compared by what the tree shows of it.
#[derive(Clone)]
pub struct RouteProp(Vec<FlowStep>);

impl PartialEq for RouteProp {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self.0.iter().zip(&other.0).all(|(a, b)| {
                (&a.node, &a.chosen, &a.how, &a.branches, &a.probabilities)
                    == (&b.node, &b.chosen, &b.how, &b.branches, &b.probabilities)
            })
    }
}

/// The route as a tree, drawn like the Flows tab's: each decision's branches on a guide line, the
/// chosen one marked and the next decision nested under it, down to where the walk ended. A
/// branch's rule checks unfold under it.
#[component]
fn RouteTree(route: RouteProp, about: String) -> Element {
    let route = route.0;
    let unfolded = use_signal(BTreeSet::<String>::new);
    let path: Vec<String> = route.iter().filter_map(|s| s.chosen.clone()).collect();
    let end = route.last().and_then(|s| s.how.clone());
    let root = route.first();
    rsx! {
        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Route" }
                    div { class: "dx-card-description", "{about}" }
                }
                div { class: "row",
                    {badge(&if path.is_empty() { "/".to_string() } else { path.join(" / ") }, "accent")}
                }
            }
            div { class: "dx-card-content",
                if let Some(end) = end {
                    p { class: "muted", "{end}" }
                }
                if let Some(first) = root {
                    div { class: "flow-tree",
                        div { class: "flow-node",
                            div { class: "flow-row", "data-route": "true",
                                span { class: "flow-spacer" }
                                span { class: "flow-label",
                                    span { class: "flow-kind", "data-kind": kind_word(first.kind), "{kind_word(first.kind)}" }
                                    span { class: "flow-name", "{first.node}" }
                                    {step_notes(first)}
                                }
                            }
                            {decision(&route, 0, unfolded)}
                        }
                    }
                }
            }
        }
    }
}

fn kind_word(kind: jevons_desktop_core::flow::Kind) -> &'static str {
    use jevons_desktop_core::flow::Kind;
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

/// How a step chose (or that the walk ended there), and what it read.
fn step_notes(step: &FlowStep) -> Element {
    let how = match (&step.chosen, &step.how) {
        (_, Some(how)) => how.clone(),
        (None, None) if step.branches.is_empty() => "the walk ends here".into(),
        _ => String::new(),
    };
    let reads = step.extracts.len() + step.investigations.len();
    rsx! {
        if !how.is_empty() {
            span { class: "flow-what", "{how}" }
        }
        if reads > 0 {
            span { class: "flow-notes", "read {reads} from the screen" }
        }
    }
}

/// Step `i`'s branches; the chosen one heads the next step, nested beneath it.
fn decision(route: &[FlowStep], i: usize, unfolded: Signal<BTreeSet<String>>) -> Element {
    let step = &route[i];
    if step.branches.is_empty() {
        return rsx! {};
    }
    rsx! {
        div { class: "flow-children",
            {step.branches.iter().map(|b| {
                let chosen = step.chosen.as_deref() == Some(b.name.as_str());
                let next = chosen.then(|| route.get(i + 1)).flatten();
                let key = format!("{i}/{}", b.name);
                let open = unfolded.read().contains(&key);
                let probability = step.probabilities.get(&b.name).map(|p| format!("{:.0}%", p * 100.0));
                let failed: Vec<&str> = b.checks.iter().filter(|c| !c.passed).map(|c| c.rule).collect();
                let has_rules = !b.checks.is_empty() || !b.prefer.is_empty();
                let toggle = key.clone();
                let mut set = unfolded;
                rsx! {
                    div { class: "flow-node", key: "{key}",
                        div { class: "flow-row",
                            "data-route": if chosen { "true" } else { "false" },
                            "data-off": if b.passed { "false" } else { "true" },
                            if has_rules {
                                button { class: "flow-toggle",
                                    onclick: move |_| {
                                        let mut s = set.write();
                                        if !s.remove(&toggle) {
                                            s.insert(toggle.clone());
                                        }
                                    },
                                    {icon(if open { Icon::ChevronDown } else { Icon::ChevronRight })}
                                }
                            } else {
                                span { class: "flow-spacer" }
                            }
                            span { class: "flow-label",
                                {icon(if b.passed { Icon::Check } else { Icon::Cross })}
                                if let Some(next) = next {
                                    span { class: "flow-kind", "data-kind": kind_word(next.kind), "{kind_word(next.kind)}" }
                                }
                                span { class: "flow-name", "{b.name}" }
                                if let Some(p) = probability {
                                    {badge(&p, if chosen { "accent" } else { "secondary" })}
                                }
                                if b.preferred {
                                    {badge("preferred", "accent")}
                                }
                                if let Some(next) = next {
                                    {step_notes(next)}
                                } else if !failed.is_empty() {
                                    span { class: "flow-notes", "failed: {failed.join(\", \")}" }
                                }
                            }
                        }
                        if open {
                            div { class: "route-checks",
                                {checks(&b.checks)}
                                if !b.prefer.is_empty() {
                                    p { class: "muted", "[prefer]" }
                                    {checks(&b.prefer)}
                                }
                            }
                        }
                        if next.is_some() {
                            {decision(route, i + 1, unfolded)}
                        }
                    }
                }
            })}
        }
    }
}

fn checks(checks: &[Check]) -> Element {
    if checks.is_empty() {
        return rsx! { p { class: "muted", "No guard: always applies." } };
    }
    rsx! {
        div {
            {checks.iter().map(|check| {
                let mark = icon(if check.passed { Icon::Check } else { Icon::Cross });
                let value = check.value.clone().unwrap_or_else(|| "(nothing)".into());
                rsx! {
                    div { class: "check",
                        span { {mark} }
                        span { class: "mono", "{check.rule}" }
                        span { class: "mono", "{check.pattern}" }
                        span { "{value}" }
                    }
                }
            })}
        }
    }
}
