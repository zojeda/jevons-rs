//! The recent takes, step by step.

use super::Ctx;
use super::components::{Collapsible, CopyButton, JsonTree, badge};
use super::context::route_card;
use dioxus::prelude::*;
use jevons_desktop_core::config::HotkeyMode;
use jevons_desktop_core::pipeline::Trace;
use jevons_desktop_core::platform::DeliveryOutcome;

#[component]
pub fn TakesPage(rev: u64) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    let view = ctx.view.lock().expect("the view lock");
    let traces: Vec<Trace> = view.traces.iter().cloned().collect();
    let hotkey = view.config.dictation.hotkey.clone();
    let how = match view.config.dictation.hotkey_mode {
        HotkeyMode::Hold => {
            format!("Hold {hotkey} while speaking; the text is inserted when you release it.")
        }
        HotkeyMode::Toggle => {
            format!("Press {hotkey}, speak, and press it again; the text is inserted then.")
        }
    };
    drop(view);
    let folder = jevons_desktop_core::config::user_dir().join("traces");

    rsx! {
        div { class: "spread",
            h2 { "Takes" }
            span { class: "muted", "Saved as JSON in {folder.display()}" }
        }
        if traces.is_empty() {
            div { class: "dx-card",
                div { class: "dx-card-content",
                    p { class: "muted", "No takes yet. {how}" }
                }
            }
        } else {
            div { class: "dx-accordion",
                {traces.into_iter().map(|trace| rsx! { TakeItem { key: "{trace.take}-{trace.turn:?}", trace: TraceProp(trace) } })}
            }
        }
    }
}

/// A trace compared by its take and turn, so an unchanged take keeps its open state.
#[derive(Clone)]
pub struct TraceProp(Trace);

impl PartialEq for TraceProp {
    fn eq(&self, other: &Self) -> bool {
        (self.0.take, self.0.turn, self.0.started_at_ms)
            == (other.0.take, other.0.turn, other.0.started_at_ms)
    }
}

#[component]
fn TakeItem(trace: TraceProp) -> Element {
    let trace = trace.0;
    let turn = trace.turn.map_or(String::new(), |t| format!(".{t} live"));
    let (status, style) = match &trace.error {
        Some(_) => ("failed", "destructive"),
        None if matches!(trace.delivery, Some(DeliveryOutcome::OnClipboard { .. })) => {
            ("clipboard", "warning")
        }
        None if trace.delivery == Some(DeliveryOutcome::Shown) => ("answered", "success"),
        None if trace.output.is_empty() => ("nothing typed", "secondary"),
        None => ("typed", "success"),
    };
    let title = format!(
        "#{}{turn} · {} · {} · {:.1} s",
        trace.take,
        ago(trace.started_at_ms),
        trace.context.app.process_name,
        trace.audio_seconds
    );
    let what: String = trace
        .error
        .clone()
        .unwrap_or_else(|| trace.output.chars().take(70).collect());
    let summary = format!("{status} · {what}");
    let mut rows: Vec<(&str, String)> = vec![("Transcript", trace.transcript.clone())];
    if let Some(path) = trace.transcription {
        rows.push((
            "Transcribed",
            match path {
                jevons_desktop_core::pipeline::TranscriptionPath::Realtime => {
                    "live, while speaking"
                }
                jevons_desktop_core::pipeline::TranscriptionPath::Upload => {
                    "after speaking (upload)"
                }
            }
            .into(),
        ));
    }
    rows.push(("Route", trace.route()));
    if let Some(leaf) = &trace.leaf {
        rows.push((
            "Leaf",
            format!("{} → {:?}, {:?}", leaf.node, leaf.output, leaf.action).to_lowercase(),
        ));
    }
    for call in &trace.calls {
        let confirmed = match call.confirmed {
            Some(true) => " (approved)",
            Some(false) => " (not approved)",
            None => "",
        };
        rows.push((
            "Tool call",
            format!(
                "{} {}{confirmed} → {}",
                call.tool,
                call.arguments,
                call.result.clone().unwrap_or_default()
            ),
        ));
    }
    rows.push(("Output", trace.output.clone()));
    if let Some(delivery) = &trace.delivery {
        rows.push((
            "Delivery",
            match delivery {
                DeliveryOutcome::Delivered { method } => format!("{method:?}"),
                DeliveryOutcome::OnClipboard { reason } => format!("on the clipboard: {reason}"),
                DeliveryOutcome::Shown => "shown in the bubble".into(),
            },
        ));
    }
    let timings: Vec<String> = trace
        .timings
        .iter()
        .map(|(step, ms)| format!("{step} {ms} ms"))
        .collect();
    rows.push(("Timings", timings.join(" · ")));
    let value = serde_json::to_value(&trace).unwrap_or_default();
    let json = serde_json::to_string_pretty(&value).unwrap_or_default();

    rsx! {
        Collapsible { title, subtitle: Some(summary), open: false, status: Some(trace.error.is_none()),
            div { class: "row", {badge(status, style)} }
            div { class: "kv",
                {rows.into_iter().map(|(k, v)| rsx! {
                    span { class: "k", "{k}" }
                    span { class: "v", "{v}" }
                })}
            }
            {trace.notes.iter().map(|note| rsx! { p { class: "warn", "{note}" } })}
            if let Some(error) = &trace.error {
                p { class: "error-text", "{error}" }
            }
            div { class: "row",
                CopyButton { text: json, label: "Copy trace as JSON".to_string() }
            }
            if !trace.flow.is_empty() {
                {route_card(&trace.flow, "How this take went through the flow tree: each decision, its branches' rules and, when the model was asked, their probabilities")}
            }
            Collapsible { title: "Full trace: context, route, decisions, prompt".to_string(), subtitle: None, open: false,
                JsonTree { value }
            }
        }
    }
}

/// How long ago a take started, such as "just now", "12 min ago" or "3 h ago".
fn ago(started_at_ms: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    let seconds = now.saturating_sub(started_at_ms) / 1000;
    match seconds {
        0..60 => "just now".into(),
        60..3600 => format!("{} min ago", seconds / 60),
        3600..86_400 => format!("{} h ago", seconds / 3600),
        _ => format!("{} d ago", seconds / 86_400),
    }
}
