//! The recent takes, step by step.

use super::Ctx;
use super::components::{Collapsible, JsonTree, badge, copy};
use dioxus::prelude::*;
use jevons_desktop_core::pipeline::Trace;
use jevons_desktop_core::platform::DeliveryOutcome;

#[component]
pub fn TakesPage(rev: u64) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    let view = ctx.view.lock().expect("the view lock");
    let traces: Vec<Trace> = view.traces.iter().cloned().collect();
    let hotkey = view.config.dictation.hotkey.clone();
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
                    p { class: "muted", "No takes yet. Hold {hotkey} while speaking; the text is inserted when you release it." }
                }
            }
        }
        div { class: "dx-accordion",
            {traces.into_iter().map(|trace| rsx! { TakeItem { key: "{trace.take}-{trace.turn:?}", trace: TraceProp(trace) } })}
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
        None if trace.output.is_empty() => ("nothing typed", "secondary"),
        None => ("typed", "success"),
    };
    let title = format!(
        "#{}{turn} · {} · {:.1} s",
        trace.take, trace.context.app.process_name, trace.audio_seconds
    );
    let summary: String = trace
        .error
        .clone()
        .unwrap_or_else(|| trace.output.chars().take(70).collect());
    let mut rows: Vec<(&str, String)> = vec![("Transcript", trace.transcript.clone())];
    if let Some(path) = trace.transcription {
        rows.push(("Transcribed by", format!("{path:?}")));
    }
    if let Some(resolution) = &trace.resolution {
        rows.push((
            "Profile",
            match &resolution.destination {
                Some(d) => format!("{} → {d}", resolution.profile),
                None => resolution.profile.clone(),
            },
        ));
    }
    if let Some(action) = trace.action {
        rows.push(("Action", format!("{action:?}")));
    }
    rows.push(("Output", trace.output.clone()));
    if let Some(delivery) = &trace.delivery {
        rows.push((
            "Delivery",
            match delivery {
                DeliveryOutcome::Delivered { method } => format!("{method:?}"),
                DeliveryOutcome::OnClipboard { reason } => format!("on the clipboard: {reason}"),
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
        Collapsible { title, subtitle: Some(summary), open: false,
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
                button {
                    class: "dx-button",
                    "data-style": "outline",
                    "data-size": "sm",
                    onclick: move |_| copy(&json),
                    "Copy trace as JSON"
                }
            }
            Collapsible { title: "Full trace: context, decision, prompt".to_string(), subtitle: None, open: false,
                JsonTree { value }
            }
        }
    }
}
