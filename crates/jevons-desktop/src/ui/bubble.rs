//! The feedback bubble: a small window above the tray icon that shows what a take hears and
//! does, from the first words to the text delivered. It never takes the focus and lets clicks
//! through, so dictation keeps going to the application underneath, except while it shows an
//! answer (with Copy, Insert and Close) or asks before a tool runs (Run or Cancel, also Enter
//! and Esc). An answer opens a larger bubble, renders its Markdown, scrolls with the wheel and
//! stays until Close (or the next take).

use super::{Ctx, markdown};
use crate::agent::{BubbleAction, Command, Feedback, StageView};
use dioxus::prelude::*;
use jevons_desktop_core::pipeline::StageKind;

/// Where the bubble's arrow points: its distance from the bubble's left edge in logical pixels,
/// and whether the icon is below the bubble (a taskbar at the bottom of the screen); and the
/// bubble's width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub tail_x: f64,
    pub icon_below: bool,
    pub width: f64,
}

/// The bubble's size in logical pixels.
pub const SIZE: (f64, f64) = (380.0, 190.0);
/// The bubble's size while it shows an answer, which may be long.
pub const ANSWER_SIZE: (f64, f64) = (520.0, 460.0);
/// The arrow's width in logical pixels.
const TAIL: f64 = 18.0;

#[component]
pub fn Bubble() -> Element {
    let ctx = use_context::<Ctx>();
    let anchor = use_context::<Anchor>();
    // An answer shown as plain, selectable text instead of its formatting.
    let selecting = use_signal(|| false);
    let feedback = ctx.view.lock().expect("the view lock").feedback.clone();
    let Some(f) = feedback else {
        return rsx! { div { class: "bubble" } };
    };
    let state = if f.failed {
        "failed"
    } else if f.done {
        "done"
    } else if f.live {
        "live"
    } else {
        "busy"
    };
    let left = (anchor.tail_x - TAIL / 2.0).clamp(8.0, anchor.width - TAIL - 8.0);
    let tail = rsx! {
        svg {
            class: "bubble-tail",
            style: "left: {left}px",
            width: "{TAIL}",
            height: "9",
            view_box: "0 0 18 9",
            path {
                d: if anchor.icon_below { "M0 0L9 9L18 0Z" } else { "M0 9L9 0L18 9Z" },
                fill: "#22e6f2",
            }
        }
    };
    // The dot breathes while the take works.
    let glow = if f.animating() {
        let phase = (f.frame % 8) as f64 / 7.0;
        0.35 + 0.65 * (1.0 - (2.0 * phase - 1.0).abs())
    } else {
        1.0
    };
    let head = rsx! {
        div { class: "bubble-head",
            span { class: "bubble-dot", style: "opacity: {glow:.2}" }
            span { class: "bubble-status", "{f.status}" }
        }
    };
    let tail_top = (!anchor.icon_below).then(|| tail.clone());
    let tail_bottom = anchor.icon_below.then_some(tail);
    if let Some(call) = f.confirm.clone() {
        let run = ctx.clone();
        let cancel = ctx.clone();
        return rsx! {
            div { class: "bubble", "data-state": "confirm",
                "data-tail": if anchor.icon_below { "down" } else { "up" },
                {tail_top}
                {head}
                div { class: "bubble-question", "Run {call.tool}?" }
                pre { class: "bubble-args", "{call.arguments}" }
                div { class: "bubble-actions",
                    button { class: "bubble-button", onclick: move |_| cancel.send(Command::Confirmed(false)), "Cancel (Esc)" }
                    button { class: "bubble-button", "data-primary": "true", onclick: move |_| run.send(Command::Confirmed(true)), "Run (Enter)" }
                }
                {tail_bottom}
            }
        };
    }
    if f.answer {
        return answer(&ctx, &f, selecting, head, tail_top, tail_bottom);
    }
    let quiet = f.heard.is_empty() && f.partial.is_empty();
    rsx! {
        div { class: "bubble", "data-state": state,
            "data-tail": if anchor.icon_below { "down" } else { "up" },
            {tail_top}
            {head}
            div { class: "bubble-text",
                p {
                    if quiet && !f.done {
                        span { class: "bubble-hint", "Speak: the words appear here as they are recognized." }
                    }
                    span { "{f.heard}" }
                    if !f.partial.is_empty() {
                        span { class: "bubble-partial",
                            if f.heard.is_empty() { "{f.partial.trim_start()}" } else { " {f.partial.trim_start()}" }
                        }
                    }
                }
            }
            if !f.output.is_empty() {
                div { class: "bubble-output", "→ {f.output}" }
            }
            if !f.stages.is_empty() {
                div { class: "bubble-stages",
                    {f.stages.iter().rev().take(3).collect::<Vec<_>>().into_iter().rev().enumerate().map(|(i, stage)| stage_row(i, stage, f.frame))}
                }
            }
            {tail_bottom}
        }
    }
}

/// The most choices a decision row shows before `+n`.
const CHIPS: usize = 5;

/// One stage: what it is, its choices or progress, and a check, a cross or moving dots.
fn stage_row(index: usize, stage: &StageView, frame: u64) -> Element {
    let kind = match stage.kind {
        StageKind::Deciding => "decide",
        StageKind::Investigating => "read",
        StageKind::Writing => "write",
        StageKind::Answering => "answer",
        StageKind::Calling => "call",
        StageKind::Agent => "agent",
    };
    let state = match stage.ok {
        None => "running",
        Some(true) => "ok",
        Some(false) => "failed",
    };
    let mark = match stage.ok {
        None => {
            let lit = (frame / 2 % 3) as usize;
            rsx! {
                span { class: "bubble-dots",
                    {(0..3).map(|d| rsx! { span { key: "{d}", "data-on": if d == lit { "true" } else { "false" } } })}
                }
            }
        }
        Some(true) => rsx! {
            svg { class: "bubble-mark", width: "12", height: "12", view_box: "0 0 12 12",
                path { d: "M2 6.5L5 9.5L10 3", stroke: "#b6fae3", stroke_width: "1.8", fill: "none" }
            }
        },
        Some(false) => rsx! {
            svg { class: "bubble-mark", width: "12", height: "12", view_box: "0 0 12 12",
                path { d: "M3 3L9 9M9 3L3 9", stroke: "#ffb4b4", stroke_width: "1.8", fill: "none" }
            }
        },
    };
    // While a decision runs, its choices light up in turn; then the chosen one stays lit.
    let scanning = (frame / 3) as usize % stage.choices.len().max(1);
    let chips: Vec<Element> = stage
        .choices
        .iter()
        .take(CHIPS)
        .enumerate()
        .map(|(i, choice)| {
            let chosen = stage.chosen.as_deref() == Some(choice.as_str());
            let lit = stage.ok.is_none() && i == scanning;
            rsx! {
                span { key: "{choice}", class: "bubble-chip",
                    "data-chosen": if chosen { "true" } else { "false" },
                    "data-lit": if lit { "true" } else { "false" },
                    "{choice}"
                }
            }
        })
        .collect();
    let more = stage.choices.len().saturating_sub(CHIPS);
    // A decision among no choices (a fallback) still shows where it went.
    let chosen_only = stage
        .choices
        .is_empty()
        .then(|| stage.chosen.clone())
        .flatten();
    rsx! {
        div { key: "{index}-{stage.label}", class: "bubble-stage", "data-state": state,
            {mark}
            span { class: "bubble-stage-kind", "{kind}" }
            span { class: "bubble-stage-label", "{stage.label}" }
            {chips.into_iter()}
            if more > 0 {
                span { class: "bubble-chip", "+{more}" }
            }
            if let Some(chosen) = chosen_only {
                span { class: "bubble-chip", "data-chosen": "true", "{chosen}" }
            }
            if !stage.detail.is_empty() {
                span { class: "bubble-stage-detail", "{stage.detail}" }
            }
        }
    }
}

/// A finished answer: the question, the answer and what to do with it. Blitz selects text only
/// in text fields, so Select text shows the answer's Markdown in one, to select any part of it
/// and copy it with Ctrl+C.
fn answer(
    ctx: &Ctx,
    f: &Feedback,
    selecting: Signal<bool>,
    head: Element,
    tail_top: Option<Element>,
    tail_bottom: Option<Element>,
) -> Element {
    let (copy, raw, insert, close) = (ctx.clone(), ctx.clone(), ctx.clone(), ctx.clone());
    let can_insert = f.window.is_some();
    let tail = if tail_bottom.is_some() { "down" } else { "up" };
    let mut toggle = selecting;
    let select = selecting();
    rsx! {
        div { class: "bubble", "data-state": "answer",
            "data-tail": tail,
            {tail_top}
            {head}
            if !f.heard.is_empty() {
                div { class: "bubble-asked", "{f.heard}" }
            }
            div { class: "bubble-answer",
                if select {
                    textarea { class: "bubble-select", value: "{f.output}" }
                } else {
                    {markdown::render(&f.output)}
                }
            }
            if f.done {
                div { class: "bubble-actions",
                    button { class: "bubble-button", onclick: move |_| close.send(Command::Bubble(BubbleAction::Close)), "Close" }
                    if can_insert {
                        button { class: "bubble-button", onclick: move |_| insert.send(Command::Bubble(BubbleAction::Insert)), "Insert" }
                    }
                    button { class: "bubble-button", "data-on": if select { "true" } else { "false" },
                        onclick: move |_| toggle.set(!select),
                        if select { "Done selecting" } else { "Select text" }
                    }
                    button { class: "bubble-button", onclick: move |_| raw.send(Command::Bubble(BubbleAction::Copy { raw: true })), "Copy raw" }
                    button { class: "bubble-button", "data-primary": "true", onclick: move |_| copy.send(Command::Bubble(BubbleAction::Copy { raw: false })), "Copy" }
                }
            }
            {tail_bottom}
        }
    }
}
