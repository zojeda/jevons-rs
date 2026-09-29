//! The feedback bubble: a small window above the tray icon that shows what a take hears and
//! does, from the first words to the text delivered. It never takes the focus and lets clicks
//! through, so dictation keeps going to the application underneath, except while it shows an
//! answer (with Copy, Insert and Close) or asks before a tool runs (Run or Cancel, also Enter
//! and Esc).

use super::Ctx;
use crate::agent::{BubbleAction, Command, Feedback};
use dioxus::prelude::*;

/// Where the bubble's arrow points: its distance from the bubble's left edge in logical pixels,
/// and whether the icon is below the bubble (a taskbar at the bottom of the screen).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub tail_x: f64,
    pub icon_below: bool,
}

/// The bubble's size in logical pixels.
pub const SIZE: (f64, f64) = (380.0, 190.0);
/// The arrow's width in logical pixels.
const TAIL: f64 = 18.0;

#[component]
pub fn Bubble() -> Element {
    let ctx = use_context::<Ctx>();
    let anchor = use_context::<Anchor>();
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
    let left = (anchor.tail_x - TAIL / 2.0).clamp(8.0, SIZE.0 - TAIL - 8.0);
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
    let head = rsx! {
        div { class: "bubble-head",
            span { class: "bubble-dot" }
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
        return answer(&ctx, &f, head, tail_top, tail_bottom);
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
            if !f.steps.is_empty() {
                div { class: "bubble-steps",
                    {f.steps.iter().map(|step| rsx! { span { class: "bubble-step", "{step}" } })}
                }
            }
            {tail_bottom}
        }
    }
}

/// A finished answer: the question, the answer and what to do with it.
fn answer(
    ctx: &Ctx,
    f: &Feedback,
    head: Element,
    tail_top: Option<Element>,
    tail_bottom: Option<Element>,
) -> Element {
    let (copy, insert, close) = (ctx.clone(), ctx.clone(), ctx.clone());
    let can_insert = f.window.is_some();
    let tail = if tail_bottom.is_some() { "down" } else { "up" };
    rsx! {
        div { class: "bubble", "data-state": "answer",
            "data-tail": tail,
            {tail_top}
            {head}
            if !f.heard.is_empty() {
                div { class: "bubble-asked", "{f.heard}" }
            }
            div { class: "bubble-answer", "{f.output}" }
            if f.done {
                div { class: "bubble-actions",
                button { class: "bubble-button", onclick: move |_| close.send(Command::Bubble(BubbleAction::Close)), "Close" }
                if can_insert {
                    button { class: "bubble-button", onclick: move |_| insert.send(Command::Bubble(BubbleAction::Insert)), "Insert" }
                }
                button { class: "bubble-button", "data-primary": "true", onclick: move |_| copy.send(Command::Bubble(BubbleAction::Copy)), "Copy" }
                }
            }
            {tail_bottom}
        }
    }
}
