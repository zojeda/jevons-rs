//! The feedback bubble: a small window above the tray icon that shows what a take hears and
//! does, from the first words to the text delivered. It never takes the focus and lets clicks
//! through, so dictation keeps going to the application underneath.

use super::Ctx;
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
    let quiet = f.heard.is_empty() && f.partial.is_empty();
    rsx! {
        div { class: "bubble", "data-state": state,
            "data-tail": if anchor.icon_below { "down" } else { "up" },
            if !anchor.icon_below { {tail.clone()} }
            div { class: "bubble-head",
                span { class: "bubble-dot" }
                span { class: "bubble-status", "{f.status}" }
            }
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
            if anchor.icon_below { {tail} }
        }
    }
}
