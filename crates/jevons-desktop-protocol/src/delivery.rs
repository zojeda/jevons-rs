//! The vocabulary of a take's two ends: the audio the client captures, and how the text the
//! server wrote reaches the application.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The rate of [`AudioEvent::Chunk`] samples: the Realtime API's default PCM16 rate.
pub const SAMPLE_RATE: u32 = 24_000;

/// What an audio source reports while capturing.
#[derive(Clone, Debug, PartialEq)]
pub enum AudioEvent {
    /// Mono PCM16 at [`SAMPLE_RATE`], about 100 ms each.
    Chunk(Vec<i16>),
    /// The meter bands for the latest audio, 0..=16.
    Level([u8; 5]),
    /// The source ended by itself (a file finished, a device went away).
    Ended,
    Failed(String),
}

/// What to do with the text in the target.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// Insert at the caret.
    #[default]
    Insert,
    /// Replace the selection.
    Replace,
    /// Replace the selection, or the whole field when nothing is selected.
    Rewrite,
}

/// How the text reaches the target.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryMethod {
    /// Put the text on the clipboard and paste it, restoring the clipboard after.
    #[default]
    Paste,
    /// Type it key by key.
    Type,
    /// Set the element's value through the accessibility API.
    SetValue,
    /// Only copy it; the user pastes.
    Clipboard,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DeliveryRequest {
    pub action: Action,
    pub text: String,
    pub method: DeliveryMethod,
    /// Whether the field's text must be selected first (a rewrite with nothing selected).
    pub select_all: bool,
    /// Characters to delete before the caret first (live dictation correcting what it typed).
    pub erase: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum DeliveryOutcome {
    Delivered {
        method: DeliveryMethod,
    },
    /// The text is on the clipboard for the user to paste.
    OnClipboard {
        reason: String,
    },
    /// Shown in the feedback bubble, where the user can copy or insert it.
    Shown,
}
