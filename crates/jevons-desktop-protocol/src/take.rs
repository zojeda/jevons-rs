//! A take as it travels: the client's settings for it, and what it is doing while it runs.

use serde::{Deserialize, Serialize};

/// The client's settings for its takes (its `[dictation]`, and its `[voice]` switch).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TakeSettings {
    /// The language spoken; detected when left out.
    pub language: Option<String>,
    /// Ask the decision model at model decisions; when off, they take their fallback.
    pub decide: bool,
    /// The most tokens a generation writes, unless a flow node sets its own.
    pub max_output_tokens: u32,
    /// Whether a take may speak its answer. Off, a leaf that asks for voice answers in the
    /// bubble: a client with no speaker, or whose user muted it, says so here.
    pub voice: bool,
}

impl Default for TakeSettings {
    fn default() -> Self {
        Self {
            language: None,
            decide: true,
            max_output_tokens: 1024,
            voice: true,
        }
    }
}

/// What a take is doing, as it does it: for the tray, the bubble and the inspector.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Update {
    Level([u8; 5]),
    /// Words recognized so far in the phrase being spoken.
    Delta(String),
    /// Live dictation: every phrase finished so far; the phrase being spoken starts over.
    Heard(String),
    /// The take stopped listening by itself, at a quiet pause: the desk closes its microphone.
    StoppedListening,
    Transcribing,
    Thinking,
    /// A stage of processing started: a decision, an investigation, a generation, a tool call,
    /// an answer's speech.
    /// It runs until a [`Update::StageDone`]; stages nest (an agent's tool calls), so a done
    /// closes the latest stage still open.
    Stage(Stage),
    /// What the running stage is doing now, such as the element an investigation reads.
    Progress(String),
    /// The latest open stage ended: what it chose or produced, and whether it worked.
    StageDone {
        detail: String,
        /// The branch a decision took.
        chosen: Option<String>,
        ok: bool,
    },
    /// Generated text.
    Output(String),
    /// The text being generated is an answer for the bubble, not text for the application.
    Answering,
    /// The desk took the first stretch of the answer's speech: it is being read aloud.
    Speaking,
    /// The machines moved: where they are now, such as `search › answering`.
    State(String),
    /// The words are for a task that was waiting: its agent handed them to it. Until then a
    /// take is not part of any task's conversation.
    Task {
        instance: u64,
    },
}

/// What a stage does, for the bubble's icon and animation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    Deciding,
    Investigating,
    Writing,
    Answering,
    Calling,
    /// The answer is synthesized and handed to the desk to play.
    Speaking,
}

/// A stage as it starts.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Stage {
    pub kind: StageKind,
    /// A few words, such as `dictate` or `conversation`.
    pub label: String,
    /// A decision's branches, before it chooses.
    pub choices: Vec<String>,
}

impl Stage {
    pub fn new(kind: StageKind, label: impl Into<String>) -> Self {
        Self {
            kind,
            label: label.into(),
            choices: Vec::new(),
        }
    }
}
