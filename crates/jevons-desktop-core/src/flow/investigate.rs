//! The context investigator's contract: a question about what is on screen, answered in a fixed
//! [`Shape`]. The walker asks it for every `[investigate.<name>]` a take reaches; the desktop
//! implements it over the platform's accessibility tree.

use super::shape::Shape;
use crate::context::ContextSnapshot;
use futures_util::future::BoxFuture;
use serde_json::Value;

/// One investigation.
#[derive(Clone, Debug)]
pub struct Inquiry<'a> {
    pub name: &'a str,
    /// The question, with its placeholders filled.
    pub question: String,
    pub shape: &'a Shape,
    /// Application globs it may read; empty for only the application the take started in.
    pub scope: &'a [String],
    pub max_steps: u32,
    /// The context when the take started.
    pub snapshot: &'a ContextSnapshot,
}

/// An investigation's answer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Found {
    /// The answer in the inquiry's shape; fields not found are `null`.
    pub value: Value,
    /// What it looked at, in order, for the trace.
    pub steps: Vec<String>,
    /// Why the answer may be incomplete, such as a window it was not allowed to read.
    pub note: Option<String>,
}

/// Answers inquiries.
pub trait Investigate: Send + Sync {
    fn investigate<'a>(&'a self, inquiry: Inquiry<'a>) -> BoxFuture<'a, Found>;
}
