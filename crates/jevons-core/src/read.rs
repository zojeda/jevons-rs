//! Restricted-canvas inputs and the diagnostics returned by a read.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub struct ReadOptions {
    pub steps: usize,
    pub samples: usize,
    pub think: usize,
    pub sequential: bool,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            steps: 1,
            samples: 1,
            think: 0,
            sequential: false,
        }
    }
}

impl ReadOptions {
    pub fn validate(self) -> crate::Result<()> {
        if !(1..=8).contains(&self.steps) || !(1..=32).contains(&self.samples) || self.think > 4096
        {
            return Err(crate::Error::InvalidInput(
                "Require steps=1..8, samples=1..32, think=0..4096".into(),
            ));
        }
        Ok(())
    }
}

/// Compressed image bytes, decoded on the inference worker with allocation limits.
#[derive(Clone, Debug)]
pub struct ImageInput {
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Slot {
    /// Fixed canvas text immediately before this slot.
    pub prefix: String,
    /// Each candidate must encode to exactly one distinct token.
    pub candidates: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReadRequest {
    pub prompt: String,
    pub slots: Vec<Slot>,
}

impl ReadRequest {
    pub fn scm(material: &str) -> Self {
        Self {
            prompt: format!(
                "{material}\n\nSCM means supplementary cementitious material. Use these answer codes:\nA = yes\nB = no"
            ),
            slots: vec![Slot {
                prefix: "Is this material an SCM?\nAnswer: ".into(),
                candidates: vec!["A".into(), "B".into()],
            }],
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SlotRead {
    pub canvas_position: usize,
    pub absolute_position: usize,
    pub initial_token: i32,
    pub candidate_tokens: Vec<i32>,
    pub logits: Vec<f64>,
    pub probabilities: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct ReadResult {
    pub slots: Vec<SlotRead>,
    pub prompt_tokens: usize,
    pub canvas_tokens: usize,
    pub output_tokens: usize,
    pub seed: u64,
    pub forward_ms: f64,
}
