//! Validation errors for the wire contract and inference mapping errors.

use serde::Serialize;
use serde_json::{Value, json};

#[derive(Debug, Serialize, thiserror::Error)]
#[error("{msg}")]
pub struct ValidationError {
    pub loc: Vec<Value>,
    pub msg: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
}

impl ValidationError {
    pub fn new(path: &[&str], message: impl Into<String>, kind: &'static str) -> Self {
        Self {
            loc: path.iter().map(|s| json!(s)).collect(),
            msg: message.into(),
            kind,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MappingError {
    #[error("Not enough verified candidate codes for this request")]
    CandidateCodes,
    #[error("Inference returned an invalid number of slots or probabilities")]
    InvalidResult,
}
