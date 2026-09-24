//! Errors shared by the inference API and the model implementations.

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Could not load the model")]
    ModelLoad,
    #[error("Unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("The canvas forward returned no logits")]
    MissingLogits,
    #[error("Candidate logits must be finite and nonempty")]
    InvalidLogits,
    #[error("GPU backend failure: {0}")]
    Backend(String),
}
