//! The Generative service: free-form chat and text answers from a diffusion model, as the
//! OpenAI-compatible Chat Completions, Completions and Responses APIs serve them.
//!
//! [`Generate`] runs on the shared [`DiffusionEngine`](jevons_diffusion::DiffusionEngine): it frames
//! the conversation, reserves an optional bounded thought, and streams the answer as it is
//! decided, holding back text that could still become a stop sequence. [`tools`] adds tool calls
//! and structured answers: the next step and labelled arguments are restricted reads, and free
//! arguments are written as JSON and checked against the tool's schema.
#![forbid(unsafe_code)]

mod generate;
mod request;
pub mod tools;

pub use generate::Generate;
pub use request::{
    FinishReason, Generation, GenerationPrompt, GenerationRequest, MAX_STOP_SEQUENCES, Message,
    Role,
};
