//! The diffusion layer shared by the Generative and Decision services.
//!
//! [`DiffusionEngine`] owns a loaded [`DiffusionModel`](jevons_core::DiffusionModel) (DiffusionGemma
//! on CubeCL, Nemotron-Labs-Diffusion on Burn) with its chat framing, verified answer codes and
//! context limits, and generates bounded token runs, such as thoughts before an answer or read, in
//! every [`Decoding`] mode. The services add their own policy on top: free-form answers
//! (`jevons-generative`) and restricted-canvas reads (`jevons-decision`). With the default
//! `models` feature, [`DiffusionEngine::load`] detects and loads a supported architecture.
#![forbid(unsafe_code)]

mod engine;
#[cfg(any(test, feature = "testing"))]
pub mod fake;
mod sampler;

pub use engine::{Decoding, DiffusionEngine, Generated, Sink, Thought, conditioning};
#[cfg(feature = "models")]
pub use jevons_models::{Architecture, default_model_id, resolve as resolve_architecture};
pub use sampler::{masked, uniform};
