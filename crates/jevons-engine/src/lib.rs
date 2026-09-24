//! Structured diffusion reads and bounded thought generation over any
//! [`DiffusionModel`](jevons_core::DiffusionModel).
//!
//! Configuration and read types form the public API. The engine handles token
//! preparation and sampling; model crates run the network. With the default `models` feature,
//! [`Engine::load`] detects and loads a supported architecture.
#![forbid(unsafe_code)]

mod engine;
#[cfg(test)]
mod fake;
mod probability;
mod sampler;

pub use engine::Engine;
pub use jevons_core::{
    DiffusionModel, Error, ImageInput, ModelConfig, ModelInfo, PrefillProfile, ReadOptions,
    ReadRequest, ReadResult, Result, Slot, SlotRead,
};
#[cfg(feature = "models")]
pub use jevons_models::{Architecture, resolve as resolve_architecture};
pub use probability::restricted_softmax;
