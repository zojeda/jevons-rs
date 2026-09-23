//! Structured diffusion reads, bounded thought generation and image prefill on the CubeCL/HIP
//! runtime.
//!
//! Configuration and read types form the public API. The engine handles token
//! preparation and inference phases; the `jevons-cubecl` crate runs the model.
#![forbid(unsafe_code)]

mod backend;
mod config;
mod cubecl;
mod denoise;
mod engine;
mod images;
mod probability;

pub use config::ModelConfig;
pub use engine::Engine;
pub use jevons_core::{Error, PrefillProfile, Result};
pub use jevons_core::{ImageInput, ReadOptions, ReadRequest, ReadResult, Slot, SlotRead};
pub use probability::restricted_softmax;
