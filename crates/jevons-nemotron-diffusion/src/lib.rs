//! NVIDIA Nemotron-Labs-Diffusion: a Ministral-3 decoder trained for masked block diffusion.
#![forbid(unsafe_code)]

pub mod config;
pub mod image;
mod model;
pub mod rope;
pub mod vision;

pub use model::Nemotron;
