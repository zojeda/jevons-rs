//! NVIDIA Nemotron-Labs-Diffusion: a Ministral-3 decoder trained for masked block diffusion.
#![forbid(unsafe_code)]

pub mod config;
mod model;
pub mod rope;

pub use model::Nemotron;
