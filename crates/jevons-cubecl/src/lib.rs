//! DiffusionGemma inference on CubeCL (HIP): GGUF loading, the Gemma 4 tokenizer, quantized
//! kernels and the prompt/canvas forward passes.
#![forbid(unsafe_code)]

pub use jevons_formats::{gguf, quant};
pub use jevons_tokenizer::gemma4 as tokenizer;
pub mod vision_input;

#[cfg(feature = "hip")]
pub mod gpu;
#[cfg(feature = "hip")]
pub mod model;
#[cfg(feature = "hip")]
pub mod q4k;
#[cfg(feature = "hip")]
pub mod vision;

#[cfg(feature = "hip")]
pub mod expert;
