//! DiffusionGemma inference on CubeCL (HIP): quantized kernels, the prompt/canvas forward
//! passes as Burn operations over them, and the Gemma 4 vision encoder.
#![forbid(unsafe_code)]

pub use jevons_formats::{gguf, quant};
pub use jevons_tokenizer::gemma4 as tokenizer;
pub mod gpu;
pub mod model;
pub mod vision;
pub mod vision_input;
