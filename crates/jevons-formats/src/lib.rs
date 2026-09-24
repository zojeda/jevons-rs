//! Model file formats: GGUF with GGML block quantization, and Hugging Face safetensors.
//!
//! Everything here is host-side and GPU-independent.
#![forbid(unsafe_code)]

pub mod gguf;
pub mod quant;
pub mod safetensors;
