//! Shared Burn runtime for the diffusion models: HIP device setup, weight streaming, and the
//! transformer building blocks (RMS norm, rotary embedding, grouped-query attention over a
//! resident KV cache).
//!
//! Numerics: the residual stream and norms are f32; weights (FP16 for the tuned GEMM), matmul
//! inputs, attention inputs and the KV cache are FP16 (Burn's flash kernel computes in FP16).
#![forbid(unsafe_code)]

pub mod device;
pub mod kernels;
pub mod layers;
pub mod weights;

pub use burn::tensor::activation;
pub use burn::tensor::module;
pub use burn::tensor::ops::ConvOptions;
pub use burn::tensor::{Bool, DType, Device, Int, Tensor, TensorData};
