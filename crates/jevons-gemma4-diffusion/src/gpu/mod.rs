//! DiffusionGemma kernels on the shared CubeCL runtime. Device buffers and the GEMM live in
//! `jevons-kernels`, shared with the Burn models.
pub mod attention;
pub mod ops;
pub mod tune;
pub mod vision;

pub use jevons_kernels::{Buf, Gpu, Hip, cache_dir, gemm};

#[cfg(test)]
mod tests;
