//! Device buffers and CubeCL kernels for the DiffusionGemma runtime (HIP).
//!
//! Kernels launch in CubeCL's checked mode: every global array access is bounds-checked
//! against the real allocation size, so a wrong logical length can never read or write
//! outside a buffer. The crate therefore stays free of `unsafe`.
pub mod attention;
pub mod gemm;
pub mod ops;
pub mod tune;
pub mod vision;

use cubecl::{
    client::ComputeClient,
    hip::{AmdDevice, HipRuntime},
    prelude::*,
    server::Handle,
};
use half::f16;
use std::marker::PhantomData;

pub type Hip = HipRuntime;

/// Owns the compute client used by every buffer and kernel on one device.
#[derive(Clone)]
pub struct Gpu {
    pub client: ComputeClient<Hip>,
}

/// Raw device allocation with a logical element count for launch metadata.
#[derive(Clone)]
pub struct Buf {
    handle: Handle,
    len: usize,
}

impl Buf {
    /// Uploads raw little-endian bytes holding `len` elements.
    pub fn from_bytes(gpu: &Gpu, bytes: &[u8], len: usize) -> Self {
        Self {
            handle: gpu.client.create_from_slice(bytes),
            len,
        }
    }

    /// Array argument of `len` scalar elements (vectorized kernels divide by the vector size).
    pub fn arg(&self) -> ArrayArg<Hip> {
        TensorBinding::<Hip> {
            handle: self.handle.clone().binding(),
            strides: [1].into(),
            shape: [self.len].into(),
            runtime: PhantomData,
        }
        .into_array_arg()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Same allocation viewed with another logical element count (e.g. f16 pairs as words).
    pub fn with_len(&self, len: usize) -> Self {
        Self {
            handle: self.handle.clone(),
            len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Directory for compiled kernels and tuned launch plans: `DIFFUSION_CUBECL_CACHE`, else
/// `$XDG_CACHE_HOME/diffusion-cubecl`, else `~/.cache/diffusion-cubecl`.
pub fn cache_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    std::env::var_os("DIFFUSION_CUBECL_CACHE")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME").map(|d| PathBuf::from(d).join("diffusion-cubecl"))
        })
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/diffusion-cubecl"))
        })
}

/// Enables CubeCL's persistent kernel compilation cache (unless configured otherwise) so
/// kernels compiled once are reused by later processes (see [`cache_dir`]).
fn configure_compilation_cache() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
        if CubeClRuntimeConfig::storage().lock().is_some() {
            return;
        }
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        if config.compilation.cache.is_none()
            && let Some(dir) = cache_dir()
        {
            config.compilation.cache = Some(CacheConfig::File(dir));
        }
        CubeClRuntimeConfig::set(config);
    });
}

impl Gpu {
    pub fn new(device: usize) -> Result<Self, String> {
        configure_compilation_cache();
        let client = <Hip as cubecl::Runtime>::client(&AmdDevice::new(device));
        let hw = &client.properties().hardware;
        if hw.plane_size_min != 32 || hw.plane_size_max != 32 {
            return Err("The CubeCL backend requires 32-lane waves".into());
        }
        Ok(Self { client })
    }

    /// Uploads an owned word vector without an intermediate host copy.
    pub fn upload_u32_owned(&self, data: Vec<u32>) -> Buf {
        let len = data.len().max(1);
        let data = if data.is_empty() { vec![0] } else { data };
        Buf {
            handle: self.client.create(cubecl::bytes::Bytes::from_elems(data)),
            len,
        }
    }

    pub fn upload_u32(&self, data: &[u32]) -> Buf {
        Buf {
            handle: self.client.create_from_slice(bytes_of_u32(data)),
            len: data.len(),
        }
    }

    pub fn upload_f32(&self, data: &[f32]) -> Buf {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Buf {
            handle: self.client.create_from_slice(&bytes),
            len: data.len(),
        }
    }

    pub fn upload_f16(&self, data: &[f32]) -> Buf {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        Buf {
            handle: self.client.create_from_slice(&bytes),
            len: data.len(),
        }
    }

    /// Zero-filled buffer of `len` elements of `elem_size` bytes.
    pub fn zeros(&self, len: usize, elem_size: usize) -> Buf {
        Buf {
            handle: self
                .client
                .create_from_slice(&vec![0u8; len.max(1) * elem_size]),
            len,
        }
    }

    /// Uninitialized buffer of `len` elements of `elem_size` bytes.
    pub fn empty(&self, len: usize, elem_size: usize) -> Buf {
        Buf {
            handle: self.client.empty(len.max(1) * elem_size),
            len,
        }
    }

    pub fn read_f32(&self, buf: &Buf) -> Vec<f32> {
        let bytes = self
            .client
            .read_one(buf.handle.clone())
            .expect("device read");
        bytes[..buf.len * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    pub fn read_f16(&self, buf: &Buf) -> Vec<f32> {
        let bytes = self
            .client
            .read_one(buf.handle.clone())
            .expect("device read");
        bytes[..buf.len * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_le_bytes(*b).to_f32())
            .collect()
    }

    pub fn read_u32(&self, buf: &Buf) -> Vec<u32> {
        let bytes = self
            .client
            .read_one(buf.handle.clone())
            .expect("device read");
        bytes[..buf.len * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }

    /// Blocks until all queued work has completed.
    /// Waits for queued work, then returns unused pooled allocations to the device. On APUs,
    /// device memory is system memory, so a dropped model must not stay reserved.
    pub fn release_memory(&self) {
        self.sync();
        self.client.memory_cleanup();
        self.sync();
    }

    pub fn sync(&self) {
        cubecl::future::block_on(self.client.sync()).expect("device sync");
    }
}

fn bytes_of_u32(data: &[u32]) -> &[u8] {
    bytemuck::cast_slice(data)
}

#[cfg(test)]
mod tests;
