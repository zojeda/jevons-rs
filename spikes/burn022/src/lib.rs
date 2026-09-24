#![forbid(unsafe_code)]
//! Shared helpers for the Burn 0.22 spike binaries.

use burn::tensor::Device;
use std::time::Instant;

/// Points CubeCL's persistent environment (compiled kernels + autotune) at a fixed directory.
/// Must run before the first device is touched.
pub fn configure_cubecl_cache() {
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    let dir = std::env::var_os("SPIKE_CUBECL_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/jevons-burn022")
        });
    config.compilation.cache = true;
    config.environment.path = CacheConfig::Directory(dir);
    CubeClRuntimeConfig::set(config);
}

pub fn rocm() -> Device {
    configure_cubecl_cache();
    Device::rocm(0)
}

pub fn vm_hwm_kb() -> u64 {
    status_field("VmHWM:")
}
pub fn vm_rss_kb() -> u64 {
    status_field("VmRSS:")
}
fn status_field(name: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find(|l| l.starts_with(name))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

pub fn mem(device: &Device, label: &str) {
    match device.memory_pool_usage() {
        Some(u) => println!(
            "[mem] {label}: allocs={} in_use={:.1} MiB reserved={:.1} MiB",
            u.number_allocs,
            u.bytes_in_use as f64 / 1048576.0,
            u.bytes_reserved as f64 / 1048576.0
        ),
        None => println!("[mem] {label}: n/a"),
    }
}

/// Time `f` (which must end with a device sync) over `iters` runs after `warm` warmups; returns ms/iter.
pub fn bench(device: &Device, warm: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warm {
        f();
    }
    device.sync().unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    device.sync().unwrap();
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

/// Deterministic pseudo-random f32 in [-1, 1).
pub fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Like [`bench`] but syncs after every iteration (defeats cross-iteration fusion).
pub fn bench_sync(device: &Device, warm: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warm {
        f();
        device.sync().unwrap();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
        device.sync().unwrap();
    }
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}
