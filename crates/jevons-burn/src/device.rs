//! HIP device selection and the persistent kernel and autotune cache.
use burn::tensor::Device;
use std::path::PathBuf;
use std::sync::Once;

/// `$JEVONS_BURN_CACHE`, or `~/.cache/jevons-burn` (the home directory is `HOME`, or
/// `USERPROFILE` on Windows).
pub fn cache_dir() -> PathBuf {
    std::env::var_os("JEVONS_BURN_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map_or_else(|| PathBuf::from("."), PathBuf::from);
            home.join(".cache/jevons-burn")
        })
}

/// The HIP device `index`. The first call points CubeCL's compiled-kernel and autotune cache at
/// [`cache_dir`]; cold starts compile kernels and autotune for minutes.
pub fn hip(index: usize) -> Device {
    static CONFIGURE: Once = Once::new();
    CONFIGURE.call_once(|| {
        use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        config.compilation.cache = true;
        config.environment.path = CacheConfig::Directory(cache_dir());
        CubeClRuntimeConfig::set(config);
    });
    Device::rocm(index)
}
