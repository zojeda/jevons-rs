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

/// Returns the device's unused pooled memory when it is dropped. Declared as a model's last
/// field, it runs once the model's tensors are gone. Burn queues a dropped tensor's release:
/// the first sync runs what is queued, the cleanup then frees the pages, so nothing of a
/// dropped model stays reserved (on APUs device memory is system memory) and a process that
/// ends holds none of it. A device that fails here is left as it is: a panic in a release that
/// runs while its thread unwinds would abort the process in the middle of a call to the driver.
/// The log says what the device held reserved before and after, or why it released nothing: a
/// release that failed must not read as one that ran.
pub struct ReleaseOnDrop(Device);

impl ReleaseOnDrop {
    pub fn new(device: &Device) -> Self {
        Self(device.clone())
    }
}

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        let device = &self.0;
        let started = std::time::Instant::now();
        let reserved = || device.memory_pool_usage().map(|usage| usage.bytes_reserved);
        let released = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let before = reserved();
            // The cleanup runs also after a sync that failed: it frees what it can.
            let queued = device.sync();
            device.memory_cleanup();
            let freed = device.sync();
            queued.and(freed).map(|()| (before, reserved()))
        }));
        let ms = started.elapsed().as_millis() as u64;
        match released {
            Ok(Ok((reserved_before, reserved_after))) => {
                tracing::info!(
                    ?reserved_before,
                    ?reserved_after,
                    ms,
                    "Device memory released"
                );
            }
            Ok(Err(error)) => tracing::warn!(
                %error,
                ms,
                "The device did not release a dropped model's memory"
            ),
            Err(panic) => tracing::warn!(
                error = jevons_kernels::panic_message(&*panic),
                ms,
                "The device did not release a dropped model's memory"
            ),
        }
    }
}

/// The HIP device `index`. The first call points CubeCL's compiled-kernel and autotune cache at
/// [`cache_dir`]; cold starts compile kernels and autotune for minutes.
pub fn hip(index: usize) -> Device {
    static CONFIGURE: Once = Once::new();
    CONFIGURE.call_once(|| {
        use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        config.compilation.cache = true;
        let dir = cache_dir();
        jevons_kernels::discard_empty_caches(&dir);
        config.environment.path = CacheConfig::Directory(dir);
        // Another model's runtime (DiffusionGemma's kernels) may have configured CubeCL first
        // in this process; keep its configuration rather than panic.
        CubeClRuntimeConfig::try_set(config);
    });
    Device::rocm(index)
}
