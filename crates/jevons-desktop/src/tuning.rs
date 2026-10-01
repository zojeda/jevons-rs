//! Notices when CubeCL autotunes GPU kernels: the first runs of a model on a machine tune every
//! new shape, which can take minutes and looks like a hang. The tray shows it instead.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Tuning counts as ongoing until this long after its last event.
const QUIET_MS: u64 = 4000;

static LAST: AtomicU64 = AtomicU64::new(0);
static NOTIFY: OnceLock<Mutex<Box<dyn Fn() + Send>>> = OnceLock::new();

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Whether kernels were tuned in the last few seconds.
pub fn active() -> bool {
    now_ms().saturating_sub(LAST.load(Ordering::Relaxed)) < QUIET_MS
}

/// Calls `notify` when tuning starts after a quiet period.
pub fn on_start(notify: impl Fn() + Send + 'static) {
    let _ = NOTIFY.set(Mutex::new(Box::new(notify)));
}

fn tuned() {
    let started = !active();
    LAST.store(now_ms(), Ordering::Relaxed);
    if started && let Some(notify) = NOTIFY.get() {
        (notify.lock().expect("the tuning lock"))();
    }
}

/// Watches every event, whatever the log filter, for CubeCL's autotuner.
pub struct Layer;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Layer {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        if event
            .metadata()
            .target()
            .starts_with("cubecl_runtime::tune")
        {
            tuned();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tuning_event_makes_tuning_active_and_notifies_once() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        on_start(move || {
            counted.fetch_add(1, Ordering::Relaxed);
        });
        LAST.store(0, Ordering::Relaxed);
        assert!(!active());
        tuned();
        tuned();
        assert!(active());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
