//! Tap-to-toggle and hold-to-dictate on one hotkey, independent of how the platform delivers
//! key events (ported from bot-rs).
//!
//! A press shorter than [`HOLD_DURATION`] toggles dictation on release. Holding past it starts
//! a take at the deadline, and releasing finishes that take only.

use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const HOLD_DURATION: Duration = Duration::from_millis(350);

/// A registered hotkey binding.
pub type Source = u32;

/// The take a release may finish.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Take {
    pub id: u64,
    /// Whether the take is still opening or recording.
    pub capturing: bool,
}

struct Press {
    at: Instant,
    pending: bool,
    capture: Option<u64>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Release {
    None,
    /// A short tap: start dictation, or stop the current take.
    Toggle,
    /// The end of a hold: finish the take it started.
    Finish,
}

#[derive(Default)]
pub struct Gestures {
    down: HashMap<Source, Press>,
}

impl Gestures {
    /// Returns true only for a fresh, non-overlapping pending gesture.
    pub fn press(&mut self, source: Source, at: Instant) -> bool {
        if self.down.contains_key(&source) {
            return false;
        }
        let pending = self.down.is_empty();
        self.down.insert(
            source,
            Press {
                at,
                pending,
                capture: None,
            },
        );
        pending
    }

    /// When [`due`](Self::due) should be called next.
    pub fn next_tick(&self) -> Option<Instant> {
        self.down
            .values()
            .filter(|press| press.pending)
            .map(|press| press.at + HOLD_DURATION)
            .min()
    }

    /// A hold that reached the deadline and should start a take; consumed once.
    pub fn due(&mut self, now: Instant) -> Option<Source> {
        for (source, press) in &mut self.down {
            if press.pending && now >= press.at + HOLD_DURATION {
                press.pending = false;
                return Some(*source);
            }
        }
        None
    }

    /// Records that the hold on `source` started `take`.
    pub fn started(&mut self, source: Source, take: Take) {
        if take.capturing
            && let Some(press) = self.down.get_mut(&source)
        {
            press.capture = Some(take.id);
        }
    }

    /// A release can finish only the still-active take started by this press.
    pub fn release(&mut self, source: Source, at: Instant, take: Take) -> Release {
        let Some(press) = self.down.remove(&source) else {
            return Release::None;
        };
        if press.pending && at.saturating_duration_since(press.at) < HOLD_DURATION {
            Release::Toggle
        } else if press.capture == Some(take.id) && take.capturing {
            Release::Finish
        } else {
            Release::None
        }
    }

    /// Keeps down keys suppressed until release, but drops their take ownership.
    pub fn cancel_capture(&mut self) {
        for press in self.down.values_mut() {
            press.pending = false;
            press.capture = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: Source = 1;

    #[test]
    fn tap_toggles_and_hold_finishes() {
        let now = Instant::now();
        let mut gestures = Gestures::default();
        assert!(gestures.press(KEY, now));
        assert_eq!(gestures.next_tick(), Some(now + HOLD_DURATION));
        assert_eq!(gestures.due(now + Duration::from_millis(349)), None);
        assert_eq!(
            gestures.release(KEY, now + Duration::from_millis(100), Take::default()),
            Release::Toggle
        );
        assert_eq!(gestures.next_tick(), None);

        assert!(gestures.press(KEY, now));
        assert_eq!(gestures.due(now + HOLD_DURATION), Some(KEY));
        let take = Take {
            id: 7,
            capturing: true,
        };
        gestures.started(KEY, take);
        assert_eq!(
            gestures.release(KEY, now + Duration::from_secs(2), take),
            Release::Finish
        );
        assert_eq!(gestures.release(KEY, now, take), Release::None);
    }

    #[test]
    fn a_late_release_without_a_started_take_does_nothing() {
        let now = Instant::now();
        let mut gestures = Gestures::default();
        gestures.press(KEY, now);
        assert_eq!(
            gestures.release(KEY, now + HOLD_DURATION, Take::default()),
            Release::None
        );
        assert_eq!(gestures.due(now + Duration::from_secs(1)), None);
    }

    #[test]
    fn overlapping_keys_do_not_restart_the_deadline() {
        let now = Instant::now();
        let mut gestures = Gestures::default();
        assert!(gestures.press(KEY, now));
        assert!(!gestures.press(KEY, now + Duration::from_millis(300)));
        assert!(!gestures.press(2, now));
        assert_eq!(gestures.due(now + HOLD_DURATION), Some(KEY));
        assert_eq!(gestures.due(now + HOLD_DURATION), None);
    }

    #[test]
    fn a_stale_release_cannot_stop_a_newer_take() {
        let now = Instant::now();
        let mut gestures = Gestures::default();
        gestures.press(KEY, now);
        gestures.due(now + HOLD_DURATION);
        gestures.started(
            KEY,
            Take {
                id: 1,
                capturing: true,
            },
        );
        let newer = Take {
            id: 2,
            capturing: true,
        };
        assert_eq!(
            gestures.release(KEY, now + HOLD_DURATION, newer),
            Release::None
        );
    }

    #[test]
    fn cancellation_suppresses_a_pending_start() {
        let now = Instant::now();
        let mut gestures = Gestures::default();
        gestures.press(KEY, now);
        gestures.cancel_capture();
        assert_eq!(gestures.next_tick(), None);
        assert_eq!(gestures.due(now + HOLD_DURATION), None);
    }
}
