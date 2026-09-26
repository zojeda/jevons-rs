//! When it is safe to deliver text into the target (ported from bot-rs).
//!
//! The text is typed or pasted only into the window the take started in, once the user has
//! let go of every key, and within [`WAIT`]; otherwise it stays on the clipboard for the user
//! to paste.

use std::time::{Duration, Instant};

/// How long delivery waits for held keys (such as the hotkey itself) to be released.
pub const WAIT: Duration = Duration::from_secs(2);

/// A delivery waiting for the right moment.
#[derive(Clone, Copy, Debug)]
pub struct Pending {
    pub take: u64,
    /// The target window when the take started; 0 when unknown.
    pub window: u64,
    pub deadline: Instant,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Decision {
    /// A newer take replaced this one.
    Cancel,
    /// Keys are still held; ask again shortly.
    Wait,
    /// Leave the text on the clipboard and tell the user.
    Manual,
    /// Deliver now.
    Deliver,
}

impl Pending {
    pub fn new(take: u64, window: u64, now: Instant) -> Self {
        Self {
            take,
            window,
            deadline: now + WAIT,
        }
    }

    pub fn decide(&self, take: u64, foreground: u64, keys_down: bool, now: Instant) -> Decision {
        if self.take != take {
            Decision::Cancel
        } else if self.window == 0 || self.window != foreground {
            Decision::Manual
        } else if !keys_down {
            Decision::Deliver
        } else if now >= self.deadline {
            Decision::Manual
        } else {
            Decision::Wait
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paste_is_skipped_when_foreground_window_changed() {
        let now = Instant::now();
        let pending = Pending::new(1, 42, now);
        assert_eq!(pending.decide(1, 43, false, now), Decision::Manual);
        assert_eq!(
            Pending::new(1, 0, now).decide(1, 0, false, now),
            Decision::Manual
        );
    }

    #[test]
    fn delivery_waits_for_held_keys_until_the_deadline() {
        let now = Instant::now();
        let pending = Pending::new(1, 42, now);
        assert_eq!(pending.decide(1, 42, true, now), Decision::Wait);
        assert_eq!(pending.decide(1, 42, false, now), Decision::Deliver);
        assert_eq!(pending.decide(1, 42, true, now + WAIT), Decision::Manual);
    }

    #[test]
    fn a_newer_take_cancels_the_delivery() {
        let now = Instant::now();
        assert_eq!(
            Pending::new(1, 42, now).decide(2, 42, false, now),
            Decision::Cancel
        );
    }
}
