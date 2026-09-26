//! Tray icon frames, drawn in code as 32×32 RGBA (ported from bot-rs): a microphone when idle,
//! a waveform that follows the microphone level while listening, and advancing dots while the
//! take is transcribed or generated.

use crate::levels::MAX_LEVEL;
use std::time::Duration;

pub const SIZE: u32 = 32;
/// How often the processing dots advance.
pub const FRAME: Duration = Duration::from_millis(180);

/// What the tray shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrayState {
    #[default]
    Idle,
    /// Capturing; `level` is the loudest meter band.
    Listening {
        level: u8,
    },
    /// Waiting for the transcript.
    Transcribing {
        frame: u8,
    },
    /// Deciding and generating.
    Thinking {
        frame: u8,
    },
    Error,
    /// The runtime is loading models or unreachable.
    Offline,
}

impl TrayState {
    /// The tooltip text.
    pub fn tooltip(self) -> &'static str {
        match self {
            Self::Idle => "jevons: ready",
            Self::Listening { .. } => "jevons: listening…",
            Self::Transcribing { .. } => "jevons: transcribing…",
            Self::Thinking { .. } => "jevons: writing…",
            Self::Error => "jevons: the last take failed",
            Self::Offline => "jevons: runtime not ready",
        }
    }

    /// Whether the icon animates on a timer (listening animates with the audio level).
    pub fn animates(self) -> bool {
        matches!(self, Self::Transcribing { .. } | Self::Thinking { .. })
    }

    /// The same state one animation frame later.
    pub fn advance(self) -> Self {
        match self {
            Self::Transcribing { frame } => Self::Transcribing {
                frame: (frame + 1) % 3,
            },
            Self::Thinking { frame } => Self::Thinking {
                frame: (frame + 1) % 3,
            },
            other => other,
        }
    }
}

/// Every distinct frame, so a tray can build its icons once and swap them.
pub fn frames() -> Vec<(TrayState, Vec<u8>)> {
    let mut frames = vec![
        (TrayState::Idle, microphone([200, 200, 210, 255])),
        (TrayState::Error, microphone([255, 110, 110, 255])),
        (TrayState::Offline, microphone([120, 120, 130, 255])),
    ];
    frames.extend((0..=MAX_LEVEL).map(|level| (TrayState::Listening { level }, wave(level))));
    frames.extend((0..3).map(|frame| {
        (
            TrayState::Transcribing { frame },
            dots(frame, [190, 255, 176, 255]),
        )
    }));
    frames.extend((0..3).map(|frame| {
        (
            TrayState::Thinking { frame },
            dots(frame, [190, 169, 255, 255]),
        )
    }));
    frames
}

/// The index of `state`'s frame in [`frames`].
pub fn frame_index(state: TrayState) -> usize {
    let listening = 3;
    let transcribing = listening + usize::from(MAX_LEVEL) + 1;
    match state {
        TrayState::Idle => 0,
        TrayState::Error => 1,
        TrayState::Offline => 2,
        TrayState::Listening { level } => listening + usize::from(level.min(MAX_LEVEL)),
        TrayState::Transcribing { frame } => transcribing + usize::from(frame % 3),
        TrayState::Thinking { frame } => transcribing + 3 + usize::from(frame % 3),
    }
}

fn put(pixels: &mut [u8], x: usize, y: usize, color: [u8; 4]) {
    let i = (y * SIZE as usize + x) * 4;
    pixels[i..i + 4].copy_from_slice(&color);
}

fn microphone(color: [u8; 4]) -> Vec<u8> {
    let mut pixels = vec![0; (SIZE * SIZE * 4) as usize];
    for y in 4..28 {
        for x in 4..28 {
            let (dx, dy) = (x as i32 - 16, y as i32);
            // Capsule body.
            let body = dx.abs() <= 4 && (8..=18).contains(&dy)
                || dx * dx + (dy - 8).pow(2) <= 16
                || dx * dx + (dy - 18).pow(2) <= 16;
            // Holder arc, stem and base.
            let r2 = dx * dx + (dy - 16).pow(2);
            let arc = dy >= 16 && (64..=100).contains(&r2);
            let stem = dx.abs() <= 1 && (26..28).contains(&dy);
            if body || arc || stem {
                put(&mut pixels, x, y, color);
            }
        }
    }
    pixels
}

fn wave(level: u8) -> Vec<u8> {
    let mut pixels = vec![0; (SIZE * SIZE * 4) as usize];
    for i in 0..5 {
        let half =
            2 + (usize::from(level.min(MAX_LEVEL)) * [5, 9, 13, 9, 5][i]) / usize::from(MAX_LEVEL);
        for y in 16 - half..16 + half {
            for x in 3 + i * 6..6 + i * 6 {
                put(&mut pixels, x, y, [190, 255, 176, 255]);
            }
        }
    }
    pixels
}

fn dots(frame: u8, color: [u8; 4]) -> Vec<u8> {
    let dim = [color[0] / 2 + 20, color[1] / 2 + 20, color[2] / 2 + 20, 255];
    let mut pixels = vec![0; (SIZE * SIZE * 4) as usize];
    for dot in 0..3 {
        let active = usize::from(frame) == dot;
        let radius = if active { 4_i32 } else { 3_i32 };
        let center = 7 + dot as i32 * 9;
        for y in 12..=20_i32 {
            for x in center - 4..=center + 4 {
                let (dx, dy) = (x - center, y - 16);
                if dx * dx + dy * dy <= radius * radius {
                    put(
                        &mut pixels,
                        x as usize,
                        y as usize,
                        if active { color } else { dim },
                    );
                }
            }
        }
    }
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_state_indexes_its_own_frame() {
        let frames = frames();
        for (i, (state, pixels)) in frames.iter().enumerate() {
            assert_eq!(frame_index(*state), i, "{state:?}");
            assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
        }
    }

    #[test]
    fn louder_audio_draws_taller_bars() {
        let lit = |p: &[u8]| p.chunks(4).filter(|c| c[3] > 0).count();
        assert!(lit(&wave(16)) > lit(&wave(4)));
    }

    #[test]
    fn processing_states_cycle_through_three_frames() {
        let start = TrayState::Thinking { frame: 0 };
        assert_eq!(start.advance().advance().advance(), start);
        assert!(start.animates());
        assert!(!TrayState::Listening { level: 3 }.animates());
    }
}
