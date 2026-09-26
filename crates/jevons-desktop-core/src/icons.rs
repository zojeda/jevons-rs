//! Icons, drawn in code as RGBA. The app icon is a text cursor speaking (voice arcs) on a violet
//! badge; the badge turns grey while the runtime is not ready and red after a failed take. While
//! listening the tray shows a waveform that follows the microphone level, and while the take is
//! transcribed or generated, advancing dots (both ported from bot-rs).

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
        (TrayState::Idle, badge(SIZE, VIOLET)),
        (TrayState::Error, badge(SIZE, RED)),
        (TrayState::Offline, badge(SIZE, GREY)),
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

/// Badge colours, top and bottom of the gradient.
type Gradient = ([f32; 3], [f32; 3]);
const VIOLET: Gradient = ([150.0, 116.0, 255.0], [88.0, 60.0, 214.0]);
const RED: Gradient = ([236.0, 112.0, 104.0], [186.0, 58.0, 58.0]);
const GREY: Gradient = ([128.0, 128.0, 140.0], [84.0, 84.0, 96.0]);

/// The app icon at `size`×`size`, for the window and the taskbar.
pub fn app_icon(size: u32) -> Vec<u8> {
    badge(size, VIOLET)
}

/// A rounded badge with the speaking-cursor glyph, anti-aliased with 4×4 supersampling.
fn badge(size: u32, (top, bottom): Gradient) -> Vec<u8> {
    const SAMPLES: u32 = 4;
    let mut pixels = vec![0; (size * size * 4) as usize];
    for py in 0..size {
        for px in 0..size {
            let mut rgb = [0.0f32; 3];
            let mut alpha = 0.0f32;
            for sy in 0..SAMPLES {
                for sx in 0..SAMPLES {
                    let x = (px as f32 + (sx as f32 + 0.5) / SAMPLES as f32) / size as f32;
                    let y = (py as f32 + (sy as f32 + 0.5) / SAMPLES as f32) / size as f32;
                    if !rounded_square(x, y) {
                        continue;
                    }
                    let color = if glyph(x, y) {
                        [255.0, 255.0, 255.0]
                    } else {
                        std::array::from_fn(|i| top[i] + (bottom[i] - top[i]) * y)
                    };
                    for i in 0..3 {
                        rgb[i] += color[i];
                    }
                    alpha += 1.0;
                }
            }
            if alpha > 0.0 {
                let i = ((py * size + px) * 4) as usize;
                for c in 0..3 {
                    pixels[i + c] = (rgb[c] / alpha).round() as u8;
                }
                pixels[i + 3] = (alpha / (SAMPLES * SAMPLES) as f32 * 255.0).round() as u8;
            }
        }
    }
    pixels
}

/// The badge: a square with rounded corners, filling the icon with a small margin.
fn rounded_square(x: f32, y: f32) -> bool {
    let (margin, radius) = (0.03, 0.24);
    let (lo, hi) = (margin + radius, 1.0 - margin - radius);
    let dx = (x.clamp(lo, hi) - x).abs();
    let dy = (y.clamp(lo, hi) - y).abs();
    x >= margin
        && x <= 1.0 - margin
        && y >= margin
        && y <= 1.0 - margin
        && dx * dx + dy * dy <= radius * radius
}

/// A text cursor (I-beam) with three voice arcs to its right.
fn glyph(x: f32, y: f32) -> bool {
    let stem_x = 0.34;
    let stem = (x - stem_x).abs() <= 0.045 && (0.24..=0.76).contains(&y);
    let serif =
        (x - stem_x).abs() <= 0.12 && ((y - 0.24).abs() <= 0.04 || (y - 0.76).abs() <= 0.04);
    let (cx, cy) = (0.40, 0.5);
    let (dx, dy) = (x - cx, y - cy);
    let r = (dx * dx + dy * dy).sqrt();
    let within_angle = dx > 0.0 && dy.abs() <= dx * 1.1;
    let arc = within_angle
        && [0.2_f32, 0.31, 0.42]
            .iter()
            .any(|radius| (r - radius).abs() <= 0.035);
    stem || serif || arc
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
    fn the_app_icon_is_an_opaque_badge_with_transparent_corners() {
        let size = 64;
        let icon = app_icon(size);
        let alpha = |x: u32, y: u32| icon[((y * size + x) * 4 + 3) as usize];
        assert_eq!(alpha(0, 0), 0);
        assert_eq!(alpha(size / 2, size / 2 + 5), 255);
        // The glyph is white on the violet badge.
        let at = |x: u32, y: u32| {
            &icon[((y * size + x) * 4) as usize..((y * size + x) * 4 + 3) as usize]
        };
        assert_eq!(at((0.34 * size as f32) as u32, size / 2), [255, 255, 255]);
        assert_ne!(at(size / 8 + 2, size / 2), [255, 255, 255]);
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
