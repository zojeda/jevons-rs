//! Icons, drawn in code as RGBA. The app icon is the jevons alien: a dark head with headphones,
//! glowing almond eyes and a waveform on its forehead, simplified from the artwork in
//! `crates/jevons-desktop/assets/jevons.png` so it reads at tray size. It glows cyan when ready,
//! blue while the models load, amber while GPU kernels are being tuned, grey when no model is
//! loaded and red after a failed take. While listening the tray shows a waveform that follows the microphone level, and
//! while the take is transcribed or generated, advancing dots (both ported from bot-rs).

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
    /// No models are loaded (none downloaded or selected, or the server is unreachable).
    Offline,
    /// The models are loading: not ready to dictate yet.
    Loading,
    /// GPU kernels are being autotuned for this model (first runs only).
    Tuning,
    /// A demonstration is being recorded.
    Recording,
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
            Self::Offline => "jevons: no models loaded (see the Models tab)",
            Self::Loading => "jevons: loading the models, not ready yet",
            Self::Tuning => {
                "jevons: tuning GPU kernels for this model. This happens on the first runs only; \
                 the results are saved"
            }
            Self::Recording => {
                "jevons: recording what you do (hold the record hotkey to say what the task is; \
                 stop from the tray menu)"
            }
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
        (TrayState::Idle, alien(SIZE, CYAN)),
        (TrayState::Error, alien(SIZE, RED)),
        (TrayState::Offline, alien(SIZE, GREY)),
        (TrayState::Tuning, alien(SIZE, AMBER)),
        (TrayState::Loading, alien(SIZE, BLUE)),
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
    frames.push((TrayState::Recording, alien(SIZE, MAGENTA)));
    frames
}

/// The index of `state`'s frame in [`frames`].
pub fn frame_index(state: TrayState) -> usize {
    let listening = 5;
    let transcribing = listening + usize::from(MAX_LEVEL) + 1;
    match state {
        TrayState::Idle => 0,
        TrayState::Error => 1,
        TrayState::Offline => 2,
        TrayState::Tuning => 3,
        TrayState::Loading => 4,
        TrayState::Listening { level } => listening + usize::from(level.min(MAX_LEVEL)),
        TrayState::Transcribing { frame } => transcribing + usize::from(frame % 3),
        TrayState::Thinking { frame } => transcribing + 3 + usize::from(frame % 3),
        TrayState::Recording => transcribing + 6,
    }
}

fn put(pixels: &mut [u8], x: usize, y: usize, color: [u8; 4]) {
    let i = (y * SIZE as usize + x) * 4;
    pixels[i..i + 4].copy_from_slice(&color);
}

/// Glow colours by state.
const CYAN: [f32; 3] = [34.0, 230.0, 242.0];
const AMBER: [f32; 3] = [255.0, 180.0, 60.0];
const RED: [f32; 3] = [255.0, 92.0, 92.0];
const GREY: [f32; 3] = [140.0, 150.0, 160.0];
const BLUE: [f32; 3] = [96.0, 132.0, 255.0];
const MAGENTA: [f32; 3] = [255.0, 72.0, 190.0];
/// The head and ear cups.
const DARK: [f32; 3] = [16.0, 28.0, 38.0];

/// The app icon at `size`×`size`, for the window and the taskbar.
pub fn app_icon(size: u32) -> Vec<u8> {
    alien(size, CYAN)
}

/// What covers a point of the alien.
#[derive(Clone, Copy, PartialEq)]
enum Paint {
    Glow,
    Dark,
    Clear,
}

/// The alien at `size`×`size` in `glow`, anti-aliased with 4×4 supersampling. Details thinner
/// than a pixel (the smile, the nose line) appear only from 64 px up.
fn alien(size: u32, glow: [f32; 3]) -> Vec<u8> {
    const SAMPLES: u32 = 4;
    let detailed = size >= 64;
    let mut pixels = vec![0; (size * size * 4) as usize];
    for py in 0..size {
        for px in 0..size {
            let mut rgb = [0.0f32; 3];
            let mut alpha = 0.0f32;
            for sy in 0..SAMPLES {
                for sx in 0..SAMPLES {
                    let x = (px as f32 + (sx as f32 + 0.5) / SAMPLES as f32) / size as f32;
                    let y = (py as f32 + (sy as f32 + 0.5) / SAMPLES as f32) / size as f32;
                    let color = match paint(x, y, detailed) {
                        Paint::Glow => glow,
                        // A faint tint of the glow keeps the head from reading as a hole.
                        Paint::Dark => std::array::from_fn(|i| DARK[i] + glow[i] * 0.08),
                        Paint::Clear => continue,
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

fn paint(x: f32, y: f32, detailed: bool) -> Paint {
    // The head: an egg narrowing towards the chin, with a glowing rim.
    let head = |margin: f32| {
        let (cx, cy, rx, ry) = (0.5, 0.47, 0.30 - margin, 0.37 - margin);
        let taper = 1.0 - 0.38 * ((y - cy) / ry).max(0.0);
        let dx = (x - cx) / (rx * taper);
        let dy = (y - cy) / ry;
        dx * dx + dy * dy <= 1.0
    };
    let in_head = head(0.0);
    let inside_rim = head(0.035);

    // Almond eyes, tilted up and outwards.
    let eye = |cx: f32, angle: f32| {
        let (dx, dy) = (x - cx, y - 0.56);
        let (s, c) = angle.sin_cos();
        let u = (dx * c + dy * s) / 0.095;
        let v = (-dx * s + dy * c) / 0.05;
        u * u + v * v <= 1.0
    };
    let eyes = eye(0.385, 0.5) || eye(0.615, -0.5);

    // The waveform on the forehead.
    let bars = [(0.43, 0.06), (0.5, 0.11), (0.57, 0.06)]
        .iter()
        .any(|&(bx, half)| (x - bx).abs() <= 0.022 && (y - 0.33).abs() <= half);

    // The nose line and smile, too thin for the tray.
    let nose = detailed && (x - 0.5).abs() <= 0.009 && (0.46..=0.62).contains(&y);
    let smile = detailed && {
        let (dx, dy) = (x - 0.5, y - 0.63);
        let r = (dx * dx + dy * dy).sqrt();
        dy > 0.03 && (r - 0.085).abs() <= 0.013
    };

    // Headphones: a band over the head and a cup on each side, with a glowing bar.
    let band = {
        let (dx, dy) = (x - 0.5, y - 0.47);
        let r = (dx * dx + dy * dy).sqrt();
        dy < -0.02 && (r - 0.42).abs() <= 0.03
    };
    // Rounded rectangles: half extents `hx`, `hy` and corner radius `r` around (cx, 0.52).
    let rounded = |cx: f32, hx: f32, hy: f32, r: f32| {
        let dx = ((x - cx).abs() - (hx - r)).max(0.0);
        let dy = ((y - 0.52).abs() - (hy - r)).max(0.0);
        dx * dx + dy * dy <= r * r
    };
    let cups = rounded(0.18, 0.065, 0.15, 0.055) || rounded(0.82, 0.065, 0.15, 0.055);
    let cup_bars = rounded(0.145, 0.016, 0.1, 0.016) || rounded(0.855, 0.016, 0.1, 0.016);

    if in_head {
        if !inside_rim || eyes || bars || nose || smile {
            Paint::Glow
        } else {
            Paint::Dark
        }
    } else if cup_bars || band {
        Paint::Glow
    } else if cups {
        Paint::Dark
    } else {
        Paint::Clear
    }
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
    fn the_app_icon_glows_at_the_eyes_on_a_dark_head_with_clear_corners() {
        let size = 64;
        let icon = app_icon(size);
        let at = |x: f32, y: f32| {
            let i = (((y * size as f32) as u32 * size + (x * size as f32) as u32) * 4) as usize;
            [icon[i], icon[i + 1], icon[i + 2], icon[i + 3]]
        };
        assert_eq!(at(0.02, 0.02)[3], 0, "corners are transparent");
        assert_eq!(at(0.385, 0.56), [34, 230, 242, 255], "the eyes glow cyan");
        let cheek = at(0.5, 0.75);
        assert_eq!(cheek[3], 255);
        assert!(cheek[1] < 60, "the head is dark: {cheek:?}");
    }

    #[test]
    fn tuning_has_its_own_amber_frame_and_explains_itself() {
        let frames = frames();
        let (_, pixels) = &frames[frame_index(TrayState::Tuning)];
        let lit = pixels
            .chunks(4)
            .find(|p| p[3] == 255 && p[0] > 200)
            .unwrap();
        assert!(lit[0] > lit[2], "amber: more red than blue");
        assert!(TrayState::Tuning.tooltip().contains("first runs"));
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
