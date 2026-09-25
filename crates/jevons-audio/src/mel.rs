//! Log-mel spectrogram features, matching the NeMo / transformers `ParakeetFeatureExtractor`:
//! pre-emphasis, a centered zero-padded STFT with a symmetric Hann window, a power spectrum,
//! a Slaney mel filterbank, a guarded natural log, and per-feature normalization.
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

const LOG_GUARD: f64 = 1.0 / 16_777_216.0; // 2^-24
const NORMALIZE_EPSILON: f64 = 1e-5;

/// Front-end parameters, from a model's `processor_config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    /// Pre-emphasis coefficient; 0 disables the filter.
    pub preemphasis: f32,
}

/// `frames × n_mels` features, row-major. Rows at and past `valid` are padding (zero).
#[derive(Clone, Debug)]
pub struct Features {
    pub data: Vec<f32>,
    pub frames: usize,
    pub valid: usize,
    pub n_mels: usize,
}

pub struct LogMel {
    config: MelConfig,
    /// The window zero-padded and centered to `n_fft`.
    window: Vec<f64>,
    /// Per mel bin: the first nonzero frequency bin and its weights.
    filters: Vec<(usize, Vec<f64>)>,
    fft: Arc<dyn RealToComplex<f64>>,
}

impl LogMel {
    pub fn new(config: MelConfig) -> Self {
        assert!(
            config.win_length <= config.n_fft && config.hop_length > 0 && config.n_mels > 0,
            "invalid mel configuration"
        );
        let n = config.win_length;
        let offset = (config.n_fft - n) / 2;
        let mut window = vec![0.0; config.n_fft];
        for i in 0..n {
            // torch.hann_window(n, periodic=False)
            let denominator = (n.max(2) - 1) as f64;
            window[offset + i] =
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denominator).cos();
        }
        let filters = slaney_filters(config.sample_rate, config.n_fft, config.n_mels)
            .into_iter()
            .map(|row| {
                let first = row.iter().position(|&w| w != 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&w| w != 0.0).map_or(first, |l| l + 1);
                (first, row[first..last].to_vec())
            })
            .collect();
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(config.n_fft);
        Self {
            config,
            window,
            filters,
            fft,
        }
    }

    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    /// Normalized log-mel features of mono samples at the configured rate.
    pub fn features(&self, samples: &[f32]) -> Features {
        let n_mels = self.config.n_mels;
        let (logs, frames, valid) = self.log_energies(samples);
        let mut data = vec![0.0f32; frames * n_mels];
        if valid > 0 {
            for m in 0..n_mels {
                let column = (0..valid).map(|f| logs[f * n_mels + m]);
                let mean = column.clone().sum::<f64>() / valid as f64;
                let variance = if valid > 1 {
                    column.map(|x| (x - mean).powi(2)).sum::<f64>() / (valid - 1) as f64
                } else {
                    0.0
                };
                let scale = 1.0 / (variance.sqrt() + NORMALIZE_EPSILON);
                for f in 0..valid {
                    data[f * n_mels + m] = ((logs[f * n_mels + m] - mean) * scale) as f32;
                }
            }
        }
        Features {
            data,
            frames,
            valid,
            n_mels,
        }
    }

    /// Unnormalized log mel energies `frames × n_mels`, with the frame and valid-frame counts.
    fn log_energies(&self, samples: &[f32]) -> (Vec<f64>, usize, usize) {
        let MelConfig {
            n_fft,
            hop_length: hop,
            n_mels,
            preemphasis,
            ..
        } = self.config;
        let len = samples.len();
        let mut emphasized: Vec<f64> = Vec::with_capacity(len);
        for (i, &x) in samples.iter().enumerate() {
            // Computed in f32, as the reference does.
            let value = if i == 0 || preemphasis == 0.0 {
                x
            } else {
                x - preemphasis * samples[i - 1]
            };
            emphasized.push(f64::from(value));
        }
        let frames = 1 + len / hop;
        let valid = len / hop;
        let pad = n_fft / 2;
        let bins = n_fft / 2 + 1;

        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0.0f64; bins];
        let mut logs = vec![0.0f64; frames * n_mels];
        for frame in 0..frames {
            let start = (frame * hop) as isize - pad as isize;
            for (i, slot) in input.iter_mut().enumerate() {
                let at = start + i as isize;
                let x = if at >= 0 && (at as usize) < len {
                    emphasized[at as usize]
                } else {
                    0.0
                };
                *slot = x * self.window[i];
            }
            self.fft
                .process_with_scratch(&mut input, &mut spectrum, &mut scratch)
                .expect("FFT buffers sized by the planner");
            for (p, c) in power.iter_mut().zip(&spectrum) {
                *p = c.norm_sqr();
            }
            for (m, (first, weights)) in self.filters.iter().enumerate() {
                let energy: f64 = weights
                    .iter()
                    .zip(&power[*first..])
                    .map(|(w, p)| w * p)
                    .sum();
                logs[frame * n_mels + m] = (energy + LOG_GUARD).ln();
            }
        }
        (logs, frames, valid)
    }
}

fn hz_to_mel(hz: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        MIN_LOG_HZ / F_SP + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        hz / F_SP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
    } else {
        F_SP * mel
    }
}

/// `librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, htk=False, norm="slaney")`,
/// rounded to f32 as librosa returns it.
fn slaney_filters(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f64>> {
    let bins = n_fft / 2 + 1;
    let nyquist = f64::from(sample_rate) / 2.0;
    let frequencies: Vec<f64> = (0..bins)
        .map(|i| nyquist * i as f64 / (bins - 1) as f64)
        .collect();
    let top = hz_to_mel(nyquist);
    let edges: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(top * i as f64 / (n_mels + 1) as f64))
        .collect();
    (0..n_mels)
        .map(|m| {
            let (low, center, high) = (edges[m], edges[m + 1], edges[m + 2]);
            let norm = 2.0 / (high - low);
            frequencies
                .iter()
                .map(|&f| {
                    let lower = (f - low) / (center - low);
                    let upper = (high - f) / (high - center);
                    f64::from((lower.min(upper).max(0.0) * norm) as f32)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parakeet() -> MelConfig {
        MelConfig {
            sample_rate: 16000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 128,
            preemphasis: 0.97,
        }
    }

    #[test]
    fn slaney_filters_match_librosa_reference_points() {
        let filters = slaney_filters(16000, 512, 128);
        assert_eq!((filters.len(), filters[0].len()), (128, 257));
        // librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney")
        for (m, bin, expected) in [
            (0, 1, 0.028_377_542),
            (0, 2, 0.0),
            (64, 55, 0.018_818_283),
            (127, 244, 4.764_134e-5),
            (127, 250, 0.005_223_188_5),
            (127, 255, 0.000_870_531_37),
        ] {
            let found = filters[m][bin];
            assert!(
                (found - expected).abs() < 1e-9,
                "[{m}][{bin}]: {found} vs {expected}"
            );
        }
        assert!(filters.iter().all(|row| row.iter().any(|&w| w > 0.0)));
    }

    #[test]
    fn frame_counts_follow_the_centered_stft() {
        let mel = LogMel::new(parakeet());
        let features = mel.features(&vec![0.0; 16000]);
        assert_eq!((features.frames, features.valid), (101, 100));
        assert_eq!(features.data.len(), 101 * 128);
    }

    #[test]
    fn a_tone_peaks_in_its_mel_band_and_padding_rows_are_zero() {
        let config = MelConfig {
            preemphasis: 0.0,
            ..parakeet()
        };
        let mel = LogMel::new(config);
        let samples: Vec<f32> = (0..16080)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let (logs, _, _) = mel.log_energies(&samples);
        let row = &logs[50 * 128..51 * 128];
        let peak = (0..128).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
        // 1 kHz is mel 15 of 45.2 at 8 kHz; bin centers are spaced top/129.
        let expected = (hz_to_mel(1000.0) / (hz_to_mel(8000.0) / 129.0)).round() as usize - 1;
        assert!(
            peak.abs_diff(expected) <= 1,
            "peak {peak}, expected {expected}"
        );
        let features = mel.features(&samples);
        assert!(
            features.data[features.valid * 128..]
                .iter()
                .all(|&x| x == 0.0)
        );
    }

    fn golden(name: &str) -> Vec<f32> {
        let dir = std::env::var_os("JEVONS_GOLDEN_DIR").map_or_else(
            || {
                std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
                    .join(".cache/jevons/golden")
            },
            std::path::PathBuf::from,
        );
        let path = dir.join("parakeet-tdt-0.6b-v3").join(format!("{name}.f32"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    #[test]
    #[ignore = "Requires the Parakeet reference dump (scripts/reference/parakeet_dump.py)"]
    fn features_match_the_reference_extractor() {
        let mel = LogMel::new(parakeet());
        for clip in ["en", "es"] {
            let features = mel.features(&golden(&format!("{clip}_audio")));
            let reference = golden(&format!("{clip}_mel"));
            assert_eq!(features.data.len(), reference.len(), "{clip}");
            let max = features
                .data
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("{clip}: {} frames, max |diff| {max:e}", features.frames);
            assert!(max < 1e-3, "{clip}: max |diff| {max}");
        }
    }
}
