//! Streaming band-limited resampling with a windowed-sinc kernel.

/// Kernel zero crossings on each side, at the lower of the two rates.
const ZERO_CROSSINGS: usize = 16;
/// Kernel phases between input samples; intermediate phases interpolate linearly.
const PHASES: usize = 256;
/// Passband edge as a fraction of the lower Nyquist rate.
const ROLLOFF: f64 = 0.95;

/// Converts a mono stream from one sample rate to another. Output sample `n` sits at input
/// position `n · from / to`, so a stream of `len` samples yields `round(len · to / from)`
/// samples once [`Resampler::finish`] runs.
pub struct Resampler {
    from: u32,
    to: u32,
    half: usize,
    table: Vec<f32>,
    /// Pending input; `history[0]` is input sample `base` (negative indices are leading zeros).
    history: Vec<f32>,
    base: i64,
    next: u64,
    consumed: u64,
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        assert!(from > 0 && to > 0, "sample rates must be positive");
        let cutoff = (f64::from(to) / f64::from(from)).min(1.0) * ROLLOFF;
        let half = (ZERO_CROSSINGS as f64 / cutoff).ceil() as usize;
        let width = 2 * half;
        let mut table = Vec::with_capacity((PHASES + 1) * width);
        for phase in 0..=PHASES {
            let frac = phase as f64 / PHASES as f64;
            let row: Vec<f64> = (0..width)
                .map(|j| kernel(frac + half as f64 - 1.0 - j as f64, cutoff, half as f64))
                .collect();
            // Unit DC gain at every phase.
            let sum: f64 = row.iter().sum();
            table.extend(row.iter().map(|w| (w / sum) as f32));
        }
        Self {
            from,
            to,
            half,
            table,
            history: vec![0.0; half - 1],
            base: 1 - half as i64,
            next: 0,
            consumed: 0,
        }
    }

    /// Resamples `input`, appending every output sample whose kernel is fully covered.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.consumed += input.len() as u64;
        if self.from == self.to {
            output.extend_from_slice(input);
            return;
        }
        self.history.extend_from_slice(input);
        self.drain(output, u64::MAX);
    }

    /// Flushes the tail of the stream, as if it ended with silence.
    pub fn finish(&mut self, output: &mut Vec<f32>) {
        if self.from == self.to {
            return;
        }
        let (from, to) = (u128::from(self.from), u128::from(self.to));
        let total = ((u128::from(self.consumed) * to + from / 2) / from) as u64;
        self.history.extend(std::iter::repeat_n(0.0, self.half + 1));
        self.drain(output, total);
    }

    fn position(&self, n: u64) -> (i64, f64) {
        let exact = u128::from(n) * u128::from(self.from);
        let to = u128::from(self.to);
        let center = (exact / to) as i64;
        (center, (exact % to) as f64 / to as f64)
    }

    fn drain(&mut self, output: &mut Vec<f32>, limit: u64) {
        let width = 2 * self.half;
        let available = self.base + self.history.len() as i64;
        while self.next < limit {
            let (center, frac) = self.position(self.next);
            if center + self.half as i64 >= available {
                break;
            }
            let phase = frac * PHASES as f64;
            let index = phase as usize;
            let t = (phase - index as f64) as f32;
            let lower = &self.table[index * width..][..width];
            let upper = &self.table[(index + 1) * width..][..width];
            let start = (center - self.half as i64 + 1 - self.base) as usize;
            let taps = &self.history[start..start + width];
            let value = taps
                .iter()
                .zip(lower.iter().zip(upper))
                .map(|(x, (a, b))| x * (a + t * (b - a)))
                .sum();
            output.push(value);
            self.next += 1;
        }
        let (center, _) = self.position(self.next);
        let first = center - self.half as i64 + 1;
        let drop = (first - self.base).clamp(0, self.history.len() as i64) as usize;
        self.history.drain(..drop);
        self.base += drop as i64;
    }
}

/// Blackman-windowed sinc low-pass with `cutoff` in cycles per input sample (×2).
fn kernel(x: f64, cutoff: f64, half: f64) -> f64 {
    if x.abs() >= half {
        return 0.0;
    }
    let sinc = if x == 0.0 {
        1.0
    } else {
        let a = std::f64::consts::PI * cutoff * x;
        a.sin() / a
    };
    let u = std::f64::consts::PI * x / half;
    let window = 0.42 + 0.5 * u.cos() + 0.08 * (2.0 * u).cos();
    cutoff * sinc * window
}

/// Resamples a whole signal.
pub fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    let mut resampler = Resampler::new(from, to);
    let mut output = Vec::new();
    resampler.process(samples, &mut output);
    resampler.finish(&mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    fn rms_error(a: &[f32], b: &[f32]) -> f32 {
        let sum: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        (sum / a.len() as f32).sqrt()
    }

    #[test]
    fn common_rates_preserve_length_and_an_in_band_tone() {
        for (from, to) in [
            (48000, 16000),
            (44100, 16000),
            (24000, 16000),
            (8000, 16000),
        ] {
            let input = tone(from, 440.0, from as usize);
            let output = resample(&input, from, to);
            assert_eq!(output.len(), to as usize, "{from} → {to}");
            let expected = tone(to, 440.0, to as usize);
            // Skip the edges, where the kernel sees the implicit silence around the signal.
            let edge = 200;
            let error = rms_error(
                &output[edge..output.len() - edge],
                &expected[edge..expected.len() - edge],
            );
            assert!(error < 5e-3, "{from} → {to}: rms error {error}");
        }
    }

    #[test]
    fn tones_above_the_output_nyquist_are_removed() {
        let output = resample(&tone(48000, 12000.0, 48000), 48000, 16000);
        let middle = &output[1000..15000];
        let rms = (middle.iter().map(|x| x * x).sum::<f32>() / middle.len() as f32).sqrt();
        assert!(rms < 1e-2, "aliased energy {rms}");
    }

    #[test]
    fn streaming_in_chunks_matches_one_pass() {
        let input = tone(24000, 300.0, 24000);
        let whole = resample(&input, 24000, 16000);
        let mut resampler = Resampler::new(24000, 16000);
        let mut chunked = Vec::new();
        for chunk in input.chunks(480) {
            resampler.process(chunk, &mut chunked);
        }
        resampler.finish(&mut chunked);
        assert_eq!(chunked, whole);
    }

    #[test]
    fn equal_rates_pass_samples_through() {
        let input = tone(16000, 100.0, 1000);
        assert_eq!(resample(&input, 16000, 16000), input);
    }
}
