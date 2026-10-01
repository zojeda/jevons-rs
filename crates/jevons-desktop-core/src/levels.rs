//! The microphone meter: five bands of loudness on a 0..=16 scale.

pub const MAX_LEVEL: u8 = 16;

/// The RMS level of five consecutive parts of `samples`, mapped from -60..0 dBFS to 0..=16.
pub fn levels(samples: &[i16]) -> [u8; 5] {
    std::array::from_fn(|i| {
        let part = &samples[samples.len() * i / 5..samples.len() * (i + 1) / 5];
        if part.is_empty() {
            return 0;
        }
        let rms = (part
            .iter()
            .map(|v| (f64::from(*v) / 32768.).powi(2))
            .sum::<f64>()
            / part.len() as f64)
            .sqrt();
        if rms < 0.001 {
            0
        } else {
            (((20. * rms.log10() + 60.) / 60.).clamp(0., 1.) * f64::from(MAX_LEVEL)).round() as u8
        }
    })
}

/// Smooths meter bands: fast attack, slower release.
#[derive(Clone, Debug, Default)]
pub struct Meter {
    smooth: [f32; 5],
}

impl Meter {
    pub fn update(&mut self, samples: &[i16]) -> [u8; 5] {
        let raw = levels(samples);
        std::array::from_fn(|i| {
            let target = f32::from(raw[i]);
            let rate = if target > self.smooth[i] { 0.65 } else { 0.3 };
            self.smooth[i] += (target - self.smooth[i]) * rate;
            self.smooth[i].round() as u8
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_zero_and_full_scale_is_the_top_level() {
        assert_eq!(levels(&[0; 500]), [0; 5]);
        assert_eq!(levels(&[i16::MAX; 500]), [MAX_LEVEL; 5]);
        assert_eq!(levels(&[]), [0; 5]);
    }

    #[test]
    fn the_meter_rises_faster_than_it_falls() {
        let mut meter = Meter::default();
        let up = meter.update(&[i16::MAX; 500])[0];
        let down = MAX_LEVEL - meter.update(&[0; 500])[0].min(MAX_LEVEL);
        assert!(up > down, "{up} {down}");
    }
}
