//! Rotary position embedding frequencies (default and YaRN) and Llama-4 query scaling, as in
//! Hugging Face `transformers` 4.57 (`_compute_default_rope_parameters`,
//! `_compute_yarn_parameters`, and `_get_llama_4_attn_scale` in the checkpoint's code).
//!
//! Frequencies are computed in f32 like the reference, which builds them from float32 tensors.
use crate::config::Config;
use std::f64::consts::PI;

/// Inverse frequencies for the `head_dim / 2` rotated pairs, and the cos/sin scale.
#[derive(Clone, Debug, PartialEq)]
pub struct Rope {
    pub inv_freq: Vec<f32>,
    pub attention_scaling: f32,
}

fn mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

impl Rope {
    pub fn new(config: &Config) -> Self {
        let p = &config.rope_parameters;
        let dim = config.head_dim;
        let base = p.rope_theta as f32;
        // base ** (arange(0, dim, 2) / dim), in float32 as in the reference.
        let pos_freqs: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| base.powf(i as f32 / dim as f32))
            .collect();
        if p.rope_type != "yarn" {
            return Self {
                inv_freq: pos_freqs.iter().map(|f| 1.0 / f).collect(),
                attention_scaling: 1.0,
            };
        }
        let factor = p.factor.unwrap_or(1.0);
        let attention_scaling = match (p.mscale, p.mscale_all_dim) {
            (Some(m), Some(all)) if m != 0.0 && all != 0.0 => {
                mscale(factor, m) / mscale(factor, all)
            }
            _ => mscale(factor, 1.0),
        };
        let original = p
            .original_max_position_embeddings
            .unwrap_or(config.max_position_embeddings) as f64;
        let (beta_fast, beta_slow) = (p.beta_fast.unwrap_or(32.0), p.beta_slow.unwrap_or(1.0));
        let correction_dim = |rotations: f64| {
            dim as f64 * (original / (rotations * 2.0 * PI)).ln() / (2.0 * f64::from(base).ln())
        };
        let (mut low, mut high) = (correction_dim(beta_fast), correction_dim(beta_slow));
        if p.truncate.unwrap_or(true) {
            (low, high) = (low.floor(), high.ceil());
        }
        let (low, high) = (low.max(0.0) as f32, high.min(dim as f64 - 1.0) as f32);
        let high = if low == high { high + 0.001 } else { high };
        let inv_freq = pos_freqs
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                let extrapolation = 1.0 - ramp;
                let interpolated = 1.0 / (factor as f32 * f);
                interpolated * (1.0 - extrapolation) + (1.0 / f) * extrapolation
            })
            .collect();
        Self {
            inv_freq,
            attention_scaling: attention_scaling as f32,
        }
    }

    /// `(cos, sin)` rows of width `head_dim` for `positions`, laid out for `rotate_half`: the
    /// frequencies repeat across both halves.
    pub fn tables(&self, positions: std::ops::Range<usize>) -> (Vec<f32>, Vec<f32>) {
        let half = self.inv_freq.len();
        let rows = positions.len();
        let (mut cos, mut sin) = (vec![0.0; rows * 2 * half], vec![0.0; rows * 2 * half]);
        for (r, pos) in positions.enumerate() {
            for (i, &f) in self.inv_freq.iter().enumerate() {
                let angle = pos as f32 * f;
                let (s, c) = angle.sin_cos();
                for j in [i, i + half] {
                    cos[r * 2 * half + j] = c * self.attention_scaling;
                    sin[r * 2 * half + j] = s * self.attention_scaling;
                }
            }
        }
        (cos, sin)
    }
}

/// Llama-4 query scale `1 + beta * ln(1 + floor(position / original))`; exactly 1 below
/// `original` positions.
pub fn llama4_query_scale(config: &Config, position: usize) -> f32 {
    let p = &config.rope_parameters;
    match (p.llama_4_scaling_beta, p.original_max_position_embeddings) {
        (Some(beta), Some(original)) => {
            1.0 + beta as f32 * (1.0 + (position / original) as f32).ln()
        }
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, VLM_8B};

    #[test]
    fn yarn_keeps_fast_frequencies_and_interpolates_slow_ones() {
        let config = Config::parse(VLM_8B).unwrap();
        let rope = Rope::new(&config);
        assert_eq!(rope.inv_freq.len(), 64);
        assert_eq!(rope.attention_scaling, 1.0);
        // Correction range for theta 1e6, dim 128, original 16384: floor(20.38)=20 .. ceil(36.44)=37.
        assert_eq!(rope.inv_freq[0], 1.0);
        let base = |i: usize| 1.0 / 1e6f32.powf((2 * i) as f32 / 128.0);
        for i in 0..=20 {
            assert_eq!(rope.inv_freq[i], base(i), "pair {i} is extrapolated");
        }
        for i in 37..64 {
            let want = base(i) / 16.0;
            assert!((rope.inv_freq[i] - want).abs() <= want * 1e-6, "pair {i}");
        }
        assert!(rope.inv_freq[28] < base(28) && rope.inv_freq[28] > base(28) / 16.0);
    }

    #[test]
    fn tables_repeat_frequencies_across_halves_and_query_scale_starts_at_one() {
        let config = Config::parse(VLM_8B).unwrap();
        let rope = Rope::new(&config);
        let (cos, sin) = rope.tables(3..5);
        assert_eq!(cos.len(), 2 * 128);
        assert_eq!(cos[5], cos[64 + 5]);
        assert_eq!(sin[128 + 1], (4.0 * rope.inv_freq[1]).sin());
        assert_eq!(llama4_query_scale(&config, 16383), 1.0);
        assert!((llama4_query_scale(&config, 16384) - (1.0 + 0.1 * 2f32.ln())).abs() < 1e-6);
    }
}
