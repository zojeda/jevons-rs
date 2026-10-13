//! Choosing a code from logits as `qwen-tts` does through `transformers`' `generate`: a
//! repetition penalty and suppressed ids for the talker, then the argmax, or a draw from the
//! `top_k` most likely codes at a temperature.

/// How codes are chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Decoding {
    /// The most likely code, for parity with the reference: only the tests ask for it.
    #[cfg_attr(not(test), allow(dead_code))]
    Greedy,
    Sample {
        temperature: f32,
        top_k: usize,
    },
}

/// SplitMix64: a small generator, seeded per utterance, so a seed reproduces a voice's take.
#[derive(Clone, Debug)]
pub(crate) struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Divides the positive logits of `seen` ids by `penalty` and multiplies the negative ones.
/// An id counts once however often it was seen, as in `transformers`, which scales the logits
/// it gathers and writes them back.
pub(crate) fn penalize(logits: &mut [f32], seen: &[i32], penalty: f32) {
    let mut ids: Vec<usize> = seen
        .iter()
        .filter_map(|id| usize::try_from(*id).ok())
        .filter(|id| *id < logits.len())
        .collect();
    ids.sort_unstable();
    ids.dedup();
    for id in ids {
        let logit = &mut logits[id];
        *logit = if *logit < 0.0 {
            *logit * penalty
        } else {
            *logit / penalty
        };
    }
}

/// The first largest logit's index.
fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, logit) in logits.iter().enumerate() {
        if *logit > logits[best] {
            best = i;
        }
    }
    best
}

/// Chooses an index of `logits`. Entries at negative infinity are never chosen.
pub(crate) fn choose(logits: &[f32], decoding: Decoding, rng: &mut Rng) -> usize {
    match decoding {
        Decoding::Greedy => argmax(logits),
        Decoding::Sample { .. } => choose_at(logits, decoding, rng.uniform()),
    }
}

/// [`choose`] with the draw given: `uniform` in `[0, 1)` lands on a code by the weights of the
/// `top_k` most likely, in code order.
pub(crate) fn choose_at(logits: &[f32], decoding: Decoding, uniform: f64) -> usize {
    let Decoding::Sample { temperature, top_k } = decoding else {
        return argmax(logits);
    };
    // The k-th largest logit: everything below it is out, ties with it stay.
    let mut sorted: Vec<f32> = logits.to_vec();
    let k = top_k.clamp(1, sorted.len());
    sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    let threshold = sorted[k - 1];
    let top = logits[argmax(logits)];
    let weight = |logit: f32| {
        if logit < threshold {
            0.0
        } else {
            f64::from((logit - top) / temperature).exp()
        }
    };
    let total: f64 = logits.iter().map(|l| weight(*l)).sum();
    let mut left = uniform * total;
    let mut last = 0;
    for (i, logit) in logits.iter().enumerate() {
        let w = weight(*logit);
        if w > 0.0 {
            last = i;
            if left < w {
                return i;
            }
            left -= w;
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_penalty_pushes_seen_codes_down_whatever_their_sign() {
        let mut logits = vec![2.0, -2.0, 1.0];
        penalize(&mut logits, &[0, 1, 7, -1], 2.0);
        assert_eq!(logits, vec![1.0, -4.0, 1.0]);
        // A code used twice is penalized once.
        let mut logits = vec![2.0, -2.0, 1.0];
        penalize(&mut logits, &[0, 1, 0, 1, 1], 2.0);
        assert_eq!(logits, vec![1.0, -4.0, 1.0]);
    }

    #[test]
    fn greedy_takes_the_first_maximum_and_skips_suppressed_codes() {
        let mut rng = Rng::new(1);
        let logits = [0.5, f32::NEG_INFINITY, 3.0, 3.0];
        assert_eq!(choose(&logits, Decoding::Greedy, &mut rng), 2);
    }

    #[test]
    fn sampling_draws_only_from_the_top_codes_in_their_proportions() {
        let mut rng = Rng::new(7);
        let decoding = Decoding::Sample {
            temperature: 1.0,
            top_k: 2,
        };
        // Odds 1 : e within the top two; the third code is out.
        let logits = [0.0, 1.0, -0.5, f32::NEG_INFINITY];
        let mut counts = [0usize; 4];
        for _ in 0..20_000 {
            counts[choose(&logits, decoding, &mut rng)] += 1;
        }
        assert_eq!(counts[2] + counts[3], 0);
        let share = counts[1] as f64 / 20_000.0;
        let want = std::f64::consts::E / (1.0 + std::f64::consts::E);
        assert!((share - want).abs() < 0.01, "{share} vs {want}");
    }

    #[test]
    fn a_seed_reproduces_its_draws() {
        let draws = |seed| {
            let mut rng = Rng::new(seed);
            (0..4).map(|_| rng.uniform()).collect::<Vec<_>>()
        };
        assert_eq!(draws(3), draws(3));
        assert_ne!(draws(3), draws(4));
        assert!(draws(3).iter().all(|u| (0.0..1.0).contains(u)));
    }
}
