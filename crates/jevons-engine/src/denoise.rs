//! Entropy-bound refinement, following the pinned native diffusion example.
use crate::{Error, Result};
use rand::Rng;

pub(crate) struct Prediction {
    pub best: i32,
    pub sampled: i32,
    pub entropy: f64,
}

fn predict(row: &[f32], temperature: f64, draw: f64) -> Result<Prediction> {
    if row.is_empty() || row.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidLogits);
    }
    let best = row
        .iter()
        .enumerate()
        .fold(0, |best, (i, x)| if *x > row[best] { i } else { best });
    let max = f64::from(row[best]);
    let z: f64 = row
        .iter()
        .map(|&x| ((f64::from(x) - max) / temperature).exp())
        .sum();
    let mut cumulative = 0.0;
    let mut sampled = None;
    let mut entropy = 0.0;
    for (i, &x) in row.iter().enumerate() {
        let p = ((f64::from(x) - max) / temperature).exp() / z;
        cumulative += p;
        if sampled.is_none() && cumulative > draw {
            sampled = Some(i);
        }
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    Ok(Prediction {
        best: best as i32,
        sampled: sampled.unwrap_or(best) as i32,
        entropy,
    })
}

/// Only mutable positions participate in the entropy budget; fixed template tokens stay intact.
pub(crate) fn refine(
    canvas: &mut [i32],
    logits: &[f32],
    positions: &[usize],
    vocab: usize,
    temperature: f64,
    mask: i32,
    rng: &mut impl Rng,
) -> Result<Vec<Prediction>> {
    if logits.len() != canvas.len() * vocab || positions.iter().any(|&p| p >= canvas.len()) {
        return Err(Error::InvalidLogits);
    }
    let predictions = positions
        .iter()
        .map(|&p| {
            predict(
                &logits[p * vocab..(p + 1) * vocab],
                temperature,
                rng.random(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut order: Vec<_> = (0..positions.len()).collect();
    order.sort_by(|&a, &b| predictions[a].entropy.total_cmp(&predictions[b].entropy));
    let mut entropy = 0.0;
    for i in order {
        canvas[positions[i]] = if entropy <= 0.1 {
            predictions[i].sampled
        } else {
            noise(vocab as i32, mask, rng)
        };
        entropy += predictions[i].entropy;
    }
    Ok(predictions)
}

pub(crate) fn noise(vocab: i32, mask: i32, rng: &mut impl Rng) -> i32 {
    loop {
        let token = rng.random_range(0..vocab);
        if token != mask {
            return token;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn refinement_preserves_fixed_tokens_and_accepts_certain_predictions() {
        let mut canvas = [9, 0, 8, 0];
        let logits = [
            0., 0., 0., -1000., 1000., -1000., 0., 0., 0., -1000., -1000., 1000.,
        ];
        let predictions = refine(
            &mut canvas,
            &logits,
            &[1, 3],
            3,
            1.0,
            -1,
            &mut ChaCha8Rng::seed_from_u64(1),
        )
        .unwrap();
        assert_eq!(canvas, [9, 1, 8, 2]);
        assert!(predictions.iter().all(|p| p.entropy < 1e-10));
    }

    #[test]
    fn entropy_and_sampling_use_stable_probabilities() {
        let p = predict(&[1000., 1000.], 1.0, 0.75).unwrap();
        assert_eq!(p.sampled, 1);
        assert!((p.entropy - 2_f64.ln()).abs() < 1e-10);
        assert!(predict(&[f32::NAN], 1.0, 0.0).is_err());
    }
}
