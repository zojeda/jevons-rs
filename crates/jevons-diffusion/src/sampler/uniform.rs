//! Entropy-bound refinement for uniform-noise diffusion, following the pinned native sampler.
use jevons_core::{DiffusionModel, Error, Prediction, Result};
use rand::Rng;

/// One refinement step over the last canvas forward's full logits. The model predicts each of
/// `positions` at `temperature`, from one draw per position in order. Positions are then taken
/// by ascending entropy: those reached while the summed entropy is at most 0.1 take their
/// sampled token and the rest fresh noise. Only `positions` change; fixed template tokens stay
/// intact.
pub fn refine(
    model: &mut dyn DiffusionModel,
    canvas: &mut [i32],
    positions: &[usize],
    temperature: f64,
    mask: i32,
    rng: &mut impl Rng,
) -> Result<Vec<Prediction>> {
    if positions.iter().any(|&p| p >= canvas.len()) {
        return Err(Error::InvalidLogits);
    }
    let draws: Vec<f64> = positions.iter().map(|_| rng.random()).collect();
    let predictions = model.predictions(positions, temperature, &draws)?;
    if predictions.len() != positions.len() {
        return Err(Error::InvalidLogits);
    }
    let vocab = model.info().n_vocab;
    let mut order: Vec<_> = (0..positions.len()).collect();
    order.sort_by(|&a, &b| predictions[a].entropy.total_cmp(&predictions[b].entropy));
    let mut entropy = 0.0;
    for i in order {
        canvas[positions[i]] = if entropy <= 0.1 {
            predictions[i].sampled
        } else {
            noise(vocab, mask, rng)
        };
        entropy += predictions[i].entropy;
    }
    Ok(predictions)
}

pub fn noise(vocab: i32, mask: i32, rng: &mut impl Rng) -> i32 {
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
    use crate::fake::{FakeModel, MASK};
    use jevons_core::{Conditioning, Logits};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn refinement_preserves_fixed_tokens_and_accepts_certain_predictions() {
        let mut model = FakeModel::new();
        model.favored = vec![0, 21, 0, 22];
        let mut canvas = [9, 0, 8, 0];
        model
            .forward_canvas(&canvas, 0, Conditioning::None, Logits::Full)
            .unwrap();
        let predictions = refine(
            &mut model,
            &mut canvas,
            &[1, 3],
            1.0,
            MASK,
            &mut ChaCha8Rng::seed_from_u64(1),
        )
        .unwrap();
        assert_eq!(canvas, [9, 21, 8, 22]);
        assert!(predictions.iter().all(|p| p.entropy < 1e-10));
        assert!(
            refine(
                &mut model,
                &mut canvas,
                &[4],
                1.0,
                MASK,
                &mut ChaCha8Rng::seed_from_u64(1)
            )
            .is_err()
        );
    }
}
