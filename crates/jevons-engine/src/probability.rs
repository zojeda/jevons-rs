//! Probability normalization over a restricted set of candidate logits.

use crate::{Error, Result};

/// Softmax over the allowed candidates, with no full-vocabulary normalization.
pub fn restricted_softmax(logits: &[f64]) -> Result<Vec<f64>> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidLogits);
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = weights.iter().sum();
    Ok(weights.into_iter().map(|x| x / sum).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_handles_extremes_and_rejects_nonfinite_values() {
        assert_eq!(restricted_softmax(&[10000.0, 10000.0]).unwrap(), [0.5, 0.5]);
        assert_eq!(
            restricted_softmax(&[10000.0, -10000.0]).unwrap(),
            [1.0, 0.0]
        );
        assert_eq!(restricted_softmax(&[42.0]).unwrap(), [1.0]);
        for values in [&[][..], &[f64::NAN], &[f64::INFINITY]] {
            assert!(restricted_softmax(values).is_err());
        }
    }
}
