//! Confidence-ordered unmasking for masked diffusion models (LLaDA-style low-confidence
//! remasking with a commit threshold).
use crate::{Error, Result};

/// A masked position's best token and that token's probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Proposal {
    pub token: i32,
    pub confidence: f64,
}

/// Indices of `proposals` to commit this step: the most confident one always, plus every other
/// at or above `threshold`. Ties keep the earlier position first.
pub(crate) fn commits(proposals: &[Proposal], threshold: f64) -> Vec<usize> {
    let mut order: Vec<_> = (0..proposals.len()).collect();
    order.sort_by(|&a, &b| proposals[b].confidence.total_cmp(&proposals[a].confidence));
    order
        .iter()
        .enumerate()
        .filter(|&(rank, &i)| rank == 0 || proposals[i].confidence >= threshold)
        .map(|(_, &i)| i)
        .collect()
}

/// The full-vocabulary argmax of one logit row and its softmax probability.
pub(crate) fn propose(row: &[f32]) -> Result<Proposal> {
    if row.is_empty() || row.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidLogits);
    }
    let best = row
        .iter()
        .enumerate()
        .fold(0, |best, (i, x)| if *x > row[best] { i } else { best });
    let max = f64::from(row[best]);
    let z: f64 = row.iter().map(|&x| (f64::from(x) - max).exp()).sum();
    Ok(Proposal {
        token: best as i32,
        confidence: 1.0 / z,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(confidence: f64) -> Proposal {
        Proposal {
            token: 0,
            confidence,
        }
    }

    #[test]
    fn the_most_confident_position_always_commits_and_others_need_the_threshold() {
        assert_eq!(commits(&[p(0.2), p(0.5), p(0.1)], 0.9), vec![1]);
        assert_eq!(commits(&[p(0.95), p(0.5), p(0.91)], 0.9), vec![0, 2]);
        assert_eq!(commits(&[p(0.3), p(0.3)], 0.9), vec![0]);
        assert!(commits(&[], 0.9).is_empty());
    }

    #[test]
    fn proposals_take_the_argmax_with_its_stable_probability() {
        let proposal = propose(&[1000.0, 1000.0 + 2f32.ln(), 0.0]).unwrap();
        assert_eq!(proposal.token, 1);
        assert!((proposal.confidence - 2.0 / 3.0).abs() < 1e-4);
        assert!(propose(&[f32::INFINITY]).is_err());
        assert!(propose(&[]).is_err());
    }
}
