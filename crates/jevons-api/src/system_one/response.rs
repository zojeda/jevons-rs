//! Map inference distributions back to protocol answers and usage.

use crate::system_one::{
    MappingError, Request,
    request::{Question, QuestionKind},
};
use jevons_decision::ReadResult;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        legend: BTreeMap<String, String>,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
}

impl Request {
    /// Restore question IDs and validate the distributions returned by inference.
    pub fn response(&self, model: &str, read: &ReadResult) -> Result<Response, MappingError> {
        if self.questions.len() != read.slots.len() {
            return Err(MappingError::InvalidResult);
        }
        let mut answers = BTreeMap::new();
        for ((id, question), slot) in self.questions.iter().zip(&read.slots) {
            answers.insert(
                id.clone(),
                Answer::from_probabilities(question, &slot.probabilities)?,
            );
        }
        Ok(Response {
            model: model.into(),
            answers,
            usage: Usage {
                input_tokens: read.prompt_tokens + read.canvas_tokens,
                output_tokens: read.output_tokens,
            },
        })
    }
}

impl Answer {
    fn from_probabilities(question: &Question, values: &[f64]) -> Result<Self, MappingError> {
        if values.len() != question.labels.len()
            || values.is_empty()
            || values
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || (values.iter().sum::<f64>() - 1.0).abs() > 1e-6
        {
            return Err(MappingError::InvalidResult);
        }
        let probabilities = question
            .labels
            .iter()
            .cloned()
            .zip(values.iter().copied())
            .collect();
        let confidence = confidence(values);
        Ok(match question.kind {
            QuestionKind::Noul => Self::Noul { noul: values[0] },
            QuestionKind::Choice => {
                let best = values
                    .iter()
                    .enumerate()
                    .fold(0, |best, (i, p)| if *p > values[best] { i } else { best });
                Self::Choice {
                    choice: question.labels[best].clone(),
                    probabilities,
                    confidence,
                }
            }
            QuestionKind::Score => {
                let legend = question
                    .labels
                    .iter()
                    .cloned()
                    .zip(
                        question
                            .descriptions
                            .iter()
                            .map(|v| v.as_str().unwrap_or_default().to_owned()),
                    )
                    .collect();
                let score = values.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
                Self::Score {
                    score,
                    legend,
                    probabilities,
                    confidence,
                }
            }
        })
    }
}

/// Normalized entropy confidence. A single possible option has confidence one.
pub fn confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() <= 1 {
        return 1.0;
    }
    let entropy: f64 = probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    (1.0 - entropy / (probabilities.len() as f64).ln()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jevons_decision::SlotRead;
    use serde_json::json;

    #[test]
    fn malformed_inference_results_cannot_become_protocol_answers() {
        let request = Request::parse(json!({
            "model": "local",
            "state": "test",
            "questions": {"q": {"type": "noul"}}
        }))
        .unwrap();
        let mut read = ReadResult {
            slots: vec![],
            prompt_tokens: 1,
            canvas_tokens: 1,
            output_tokens: 0,
            seed: 42,
            forward_ms: 0.0,
        };
        assert!(matches!(
            request.response("local", &read),
            Err(MappingError::InvalidResult)
        ));
        read.slots.push(SlotRead {
            canvas_position: 0,
            absolute_position: 1,
            initial_token: 0,
            candidate_tokens: vec![1, 2],
            logits: vec![0.0, 0.0],
            probabilities: vec![],
        });
        for probabilities in [
            vec![],
            vec![1.0],
            vec![0.2, 0.3, 0.5],
            vec![f64::NAN, 0.5],
            vec![f64::INFINITY, 0.0],
            vec![-0.1, 1.1],
            vec![0.2, 0.2],
        ] {
            read.slots[0].probabilities = probabilities;
            assert!(matches!(
                request.response("local", &read),
                Err(MappingError::InvalidResult)
            ));
        }
        read.slots[0].probabilities = vec![0.25, 0.75];
        assert!(matches!(
            request.response("local", &read).unwrap().answers["q"],
            Answer::Noul { noul: 0.25 }
        ));
    }

    #[test]
    fn entropy_confidence_handles_uniform_certain_and_singleton_answers() {
        assert_eq!(confidence(&[1.0]), 1.0);
        assert_eq!(confidence(&[0.0, 1.0]), 1.0);
        assert!(confidence(&[0.5, 0.5]).abs() < 1e-12);
        assert!((confidence(&[0.84, 0.159, 0.001]) - 0.5942682179296132).abs() < 1e-12);
    }
}
