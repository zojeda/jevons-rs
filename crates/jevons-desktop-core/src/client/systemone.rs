//! `POST /v1/systemone`: questions about a state, answered with probabilities.

use super::{Client, ClientError, checked};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub struct DecisionRequest {
    pub model: String,
    pub state: String,
    pub questions: BTreeMap<String, Question>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samples: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes or no.
    Noul {
        #[serde(skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// One label among 1 to 128, each with its description.
    Choice {
        #[serde(skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
        criteria: BTreeMap<String, String>,
    },
    /// 2 to 10 levels, lowest first.
    Score {
        #[serde(skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
        criteria: Vec<String>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", skip_serializing_if = "Option::is_none")]
    pub yes: Option<String>,
    #[serde(rename = "false", skip_serializing_if = "Option::is_none")]
    pub no: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        /// The probability of yes.
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

impl Client {
    pub async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse, ClientError> {
        let call = super::log::Call::start("POST /v1/systemone", request);
        let body = async {
            let response = self
                .request(reqwest::Method::POST, "/v1/systemone")
                .json(request)
                .send()
                .await?;
            Ok::<_, ClientError>(checked(response).await?.text().await?)
        }
        .await;
        let parsed = body.and_then(|body| {
            serde_json::from_str::<serde_json::Value>(&body)
                .map_err(|e| ClientError::Protocol(format!("System One answer: {e}: {body}")))
        });
        match &parsed {
            Ok(value) => call.end(value.clone(), None),
            Err(e) => call.end(serde_json::Value::Null, Some(e)),
        }
        serde_json::from_value(parsed?)
            .map_err(|e| ClientError::Protocol(format!("System One answer: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systemone_questions_round_trip_the_server_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../../examples/system-one.json")).unwrap();
        let questions: BTreeMap<String, Question> =
            serde_json::from_value(fixture["questions"].clone()).unwrap();
        assert!(matches!(questions["is_scm"], Question::Noul { .. }));
        let request = DecisionRequest {
            model: fixture["model"].as_str().unwrap().into(),
            state: fixture["state"].as_str().unwrap().into(),
            questions,
            steps: None,
            samples: None,
            think: None,
        };
        assert_eq!(serde_json::to_value(&request).unwrap(), fixture);
    }

    #[test]
    fn systemone_answers_parse_every_type() {
        let body = r#"{"model":"jev","usage":{"input_tokens":3,"output_tokens":1},"answers":{
            "a":{"type":"noul","noul":0.9},
            "b":{"type":"choice","choice":"x","probabilities":{"x":0.8,"y":0.2},"confidence":0.8},
            "c":{"type":"score","score":1.2,"legend":{"0":"low","1":"mid","2":"high"},
                 "probabilities":{"0":0.1,"1":0.6,"2":0.3},"confidence":0.6}}}"#;
        let response: DecisionResponse = serde_json::from_str(body).unwrap();
        assert_eq!(response.answers["a"], Answer::Noul { noul: 0.9 });
        assert!(
            matches!(&response.answers["c"], Answer::Score { legend, .. } if legend["2"] == "high")
        );
    }
}
