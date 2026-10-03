//! Typed requests to the jevons API and to other providers of the same APIs, and the routes
//! that send each capability to its provider. The jevons server parses requests by hand, so
//! these mirror its schemas rather than share types with it (which would also pull in the
//! inference stack).

mod chat;
pub mod log;
mod realtime;
mod responses;
mod route;
mod systemone;
mod transcriptions;

pub use chat::{ChatMessage, ChatReply, ChatRequest};
pub use realtime::{RealtimeEvent, RealtimeReader, RealtimeWriter, Turns};
pub use responses::{Reasoning, ResponseRequest};
pub use route::{EXTERNAL_MAX_QUESTIONS, MIN_PROBABILITY, Profile, Route, Routes};
pub use systemone::{
    Answer, DecisionRequest, DecisionResponse, NoulCriteria, Question, Usage, probability_of,
};
pub use transcriptions::Transcription;

use serde::Deserialize;
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("cannot reach the API: {0}")]
    Http(#[from] reqwest::Error),
    #[error("the API answered {status}: {message}")]
    Api { status: u16, message: String },
    #[error("{0} is not served by this runtime")]
    NotServed(&'static str),
    #[error("Realtime session: {0}")]
    Realtime(String),
    #[error("unexpected response: {0}")]
    Protocol(String),
}

/// A provider's endpoint and its key.
#[derive(Clone, Debug)]
pub struct Client {
    http: reqwest::Client,
    base: String,
    key: Option<String>,
}

/// `GET /health`: the model serving each service.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct Health {
    pub status: String,
    pub services: Services,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct Services {
    pub generative: Option<String>,
    pub decision: Option<String>,
    pub speech: Option<String>,
}

impl Client {
    /// `base` is the server root, such as `http://127.0.0.1:8080`.
    pub fn new(base: &str, key: Option<String>) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("the HTTP client builds");
        Self {
            http,
            base: base.trim_end_matches('/').to_string(),
            key: key.filter(|k| !k.is_empty()),
        }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let request = self.http.request(method, self.url(path));
        match &self.key {
            Some(key) => request.bearer_auth(key),
            None => request,
        }
    }

    pub async fn health(&self) -> Result<Health, ClientError> {
        let response = self.request(reqwest::Method::GET, "/health").send().await?;
        Ok(checked(response).await?.json().await?)
    }
}

/// Turns an error status into [`ClientError::Api`] with the server's message.
async fn checked(response: reqwest::Response) -> Result<reqwest::Response, ClientError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    Err(ClientError::Api {
        status: status.as_u16(),
        message: error_message(&body),
    })
}

/// The message of an OpenAI-style (`error.message`) or System One (`detail[].msg`) error body.
fn error_message(body: &str) -> String {
    let value: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    value["error"]["message"]
        .as_str()
        .map(String::from)
        .or_else(|| {
            value["detail"].as_array().map(|d| {
                d.iter()
                    .filter_map(|e| e["msg"].as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            })
        })
        .or_else(|| value["detail"].as_str().map(String::from))
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| body.chars().take(200).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_come_from_either_error_shape() {
        assert_eq!(
            error_message(r#"{"error":{"message":"No API key provided","type":"x"}}"#),
            "No API key provided"
        );
        assert_eq!(
            error_message(r#"{"detail":[{"loc":["body"],"msg":"Unknown field","type":"x"}]}"#),
            "Unknown field"
        );
        assert_eq!(error_message("plain"), "plain");
    }
}
