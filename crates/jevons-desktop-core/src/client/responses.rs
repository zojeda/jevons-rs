//! `POST /v1/responses`, streamed: the generated text arrives as deltas.

use super::{Client, ClientError, checked};
use futures_util::StreamExt;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct ResponseRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
}

/// `reasoning.effort`: the server thinks up to 64 (minimal), 256 (low), 1024 (medium) or 4096
/// (high) tokens before answering.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Reasoning {
    pub effort: String,
}

impl Reasoning {
    /// The smallest effort whose thought budget covers `tokens`.
    pub fn for_budget(tokens: u32) -> Self {
        let effort = match tokens {
            0 => "none",
            1..=64 => "minimal",
            65..=256 => "low",
            257..=1024 => "medium",
            _ => "high",
        };
        Self {
            effort: effort.into(),
        }
    }
}

#[derive(Serialize)]
struct Streamed<'a> {
    #[serde(flatten)]
    request: &'a ResponseRequest,
    stream: bool,
}

impl Client {
    /// Generates a response, calling `on_delta` with each piece of text; returns the whole text.
    pub async fn respond(
        &self,
        request: &ResponseRequest,
        mut on_delta: impl FnMut(&str),
    ) -> Result<String, ClientError> {
        let response = self
            .request(reqwest::Method::POST, "/v1/responses")
            .json(&Streamed {
                request,
                stream: true,
            })
            .send()
            .await?;
        let mut body = checked(response).await?.bytes_stream();
        let mut events = SseParser::default();
        let mut text = String::new();
        while let Some(chunk) = body.next().await {
            for data in events.push(&chunk?) {
                let event: serde_json::Value = serde_json::from_str(&data)
                    .map_err(|e| ClientError::Protocol(format!("Responses event: {e}")))?;
                match event["type"].as_str() {
                    Some("response.output_text.delta") => {
                        let delta = event["delta"].as_str().unwrap_or_default();
                        text.push_str(delta);
                        on_delta(delta);
                    }
                    Some("response.output_text.done") => {
                        if let Some(done) = event["text"].as_str() {
                            text = done.to_string();
                        }
                    }
                    Some("error" | "response.failed") => {
                        let message = event["error"]["message"]
                            .as_str()
                            .or(event["response"]["error"]["message"].as_str())
                            .unwrap_or("generation failed");
                        return Err(ClientError::Api {
                            status: 500,
                            message: message.into(),
                        });
                    }
                    _ => {}
                }
            }
        }
        Ok(text)
    }
}

/// Splits a server-sent event stream into the `data` of each event.
#[derive(Default)]
struct SseParser {
    buffer: String,
}

impl SseParser {
    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));
        let mut events = Vec::new();
        while let Some(end) = self.buffer.find("\n\n") {
            let block: String = self.buffer.drain(..end + 2).collect();
            let data: Vec<&str> = block
                .lines()
                .filter_map(|l| l.strip_prefix("data:"))
                .map(|d| d.strip_prefix(' ').unwrap_or(d))
                .collect();
            if !data.is_empty() && data != ["[DONE]"] {
                events.push(data.join("\n"));
            }
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_split_across_chunks_are_joined() {
        let mut parser = SseParser::default();
        assert!(parser.push(b"event: x\ndata: {\"a\"").is_empty());
        assert_eq!(parser.push(b":1}\n\ndata: [DONE]\n\n"), ["{\"a\":1}"]);
    }
}
