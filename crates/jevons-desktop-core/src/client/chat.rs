//! `POST /v1/chat/completions` with function tools, streamed: text arrives as deltas, and a tool
//! call arrives whole once the model has decided it.

use super::responses::SseParser;
use super::{Client, ClientError, checked};
use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ChatMessage {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// An assistant turn's calls: `{"id", "type": "function", "function": {"name", "arguments"}}`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<Value>,
    /// A tool turn: the call it answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn text(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Function tools: `{"type": "function", "function": {"name", "description", "parameters"}}`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// `{"type": "json_schema", "json_schema": {"name", "schema"}}` for a structured answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

impl ChatRequest {
    /// A structured answer in `schema`'s shape.
    pub fn answer_schema(mut self, schema: Value) -> Self {
        self.response_format = Some(json!({
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        }));
        self
    }
}

/// How a chat turn ended.
#[derive(Clone, Debug, PartialEq)]
pub enum ChatReply {
    Text(String),
    Call {
        id: String,
        name: String,
        arguments: Value,
    },
}

#[derive(Serialize)]
struct Streamed<'a> {
    #[serde(flatten)]
    request: &'a ChatRequest,
    stream: bool,
}

impl Client {
    /// One chat turn, calling `on_delta` with each piece of answer text.
    pub async fn chat(
        &self,
        request: &ChatRequest,
        mut on_delta: impl FnMut(&str),
    ) -> Result<ChatReply, ClientError> {
        let response = self
            .request(reqwest::Method::POST, "/v1/chat/completions")
            .json(&Streamed {
                request,
                stream: true,
            })
            .send()
            .await?;
        let mut body = checked(response).await?.bytes_stream();
        let mut events = SseParser::default();
        let mut text = String::new();
        let (mut id, mut name, mut arguments) = (String::new(), String::new(), String::new());
        while let Some(chunk) = body.next().await {
            for data in events.push(&chunk?) {
                let event: Value = serde_json::from_str(&data)
                    .map_err(|e| ClientError::Protocol(format!("chat chunk: {e}")))?;
                if let Some(message) = event["error"]["message"].as_str() {
                    return Err(ClientError::Api {
                        status: 500,
                        message: message.into(),
                    });
                }
                let delta = &event["choices"][0]["delta"];
                if let Some(piece) = delta["content"].as_str().filter(|p| !p.is_empty()) {
                    text.push_str(piece);
                    on_delta(piece);
                }
                for call in delta["tool_calls"].as_array().into_iter().flatten() {
                    if let Some(value) = call["id"].as_str() {
                        id = value.into();
                    }
                    if let Some(value) = call["function"]["name"].as_str() {
                        name.push_str(value);
                    }
                    if let Some(value) = call["function"]["arguments"].as_str() {
                        arguments.push_str(value);
                    }
                }
            }
        }
        if name.is_empty() {
            return Ok(ChatReply::Text(text));
        }
        let arguments = if arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(&arguments)
                .map_err(|e| ClientError::Protocol(format!("tool arguments: {e}")))?
        };
        Ok(ChatReply::Call {
            id,
            name,
            arguments,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Json;
    use axum::routing::post;

    async fn server(
        events: &'static [&'static str],
    ) -> (Client, std::sync::Arc<std::sync::Mutex<Value>>) {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Value::Null));
        let s = seen.clone();
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<Value>| {
                let s = s.clone();
                async move {
                    *s.lock().unwrap() = body;
                    let body: String = events
                        .iter()
                        .map(|e| format!("data: {e}\n\n"))
                        .chain(["data: [DONE]\n\n".to_string()])
                        .collect();
                    ([("content-type", "text/event-stream")], body)
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (Client::new(&base, None), seen)
    }

    #[tokio::test]
    async fn a_streamed_tool_call_is_joined_and_its_arguments_parsed() {
        let (client, seen) = server(&[
            r#"{"choices":[{"delta":{"role":"assistant","content":null}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":\"status\"}"}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ])
        .await;
        let request = ChatRequest {
            model: "jev".into(),
            messages: vec![ChatMessage::text("user", "Is it up?")],
            tools: vec![json!({"type": "function", "function": {"name": "lookup"}})],
            ..ChatRequest::default()
        };
        let reply = client.chat(&request, |_| {}).await.unwrap();
        assert_eq!(
            reply,
            ChatReply::Call {
                id: "call_1".into(),
                name: "lookup".into(),
                arguments: json!({"q": "status"})
            }
        );
        let sent = seen.lock().unwrap().clone();
        assert_eq!(sent["stream"], true);
        assert_eq!(sent["tools"][0]["function"]["name"], "lookup");
        assert!(sent.get("response_format").is_none());
    }

    #[tokio::test]
    async fn text_streams_through_the_delta_callback() {
        let (client, seen) = server(&[
            r#"{"choices":[{"delta":{"role":"assistant","content":""}}]}"#,
            r#"{"choices":[{"delta":{"content":"All "}}]}"#,
            r#"{"choices":[{"delta":{"content":"green."}}]}"#,
        ])
        .await;
        let request = ChatRequest {
            model: "jev".into(),
            messages: vec![ChatMessage::text("user", "Status?")],
            ..ChatRequest::default()
        }
        .answer_schema(json!({"type": "string"}));
        let mut deltas = Vec::new();
        let reply = client
            .chat(&request, |d| deltas.push(d.to_string()))
            .await
            .unwrap();
        assert_eq!(reply, ChatReply::Text("All green.".into()));
        assert_eq!(deltas, ["All ", "green."]);
        assert_eq!(
            seen.lock().unwrap()["response_format"]["type"],
            "json_schema"
        );
    }
}
