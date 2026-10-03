//! The jevons runtime as an adk-rust model: agents built with adk (the context investigator,
//! `loop.toml` nodes) run their tool-calling loop over the API's Chat Completions tools, where
//! the next step and the labelled arguments are restricted reads.

use crate::client::{ChatMessage, ChatReply, ChatRequest, Client, Reasoning};
use adk_core::{
    AdkError, Content, FinishReason, Llm, LlmRequest, LlmResponse, LlmResponseStream, Part,
    async_trait,
};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Receives answer text as it is written.
pub type Deltas = Arc<dyn Fn(&str) + Send + Sync>;

/// A jevons generative model behind the API.
#[derive(Clone)]
pub struct JevonsLlm {
    client: Client,
    model: String,
    /// A thought budget before each answer, in tokens.
    think: u32,
    deltas: Option<Deltas>,
}

impl JevonsLlm {
    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            think: 0,
            deltas: None,
        }
    }

    /// Also sends the answer text to `deltas` as it is written, such as into the bubble.
    pub fn with_deltas(mut self, deltas: Deltas) -> Self {
        self.deltas = Some(deltas);
        self
    }

    pub fn with_think(mut self, tokens: u32) -> Self {
        self.think = tokens;
        self
    }
}

/// adk's conversation as Chat Completions messages.
fn messages(contents: &[Content]) -> Vec<ChatMessage> {
    let mut out = Vec::new();
    for content in contents {
        let text: Vec<&str> = content
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let text = (!text.is_empty()).then(|| text.join("\n"));
        let calls: Vec<Value> = content
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::FunctionCall { name, args, id, .. } => Some(json!({
                    "id": id.clone().unwrap_or_else(|| name.clone()),
                    "type": "function",
                    "function": {"name": name, "arguments": args.to_string()},
                })),
                _ => None,
            })
            .collect();
        let results: Vec<ChatMessage> = content
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::FunctionResponse {
                    function_response,
                    id,
                    ..
                } => Some(ChatMessage {
                    role: "tool".into(),
                    content: Some(match &function_response.response {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    }),
                    tool_call_id: Some(
                        id.clone().unwrap_or_else(|| function_response.name.clone()),
                    ),
                    ..ChatMessage::default()
                }),
                _ => None,
            })
            .collect();
        if !results.is_empty() {
            out.extend(results);
            continue;
        }
        let role = match content.role.as_str() {
            "model" | "assistant" => "assistant",
            "system" => "system",
            _ => "user",
        };
        if !calls.is_empty() {
            out.push(ChatMessage {
                role: "assistant".into(),
                content: text,
                tool_calls: calls,
                ..ChatMessage::default()
            });
        } else if let Some(text) = text {
            out.push(ChatMessage::text(role, text));
        }
    }
    out
}

/// The Chat Completions request for an adk model request.
fn chat_request(model: &str, think: u32, request: &LlmRequest) -> ChatRequest {
    let mut tools: Vec<(&String, &Value)> = request.tools.iter().collect();
    tools.sort_by_key(|(name, _)| *name);
    let config = request.config.as_ref();
    let mut chat = ChatRequest {
        model: model.into(),
        messages: messages(&request.contents),
        tools: tools
            .into_iter()
            .map(|(name, declaration)| {
                let mut function = json!({
                    "name": name,
                    "description": declaration["description"].as_str().unwrap_or_default(),
                });
                if let Some(parameters) = declaration.get("parameters").filter(|p| !p.is_null()) {
                    function["parameters"] = parameters.clone();
                }
                json!({"type": "function", "function": function})
            })
            .collect(),
        max_completion_tokens: config
            .and_then(|c| c.max_output_tokens)
            .and_then(|t| u32::try_from(t).ok()),
        reasoning_effort: (think > 0).then(|| Reasoning::for_budget(think).effort),
        ..ChatRequest::default()
    };
    if let Some(schema) = config.and_then(|c| c.response_schema.clone()) {
        chat = chat.answer_schema(schema);
    }
    chat
}

fn text_chunk(text: &str) -> LlmResponse {
    LlmResponse {
        content: Some(Content {
            role: "model".into(),
            parts: vec![Part::Text { text: text.into() }],
        }),
        partial: true,
        ..LlmResponse::default()
    }
}

#[async_trait]
impl Llm for JevonsLlm {
    fn name(&self) -> &str {
        &self.model
    }

    async fn generate_content(
        &self,
        request: LlmRequest,
        _stream: bool,
    ) -> adk_core::Result<LlmResponseStream> {
        let chat = chat_request(&self.model, self.think, &request);
        let client = self.client.clone();
        let watcher = self.deltas.clone();
        let (sender, receiver) = mpsc::unbounded_channel::<adk_core::Result<LlmResponse>>();
        tokio::spawn(async move {
            let deltas = sender.clone();
            let reply = client
                .chat(&chat, |delta| {
                    if let Some(watcher) = &watcher {
                        watcher(delta);
                    }
                    let _ = deltas.send(Ok(text_chunk(delta)));
                })
                .await;
            let last = match reply {
                // The text went out as deltas: the last chunk only ends the turn.
                Ok(ChatReply::Text(_)) => Ok(LlmResponse {
                    turn_complete: true,
                    finish_reason: Some(FinishReason::Stop),
                    ..LlmResponse::default()
                }),
                Ok(ChatReply::Call {
                    id,
                    name,
                    arguments,
                }) => Ok(LlmResponse {
                    content: Some(Content {
                        role: "model".into(),
                        parts: vec![Part::FunctionCall {
                            name,
                            args: arguments,
                            id: Some(id),
                            thought_signature: None,
                        }],
                    }),
                    turn_complete: true,
                    finish_reason: Some(FinishReason::Stop),
                    ..LlmResponse::default()
                }),
                Err(e) => Err(AdkError::model(e.to_string())),
            };
            let _ = sender.send(last);
        });
        Ok(Box::pin(futures_util::stream::unfold(
            receiver,
            |mut receiver| async move { receiver.recv().await.map(|item| (item, receiver)) },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_core::{FunctionResponseData, GenerateContentConfig};
    use std::collections::HashMap;

    #[test]
    fn adk_contents_become_chat_turns_with_calls_and_results() {
        let contents = vec![
            Content::new("user").with_text("Is it up?"),
            Content {
                role: "model".into(),
                parts: vec![Part::FunctionCall {
                    name: "lookup".into(),
                    args: json!({"q": "status"}),
                    id: Some("call_1".into()),
                    thought_signature: None,
                }],
            },
            Content {
                role: "function".into(),
                parts: vec![Part::FunctionResponse {
                    function_response: FunctionResponseData::new(
                        "lookup",
                        json!({"state": "green"}),
                    ),
                    id: Some("call_1".into()),
                    annotations: None,
                }],
            },
        ];
        let request = LlmRequest {
            model: "jev".into(),
            contents,
            config: Some(GenerateContentConfig {
                max_output_tokens: Some(300),
                response_schema: Some(json!({"type": "object"})),
                ..GenerateContentConfig::default()
            }),
            tools: HashMap::from([(
                "lookup".to_string(),
                json!({"name": "lookup", "description": "Looks up", "parameters": {"type": "object"}}),
            )]),
            previous_response_id: None,
        };
        let chat = chat_request("jev", 256, &request);
        assert_eq!(chat.messages.len(), 3);
        assert_eq!(chat.messages[1].role, "assistant");
        assert_eq!(
            chat.messages[1].tool_calls[0]["function"]["arguments"],
            "{\"q\":\"status\"}"
        );
        assert_eq!(chat.messages[2].role, "tool");
        assert_eq!(chat.messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(chat.tools[0]["function"]["description"], "Looks up");
        assert_eq!(chat.max_completion_tokens, Some(300));
        assert_eq!(chat.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(
            chat.response_format.unwrap()["json_schema"]["schema"],
            json!({"type": "object"})
        );
    }
}
