//! Request validation for the three APIs.
use crate::OpenAiError;
use jevons_core::{GenerationPrompt, GenerationRequest, MAX_STOP_SEQUENCES, Message, Role};
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    /// `POST /v1/chat/completions`
    ChatCompletions,
    /// `POST /v1/completions` (legacy text completions)
    Completions,
    /// `POST /v1/responses`
    Responses,
}

/// A validated request: the generation to run and how to answer.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiRequest {
    pub api: Api,
    pub model: String,
    pub generation: GenerationRequest,
    pub stream: bool,
    /// Chat and text completions: add a usage chunk to the stream.
    pub include_usage: bool,
    pub seed: Option<u64>,
    /// Echoed by the Responses API.
    pub(crate) instructions: Option<String>,
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) effort: Option<String>,
    pub(crate) metadata: Value,
}

/// Thought budgets for `reasoning_effort` / `reasoning.effort`.
fn thought_budget(effort: &str, param: &str) -> Result<usize, OpenAiError> {
    Ok(match effort {
        "none" => 0,
        "minimal" => 64,
        "low" => 256,
        "medium" => 1024,
        "high" | "xhigh" => 4096,
        _ => {
            return Err(OpenAiError::invalid(
                format!("{param} must be none, minimal, low, medium or high"),
                Some(param),
            ));
        }
    })
}

/// Reads the fields of one request object, tracking which were used.
struct Fields<'a> {
    object: &'a Map<String, Value>,
    known: Vec<&'static str>,
}

impl<'a> Fields<'a> {
    fn get(&mut self, key: &'static str) -> Option<&'a Value> {
        self.known.push(key);
        self.object.get(key).filter(|v| !v.is_null())
    }

    fn string(&mut self, key: &'static str) -> Result<Option<&'a str>, OpenAiError> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(_) => Err(OpenAiError::invalid(
                format!("{key} must be a string"),
                Some(key),
            )),
        }
    }

    fn boolean(&mut self, key: &'static str) -> Result<bool, OpenAiError> {
        match self.get(key) {
            None => Ok(false),
            Some(Value::Bool(b)) => Ok(*b),
            Some(_) => Err(OpenAiError::invalid(
                format!("{key} must be a boolean"),
                Some(key),
            )),
        }
    }

    fn integer(
        &mut self,
        key: &'static str,
        min: u64,
        max: u64,
    ) -> Result<Option<u64>, OpenAiError> {
        match self.get(key) {
            None => Ok(None),
            Some(value) => match value.as_u64() {
                Some(n) if (min..=max).contains(&n) => Ok(Some(n)),
                _ => Err(OpenAiError::invalid(
                    format!("{key} must be an integer from {min} to {max}"),
                    Some(key),
                )),
            },
        }
    }

    fn number(
        &mut self,
        key: &'static str,
        min: f64,
        max: f64,
    ) -> Result<Option<f64>, OpenAiError> {
        match self.get(key) {
            None => Ok(None),
            Some(value) => match value.as_f64() {
                Some(x) if (min..=max).contains(&x) => Ok(Some(x)),
                _ => Err(OpenAiError::invalid(
                    format!("{key} must be a number from {min} to {max}"),
                    Some(key),
                )),
            },
        }
    }

    /// Accepts the field only at its no-op value.
    fn only(
        &mut self,
        key: &'static str,
        accepted: &[Value],
        detail: &str,
    ) -> Result<(), OpenAiError> {
        match self.get(key) {
            Some(value) if !accepted.contains(value) => Err(OpenAiError::unsupported(key, detail)),
            _ => Ok(()),
        }
    }

    /// Accepts the field only when it is absent or an empty array.
    fn none_or_empty(&mut self, key: &'static str, detail: &str) -> Result<(), OpenAiError> {
        match self.get(key) {
            Some(Value::Array(items)) if items.is_empty() => Ok(()),
            Some(_) => Err(OpenAiError::unsupported(key, detail)),
            None => Ok(()),
        }
    }

    /// Accepted for compatibility; has no effect.
    fn ignore(&mut self, keys: &[&'static str]) {
        self.known.extend(keys);
    }

    fn finish(self) -> Result<(), OpenAiError> {
        match self
            .object
            .keys()
            .find(|k| !self.known.contains(&k.as_str()))
        {
            Some(key) => Err(OpenAiError::unsupported(key, "unknown parameter")),
            None => Ok(()),
        }
    }
}

fn stop_sequences(value: Option<&Value>) -> Result<Vec<String>, OpenAiError> {
    let invalid = || {
        OpenAiError::invalid(
            format!("stop must be a string or up to {MAX_STOP_SEQUENCES} nonempty strings"),
            Some("stop"),
        )
    };
    let stops = match value {
        None => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().map(String::from))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(invalid)?,
        Some(_) => return Err(invalid()),
    };
    if stops.len() > MAX_STOP_SEQUENCES || stops.iter().any(String::is_empty) {
        return Err(invalid());
    }
    Ok(stops)
}

fn role(name: &str, param: &str) -> Result<Role, OpenAiError> {
    match name {
        "system" | "developer" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" | "function" => Err(OpenAiError::unsupported(param, "tool messages")),
        _ => Err(OpenAiError::invalid(
            format!("unknown role {name:?}"),
            Some(param),
        )),
    }
}

/// Text of message content: a string, or text parts of the given types.
fn content_text(content: &Value, part_types: &[&str], param: &str) -> Result<String, OpenAiError> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let kind = part["type"].as_str().unwrap_or_default();
                if !part_types.contains(&kind) {
                    return Err(OpenAiError::unsupported(
                        param,
                        &format!("content part {kind:?}; only text is supported"),
                    ));
                }
                text.push_str(part["text"].as_str().ok_or_else(|| {
                    OpenAiError::invalid("text parts need a text string", Some(param))
                })?);
            }
            Ok(text)
        }
        _ => Err(OpenAiError::invalid(
            "content must be a string or an array of parts",
            Some(param),
        )),
    }
}

fn chat_messages(value: Option<&Value>) -> Result<Vec<Message>, OpenAiError> {
    let Some(Value::Array(items)) = value else {
        return Err(OpenAiError::invalid(
            "messages must be a nonempty array",
            Some("messages"),
        ));
    };
    items
        .iter()
        .map(|item| {
            if item.get("tool_calls").is_some_and(|v| !v.is_null()) {
                return Err(OpenAiError::unsupported("messages", "tool calls"));
            }
            let role = role(item["role"].as_str().unwrap_or_default(), "messages")?;
            let text = content_text(&item["content"], &["text"], "messages")?;
            Ok(Message { role, text })
        })
        .collect()
}

fn response_input(
    value: Option<&Value>,
    instructions: Option<&str>,
) -> Result<Vec<Message>, OpenAiError> {
    let mut messages: Vec<Message> = instructions
        .map(|text| Message {
            role: Role::System,
            text: text.into(),
        })
        .into_iter()
        .collect();
    match value {
        Some(Value::String(text)) => messages.push(Message {
            role: Role::User,
            text: text.clone(),
        }),
        Some(Value::Array(items)) => {
            for item in items {
                match item["type"].as_str() {
                    None | Some("message") => {}
                    Some(kind) => {
                        return Err(OpenAiError::unsupported("input", &format!("{kind} items")));
                    }
                }
                messages.push(Message {
                    role: role(item["role"].as_str().unwrap_or_default(), "input")?,
                    text: content_text(&item["content"], &["input_text", "output_text"], "input")?,
                });
            }
        }
        _ => {
            return Err(OpenAiError::invalid(
                "input must be a string or an array",
                Some("input"),
            ));
        }
    }
    Ok(messages)
}

impl OpenAiRequest {
    pub fn parse(api: Api, body: &Value) -> Result<Self, OpenAiError> {
        let Some(object) = body.as_object() else {
            return Err(OpenAiError::invalid(
                "The request body must be a JSON object",
                None,
            ));
        };
        let mut f = Fields {
            object,
            known: Vec::new(),
        };
        let model = f
            .string("model")?
            .ok_or_else(|| OpenAiError::invalid("model is required", Some("model")))?
            .to_string();
        let stream = f.boolean("stream")?;
        let seed = f.integer("seed", 0, u64::MAX)?;
        let temperature = f.number("temperature", 0.0, 2.0)?;
        let top_p = f.number("top_p", 0.0, 1.0)?;
        f.ignore(&[
            "user",
            "metadata",
            "store",
            "service_tier",
            "safety_identifier",
            "prompt_cache_key",
        ]);
        let metadata = object.get("metadata").cloned().unwrap_or(Value::Null);
        let mut include_usage = false;
        let (prompt, max_tokens, think, stop, instructions, effort) = match api {
            Api::ChatCompletions => {
                let messages = chat_messages(f.get("messages"))?;
                if messages.is_empty() {
                    return Err(OpenAiError::invalid(
                        "messages must be a nonempty array",
                        Some("messages"),
                    ));
                }
                let newer = f.integer("max_completion_tokens", 1, u32::MAX as u64)?;
                let max = newer.or(f.integer("max_tokens", 1, u32::MAX as u64)?);
                let effort = f.string("reasoning_effort")?.map(String::from);
                let think = effort
                    .as_deref()
                    .map_or(Ok(0), |e| thought_budget(e, "reasoning_effort"))?;
                let stop = stop_sequences(f.get("stop"))?;
                f.only("n", &[1.into()], "only one choice is generated")?;
                f.only("logprobs", &[false.into()], "log probabilities")?;
                f.only("top_logprobs", &[0.into()], "log probabilities")?;
                f.only("frequency_penalty", &[0.into(), 0.0.into()], "penalties")?;
                f.only("presence_penalty", &[0.into(), 0.0.into()], "penalties")?;
                f.none_or_empty("tools", "tools")?;
                f.only("tool_choice", &["none".into(), "auto".into()], "tools")?;
                f.only("parallel_tool_calls", &[true.into(), false.into()], "tools")?;
                f.only(
                    "response_format",
                    &[serde_json::json!({"type": "text"})],
                    "structured output",
                )?;
                f.only(
                    "modalities",
                    &[serde_json::json!(["text"])],
                    "only text output",
                )?;
                f.none_or_empty("logit_bias", "logit bias")?;
                if let Some(options) = f.get("stream_options") {
                    include_usage = options["include_usage"].as_bool().unwrap_or(false);
                }
                (
                    GenerationPrompt::Chat(messages),
                    max,
                    think,
                    stop,
                    None,
                    effort,
                )
            }
            Api::Completions => {
                let prompt = match f.get("prompt") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(items)) if items.len() == 1 && items[0].is_string() => {
                        items[0].as_str().unwrap_or_default().to_string()
                    }
                    Some(Value::Array(_)) => {
                        return Err(OpenAiError::unsupported("prompt", "only one text prompt"));
                    }
                    _ => {
                        return Err(OpenAiError::invalid(
                            "prompt must be a string",
                            Some("prompt"),
                        ));
                    }
                };
                let max = f.integer("max_tokens", 1, u32::MAX as u64)?.unwrap_or(16);
                let stop = stop_sequences(f.get("stop"))?;
                f.only("n", &[1.into()], "only one choice is generated")?;
                f.only("best_of", &[1.into()], "only one choice is generated")?;
                f.only("echo", &[false.into()], "echoing the prompt")?;
                f.only("logprobs", &[0.into()], "log probabilities")?;
                f.only("frequency_penalty", &[0.into(), 0.0.into()], "penalties")?;
                f.only("presence_penalty", &[0.into(), 0.0.into()], "penalties")?;
                f.only("suffix", &[], "insertion")?;
                f.none_or_empty("logit_bias", "logit bias")?;
                if let Some(options) = f.get("stream_options") {
                    include_usage = options["include_usage"].as_bool().unwrap_or(false);
                }
                (
                    GenerationPrompt::Text(prompt),
                    Some(max),
                    0,
                    stop,
                    None,
                    None,
                )
            }
            Api::Responses => {
                let instructions = f.string("instructions")?.map(String::from);
                let messages = response_input(f.get("input"), instructions.as_deref())?;
                if !messages.iter().any(|m| m.role != Role::System) {
                    return Err(OpenAiError::invalid(
                        "input needs at least one message",
                        Some("input"),
                    ));
                }
                let max = f.integer("max_output_tokens", 1, u32::MAX as u64)?;
                let effort = match f.get("reasoning") {
                    None => None,
                    Some(reasoning) => {
                        if reasoning.get("summary").is_some_and(|s| !s.is_null()) {
                            return Err(OpenAiError::unsupported(
                                "reasoning.summary",
                                "thoughts stay internal",
                            ));
                        }
                        reasoning["effort"].as_str().map(String::from)
                    }
                };
                let think = effort
                    .as_deref()
                    .map_or(Ok(0), |e| thought_budget(e, "reasoning.effort"))?;
                f.only("previous_response_id", &[], "responses are not stored")?;
                f.only("conversation", &[], "responses are not stored")?;
                f.only("background", &[false.into()], "background responses")?;
                f.none_or_empty("tools", "tools")?;
                f.only("tool_choice", &["none".into(), "auto".into()], "tools")?;
                f.only("parallel_tool_calls", &[true.into(), false.into()], "tools")?;
                f.only(
                    "text",
                    &[serde_json::json!({"format": {"type": "text"}})],
                    "structured output",
                )?;
                f.only("truncation", &["disabled".into()], "truncation")?;
                f.none_or_empty("include", "extra output")?;
                f.only("top_logprobs", &[0.into()], "log probabilities")?;
                (
                    GenerationPrompt::Chat(messages),
                    max,
                    think,
                    Vec::new(),
                    instructions,
                    effort,
                )
            }
        };
        f.finish()?;
        let generation = GenerationRequest {
            prompt,
            max_tokens: max_tokens.map(|n| n as usize),
            think,
            stop,
        };
        generation
            .validate()
            .map_err(|e| OpenAiError::invalid(e.to_string(), None))?;
        Ok(Self {
            api,
            model,
            generation,
            stream,
            include_usage,
            seed,
            instructions,
            temperature,
            top_p,
            effort,
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat(extra: Value) -> Result<OpenAiRequest, OpenAiError> {
        let mut body = json!({"model": "m", "messages": [
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "text", "text": "!"}]},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Again"}
        ]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        OpenAiRequest::parse(Api::ChatCompletions, &body)
    }

    #[test]
    fn chat_messages_map_roles_and_text_parts() {
        let request = chat(json!({"max_tokens": 32, "stop": "\n\n", "stream": true,
            "stream_options": {"include_usage": true}, "temperature": 0.7, "seed": 5}))
        .unwrap();
        let GenerationPrompt::Chat(messages) = &request.generation.prompt else {
            panic!()
        };
        let roles: Vec<_> = messages.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            [Role::System, Role::User, Role::Assistant, Role::User]
        );
        assert_eq!(messages[1].text, "Hi!");
        assert_eq!(request.generation.max_tokens, Some(32));
        assert_eq!(request.generation.stop, ["\n\n"]);
        assert!(request.stream && request.include_usage);
        assert_eq!(request.seed, Some(5));
        assert_eq!(
            chat(json!({"max_completion_tokens": 8, "max_tokens": 99}))
                .unwrap()
                .generation
                .max_tokens,
            Some(8)
        );
        assert_eq!(
            chat(json!({"reasoning_effort": "low"}))
                .unwrap()
                .generation
                .think,
            256
        );
    }

    #[test]
    fn unsupported_parameters_are_rejected_not_ignored() {
        for (extra, param) in [
            (json!({"n": 2}), "n"),
            (json!({"tools": [{"type": "function"}]}), "tools"),
            (json!({"logprobs": true}), "logprobs"),
            (
                json!({"response_format": {"type": "json_object"}}),
                "response_format",
            ),
            (json!({"frequency_penalty": 0.5}), "frequency_penalty"),
            (json!({"surprise": 1}), "surprise"),
        ] {
            let error = chat(extra).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param), "{error:?}");
            assert_eq!(error.status, 400);
        }
        for extra in [
            json!({"n": 1, "tools": [], "logprobs": false, "tool_choice": "none", "user": "u"}),
            json!({"response_format": {"type": "text"}, "presence_penalty": 0}),
        ] {
            assert!(chat(extra).is_ok());
        }
        let image = json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:"}}]}]});
        assert!(OpenAiRequest::parse(Api::ChatCompletions, &image).is_err());
        let tool = json!({"model": "m", "messages": [{"role": "tool", "content": "x"}]});
        assert!(OpenAiRequest::parse(Api::ChatCompletions, &tool).is_err());
        assert!(chat(json!({"stop": ["a", "b", "c", "d", "e"]})).is_err());
        assert!(chat(json!({"max_tokens": 0})).is_err());
    }

    #[test]
    fn completions_take_one_text_prompt_and_default_to_16_tokens() {
        let request =
            OpenAiRequest::parse(Api::Completions, &json!({"model": "m", "prompt": ["Once"]}))
                .unwrap();
        assert_eq!(
            request.generation.prompt,
            GenerationPrompt::Text("Once".into())
        );
        assert_eq!(request.generation.max_tokens, Some(16));
        for bad in [
            json!({"model": "m", "prompt": [[1, 2]]}),
            json!({"model": "m", "prompt": "x", "echo": true}),
            json!({"model": "m", "prompt": "x", "suffix": "y"}),
            json!({"model": "m"}),
        ] {
            assert!(
                OpenAiRequest::parse(Api::Completions, &bad).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn responses_prepend_instructions_and_accept_message_items() {
        let request = OpenAiRequest::parse(
            Api::Responses,
            &json!({
                "model": "m", "instructions": "Be brief.", "max_output_tokens": 20,
                "reasoning": {"effort": "medium"},
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
                          {"type": "message", "role": "assistant", "content": "Hello."},
                          {"role": "user", "content": "Again"}]
            }),
        )
        .unwrap();
        let GenerationPrompt::Chat(messages) = &request.generation.prompt else {
            panic!()
        };
        assert_eq!(
            messages[0],
            Message {
                role: Role::System,
                text: "Be brief.".into()
            }
        );
        assert_eq!(messages.len(), 4);
        assert_eq!(request.generation.think, 1024);
        assert_eq!(request.generation.max_tokens, Some(20));
        let simple =
            OpenAiRequest::parse(Api::Responses, &json!({"model": "m", "input": "Hi"})).unwrap();
        assert_eq!(simple.generation.max_tokens, None);
        for bad in [
            json!({"model": "m", "input": "Hi", "previous_response_id": "resp_1"}),
            json!({"model": "m", "input": [{"type": "function_call_output", "output": "x"}]}),
            json!({"model": "m", "input": "Hi", "reasoning": {"summary": "auto"}}),
            json!({"model": "m", "instructions": "only instructions", "input": []}),
        ] {
            assert!(OpenAiRequest::parse(Api::Responses, &bad).is_err(), "{bad}");
        }
    }
}
