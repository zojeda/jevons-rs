//! Request validation for the three APIs.
use crate::openai::OpenAiError;
use crate::openai::tools::{History, parse_format, parse_tool_choice, parse_tools};
use jevons_generative::tools::{Schema, Tool, ToolChoice, ToolRequest};
use jevons_generative::{GenerationPrompt, GenerationRequest, MAX_STOP_SEQUENCES, Message, Role};
use serde_json::{Map, Value, json};

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
    /// Function tools the model may call.
    pub tools: Vec<Tool>,
    pub tool_choice: ToolChoice,
    /// The shape of a structured answer (`json_schema` or `json_object` formats).
    pub answer: Option<Schema>,
    /// Echoed by the Responses API.
    pub(crate) instructions: Option<String>,
    pub(crate) echo_tools: Value,
    pub(crate) echo_tool_choice: Value,
    pub(crate) echo_format: Value,
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
        "function" => Err(OpenAiError::unsupported(
            param,
            "function messages; use tool",
        )),
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
    let mut history = History::default();
    items
        .iter()
        .map(|item| {
            let role_name = item["role"].as_str().unwrap_or_default();
            let text = || match &item["content"] {
                Value::Null => Ok(String::new()),
                content => content_text(content, &["text"], "messages"),
            };
            if let Some(calls) = item.get("tool_calls").filter(|v| !v.is_null()) {
                let calls = calls.as_array().ok_or_else(|| {
                    OpenAiError::invalid("tool_calls must be an array", Some("messages"))
                })?;
                let calls: Vec<(&str, &str, &str)> = calls
                    .iter()
                    .map(|c| {
                        (
                            c["id"].as_str().unwrap_or_default(),
                            c["function"]["name"].as_str().unwrap_or_default(),
                            c["function"]["arguments"].as_str().unwrap_or("{}"),
                        )
                    })
                    .collect();
                return Ok(history.calls(&text()?, calls));
            }
            if role_name == "tool" {
                let id = item["tool_call_id"].as_str().ok_or_else(|| {
                    OpenAiError::invalid("tool messages need a tool_call_id", Some("messages"))
                })?;
                return Ok(history.result(id, &text()?));
            }
            let role = role(role_name, "messages")?;
            Ok(Message {
                role,
                text: text()?,
            })
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
            let mut history = History::default();
            for item in items {
                match item["type"].as_str() {
                    None | Some("message") => {}
                    Some("function_call") => {
                        messages.push(history.calls(
                            "",
                            [(
                                item["call_id"].as_str().unwrap_or_default(),
                                item["name"].as_str().unwrap_or_default(),
                                item["arguments"].as_str().unwrap_or("{}"),
                            )],
                        ));
                        continue;
                    }
                    Some("function_call_output") => {
                        let output = match &item["output"] {
                            Value::String(s) => s.clone(),
                            other => content_text(other, &["input_text", "output_text"], "input")?,
                        };
                        messages.push(
                            history.result(item["call_id"].as_str().unwrap_or_default(), &output),
                        );
                        continue;
                    }
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
        let mut tools = Vec::new();
        let mut tool_choice = ToolChoice::Auto;
        let mut answer = None;
        let mut echo_tools = json!([]);
        let mut echo_tool_choice = json!("none");
        let mut echo_format = json!({"type": "text"});
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
                tools = parse_tools(f.get("tools"), "tools")?;
                tool_choice = parse_tool_choice(f.get("tool_choice"), &tools)?;
                f.boolean("parallel_tool_calls")?;
                if let Some(format) = f.get("response_format") {
                    answer = parse_format(format, "response_format")?;
                }
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
                tools = parse_tools(f.get("tools"), "tools")?;
                tool_choice = parse_tool_choice(f.get("tool_choice"), &tools)?;
                f.boolean("parallel_tool_calls")?;
                echo_tools = object.get("tools").cloned().unwrap_or_else(|| json!([]));
                echo_tool_choice = object
                    .get("tool_choice")
                    .cloned()
                    .unwrap_or_else(|| json!(if tools.is_empty() { "none" } else { "auto" }));
                if let Some(text) = f.get("text") {
                    if let Some(key) = text
                        .as_object()
                        .and_then(|t| t.keys().find(|k| *k != "format"))
                    {
                        return Err(OpenAiError::unsupported(
                            &format!("text.{key}"),
                            "only text.format",
                        ));
                    }
                    if let Some(format) = text.get("format") {
                        answer = parse_format(format, "text.format")?;
                        echo_format = format.clone();
                    }
                }
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
            tools,
            tool_choice,
            answer,
            instructions,
            echo_tools,
            echo_tool_choice,
            echo_format,
            temperature,
            top_p,
            effort,
            metadata,
        })
    }
}

impl OpenAiRequest {
    /// Whether the answer needs the tool-calling steps: tools that may be called, or a
    /// structured format.
    pub fn uses_tools(&self) -> bool {
        (!self.tools.is_empty() && self.tool_choice != ToolChoice::None) || self.answer.is_some()
    }

    /// The Generative service's tool request for this conversation.
    pub fn tool_request(&self) -> ToolRequest {
        let messages = match &self.generation.prompt {
            GenerationPrompt::Chat(messages) => messages.clone(),
            GenerationPrompt::Text(text) => vec![Message {
                role: Role::User,
                text: text.clone(),
            }],
        };
        ToolRequest {
            messages,
            tools: self.tools.clone(),
            choice: self.tool_choice.clone(),
            answer: self.answer.clone(),
            max_tokens: self.generation.max_tokens,
            think: self.generation.think,
            stop: self.generation.stop.clone(),
        }
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
            (json!({"tools": [{"type": "web_search"}]}), "tools"),
            (json!({"logprobs": true}), "logprobs"),
            (
                json!({"response_format": {"type": "grammar"}}),
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
            json!({"model": "m", "input": [{"type": "web_search_call", "id": "x"}]}),
            json!({"model": "m", "input": "Hi", "text": {"verbosity": "low"}}),
            json!({"model": "m", "input": "Hi", "reasoning": {"summary": "auto"}}),
            json!({"model": "m", "instructions": "only instructions", "input": []}),
        ] {
            assert!(OpenAiRequest::parse(Api::Responses, &bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn chat_tool_history_and_tools_reach_the_tool_request() {
        let request = OpenAiRequest::parse(
            Api::ChatCompletions,
            &json!({"model": "m",
                "tools": [{"type": "function", "function": {"name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
                "tool_choice": "required", "parallel_tool_calls": false,
                "messages": [
                    {"role": "user", "content": "Weather in Oslo?"},
                    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "3 °C"}
                ]}),
        )
        .unwrap();
        assert!(request.uses_tools());
        let tools = request.tool_request();
        assert_eq!(tools.choice, ToolChoice::Required);
        assert_eq!(tools.messages.len(), 3);
        assert!(
            tools.messages[1]
                .text
                .contains("[Called the tool get_weather")
        );
        assert_eq!(tools.messages[2].role, Role::User);
        assert_eq!(
            tools.messages[2].text,
            "[The tool get_weather returned: 3 °C]"
        );
        let plain = chat(json!({"tools": [{"type": "function", "function": {"name": "x"}}], "tool_choice": "none"})).unwrap();
        assert!(!plain.uses_tools(), "tool_choice none generates as usual");
        let structured = chat(json!({"response_format": {"type": "json_schema",
            "json_schema": {"name": "a", "schema": {"type": "object", "properties": {"n": {"type": "integer"}}}}}}))
        .unwrap();
        assert!(structured.uses_tools() && structured.answer.is_some());
    }

    #[test]
    fn responses_function_items_and_text_format_parse() {
        let request = OpenAiRequest::parse(
            Api::Responses,
            &json!({"model": "m",
                "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object",
                    "properties": {"q": {"type": "string"}}, "required": ["q"]}}],
                "text": {"format": {"type": "json_schema", "name": "out", "schema": {"type": "boolean"}}},
                "input": [
                    {"role": "user", "content": "Is it up?"},
                    {"type": "function_call", "call_id": "c9", "name": "lookup", "arguments": "{\"q\":\"status\"}"},
                    {"type": "function_call_output", "call_id": "c9", "output": "all green"}
                ]}),
        )
        .unwrap();
        let tools = request.tool_request();
        assert_eq!(
            tools.messages[2].text,
            "[The tool lookup returned: all green]"
        );
        assert!(matches!(tools.answer, Some(Schema::Boolean { .. })));
        assert_eq!(request.echo_tool_choice, json!("auto"));
        assert_eq!(request.echo_format["name"], "out");
    }
}
