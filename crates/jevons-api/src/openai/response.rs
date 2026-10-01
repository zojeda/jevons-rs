//! Response bodies and streaming events.
use crate::openai::{Api, OpenAiRequest};
use jevons_generative::{FinishReason, Generation};
use serde_json::{Value, json};

/// One server-sent event: an optional event name and its data line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub name: Option<&'static str>,
    pub data: String,
}

impl Event {
    pub(crate) fn data(value: Value) -> Self {
        Self {
            name: None,
            data: value.to_string(),
        }
    }
}

fn finish_reason(finish: FinishReason) -> &'static str {
    match finish {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

/// A tool call the model made: the answer of a turn that calls a tool instead of replying.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub name: String,
    /// The arguments as a JSON object, serialized.
    pub arguments: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

impl Call {
    fn usage(&self) -> Generation {
        Generation {
            text: String::new(),
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            reasoning_tokens: 0,
            finish: FinishReason::Stop,
        }
    }
}

fn completion_usage(g: &Generation) -> Value {
    json!({
        "prompt_tokens": g.prompt_tokens,
        "completion_tokens": g.completion_tokens,
        "total_tokens": g.prompt_tokens + g.completion_tokens,
        "completion_tokens_details": {"reasoning_tokens": g.reasoning_tokens},
    })
}

impl OpenAiRequest {
    /// The complete (non-streaming) response. `id` is unique; `created` is Unix seconds.
    pub fn response(&self, id: &str, created: u64, model: &str, g: &Generation) -> Value {
        match self.api {
            Api::ChatCompletions => json!({
                "id": format!("chatcmpl-{id}"),
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": g.text, "refusal": null, "annotations": []},
                    "logprobs": null,
                    "finish_reason": finish_reason(g.finish),
                }],
                "usage": completion_usage(g),
            }),
            Api::Completions => json!({
                "id": format!("cmpl-{id}"),
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"text": g.text, "index": 0, "logprobs": null, "finish_reason": finish_reason(g.finish)}],
                "usage": completion_usage(g),
            }),
            Api::Responses => self.response_object(id, created, model, Some(g)),
        }
    }

    /// The complete response of a turn that calls a tool.
    pub fn call_response(&self, id: &str, created: u64, model: &str, call: &Call) -> Value {
        match self.api {
            Api::Responses => {
                let mut body = self.response_object(id, created, model, Some(&call.usage()));
                body["output"] = json!([Self::call_item(id, call, "completed")]);
                body
            }
            _ => json!({
                "id": format!("chatcmpl-{id}"),
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": null, "refusal": null,
                        "annotations": [], "tool_calls": [Self::chat_call(id, call)]},
                    "logprobs": null,
                    "finish_reason": "tool_calls",
                }],
                "usage": completion_usage(&call.usage()),
            }),
        }
    }

    fn chat_call(id: &str, call: &Call) -> Value {
        json!({"id": format!("call_{id}"), "type": "function",
            "function": {"name": call.name, "arguments": call.arguments}})
    }

    fn call_item(id: &str, call: &Call, status: &str) -> Value {
        let arguments = if status == "completed" {
            call.arguments.as_str()
        } else {
            ""
        };
        json!({"type": "function_call", "id": format!("fc_{id}"), "call_id": format!("call_{id}"),
            "name": call.name, "arguments": arguments, "status": status})
    }

    fn message_item(id: &str, text: &str, status: &str) -> Value {
        let content = if status == "in_progress" {
            json!([])
        } else {
            json!([{"type": "output_text", "text": text, "annotations": [], "logprobs": []}])
        };
        json!({"type": "message", "id": format!("msg_{id}"), "status": status, "role": "assistant", "content": content})
    }

    /// A Responses API `response` object; in progress when `g` is `None`.
    fn response_object(
        &self,
        id: &str,
        created: u64,
        model: &str,
        g: Option<&Generation>,
    ) -> Value {
        let (status, incomplete, output, usage) = match g {
            None => ("in_progress", Value::Null, json!([]), Value::Null),
            Some(g) => {
                let (status, incomplete) = match g.finish {
                    FinishReason::Stop => ("completed", Value::Null),
                    FinishReason::Length => ("incomplete", json!({"reason": "max_output_tokens"})),
                };
                let usage = json!({
                    "input_tokens": g.prompt_tokens,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": g.completion_tokens,
                    "output_tokens_details": {"reasoning_tokens": g.reasoning_tokens},
                    "total_tokens": g.prompt_tokens + g.completion_tokens,
                });
                (
                    status,
                    incomplete,
                    json!([Self::message_item(id, &g.text, status)]),
                    usage,
                )
            }
        };
        json!({
            "id": format!("resp_{id}"),
            "object": "response",
            "created_at": created,
            "status": status,
            "error": null,
            "incomplete_details": incomplete,
            "instructions": self.instructions,
            "max_output_tokens": self.generation.max_tokens,
            "model": model,
            "output": output,
            "parallel_tool_calls": false,
            "previous_response_id": null,
            "reasoning": {"effort": self.effort, "summary": null},
            "store": false,
            "temperature": self.temperature.unwrap_or(1.0),
            "text": {"format": self.echo_format},
            "tool_choice": self.echo_tool_choice,
            "tools": self.echo_tools,
            "top_p": self.top_p.unwrap_or(1.0),
            "truncation": "disabled",
            "usage": usage,
            "metadata": if self.metadata.is_object() { self.metadata.clone() } else { json!({}) },
        })
    }

    /// Event renderer for a streamed answer.
    pub fn stream(&self, id: &str, created: u64, model: &str) -> Stream {
        Stream {
            request: self.clone(),
            id: id.into(),
            created,
            model: model.into(),
            sequence: 0,
            text: String::new(),
        }
    }
}

/// Renders the events of one streamed answer: [`Stream::start`], a [`Stream::delta`] per text
/// piece, then [`Stream::finish`] (or [`Stream::error`]).
pub struct Stream {
    request: OpenAiRequest,
    id: String,
    created: u64,
    model: String,
    sequence: u64,
    text: String,
}

impl Stream {
    fn chunk(&self, choice: Value, usage: Option<Value>) -> Value {
        let mut chunk = match self.request.api {
            Api::ChatCompletions => json!({
                "id": format!("chatcmpl-{}", self.id),
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": choice,
            }),
            _ => json!({
                "id": format!("cmpl-{}", self.id),
                "object": "text_completion",
                "created": self.created,
                "model": self.model,
                "choices": choice,
            }),
        };
        if self.request.include_usage {
            chunk["usage"] = usage.unwrap_or(Value::Null);
        }
        chunk
    }

    fn event(&mut self, name: &'static str, mut body: Value) -> Event {
        body["type"] = name.into();
        body["sequence_number"] = self.sequence.into();
        self.sequence += 1;
        Event {
            name: Some(name),
            data: body.to_string(),
        }
    }

    fn item_id(&self) -> String {
        format!("msg_{}", self.id)
    }

    pub fn start(&mut self) -> Vec<Event> {
        match self.request.api {
            Api::ChatCompletions => vec![Event::data(self.chunk(
                json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null}]),
                None,
            ))],
            Api::Completions => Vec::new(),
            Api::Responses => {
                let response = self.request.response_object(&self.id, self.created, &self.model, None);
                let item = OpenAiRequest::message_item(&self.id, "", "in_progress");
                let part = json!({"type": "output_text", "text": "", "annotations": [], "logprobs": []});
                let item_id = self.item_id();
                vec![
                    self.event("response.created", json!({"response": response})),
                    self.event("response.in_progress", json!({"response": response})),
                    self.event("response.output_item.added", json!({"output_index": 0, "item": item})),
                    self.event("response.content_part.added", json!({
                        "item_id": item_id, "output_index": 0, "content_index": 0, "part": part})),
                ]
            }
        }
    }

    pub fn delta(&mut self, text: &str) -> Vec<Event> {
        self.text.push_str(text);
        match self.request.api {
            Api::ChatCompletions => vec![Event::data(self.chunk(
                json!([{"index": 0, "delta": {"content": text}, "logprobs": null, "finish_reason": null}]),
                None,
            ))],
            Api::Completions => vec![Event::data(self.chunk(
                json!([{"text": text, "index": 0, "logprobs": null, "finish_reason": null}]),
                None,
            ))],
            Api::Responses => {
                let item_id = self.item_id();
                vec![self.event("response.output_text.delta", json!({
                    "item_id": item_id, "output_index": 0, "content_index": 0, "delta": text, "logprobs": []}))]
            }
        }
    }

    /// Final events, ending with `[DONE]` for the completion APIs.
    pub fn finish(&mut self, g: &Generation) -> Vec<Event> {
        let reason = finish_reason(g.finish);
        let done = Event {
            name: None,
            data: "[DONE]".into(),
        };
        match self.request.api {
            Api::ChatCompletions | Api::Completions => {
                let choice = if self.request.api == Api::ChatCompletions {
                    json!([{"index": 0, "delta": {}, "logprobs": null, "finish_reason": reason}])
                } else {
                    json!([{"text": "", "index": 0, "logprobs": null, "finish_reason": reason}])
                };
                let mut events = vec![Event::data(self.chunk(choice, None))];
                if self.request.include_usage {
                    events.push(Event::data(
                        self.chunk(json!([]), Some(completion_usage(g))),
                    ));
                }
                events.push(done);
                events
            }
            Api::Responses => {
                let response =
                    self.request
                        .response_object(&self.id, self.created, &self.model, Some(g));
                let item = response["output"][0].clone();
                let part = item["content"][0].clone();
                let item_id = self.item_id();
                let text = g.text.clone();
                let last = if g.finish == FinishReason::Stop {
                    "response.completed"
                } else {
                    "response.incomplete"
                };
                vec![
                    self.event("response.output_text.done", json!({
                        "item_id": item_id, "output_index": 0, "content_index": 0, "text": text, "logprobs": []})),
                    self.event("response.content_part.done", json!({
                        "item_id": item_id, "output_index": 0, "content_index": 0, "part": part})),
                    self.event("response.output_item.done", json!({"output_index": 0, "item": item})),
                    self.event(last, json!({"response": response})),
                ]
            }
        }
    }

    /// Every event of a streamed turn that calls a tool: it is known whole once decided, so the
    /// arguments come in one delta.
    pub fn call(&mut self, call: &Call) -> Vec<Event> {
        let done = Event {
            name: None,
            data: "[DONE]".into(),
        };
        match self.request.api {
            Api::Responses => {
                let started =
                    self.request
                        .response_object(&self.id, self.created, &self.model, None);
                let mut finished = self.request.response_object(
                    &self.id,
                    self.created,
                    &self.model,
                    Some(&call.usage()),
                );
                let item = OpenAiRequest::call_item(&self.id, call, "completed");
                finished["output"] = json!([item]);
                let item_id = format!("fc_{}", self.id);
                let arguments = call.arguments.clone();
                vec![
                    self.event("response.created", json!({"response": started})),
                    self.event("response.in_progress", json!({"response": started})),
                    self.event("response.output_item.added", json!({"output_index": 0,
                        "item": OpenAiRequest::call_item(&self.id, call, "in_progress")})),
                    self.event("response.function_call_arguments.delta", json!({
                        "item_id": item_id, "output_index": 0, "delta": arguments})),
                    self.event("response.function_call_arguments.done", json!({
                        "item_id": item_id, "output_index": 0, "name": call.name, "arguments": call.arguments})),
                    self.event("response.output_item.done", json!({"output_index": 0, "item": item})),
                    self.event("response.completed", json!({"response": finished})),
                ]
            }
            _ => {
                let opening = json!({"index": 0, "id": format!("call_{}", self.id), "type": "function",
                    "function": {"name": call.name, "arguments": ""}});
                let mut events = vec![
                    Event::data(self.chunk(
                        json!([{"index": 0, "delta": {"role": "assistant", "content": null}, "logprobs": null, "finish_reason": null}]),
                        None,
                    )),
                    Event::data(self.chunk(
                        json!([{"index": 0, "delta": {"tool_calls": [opening]}, "logprobs": null, "finish_reason": null}]),
                        None,
                    )),
                    Event::data(self.chunk(
                        json!([{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": call.arguments}}]},
                            "logprobs": null, "finish_reason": null}]),
                        None,
                    )),
                    Event::data(self.chunk(
                        json!([{"index": 0, "delta": {}, "logprobs": null, "finish_reason": "tool_calls"}]),
                        None,
                    )),
                ];
                if self.request.include_usage {
                    events.push(Event::data(
                        self.chunk(json!([]), Some(completion_usage(&call.usage()))),
                    ));
                }
                events.push(done);
                events
            }
        }
    }

    /// A failure after the stream started.
    pub fn error(&mut self, error: &crate::openai::OpenAiError) -> Vec<Event> {
        match self.request.api {
            Api::Responses => vec![self.event("error", json!({
                "code": error.code.unwrap_or(error.kind), "message": error.message, "param": error.param}))],
            _ => vec![Event::data(error.body()), Event { name: None, data: "[DONE]".into() }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::Api;

    fn request(api: Api, body: Value) -> OpenAiRequest {
        OpenAiRequest::parse(api, &body).unwrap()
    }

    fn generation(finish: FinishReason) -> Generation {
        Generation {
            text: "Hello there".into(),
            prompt_tokens: 10,
            completion_tokens: 3,
            reasoning_tokens: 0,
            finish,
        }
    }

    fn data(events: &[Event]) -> Vec<Value> {
        events
            .iter()
            .filter(|e| e.data != "[DONE]")
            .map(|e| serde_json::from_str(&e.data).unwrap())
            .collect()
    }

    #[test]
    fn chat_responses_and_chunks_follow_the_openai_shapes() {
        let chat = request(
            Api::ChatCompletions,
            json!({"model": "m", "messages": [{"role": "user", "content": "Hi"}],
            "stream": true, "stream_options": {"include_usage": true}}),
        );
        let body = chat.response("abc", 7, "local", &generation(FinishReason::Length));
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello there");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["total_tokens"], 13);
        let mut stream = chat.stream("abc", 7, "local");
        let mut events = stream.start();
        events.extend(stream.delta("Hello"));
        events.extend(stream.delta(" there"));
        events.extend(stream.finish(&generation(FinishReason::Stop)));
        assert_eq!(events.last().unwrap().data, "[DONE]");
        let chunks = data(&events);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let text: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, "Hello there");
        assert_eq!(chunks[3]["choices"][0]["finish_reason"], "stop");
        assert!(chunks[3]["usage"].is_null());
        assert_eq!(chunks[4]["choices"], json!([]));
        assert_eq!(chunks[4]["usage"]["completion_tokens"], 3);
        assert!(events.iter().all(|e| e.name.is_none()));
    }

    #[test]
    fn text_completions_stream_text_choices() {
        let completion = request(
            Api::Completions,
            json!({"model": "m", "prompt": "Once", "stream": true}),
        );
        let mut stream = completion.stream("x", 1, "local");
        let mut events = stream.start();
        events.extend(stream.delta("upon"));
        events.extend(stream.finish(&generation(FinishReason::Length)));
        let chunks = data(&events);
        assert_eq!(chunks[0]["object"], "text_completion");
        assert_eq!(chunks[0]["choices"][0]["text"], "upon");
        assert_eq!(chunks[1]["choices"][0]["finish_reason"], "length");
        assert!(chunks[0].get("usage").is_none());
        let body = completion.response("x", 1, "local", &generation(FinishReason::Stop));
        assert_eq!(body["choices"][0]["text"], "Hello there");
    }

    fn call() -> Call {
        Call {
            name: "get_weather".into(),
            arguments: "{\"city\":\"Oslo\"}".into(),
            prompt_tokens: 30,
            completion_tokens: 8,
        }
    }

    #[test]
    fn chat_tool_calls_follow_the_openai_shapes() {
        let chat = request(
            Api::ChatCompletions,
            json!({"model": "m", "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "stream": true, "stream_options": {"include_usage": true}}),
        );
        let body = chat.call_response("abc", 7, "local", &call());
        let message = &body["choices"][0]["message"];
        assert!(message["content"].is_null());
        assert_eq!(message["tool_calls"][0]["id"], "call_abc");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Oslo\"}"
        );
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(body["usage"]["total_tokens"], 38);
        let chunks = data(&chat.stream("abc", 7, "local").call(&call()));
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        let arguments: String = chunks
            .iter()
            .filter_map(|c| {
                c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
            })
            .collect();
        assert_eq!(arguments, "{\"city\":\"Oslo\"}");
        assert_eq!(chunks[3]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(chunks[4]["usage"]["prompt_tokens"], 30);
    }

    #[test]
    fn responses_tool_calls_are_function_call_items() {
        let responses = request(
            Api::Responses,
            json!({"model": "m", "input": "Weather?", "stream": true,
            "tools": [{"type": "function", "name": "get_weather"}]}),
        );
        let body = responses.call_response("r2", 5, "local", &call());
        assert_eq!(body["output"][0]["type"], "function_call");
        assert_eq!(body["output"][0]["call_id"], "call_r2");
        assert_eq!(body["tools"][0]["name"], "get_weather");
        assert_eq!(body["tool_choice"], "auto");
        let events = responses.stream("r2", 5, "local").call(&call());
        let names: Vec<_> = events.iter().map(|e| e.name.unwrap()).collect();
        assert_eq!(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let bodies = data(&events);
        assert_eq!(bodies[2]["item"]["arguments"], "");
        assert_eq!(bodies[4]["arguments"], "{\"city\":\"Oslo\"}");
        assert_eq!(bodies[6]["response"]["output"][0]["status"], "completed");
    }

    #[test]
    fn responses_emit_the_item_lifecycle_with_sequence_numbers() {
        let responses = request(
            Api::Responses,
            json!({"model": "m", "input": "Hi", "instructions": "Brief.",
            "max_output_tokens": 3, "stream": true}),
        );
        let body = responses.response("r1", 5, "local", &generation(FinishReason::Length));
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "incomplete");
        assert_eq!(body["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(body["output"][0]["content"][0]["text"], "Hello there");
        assert_eq!(body["instructions"], "Brief.");
        assert_eq!(body["usage"]["output_tokens"], 3);
        let mut stream = responses.stream("r1", 5, "local");
        let mut events = stream.start();
        events.extend(stream.delta("Hello there"));
        events.extend(stream.finish(&generation(FinishReason::Stop)));
        let names: Vec<_> = events.iter().map(|e| e.name.unwrap()).collect();
        assert_eq!(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let bodies = data(&events);
        for (i, body) in bodies.iter().enumerate() {
            assert_eq!(body["sequence_number"], i as u64);
            assert_eq!(body["type"], names[i]);
        }
        assert_eq!(bodies[0]["response"]["status"], "in_progress");
        assert_eq!(bodies[8]["response"]["status"], "completed");
        assert_eq!(bodies[4]["item_id"], "msg_r1");
    }
}
