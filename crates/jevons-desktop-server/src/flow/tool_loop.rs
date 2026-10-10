//! The context investigator's model loop: each turn is one chat request over the tools. A call
//! the model makes runs, and its result joins the conversation; the first turn without a call is
//! the answer, in the JSON Schema asked for. The request constrains the answer to the schema, and
//! the conversation also states it in words, which is what tells the model what each field is
//! for.

use super::tools::Tool;
use crate::client::{ChatMessage, ChatReply, ChatRequest, Client};
use serde_json::{Value, json};
use std::sync::Arc;

/// How many times an answer that is not JSON is asked for again.
const ANSWER_RETRIES: u32 = 3;

/// What the model is asked to do.
pub struct Task {
    /// What it is and how it works. It goes as the first user turn, before the schema in words.
    pub instruction: String,
    /// The user turn: the task and its context.
    pub input: String,
    pub tools: Vec<Arc<dyn Tool>>,
    /// Model turns before it must have answered.
    pub max_steps: u32,
    /// The JSON Schema of the answer.
    pub schema: Value,
}

/// The tools as Chat Completions declares them, sorted by name.
fn declarations(tools: &[Arc<dyn Tool>]) -> Vec<Value> {
    let mut tools: Vec<&Arc<dyn Tool>> = tools.iter().collect();
    tools.sort_by(|a, b| a.name().cmp(b.name()));
    tools
        .into_iter()
        .map(|tool| {
            let mut function = json!({"name": tool.name(), "description": tool.description()});
            if let Some(parameters) = tool.parameters() {
                function["parameters"] = parameters;
            }
            json!({"type": "function", "function": function})
        })
        .collect()
}

/// The user turn that states the answer's schema.
fn schema_in_words(schema: &Value) -> String {
    format!(
        "You MUST respond with valid JSON conforming to this schema: {schema}. Do not include any \
         text outside the JSON object."
    )
}

/// Runs `task` on `model` to its answer.
pub async fn run(client: &Client, model: &str, task: Task) -> Result<Value, String> {
    let mut request = ChatRequest {
        model: model.into(),
        messages: vec![
            ChatMessage::text("user", task.instruction),
            ChatMessage::text("user", schema_in_words(&task.schema)),
            ChatMessage::text("user", task.input),
        ],
        ..ChatRequest::default()
    }
    .answer_schema(task.schema);
    let steps = task.max_steps.max(1);
    let mut asked_again = 0;
    for _ in 0..steps {
        // A tool's arguments may depend on what the last call showed.
        request.tools = declarations(&task.tools);
        let reply = client
            .chat(&request, |_| {})
            .await
            .map_err(|e| e.to_string())?;
        match reply {
            ChatReply::Text(text) => {
                if let Ok(answer) = serde_json::from_str(&text) {
                    return Ok(answer);
                }
                if asked_again == ANSWER_RETRIES {
                    return Err(format!(
                        "the answer was not JSON after {ANSWER_RETRIES} more tries"
                    ));
                }
                asked_again += 1;
                request.messages.push(ChatMessage::text("assistant", text));
                request.messages.push(ChatMessage::text(
                    "user",
                    "Your answer was not JSON. Answer again, with JSON in the schema asked for.",
                ));
            }
            ChatReply::Call {
                id,
                name,
                arguments,
            } => {
                let id = if id.is_empty() { name.clone() } else { id };
                let result = match task.tools.iter().find(|tool| tool.name() == name) {
                    Some(tool) => tool
                        .call(arguments.clone())
                        .await
                        .unwrap_or_else(|e| json!({"error": e})),
                    None => json!({"error": format!("There is no tool {name}")}),
                };
                request.messages.push(ChatMessage {
                    role: "assistant".into(),
                    tool_calls: vec![json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments.to_string()},
                    })],
                    ..ChatMessage::default()
                });
                request.messages.push(ChatMessage {
                    role: "tool".into(),
                    content: Some(match result {
                        Value::String(text) => text,
                        other => other.to_string(),
                    }),
                    tool_call_id: Some(id),
                    ..ChatMessage::default()
                });
            }
        }
    }
    Err(format!("no answer in {steps} steps"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use axum::Json;
    use axum::routing::post;
    use futures_util::future::BoxFuture;
    use std::sync::Mutex;

    /// A chat server that replies with `replies` in order, as streamed chunks: text, or a call
    /// (`{"call": name, "arguments": {...}}`). It records every request.
    pub(crate) async fn chat_server(replies: Vec<Value>) -> (Client, Arc<Mutex<Vec<Value>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let queue = Arc::new(Mutex::new(std::collections::VecDeque::from(replies)));
        let s = seen.clone();
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<Value>| {
                let s = s.clone();
                let queue = queue.clone();
                async move {
                    s.lock().unwrap().push(body);
                    let reply = queue.lock().unwrap().pop_front().unwrap_or(json!("(no reply)"));
                    let chunk = match &reply {
                        Value::String(text) => json!({"choices": [{"delta": {"content": text}}]}),
                        call => json!({"choices": [{"delta": {"tool_calls": [{"index": 0,
                            "id": format!("call_{}", call["call"].as_str().unwrap()), "type": "function",
                            "function": {"name": call["call"], "arguments": call["arguments"].to_string()}}]}}]}),
                    };
                    let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
                    ([("content-type", "text/event-stream")], body)
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (Client::new(&base, None), seen)
    }

    /// A tool that echoes its `text` argument, and fails on `fail`. Each call adds a word its
    /// `text` may be next.
    #[derive(Default)]
    struct Echo {
        words: Mutex<Vec<String>>,
    }

    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes the text."
        }
        fn parameters(&self) -> Option<Value> {
            let words = self.words.lock().unwrap().clone();
            Some(
                json!({"type": "object", "properties": {"text": {"type": "string", "enum": words}},
                "required": ["text"]}),
            )
        }
        fn call(&self, arguments: Value) -> BoxFuture<'_, Result<Value, String>> {
            Box::pin(async move {
                let text = arguments["text"].as_str().unwrap_or_default().to_string();
                if text == "fail" {
                    return Err("nothing to echo".into());
                }
                self.words.lock().unwrap().push(format!("after {text}"));
                Ok(json!(format!("echoed {text}")))
            })
        }
    }

    /// A tool that only has a name.
    struct Bare;

    impl Tool for Bare {
        fn name(&self) -> &str {
            "bare"
        }
        fn description(&self) -> &str {
            ""
        }
        fn parameters(&self) -> Option<Value> {
            None
        }
        fn call(&self, _: Value) -> BoxFuture<'_, Result<Value, String>> {
            Box::pin(async { Ok(json!({"done": true})) })
        }
    }

    fn task(max_steps: u32) -> Task {
        Task {
            instruction: "Use the tools.".into(),
            input: "Say hi through echo.".into(),
            tools: vec![Arc::new(Echo::default()), Arc::new(Bare)],
            max_steps,
            schema: json!({"type": "object", "properties": {"said": {"type": "string"}}}),
        }
    }

    #[tokio::test]
    async fn a_call_and_its_result_reach_the_model_as_chat_messages() {
        let (client, seen) = chat_server(vec![
            json!({"call": "echo", "arguments": {"text": "hi"}}),
            json!({"call": "bare", "arguments": {}}),
            json!(r#"{"said": "hi"}"#),
        ])
        .await;
        let answer = run(&client, "jev", task(4)).await.unwrap();
        assert_eq!(answer, json!({"said": "hi"}));
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 3);
        let first = &requests[0];
        assert_eq!(first["model"], "jev");
        assert_eq!(first["stream"], true);
        assert_eq!(first["response_format"]["type"], "json_schema");
        assert_eq!(
            first["messages"],
            json!([
                {"role": "user", "content": "Use the tools."},
                {"role": "user", "content": "You MUST respond with valid JSON conforming to this \
                    schema: {\"type\":\"object\",\"properties\":{\"said\":{\"type\":\"string\"}}}. Do not \
                    include any text outside the JSON object."},
                {"role": "user", "content": "Say hi through echo."},
            ])
        );
        // Sorted by name, each with what it has.
        assert_eq!(
            first["tools"],
            json!([
                {"type": "function", "function": {"name": "bare", "description": ""}},
                {"type": "function", "function": {"name": "echo", "description": "Echoes the text.",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string", "enum": []}},
                        "required": ["text"]}}},
            ])
        );
        let third = requests[2]["messages"].as_array().unwrap();
        assert_eq!(
            third[3..],
            [
                json!({"role": "assistant", "tool_calls": [{"id": "call_echo", "type": "function",
                    "function": {"name": "echo", "arguments": "{\"text\":\"hi\"}"}}]}),
                // Text goes as it is, anything else as JSON.
                json!({"role": "tool", "content": "echoed hi", "tool_call_id": "call_echo"}),
                json!({"role": "assistant", "tool_calls": [{"id": "call_bare", "type": "function",
                    "function": {"name": "bare", "arguments": "{}"}}]}),
                json!({"role": "tool", "content": "{\"done\":true}", "tool_call_id": "call_bare"}),
            ]
        );
    }

    #[tokio::test]
    async fn tools_are_described_again_before_each_turn() {
        let (client, seen) = chat_server(vec![
            json!({"call": "echo", "arguments": {"text": "hi"}}),
            json!("{}"),
        ])
        .await;
        run(&client, "jev", task(4)).await.unwrap();
        let requests = seen.lock().unwrap();
        let words = |request: &Value| {
            request["tools"][1]["function"]["parameters"]["properties"]["text"]["enum"].clone()
        };
        assert_eq!(words(&requests[0]), json!([]));
        assert_eq!(words(&requests[1]), json!(["after hi"]));
    }

    #[tokio::test]
    async fn a_tool_s_error_reaches_the_model_and_the_run_goes_on() {
        let (client, seen) = chat_server(vec![
            json!({"call": "echo", "arguments": {"text": "fail"}}),
            json!({"call": "nope", "arguments": {}}),
            json!(r#"{"said": null}"#),
        ])
        .await;
        let answer = run(&client, "jev", task(4)).await.unwrap();
        assert_eq!(answer, json!({"said": null}));
        let requests = seen.lock().unwrap();
        let last = requests[2]["messages"].as_array().unwrap();
        assert_eq!(last[4]["content"], r#"{"error":"nothing to echo"}"#);
        assert_eq!(last[6]["content"], r#"{"error":"There is no tool nope"}"#);
    }

    #[tokio::test]
    async fn a_run_past_its_steps_fails() {
        let call = || json!({"call": "echo", "arguments": {"text": "hi"}});
        let (client, seen) = chat_server(vec![call(), call(), call()]).await;
        let failed = run(&client, "jev", task(2)).await.unwrap_err();
        assert_eq!(failed, "no answer in 2 steps");
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn an_answer_that_is_not_json_is_asked_for_again() {
        let (client, seen) =
            chat_server(vec![json!("It said hi."), json!(r#"{"said": "hi"}"#)]).await;
        let answer = run(&client, "jev", task(4)).await.unwrap();
        assert_eq!(answer, json!({"said": "hi"}));
        let second = seen.lock().unwrap()[1]["messages"].clone();
        assert_eq!(
            second[3],
            json!({"role": "assistant", "content": "It said hi."})
        );
        assert_eq!(second[4]["role"], "user");

        // Three more tries, then the run fails.
        let (client, seen) = chat_server(vec![json!("no"); 5]).await;
        let failed = run(&client, "jev", task(8)).await.unwrap_err();
        assert_eq!(failed, "the answer was not JSON after 3 more tries");
        assert_eq!(seen.lock().unwrap().len(), 4);
    }
}
