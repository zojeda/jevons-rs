//! One bounded agent run on adk-rust: an [`LlmAgent`](adk_agent::LlmAgent) over the jevons model,
//! with tools, an optional structured answer, and confirmation before the tools that need it.
//! The context investigator and `agent.toml` nodes both run through [`run`].

use adk_agent::LlmAgentBuilder;
use adk_core::{
    Content, Llm, Part, RunConfig, StreamingMode, Tool, ToolConfirmationHandler,
    ToolConfirmationPolicy, Toolset,
};
use adk_runner::Runner;
use adk_session::{CreateRequest, InMemorySessionService, SessionService};
use futures_util::StreamExt;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::Arc;

/// The most characters of a tool result the trace keeps.
const TRACED_RESULT: usize = 600;

/// What an agent is asked to do.
pub struct Task {
    /// A short identifier, such as `investigator`.
    pub name: String,
    /// The system instruction.
    pub instruction: String,
    /// The user turn: the task and its context.
    pub input: String,
    pub tools: Vec<Arc<dyn Tool>>,
    pub toolsets: Vec<Arc<dyn Toolset>>,
    /// Model turns before the agent must answer.
    pub max_steps: u32,
    /// The JSON Schema of a structured answer.
    pub output_schema: Option<Value>,
    /// Tools that ask before they run.
    pub confirm: BTreeSet<String>,
    /// Who approves them; without one they are denied.
    pub confirmer: Option<Arc<dyn ToolConfirmationHandler>>,
}

/// A tool call the agent made.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CallRecord {
    pub tool: String,
    pub arguments: Value,
    /// The result, truncated for the trace.
    pub result: Option<String>,
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Outcome {
    /// The final answer: text, or JSON text with an output schema.
    pub text: String,
    pub calls: Vec<CallRecord>,
}

fn truncated(value: &Value) -> String {
    let text = match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if text.chars().count() <= TRACED_RESULT {
        text
    } else {
        let mut short: String = text.chars().take(TRACED_RESULT).collect();
        short.push('…');
        short
    }
}

/// Runs `task` on `model` to its answer.
pub async fn run(model: Arc<dyn Llm>, task: Task) -> Result<Outcome, String> {
    let mut builder = LlmAgentBuilder::new(&task.name)
        .instruction(task.instruction)
        .model(model)
        .max_iterations(task.max_steps.max(1));
    for tool in task.tools {
        builder = builder.tool(tool);
    }
    for toolset in task.toolsets {
        builder = builder.toolset(toolset);
    }
    if let Some(schema) = task.output_schema {
        builder = builder.output_schema(schema);
    }
    if !task.confirm.is_empty() {
        builder = builder.tool_confirmation_policy(ToolConfirmationPolicy::PerTool(task.confirm));
    }
    let agent = Arc::new(builder.build().map_err(|e| e.to_string())?);
    let sessions = Arc::new(InMemorySessionService::new());
    let (app, user, session) = ("jevons", "user", "take");
    sessions
        .create(CreateRequest {
            app_name: app.into(),
            user_id: user.into(),
            session_id: Some(session.into()),
            state: Default::default(),
        })
        .await
        .map_err(|e| e.to_string())?;
    let mut config = RunConfig::builder().streaming_mode(StreamingMode::None);
    if let Some(confirmer) = task.confirmer {
        config = config.tool_confirmation_handler(confirmer);
    }
    let runner = Runner::builder()
        .app_name(app)
        .agent(agent)
        .session_service(sessions)
        .run_config(config.build())
        .build()
        .map_err(|e| e.to_string())?;
    let mut events = runner
        .run_str(user, session, Content::new("user").with_text(task.input))
        .await
        .map_err(|e| e.to_string())?;
    let mut calls: Vec<CallRecord> = Vec::new();
    let mut text = String::new();
    while let Some(event) = events.next().await {
        let event = event.map_err(|e| e.to_string())?;
        let Some(content) = event.llm_response.content.as_ref() else {
            continue;
        };
        let mut said = String::new();
        let mut called = false;
        for part in &content.parts {
            match part {
                Part::Text { text } => said.push_str(text),
                Part::FunctionCall { name, args, .. } => {
                    called = true;
                    calls.push(CallRecord {
                        tool: name.clone(),
                        arguments: args.clone(),
                        result: None,
                    });
                }
                Part::FunctionResponse {
                    function_response, ..
                } => {
                    if let Some(call) = calls
                        .iter_mut()
                        .rev()
                        .find(|c| c.tool == function_response.name && c.result.is_none())
                    {
                        call.result = Some(truncated(&function_response.response));
                    }
                }
                _ => {}
            }
        }
        if !called && !said.trim().is_empty() && event.author != "user" {
            text = said;
        }
    }
    Ok(Outcome { text, calls })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::client::Client;
    use crate::flow::llm::JevonsLlm;
    use adk_core::{ToolConfirmationDecision, ToolConfirmationRequest, ToolContext, async_trait};
    use axum::Json;
    use axum::routing::post;
    use serde_json::json;
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

    /// A tool that echoes its `text` argument.
    struct Echo;

    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes the text."
        }
        fn parameters_schema(&self) -> Option<Value> {
            Some(
                json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
            )
        }
        async fn execute(&self, _: Arc<dyn ToolContext>, args: Value) -> adk_core::Result<Value> {
            Ok(json!({"echoed": args["text"]}))
        }
    }

    #[derive(Debug)]
    struct Deny;

    #[async_trait]
    impl ToolConfirmationHandler for Deny {
        async fn decide(
            &self,
            _: &ToolConfirmationRequest,
        ) -> adk_core::Result<ToolConfirmationDecision> {
            Ok(ToolConfirmationDecision::Deny)
        }
    }

    fn task() -> Task {
        Task {
            name: "test".into(),
            instruction: "Use the tools.".into(),
            input: "Say hi through echo.".into(),
            tools: vec![Arc::new(Echo)],
            toolsets: Vec::new(),
            max_steps: 4,
            output_schema: None,
            confirm: BTreeSet::new(),
            confirmer: None,
        }
    }

    #[tokio::test]
    async fn an_agent_calls_a_tool_reads_its_result_and_answers() {
        let (client, seen) = chat_server(vec![
            json!({"call": "echo", "arguments": {"text": "hi"}}),
            json!("It said hi."),
        ])
        .await;
        let outcome = run(Arc::new(JevonsLlm::new(client, "jev")), task())
            .await
            .unwrap();
        assert_eq!(outcome.text, "It said hi.");
        assert_eq!(outcome.calls.len(), 1);
        assert_eq!(outcome.calls[0].arguments, json!({"text": "hi"}));
        assert!(
            outcome.calls[0]
                .result
                .as_deref()
                .unwrap()
                .contains("\"hi\"")
        );
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0]["tools"][0]["function"]["name"], "echo");
        let second = requests[1]["messages"].as_array().unwrap();
        assert!(second.iter().any(|m| m["role"] == "tool"), "{second:?}");
    }

    #[tokio::test]
    async fn a_denied_confirmation_keeps_the_tool_from_running() {
        let (client, _) = chat_server(vec![
            json!({"call": "echo", "arguments": {"text": "hi"}}),
            json!("I was not allowed."),
        ])
        .await;
        let mut guarded = task();
        guarded.confirm = BTreeSet::from(["echo".to_string()]);
        guarded.confirmer = Some(Arc::new(Deny));
        let outcome = run(Arc::new(JevonsLlm::new(client, "jev")), guarded)
            .await
            .unwrap();
        assert_eq!(outcome.text, "I was not allowed.");
        let result = outcome.calls[0].result.clone().unwrap_or_default();
        assert!(!result.contains("echoed"), "{result}");
    }
}
