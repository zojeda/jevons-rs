//! The API log: with `privacy.log_api` on, every decision and generation call's exact request
//! body and its response, one pretty-printed JSON record after another (`jq` reads the file as
//! a stream). Off by default, because the bodies hold what the user said and the text of their
//! screen; keys are headers, never bodies, so they are never written.

use serde::Serialize;
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The size at which the log starts over; the full one is kept as `<name>.previous.log`.
const MAX_BYTES: u64 = 32 << 20;

static ON: AtomicBool = AtomicBool::new(false);
static FILE: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Writes the API log to `file`, or stops writing it with `None`.
pub fn set(file: Option<PathBuf>) {
    let mut current = FILE.lock().expect("the API log lock");
    ON.store(file.is_some(), Ordering::Relaxed);
    *current = file;
}

/// Whether calls are being written.
pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// The API log's file in the logs folder.
pub fn default_file() -> PathBuf {
    crate::config::user_dir().join("logs").join("api.log")
}

/// One call on its way: its request, if the log is on.
pub(super) struct Call {
    api: &'static str,
    began: Instant,
    at_ms: u128,
    request: Option<Value>,
}

impl Call {
    pub(super) fn start(api: &'static str, request: &impl Serialize) -> Self {
        Self {
            api,
            began: Instant::now(),
            at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_millis()),
            request: enabled().then(|| serde_json::to_value(request).unwrap_or_default()),
        }
    }

    /// Records the call with what came back, and the error that ended it if one did.
    pub(super) fn end(mut self, response: Value, error: Option<&super::ClientError>) {
        self.write(response, error.map(|e| e.to_string()));
    }

    fn write(&mut self, response: Value, error: Option<String>) {
        let Some(request) = self.request.take() else {
            return;
        };
        let mut record = json!({
            "at_ms": self.at_ms,
            "api": self.api,
            "ms": self.began.elapsed().as_millis(),
            "request": request,
            "response": response,
        });
        if let Some(error) = error {
            record["error"] = json!(error);
        }
        let file = FILE.lock().expect("the API log lock");
        if let Some(file) = file.as_deref()
            && let Err(e) = append(file, &record)
        {
            tracing::warn!(error = %e, "Cannot write the API log");
        }
    }
}

/// A call dropped before its answer: a time limit ran out or the take was cancelled.
impl Drop for Call {
    fn drop(&mut self) {
        self.write(
            Value::Null,
            Some(
                "abandoned before the answer: a time limit ran out or the take was cancelled"
                    .into(),
            ),
        );
    }
}

fn append(file: &Path, record: &Value) -> std::io::Result<()> {
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if std::fs::metadata(file).is_ok_and(|m| m.len() > MAX_BYTES) {
        std::fs::rename(file, file.with_extension("previous.log"))?;
    }
    let mut text = serde_json::to_string_pretty(record).unwrap_or_default();
    text.push('\n');
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)?
        .write_all(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{Client, DecisionRequest, ResponseRequest};
    use axum::routing::post;

    #[tokio::test]
    async fn the_log_holds_each_call_s_request_and_response_only_while_it_is_on() {
        let app = axum::Router::new()
            .route(
                "/v1/systemone",
                post(|| async {
                    axum::Json(json!({"model": "jev", "usage": {"input_tokens": 1, "output_tokens": 1},
                        "answers": {"q": {"type": "noul", "noul": 0.9}}}))
                }),
            )
            .route(
                "/v1/responses",
                post(|| async {
                    let body = [
                        json!({"type": "response.created"}),
                        json!({"type": "response.output_text.delta", "delta": "Hel"}),
                        json!({"type": "response.output_text.delta", "delta": "lo"}),
                        json!({"type": "response.completed", "response": {"usage": {"output_tokens": 2}}}),
                    ]
                    .iter()
                    .map(|e| format!("data: {e}\n\n"))
                    .collect::<String>();
                    ([("content-type", "text/event-stream")], body)
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await });
        let client = Client::new(&base, Some("secret-key".into()));
        let model = format!("log-test-{}", std::process::id());
        let decision = DecisionRequest {
            model: model.clone(),
            state: "The user said: \"hi\"".into(),
            questions: Default::default(),
            steps: None,
            samples: None,
            think: None,
        };
        let generation = ResponseRequest {
            model: model.clone(),
            instructions: Some("Be brief.".into()),
            input: "hi".into(),
            max_output_tokens: None,
            reasoning: None,
        };
        let file = std::env::temp_dir().join(format!("jevons-api-log-{}.log", std::process::id()));
        let _ = std::fs::remove_file(&file);
        set(Some(file.clone()));
        client.decide(&decision).await.unwrap();
        assert_eq!(client.respond(&generation, |_| {}).await.unwrap(), "Hello");
        set(None);
        client.decide(&decision).await.unwrap();

        // Other tests' calls may land in the file while it is on: only this test's count.
        let text = std::fs::read_to_string(&file).unwrap();
        let records: Vec<Value> = serde_json::Deserializer::from_str(&text)
            .into_iter::<Value>()
            .map(Result::unwrap)
            .filter(|r| r["request"]["model"] == json!(model))
            .collect();
        assert_eq!(records.len(), 2, "{text}");
        assert_eq!(records[0]["api"], "POST /v1/systemone");
        assert_eq!(records[0]["request"]["state"], "The user said: \"hi\"");
        assert_eq!(records[0]["response"]["answers"]["q"]["noul"], 0.9);
        assert_eq!(records[1]["api"], "POST /v1/responses");
        assert_eq!(records[1]["request"]["stream"], true);
        assert_eq!(records[1]["response"]["text"], "Hello");
        assert_eq!(records[1]["response"]["events"], 4);
        assert_eq!(
            records[1]["response"]["other_events"][1]["type"],
            "response.completed"
        );
        assert!(!text.contains("secret-key"));
        std::fs::remove_file(file).unwrap();
    }
}
