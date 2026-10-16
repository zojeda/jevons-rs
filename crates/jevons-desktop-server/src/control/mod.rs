//! The control API: what outside agents (Claude Code, scripts) reach through the server. One
//! table of operations ([`ops`]) and two doors built from it: MCP over streamable HTTP at
//! `/mcp` ([`mcp`]) and REST at `/api/v1/<op>`, described by `/api/openapi.json` ([`rest`]).
//!
//! Every request needs the server's key, as a bearer token, and a request a browser sends (one
//! with an `Origin`) is refused: a web page cannot drive the desk. The server only carries the
//! desk's calls: the client checks each against its own `[control]` settings.

pub mod mcp;
pub mod ops;
pub mod rest;

use crate::session::Session;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use jevons_desktop_protocol::control::{ControlError, ErrorKind};
use std::sync::Arc;

/// The control API over one session.
pub struct ControlApi {
    session: Arc<Session>,
    key: String,
}

/// Why the control API is not served: it has no key.
pub const NO_KEY: &str = "The control API needs a key: set TYPESAFE_API_KEY or [server] api_key";

impl ControlApi {
    /// The control API over `session`, behind `key`; none without a key.
    pub fn new(session: Arc<Session>, key: Option<String>) -> Result<Arc<Self>, String> {
        let key = key.filter(|k| !k.trim().is_empty()).ok_or(NO_KEY)?;
        Ok(Arc::new(Self { session, key }))
    }

    pub fn session(&self) -> &Arc<Session> {
        &self.session
    }

    /// `/mcp`, `/api/v1/<op>` and `/api/openapi.json`, behind the key.
    pub fn router(self: &Arc<Self>) -> axum::Router {
        axum::Router::new()
            .nest_service("/mcp", mcp::service(self.clone()))
            .merge(rest::router(self.clone()))
            .layer(axum::middleware::from_fn_with_state(self.clone(), guard))
    }

    fn admits(&self, headers: &HeaderMap) -> Result<(), ControlError> {
        if headers.contains_key(header::ORIGIN) {
            return Err(ControlError::denied(
                "Requests from a browser are refused: the control API drives the desktop",
            ));
        }
        let bearer = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        match bearer {
            Some(key) if same(key.trim(), &self.key) => Ok(()),
            Some(_) => Err(ControlError::new(
                ErrorKind::Unauthorized,
                "The key is not the server's",
            )),
            None => Err(ControlError::new(
                ErrorKind::Unauthorized,
                "No key: send it as `Authorization: Bearer <key>`",
            )),
        }
    }
}

/// Compares keys without stopping at the first difference.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0u8, |diff, (x, y)| diff | (x ^ y))
            == 0
}

/// The status a failed call answers with over HTTP.
pub fn status(kind: ErrorKind) -> StatusCode {
    match kind {
        ErrorKind::Invalid => StatusCode::BAD_REQUEST,
        ErrorKind::Unauthorized => StatusCode::UNAUTHORIZED,
        ErrorKind::Denied => StatusCode::FORBIDDEN,
        ErrorKind::NotFound => StatusCode::NOT_FOUND,
        ErrorKind::Busy => StatusCode::CONFLICT,
        ErrorKind::Failed => StatusCode::INTERNAL_SERVER_ERROR,
        ErrorKind::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        ErrorKind::Timeout => StatusCode::GATEWAY_TIMEOUT,
    }
}

/// A failed call as REST answers it: `{ "error": { "kind", "message" } }`.
pub fn refusal(error: ControlError) -> Response {
    (
        status(error.kind),
        axum::Json(serde_json::json!({ "error": error })),
    )
        .into_response()
}

async fn guard(State(api): State<Arc<ControlApi>>, request: Request, next: Next) -> Response {
    match api.admits(request.headers()) {
        Ok(()) => next.run(request).await,
        Err(error) => {
            tracing::info!(kind = ?error.kind, "Control API request refused");
            refusal(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::{Catalog, FlowTree, defaults};
    use crate::pipeline::Settings;
    use jevons_desktop_core::control::{Control, ControlSettings};
    use jevons_desktop_core::desk::LocalDesk;
    use jevons_desktop_core::fake::slack_demonstration;
    use jevons_desktop_core::recorded::ReplayActor;
    use jevons_desktop_protocol::control::Confirm;
    use serde_json::{Value, json};

    #[test]
    fn keys_compare_whole() {
        assert!(same("abc", "abc"));
        assert!(!same("abc", "abd"));
        assert!(!same("abc", "abcd"));
        assert!(!same("", "a"));
    }

    /// A server whose desk lets agents read and act in Slack, as recorded, without asking.
    async fn serving() -> (String, Arc<ReplayActor>) {
        let (demonstration, _, _) = slack_demonstration();
        let replay = Arc::new(ReplayActor::new(demonstration));
        let settings = ControlSettings {
            apps: vec!["slack.exe".into()],
            confirm: Confirm::Never,
            ..ControlSettings::default()
        };
        let control = Arc::new(Control::new(replay.clone(), replay.clone(), settings));
        let desk = Arc::new(LocalDesk::default().with_control(control));
        let flows = Arc::new(FlowTree::load(&defaults::builtin(), &Catalog::default()));
        let (session, _) = Session::open(desk, flows, Settings::default());
        assert!(
            ControlApi::new(session.clone(), None).is_err(),
            "no key, no API"
        );
        let api = ControlApi::new(session, Some("secret".into())).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(crate::serve::listen(
            listener,
            None,
            None,
            Some(api),
            std::future::pending(),
        ));
        (format!("http://{address}"), replay)
    }

    #[tokio::test]
    async fn rest_needs_the_key_refuses_browsers_and_reaches_the_desk() {
        let (base, replay) = serving().await;
        let http = reqwest::Client::new();
        let windows = format!("{base}/api/v1/windows");
        assert_eq!(http.get(&windows).send().await.unwrap().status(), 401);
        let wrong = http
            .get(&windows)
            .bearer_auth("guess")
            .send()
            .await
            .unwrap();
        assert_eq!(wrong.status(), 401);
        let browser = http
            .get(&windows)
            .bearer_auth("secret")
            .header("Origin", "https://example.com")
            .send()
            .await
            .unwrap();
        assert_eq!(browser.status(), 403);
        let listed: Value = http
            .get(&windows)
            .bearer_auth("secret")
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(listed["windows"][0]["app"], "slack.exe");
        let post = |op: &str, body: Value| {
            http.post(format!("{base}/api/v1/{op}"))
                .bearer_auth("secret")
                .header(rest::AGENT_HEADER, "script")
                .json(&body)
                .send()
        };
        let found: Value = post("find", json!({"role": "TreeItem", "text": "random"}))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let random = found["elements"][0]["id"].as_str().unwrap().to_string();
        let done = post("act", json!({"node": random, "action": "click"}))
            .await
            .unwrap();
        assert_eq!(done.status(), 200);
        assert_eq!(replay.done(), 1);
        // A wrong input says what is wrong; an operation that takes input takes no GET.
        let bad = post("act", json!({"node": random})).await.unwrap();
        assert_eq!(bad.status(), 400);
        let refusal: Value = bad.json().await.unwrap();
        assert_eq!(refusal["error"]["kind"], "invalid");
        let get = http
            .get(format!("{base}/api/v1/act"))
            .bearer_auth("secret")
            .send()
            .await
            .unwrap();
        assert_eq!(get.status(), 400);
        let status: Value = post("status", Value::Null)
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(status["agent"], "script");
        assert_eq!(status["desk"]["apps"], json!(["slack.exe"]));
        let document = http
            .get(format!("{base}/api/openapi.json"))
            .bearer_auth("secret")
            .send()
            .await
            .unwrap();
        assert_eq!(document.status(), 200);
    }

    #[tokio::test]
    async fn mcp_lists_the_operations_and_calls_them_as_the_client_that_says_its_name() {
        let (base, _) = serving().await;
        let http = reqwest::Client::new();
        let rpc = |body: Value, method: &'static str| {
            let mut request = http
                .post(format!("{base}/mcp"))
                .bearer_auth("secret")
                .header("Accept", "application/json, text/event-stream")
                .header("MCP-Protocol-Version", "2026-07-28")
                .header("Mcp-Method", method);
            if let Some(name) = body["params"]["name"].as_str() {
                request = request.header("Mcp-Name", name);
            }
            request.json(&body).send()
        };
        let meta = json!({
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientInfo": {"name": "claude-code", "version": "2"},
            "io.modelcontextprotocol/clientCapabilities": {},
        });
        let listed: Value = rpc(
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                "params": {"_meta": meta}}),
            "tools/list",
        )
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
        let tools = listed["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), ops::OPS.len(), "{listed}");
        // The revision requires how long the list holds and who may keep it.
        assert!(listed["result"]["ttlMs"].is_u64(), "{listed}");
        assert_eq!(listed["result"]["cacheScope"], "private", "{listed}");
        let called: Value = rpc(
            json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "status", "arguments": {}, "_meta": meta}}),
            "tools/call",
        )
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
        let text = called["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("{called}"));
        let status: Value = serde_json::from_str(text).unwrap();
        assert_eq!(status["agent"], "claude-code");
        // A refusal is a tool error that names its kind.
        let refused: Value = rpc(
            json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "read", "arguments": {"node": "e404"}, "_meta": meta}}),
            "tools/call",
        )
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
        assert_eq!(refused["result"]["isError"], true, "{refused}");
        let text = refused["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("not_found: "), "{text}");
        // A client of an earlier revision starts with `initialize`, and reads the instructions.
        let started = http
            .post(format!("{base}/mcp"))
            .bearer_auth("secret")
            .header("Accept", "application/json, text/event-stream")
            .json(
                &json!({"jsonrpc": "2.0", "id": 5, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "claude-code", "version": "2"}}}),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(started.status(), 200);
        let body = started.text().await.unwrap();
        let answer: Value = body
            .lines()
            .find_map(|line| serde_json::from_str(line.trim_start_matches("data:").trim()).ok())
            .unwrap_or_else(|| panic!("{body}"));
        assert_eq!(answer["result"]["serverInfo"]["name"], "jevons-desktop");
        assert!(
            answer["result"]["instructions"]
                .as_str()
                .unwrap()
                .contains("`status` first")
        );
        // Without the key nothing answers.
        let anonymous = http
            .post(format!("{base}/mcp"))
            .json(&json!({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}))
            .send()
            .await
            .unwrap();
        assert_eq!(anonymous.status(), 401);
    }
}
