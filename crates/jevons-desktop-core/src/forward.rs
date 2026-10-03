//! Forwarding: the API the app serves to other clients. A `/v1/*` request goes to the provider
//! that serves its model, with that provider's key in place of the client's, and the answer
//! streams back as it comes. Nothing is rewritten, so a client sees what the provider said.
//!
//! The provider is the one a route asks for the request's model; a model no route names goes
//! to the route of the path's capability (`/v1/systemone` to the decision route), which is how
//! a provider's other names for its models (`jev-latest`) still reach it.

use crate::client::Routes;
use crate::client::log::{self, Call};
use crate::config::Capability;
use axum::body::{Body, Bytes};
use axum::extract::ws::{self, WebSocket, WebSocketUpgrade};
use axum::extract::{FromRequestParts, Request, State};
use axum::http::{HeaderMap, HeaderName, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get};
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::sync::{Arc, RwLock};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{self, protocol::CloseFrame};

/// The largest request body forwarded, as jevons-api's own limit.
const MAX_BODY: usize = 64 * 1024 * 1024;

/// The most of an answer the API log keeps.
const MAX_LOGGED: usize = 1024 * 1024;

/// Headers that belong to one connection, or that the forwarder sets itself. The answer is
/// asked for uncompressed, so the API log reads it.
const NOT_FORWARDED: [HeaderName; 9] = [
    header::HOST,
    header::AUTHORIZATION,
    header::ACCEPT_ENCODING,
    header::CONTENT_LENGTH,
    header::CONNECTION,
    header::TRANSFER_ENCODING,
    header::UPGRADE,
    header::TE,
    header::TRAILER,
];

/// Where one capability's requests go.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub capability: Capability,
    /// The model its route asks the provider for.
    pub model: String,
    /// The provider's name in the settings.
    pub provider: String,
    /// The provider's root, without `/v1`.
    pub base: String,
    pub key: Option<String>,
}

/// The routes as forwarding targets.
pub fn targets(routes: &Routes) -> Vec<Target> {
    [
        (Capability::Speech, &routes.speech),
        (Capability::Realtime, &routes.realtime),
        (Capability::Decision, &routes.decision),
        (Capability::Generation, &routes.generation),
    ]
    .into_iter()
    .filter_map(|(capability, route)| {
        let route = route.as_ref()?;
        Some(Target {
            capability,
            model: route.model.clone(),
            provider: route.provider.clone(),
            base: route.client.base().to_string(),
            key: route.client.key().map(String::from),
        })
    })
    .collect()
}

/// The capability a path asks for.
fn capability(path: &str) -> Option<Capability> {
    match path {
        "/v1/audio/transcriptions" => Some(Capability::Speech),
        "/v1/realtime" => Some(Capability::Realtime),
        "/v1/systemone" => Some(Capability::Decision),
        "/v1/responses" | "/v1/chat/completions" | "/v1/completions" => {
            Some(Capability::Generation)
        }
        _ => None,
    }
}

/// Where a request goes: the route that asks for its model, else the route of its path's
/// capability.
fn pick<'a>(targets: &'a [Target], path: &str, model: Option<&str>) -> Option<&'a Target> {
    model
        .and_then(|model| targets.iter().find(|t| t.model == model))
        .or_else(|| {
            let capability = capability(path)?;
            targets.iter().find(|t| t.capability == capability)
        })
}

/// The model a request names: in its JSON body, in a multipart form's `model` field, or in
/// the query.
fn model_of(
    headers: &HeaderMap,
    query: Option<&str>,
    body: &[u8],
    json: Option<&Value>,
) -> Option<String> {
    let form = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|kind| kind.starts_with("multipart/form-data"));
    json.and_then(|body| body["model"].as_str().map(String::from))
        .or_else(|| form.then(|| form_field(body, "model")).flatten())
        .or_else(|| {
            query?
                .split('&')
                .find_map(|pair| pair.strip_prefix("model="))
                .map(String::from)
        })
}

/// A text field of a multipart form, without parsing the files around it.
fn form_field(body: &[u8], name: &str) -> Option<String> {
    let marker = format!("name=\"{name}\"");
    let at = find(body, marker.as_bytes())?;
    let rest = &body[at..];
    let value = &rest[find(rest, b"\r\n\r\n")? + 4..];
    let value = &value[..find(value, b"\r\n")?];
    Some(String::from_utf8_lossy(value).trim().to_string()).filter(|v| !v.is_empty())
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// An error of the forwarder's own, in jevons-api's shape.
fn error(status: StatusCode, kind: &str, message: impl AsRef<str>) -> Response {
    let body = json!({"detail": {"error_type": kind, "message": message.as_ref()}});
    (status, Json(body)).into_response()
}

/// The API other clients use: what the routes serve, behind the app's own key.
#[derive(Clone)]
pub struct Forwarder {
    targets: Arc<RwLock<Arc<Vec<Target>>>>,
    key: Option<Arc<str>>,
    http: reqwest::Client,
}

impl Forwarder {
    /// `key` is what clients must send; without one the API is open, as jevons-rs is.
    pub fn new(key: Option<String>) -> Self {
        Self {
            targets: Arc::default(),
            key: key.filter(|k| !k.is_empty()).map(Arc::from),
            http: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(5))
                .build()
                .expect("the HTTP client builds"),
        }
    }

    /// Replaces where requests go, keeping the listener.
    pub fn route(&self, targets: Vec<Target>) {
        *self.targets.write().expect("the targets lock") = Arc::new(targets);
    }

    fn targets(&self) -> Arc<Vec<Target>> {
        self.targets.read().expect("the targets lock").clone()
    }

    pub fn app(&self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models))
            .route("/v1/realtime", get(realtime))
            .route("/v1/{*path}", any(forward))
            .with_state(self.clone())
    }

    /// Serves until `shutdown` resolves.
    pub async fn serve(
        self,
        listener: tokio::net::TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        axum::serve(listener, self.app())
            .with_graceful_shutdown(shutdown)
            .await
    }

    /// The refusal of a request without the app's key, as jevons-api gives it. The key comes
    /// in `Authorization`, or for a WebSocket a browser opened, as the subprotocol
    /// `openai-insecure-api-key.<key>`.
    fn refused(&self, headers: &HeaderMap) -> Option<Response> {
        let expected = self.key.as_deref()?;
        let offered = headers
            .get(header::SEC_WEBSOCKET_PROTOCOL)
            .and_then(|v| v.to_str().ok())
            .and_then(|protocols| {
                protocols
                    .split(',')
                    .map(str::trim)
                    .find_map(|p| p.strip_prefix("openai-insecure-api-key."))
            });
        let bearer = headers
            .get(header::AUTHORIZATION)
            .map(|v| v.to_str().ok().and_then(|v| v.strip_prefix("Bearer ")));
        match (bearer, offered) {
            (Some(Some(key)), _) | (None, Some(key)) if key == expected => None,
            (None, None) => Some(error(
                StatusCode::FORBIDDEN,
                "authentication_error",
                "No API key provided",
            )),
            _ => Some(error(
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "Invalid API key",
            )),
        }
    }
}

/// `GET /health`: the model serving each service, as jevons-api answers, so another jevons
/// app can use this one as a `jevons` provider.
async fn health(State(forwarder): State<Forwarder>) -> Json<Value> {
    let targets = forwarder.targets();
    let model = |capability: Capability| {
        targets
            .iter()
            .find(|t| t.capability == capability)
            .map(|t| t.model.clone())
    };
    Json(json!({
        "status": "ok",
        "services": {
            "generative": model(Capability::Generation),
            "decision": model(Capability::Decision),
            "speech": model(Capability::Speech),
        },
    }))
}

/// `GET /v1/models`: the models the routes ask for, in System One's listing (`models`) and
/// OpenAI's (`object`, `data`). A request names no model, so no provider is asked.
async fn models(State(forwarder): State<Forwarder>, headers: HeaderMap) -> Response {
    if let Some(refused) = forwarder.refused(&headers) {
        return refused;
    }
    let targets = forwarder.targets();
    let mut listed: Vec<&Target> = Vec::new();
    for target in targets.iter() {
        if !listed.iter().any(|t| t.model == target.model) {
            listed.push(target);
        }
    }
    Json(json!({
        "models": listed.iter().map(|t| json!({
            "name": t.model,
            "description": format!("Served by {}", t.provider),
        })).collect::<Vec<_>>(),
        "object": "list",
        "data": listed.iter().map(|t| json!({
            "id": t.model,
            "object": "model",
            "created": 0,
            "owned_by": t.provider,
        })).collect::<Vec<_>>(),
    }))
    .into_response()
}

/// Any other `/v1/*` request: forwarded to the provider that serves its model.
async fn forward(State(forwarder): State<Forwarder>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    if let Some(refused) = forwarder.refused(&parts.headers) {
        return refused;
    }
    let body = match axum::body::to_bytes(body, MAX_BODY).await {
        Ok(body) => body,
        Err(_) => {
            return error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
                "The request body is too large",
            );
        }
    };
    let path = parts.uri.path();
    let json = serde_json::from_slice::<Value>(&body).ok();
    let model = model_of(&parts.headers, parts.uri.query(), &body, json.as_ref());
    let targets = forwarder.targets();
    let Some(target) = pick(&targets, path, model.as_deref()) else {
        let asked = model.map_or_else(String::new, |m| format!(" for the model {m:?}"));
        return error(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("No route serves {path}{asked}"),
        );
    };
    let url = match parts.uri.query() {
        Some(query) => format!("{}{path}?{query}", target.base),
        None => format!("{}{path}", target.base),
    };
    // The API log keeps decision and generation calls: JSON bodies, never audio.
    let logged = json.filter(|_| log::enabled()).map(|request| {
        Call::start(
            format!("{} {path} → {}", parts.method, target.provider),
            &request,
        )
    });
    let mut upstream = forwarder.http.request(parts.method.clone(), url);
    for (name, value) in &parts.headers {
        if !NOT_FORWARDED.contains(name) {
            upstream = upstream.header(name, value);
        }
    }
    if let Some(key) = &target.key {
        upstream = upstream.bearer_auth(key);
    }
    let answer = match upstream.body(body).send().await {
        Ok(answer) => answer,
        Err(e) => {
            let message = format!("Cannot reach {}: {e}", target.provider);
            if let Some(call) = logged {
                call.finish(Value::Null, Some(message.clone()));
            }
            return error(StatusCode::BAD_GATEWAY, "api_error", message);
        }
    };
    let status = answer.status();
    let mut response = Response::builder().status(status);
    for (name, value) in answer.headers() {
        if !NOT_FORWARDED.contains(name) {
            response = response.header(name, value);
        }
    }
    let kept = logged.map(|call| Kept {
        call,
        status: status.as_u16(),
        body: Vec::new(),
        truncated: false,
    });
    // The answer goes on as it arrives; the log, when on, keeps a copy of what went by.
    let stream = futures_util::stream::unfold(
        (Box::pin(answer.bytes_stream()), kept),
        |(mut answer, mut kept)| async move {
            match answer.next().await {
                Some(Ok(chunk)) => {
                    if let Some(kept) = kept.as_mut() {
                        kept.keep(&chunk);
                    }
                    Some((Ok(chunk), (answer, kept)))
                }
                Some(Err(e)) => {
                    if let Some(kept) = kept.take() {
                        kept.end(Some(e.to_string()));
                    }
                    Some((Err(e), (answer, None)))
                }
                None => {
                    if let Some(kept) = kept.take() {
                        kept.end(None);
                    }
                    None
                }
            }
        },
    );
    response
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

/// A forwarded answer on its way to the API log.
struct Kept {
    call: Call,
    status: u16,
    body: Vec<u8>,
    truncated: bool,
}

impl Kept {
    fn keep(&mut self, chunk: &Bytes) {
        let room = MAX_LOGGED.saturating_sub(self.body.len());
        self.body.extend_from_slice(&chunk[..chunk.len().min(room)]);
        self.truncated |= chunk.len() > room;
    }

    /// Records the answer: its status, and its body as JSON when it is, as text otherwise (a
    /// stream of events).
    fn end(self, error: Option<String>) {
        let body = serde_json::from_slice::<Value>(&self.body)
            .unwrap_or_else(|_| json!(String::from_utf8_lossy(&self.body)));
        let mut response = json!({"status": self.status, "body": body});
        if self.truncated {
            response["truncated"] = json!(true);
        }
        self.call.finish(response, error);
    }
}

/// `GET /v1/realtime`: the session is opened with the Realtime route's provider first, so a
/// provider that does not serve it answers the client with its own status, and then messages
/// pass both ways until either side closes.
async fn realtime(State(forwarder): State<Forwarder>, request: Request) -> Response {
    let (mut parts, _) = request.into_parts();
    if let Some(refused) = forwarder.refused(&parts.headers) {
        return refused;
    }
    let targets = forwarder.targets();
    let Some(target) = pick(&targets, "/v1/realtime", None) else {
        return error(
            StatusCode::NOT_FOUND,
            "not_found_error",
            "No route serves /v1/realtime",
        );
    };
    let base = target
        .base
        .replacen("http://", "ws://", 1)
        .replacen("https://", "wss://", 1);
    let url = match parts.uri.query() {
        Some(query) => format!("{base}/v1/realtime?{query}"),
        None => format!("{base}/v1/realtime"),
    };
    let Ok(mut upstream) = url.into_client_request() else {
        return error(StatusCode::BAD_GATEWAY, "api_error", "Not an address");
    };
    upstream.headers_mut().insert(
        header::SEC_WEBSOCKET_PROTOCOL,
        header::HeaderValue::from_static("realtime"),
    );
    if let Some(key) = &target.key
        && let Ok(value) = format!("Bearer {key}").parse()
    {
        upstream.headers_mut().insert(header::AUTHORIZATION, value);
    }
    let provider = match tokio_tungstenite::connect_async(upstream).await {
        Ok((provider, _)) => provider,
        Err(tungstenite::Error::Http(refused)) => {
            let body = refused.body().clone().unwrap_or_default();
            let kind = refused.headers().get(header::CONTENT_TYPE).cloned();
            let mut response = (refused.status(), body).into_response();
            if let Some(kind) = kind {
                response.headers_mut().insert(header::CONTENT_TYPE, kind);
            }
            return response;
        }
        Err(e) => {
            return error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("Cannot reach {}: {e}", target.provider),
            );
        }
    };
    match WebSocketUpgrade::from_request_parts(&mut parts, &forwarder).await {
        Ok(upgrade) => upgrade
            .protocols(["realtime"])
            .on_upgrade(move |client| relay(client, provider)),
        Err(refused) => refused.into_response(),
    }
}

type Provider =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Passes messages both ways until either side closes or fails.
async fn relay(client: WebSocket, provider: Provider) {
    let (mut to_client, mut from_client) = client.split();
    let (mut to_provider, mut from_provider) = provider.split();
    let up = async {
        while let Some(Ok(message)) = from_client.next().await {
            let message = match message {
                ws::Message::Text(text) => tungstenite::Message::text(text.as_str()),
                ws::Message::Binary(bytes) => tungstenite::Message::Binary(bytes),
                ws::Message::Ping(bytes) => tungstenite::Message::Ping(bytes),
                ws::Message::Pong(bytes) => tungstenite::Message::Pong(bytes),
                ws::Message::Close(frame) => {
                    tungstenite::Message::Close(frame.map(|frame| CloseFrame {
                        code: frame.code.into(),
                        reason: frame.reason.as_str().into(),
                    }))
                }
            };
            if to_provider.send(message).await.is_err() {
                break;
            }
        }
        let _ = to_provider.close().await;
    };
    let down = async {
        while let Some(Ok(message)) = from_provider.next().await {
            let message = match message {
                tungstenite::Message::Text(text) => ws::Message::Text(text.as_str().into()),
                tungstenite::Message::Binary(bytes) => ws::Message::Binary(bytes),
                tungstenite::Message::Ping(bytes) => ws::Message::Ping(bytes),
                tungstenite::Message::Pong(bytes) => ws::Message::Pong(bytes),
                tungstenite::Message::Close(frame) => {
                    ws::Message::Close(frame.map(|frame| ws::CloseFrame {
                        code: frame.code.into(),
                        reason: frame.reason.as_str().into(),
                    }))
                }
                tungstenite::Message::Frame(_) => continue,
            };
            if to_client.send(message).await.is_err() {
                break;
            }
        }
        let _ = to_client.close().await;
    };
    tokio::select! {
        _ = up => {}
        _ = down => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{Client, ClientError, Route, Turns};
    use std::sync::Mutex;

    /// What a fake provider got: the path, the key it came with, and the body.
    type Got = Arc<Mutex<Vec<(String, String, Vec<u8>)>>>;

    /// A provider that answers every `/v1/*` POST with its name, an event stream on
    /// `/v1/responses`, a 422 on `/v1/refused`, the headers it got on `/v1/headers`, and
    /// echoes a Realtime session when `streams`.
    async fn provider(name: &'static str, streams: bool) -> (String, Got) {
        let got = Got::default();
        let seen = got.clone();
        let answer = move |request: Request| {
            let seen = seen.clone();
            async move {
                let (parts, body) = request.into_parts();
                // Only POST is served, as on a provider with no such path.
                if parts.method != axum::http::Method::POST {
                    return StatusCode::NOT_FOUND.into_response();
                }
                let key = parts
                    .headers
                    .get(header::AUTHORIZATION)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_string();
                let body = axum::body::to_bytes(body, MAX_BODY).await.unwrap().to_vec();
                let path = parts.uri.path().to_string();
                seen.lock().unwrap().push((path.clone(), key, body));
                match path.as_str() {
                    "/v1/responses" => (
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
                    )
                        .into_response(),
                    // Which of the client's headers arrived.
                    "/v1/headers" => Json(json!({
                        "tag": parts.headers.get("x-tag").and_then(|v| v.to_str().ok()),
                        "accept_encoding": parts.headers.contains_key(header::ACCEPT_ENCODING),
                    }))
                    .into_response(),
                    "/v1/refused" => (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        Json(json!({"detail": [{"msg": "Unknown field"}]})),
                    )
                        .into_response(),
                    _ => Json(json!({"provider": name})).into_response(),
                }
            }
        };
        let mut app = Router::new().route("/v1/{*path}", any(answer));
        if streams {
            app = app.route(
                "/v1/realtime",
                get(|headers: HeaderMap, upgrade: WebSocketUpgrade| async move {
                    let key = headers[header::AUTHORIZATION].to_str().unwrap().to_string();
                    upgrade
                        .protocols(["realtime"])
                        .on_upgrade(move |mut socket| async move {
                            // Says who it was asked as, then echoes.
                            let _ = socket.send(ws::Message::Text(key.as_str().into())).await;
                            while let Some(Ok(message)) = socket.recv().await {
                                if socket.send(message).await.is_err() {
                                    break;
                                }
                            }
                        })
                }),
            );
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (base, got)
    }

    fn target(capability: Capability, model: &str, provider: &str, base: &str) -> Target {
        Target {
            capability,
            model: model.into(),
            provider: provider.into(),
            base: base.into(),
            key: Some(format!("key-of-{provider}")),
        }
    }

    /// The forwarder behind the key `app-key`, with speech and Realtime on `box`, decisions on
    /// `openrouter` and generation on `embedded`.
    async fn forwarder() -> (String, [Got; 3]) {
        let (speech, heard) = provider("box", true).await;
        let (decision, decided) = provider("openrouter", false).await;
        let (generation, written) = provider("embedded", false).await;
        let forwarder = Forwarder::new(Some("app-key".into()));
        forwarder.route(vec![
            target(Capability::Speech, "parakeet", "box", &speech),
            target(Capability::Realtime, "parakeet", "box", &speech),
            target(
                Capability::Decision,
                "typesafe/jev-1.13",
                "openrouter",
                &decision,
            ),
            target(Capability::Generation, "gemma", "embedded", &generation),
        ]);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(forwarder.serve(listener, std::future::pending()));
        (base, [heard, decided, written])
    }

    async fn post_json(base: &str, path: &str, body: Value) -> reqwest::Response {
        reqwest::Client::new()
            .post(format!("{base}{path}"))
            .bearer_auth("app-key")
            .json(&body)
            .send()
            .await
            .unwrap()
    }

    #[test]
    fn the_routes_are_the_forwarder_s_targets() {
        let client = Client::new("http://box:8080/", Some("secret".into()));
        let routes = Routes {
            speech: Some(Route::ours(client, "box", "parakeet")),
            ..Routes::default()
        };
        assert_eq!(
            targets(&routes),
            [Target {
                capability: Capability::Speech,
                model: "parakeet".into(),
                provider: "box".into(),
                base: "http://box:8080".into(),
                key: Some("secret".into()),
            }]
        );
    }

    #[tokio::test]
    async fn a_request_goes_to_the_provider_of_its_model_with_that_provider_s_key() {
        let (base, [heard, decided, written]) = forwarder().await;
        // By model name: the decision model's provider, with its key in place of the app's,
        // and the body as it was sent.
        let body = json!({"model": "typesafe/jev-1.13", "state": "s", "questions": {}});
        let answer = post_json(&base, "/v1/systemone", body.clone()).await;
        assert_eq!(answer.status(), 200);
        assert_eq!(
            answer.json::<Value>().await.unwrap()["provider"],
            "openrouter"
        );
        let (path, key, sent) = decided.lock().unwrap()[0].clone();
        assert_eq!(
            (path.as_str(), key.as_str()),
            ("/v1/systemone", "Bearer key-of-openrouter")
        );
        assert_eq!(serde_json::from_slice::<Value>(&sent).unwrap(), body);
        // A model no route names goes by the path: another name of the provider's model
        // still reaches it.
        let alias = json!({"model": "jev-latest", "state": "s", "questions": {}});
        let answer = post_json(&base, "/v1/systemone", alias).await;
        assert_eq!(
            answer.json::<Value>().await.unwrap()["provider"],
            "openrouter"
        );
        let chat = json!({"model": "some-other-model", "messages": []});
        let answer = post_json(&base, "/v1/chat/completions", chat).await;
        assert_eq!(
            answer.json::<Value>().await.unwrap()["provider"],
            "embedded"
        );
        // The model's provider wins over the path's: any `/v1` path follows the model.
        let embed = json!({"model": "gemma", "input": "x"});
        let answer = post_json(&base, "/v1/embeddings", embed).await;
        assert_eq!(
            answer.json::<Value>().await.unwrap()["provider"],
            "embedded"
        );
        assert_eq!(written.lock().unwrap().len(), 2);
        // With neither, nothing serves it, and no provider is asked.
        let nobody = json!({"model": "unknown", "input": "x"});
        let answer = post_json(&base, "/v1/embeddings", nobody).await;
        assert_eq!(answer.status(), 404);
        assert_eq!(
            answer.json::<Value>().await.unwrap()["detail"]["message"],
            "No route serves /v1/embeddings for the model \"unknown\""
        );
        assert!(heard.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn an_answer_comes_back_as_the_provider_gave_it() {
        let (base, _) = forwarder().await;
        // A stream of events, with its content type.
        let answer = post_json(&base, "/v1/responses", json!({"model": "gemma"})).await;
        assert_eq!(answer.headers()[header::CONTENT_TYPE], "text/event-stream");
        assert_eq!(
            answer.text().await.unwrap(),
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n"
        );
        // An error status and its body.
        let answer = post_json(&base, "/v1/refused", json!({"model": "gemma"})).await;
        assert_eq!(answer.status(), 422);
        assert_eq!(
            answer.json::<Value>().await.unwrap()["detail"][0]["msg"],
            "Unknown field"
        );
        // The client's own headers go on, but for the encoding it accepts: the answer is
        // asked for uncompressed.
        let answer = reqwest::Client::new()
            .post(format!("{base}/v1/headers"))
            .bearer_auth("app-key")
            .header("x-tag", "7")
            .header(header::ACCEPT_ENCODING, "gzip, deflate")
            .json(&json!({"model": "gemma"}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            answer.json::<Value>().await.unwrap(),
            json!({"tag": "7", "accept_encoding": false})
        );
        // A provider that does not answer is a 502 that names it.
        let closed = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gone = format!("http://{}", closed.local_addr().unwrap());
        drop(closed);
        let down = Forwarder::new(None);
        down.route(vec![target(Capability::Decision, "jev", "box", &gone)]);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(down.serve(listener, std::future::pending()));
        let answer = reqwest::Client::new()
            .post(format!("{base}/v1/systemone"))
            .json(&json!({"model": "jev"}))
            .send()
            .await
            .unwrap();
        assert_eq!(answer.status(), 502);
        let message = answer.json::<Value>().await.unwrap()["detail"]["message"].clone();
        assert!(
            message.as_str().unwrap().starts_with("Cannot reach box"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn the_app_s_key_is_asked_for_and_the_providers_keys_stay_with_the_app() {
        let (base, [_, decided, _]) = forwarder().await;
        let http = reqwest::Client::new();
        let ask = |key: Option<&'static str>| {
            let request = http
                .post(format!("{base}/v1/systemone"))
                .json(&json!({"model": "typesafe/jev-1.13"}));
            match key {
                Some(key) => request.bearer_auth(key),
                None => request,
            }
            .send()
        };
        let none = ask(None).await.unwrap();
        assert_eq!(none.status(), 403);
        assert_eq!(
            none.json::<Value>().await.unwrap()["detail"]["message"],
            "No API key provided"
        );
        // A provider's key does not open the app's API.
        let wrong = ask(Some("key-of-openrouter")).await.unwrap();
        assert_eq!(wrong.status(), 401);
        assert!(decided.lock().unwrap().is_empty());
        assert_eq!(ask(Some("app-key")).await.unwrap().status(), 200);
        // Health needs no key, and names the model of each service as jevons-api does.
        let health: Value = http
            .get(format!("{base}/health"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(
            health,
            json!({"status": "ok", "services": {"generative": "gemma",
                "decision": "typesafe/jev-1.13", "speech": "parakeet"}})
        );
        // The models listed are the routes', each once, and no key is in the listing.
        let listing = http.get(format!("{base}/v1/models")).send().await.unwrap();
        assert_eq!(listing.status(), 403);
        let listing = http
            .get(format!("{base}/v1/models"))
            .bearer_auth("app-key")
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        let models: Value = serde_json::from_str(&listing).unwrap();
        let ids: Vec<&str> = models["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["parakeet", "typesafe/jev-1.13", "gemma"]);
        assert_eq!(models["models"][1]["name"], "typesafe/jev-1.13");
        assert!(!listing.contains("key-of"), "{listing}");
    }

    #[tokio::test]
    async fn an_upload_goes_to_the_speech_provider_as_it_was_sent() {
        let (base, [heard, _, _]) = forwarder().await;
        let audio = vec![7u8; 4096];
        let form = reqwest::multipart::Form::new()
            .part(
                "file",
                reqwest::multipart::Part::bytes(audio.clone()).file_name("take.wav"),
            )
            .text("model", "parakeet");
        let answer = reqwest::Client::new()
            .post(format!("{base}/v1/audio/transcriptions"))
            .bearer_auth("app-key")
            .multipart(form)
            .send()
            .await
            .unwrap();
        assert_eq!(answer.json::<Value>().await.unwrap()["provider"], "box");
        let (path, key, sent) = heard.lock().unwrap()[0].clone();
        assert_eq!(path, "/v1/audio/transcriptions");
        assert_eq!(key, "Bearer key-of-box");
        assert!(find(&sent, &audio).is_some(), "the audio arrives whole");
        assert_eq!(form_field(&sent, "model").as_deref(), Some("parakeet"));
        assert_eq!(form_field(&sent, "language"), None);
    }

    #[tokio::test]
    async fn a_realtime_session_passes_both_ways_through_the_forwarder() {
        let (base, _) = forwarder().await;
        let url = format!(
            "{}/v1/realtime?intent=transcription",
            base.replacen("http", "ws", 1)
        );
        let mut request = url.into_client_request().unwrap();
        let headers = request.headers_mut();
        headers.insert(header::SEC_WEBSOCKET_PROTOCOL, "realtime".parse().unwrap());
        headers.insert(header::AUTHORIZATION, "Bearer app-key".parse().unwrap());
        let (mut socket, response) = tokio_tungstenite::connect_async(request).await.unwrap();
        assert_eq!(
            response.headers()[header::SEC_WEBSOCKET_PROTOCOL],
            "realtime"
        );
        // The provider was asked with its own key.
        let first = socket.next().await.unwrap().unwrap();
        assert_eq!(first.to_text().unwrap(), "Bearer key-of-box");
        socket
            .send(tungstenite::Message::text("{\"type\":\"session.update\"}"))
            .await
            .unwrap();
        let echoed = socket.next().await.unwrap().unwrap();
        assert_eq!(echoed.to_text().unwrap(), "{\"type\":\"session.update\"}");
        socket
            .send(tungstenite::Message::Binary(vec![1u8, 2, 3].into()))
            .await
            .unwrap();
        let echoed = socket.next().await.unwrap().unwrap();
        assert_eq!(echoed.into_data().as_ref(), [1u8, 2, 3]);
        // Without the app's key the session is refused before the provider is asked.
        let refused = tokio_tungstenite::connect_async(format!(
            "{}/v1/realtime",
            base.replacen("http", "ws", 1)
        ))
        .await;
        assert!(
            matches!(&refused, Err(tungstenite::Error::Http(r)) if r.status() == 403),
            "{refused:?}"
        );
        // A provider that does not stream answers 404, which the client reads as "not
        // served" and falls back to uploads.
        let (uploads, _) = provider("cloud", false).await;
        let forwarder = Forwarder::new(None);
        forwarder.route(vec![target(
            Capability::Realtime,
            "whisper",
            "cloud",
            &uploads,
        )]);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(forwarder.serve(listener, std::future::pending()));
        let client = Client::new(&base, None);
        let opened = client.realtime(Some("whisper"), None, Turns::Client).await;
        assert!(
            matches!(opened, Err(ClientError::NotServed(_))),
            "{:?}",
            opened.err()
        );
    }

    #[tokio::test]
    async fn the_api_log_holds_forwarded_calls_and_never_a_key_or_audio() {
        let (base, _) = forwarder().await;
        let file =
            std::env::temp_dir().join(format!("jevons-forward-log-{}.log", std::process::id()));
        let _ = std::fs::remove_file(&file);
        let marker = format!("forward-test-{}", std::process::id());
        log::set(Some(file.clone()));
        let decision = json!({"model": "typesafe/jev-1.13", "state": marker, "questions": {}});
        post_json(&base, "/v1/systemone", decision)
            .await
            .bytes()
            .await
            .unwrap();
        let generation = json!({"model": "gemma", "input": marker});
        post_json(&base, "/v1/responses", generation)
            .await
            .bytes()
            .await
            .unwrap();
        let form = reqwest::multipart::Form::new()
            .part(
                "file",
                reqwest::multipart::Part::bytes(marker.clone().into_bytes()),
            )
            .text("model", "parakeet");
        reqwest::Client::new()
            .post(format!("{base}/v1/audio/transcriptions"))
            .bearer_auth("app-key")
            .multipart(form)
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        log::set(None);
        // Other tests' calls may land in the file while it is on: only this test's count.
        let text = std::fs::read_to_string(&file).unwrap();
        let records: Vec<Value> = serde_json::Deserializer::from_str(&text)
            .into_iter::<Value>()
            .map(Result::unwrap)
            .filter(|r| r["request"].to_string().contains(&marker))
            .collect();
        assert_eq!(records.len(), 2, "{text}");
        assert_eq!(records[0]["api"], "POST /v1/systemone → openrouter");
        assert_eq!(records[0]["request"]["model"], "typesafe/jev-1.13");
        assert_eq!(
            records[0]["response"],
            json!({"status": 200, "body": {"provider": "openrouter"}})
        );
        assert_eq!(records[1]["api"], "POST /v1/responses → embedded");
        let events = records[1]["response"]["body"].as_str().unwrap();
        assert!(events.starts_with("data: {"), "{events}");
        assert!(
            !text.contains("key-of") && !text.contains("app-key"),
            "{text}"
        );
        std::fs::remove_file(file).unwrap();
    }
}
