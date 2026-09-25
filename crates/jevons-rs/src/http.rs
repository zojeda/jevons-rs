//! Compose routes, middleware, and body limits around shared application state.

use crate::{error::ApiError, handlers, middleware::request_context, worker};
use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::StatusCode,
    middleware,
    routing::{get, post},
};
use std::sync::Arc;

pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    pub worker: worker::Client,
    pub model_id: String,
    /// Routing aliases accepted in place of `model_id`.
    pub aliases: Arc<[String]>,
    /// Model listing description.
    pub description: String,
    pub api_key: Option<Arc<str>>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::models))
        .route("/v1/systemone", post(handlers::system_one))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/responses", post(handlers::responses))
        .fallback(|| async {
            ApiError::new(StatusCode::NOT_FOUND, "not_found_error", "Unknown path")
        })
        .method_not_allowed_fallback(|| async {
            ApiError::new(
                StatusCode::METHOD_NOT_ALLOWED,
                "invalid_request_error",
                "Method not allowed",
            )
        })
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            request_context,
        ))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request as HttpRequest,
    };
    use serde_json::{Value, json};
    use tokio::sync::mpsc;
    use tower::ServiceExt;

    fn app(key: Option<&str>) -> (Router, mpsc::Receiver<worker::Job>) {
        let (sender, receiver) = mpsc::channel(1);
        (
            router(AppState {
                worker: worker::Client { sender },
                model_id: "local".into(),
                aliases: ["jev-latest".to_string()].into(),
                description: "Local test model.".into(),
                api_key: key.map(Arc::from),
            }),
            receiver,
        )
    }

    async fn send(app: Router, path: &str, body: &str, auth: Option<&str>) -> (StatusCode, Value) {
        let mut request = HttpRequest::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json");
        if let Some(auth) = auth {
            request = request.header("authorization", auth);
        }
        let response = app
            .oneshot(request.body(Body::from(body.to_owned())).unwrap())
            .await
            .unwrap();
        assert!(response.headers().contains_key("x-typesafe-request-id"));
        let status = response.status();
        let bytes = to_bytes(response.into_body(), MAX_BODY_BYTES + 1024)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn health_bypasses_authentication_and_reflects_worker_liveness() {
        let (app, mut receiver) = app(Some("secret"));
        for expected in [StatusCode::OK, StatusCode::SERVICE_UNAVAILABLE] {
            let response = app
                .clone()
                .oneshot(
                    HttpRequest::builder()
                        .uri("/health")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
            assert!(response.headers().contains_key("x-typesafe-request-id"));
            receiver.close();
        }
    }

    #[tokio::test]
    async fn authentication_and_validation_match_the_wire_contract() {
        let (app, _receiver) = app(Some("secret"));
        assert_eq!(
            send(app.clone(), "/v1/systemone", "{}", None).await.0,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            send(app.clone(), "/v1/systemone", "{}", Some("Bearer bad"))
                .await
                .0,
            StatusCode::UNAUTHORIZED
        );
        let (status, body) = send(app.clone(), "/v1/systemone", "{}", Some("Bearer secret")).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"][0]["loc"], json!(["body", "model"]));
        let (status, body) = send(app.clone(), "/v1/systemone", "{", Some("Bearer secret")).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body["detail"].is_array());
        assert_eq!(
            send(app, "/missing", "{}", Some("Bearer secret")).await.0,
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn unsupported_extensions_unknown_models_and_large_bodies_fail_before_inference() {
        let (app, mut receiver) = app(None);
        for (model, steps, expected) in [
            ("missing", 1, StatusCode::NOT_FOUND),
            ("local", 9, StatusCode::UNPROCESSABLE_ENTITY),
        ] {
            let body =
                json!({"model":model,"state":"x","questions":{"q":{"type":"noul"}},"steps":steps})
                    .to_string();
            assert_eq!(
                send(app.clone(), "/v1/systemone", &body, None).await.0,
                expected
            );
        }
        assert_eq!(
            send(app, "/v1/systemone", &" ".repeat(MAX_BODY_BYTES + 1), None)
                .await
                .0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
        assert!(receiver.try_recv().is_err());
    }

    #[tokio::test]
    async fn extensions_reach_the_worker_and_thought_usage_reaches_the_client() {
        let (app, mut receiver) = app(None);
        let responder = tokio::spawn(async move {
            let Some(worker::Job::Read { request, reply }) = receiver.recv().await else {
                panic!("a read");
            };
            let options = request.options();
            assert_eq!(
                (
                    options.steps,
                    options.samples,
                    options.think,
                    options.sequential
                ),
                (3, 2, 8, true)
            );
            reply
                .send(Ok(jevons_system_one::Response {
                    model: "local".into(),
                    answers: Default::default(),
                    usage: jevons_system_one::Usage {
                        input_tokens: 123,
                        output_tokens: 8,
                    },
                }))
                .unwrap();
        });
        let (status, body) = send(app, "/v1/systemone", r#"{"model":"local","state":"x","questions":{"q":{"type":"noul"}},"steps":3,"samples":2,"think":8,"sequential":true}"#, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["usage"]["output_tokens"], 8);
        responder.await.unwrap();
    }

    #[tokio::test]
    async fn model_listing_and_successful_response_preserve_sdk_shapes() {
        let (app, mut receiver) = app(None);
        let response = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert!(
            body["models"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["name"] == "jev-latest")
        );
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["id"], "local");
        assert_eq!(body["data"][1]["object"], "model");
        let responder = tokio::spawn(async move {
            let Some(worker::Job::Read { request, reply }) = receiver.recv().await else {
                panic!("a read");
            };
            assert_eq!(request.model(), "jev-latest");
            let response = jevons_system_one::Response {
                model: "local".into(),
                answers: [("q".into(), jevons_system_one::Answer::Noul { noul: 0.75 })].into(),
                usage: jevons_system_one::Usage {
                    input_tokens: 10,
                    output_tokens: 0,
                },
            };
            reply.send(Ok(response)).unwrap();
        });
        let (status, body) = send(
            app,
            "/v1/systemone",
            r#"{"model":"jev-latest","state":"x","questions":{"q":{"type":"noul"}}}"#,
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["answers"]["q"]["noul"], 0.75);
        responder.await.unwrap();
    }

    /// Answers the next generation job with `pieces` of text, then `done`.
    fn generation_worker(
        mut receiver: mpsc::Receiver<worker::Job>,
        pieces: &'static [&'static str],
        done: jevons_engine::Result<jevons_engine::Generation>,
    ) -> tokio::task::JoinHandle<jevons_engine::GenerationRequest> {
        tokio::spawn(async move {
            let Some(worker::Job::Generate {
                request, updates, ..
            }) = receiver.recv().await
            else {
                panic!("a generation");
            };
            for piece in pieces {
                updates
                    .send(worker::Update::Text(piece.to_string()))
                    .unwrap();
            }
            updates.send(worker::Update::Done(done)).unwrap();
            request
        })
    }

    fn answer(text: &str) -> jevons_engine::Generation {
        jevons_engine::Generation {
            text: text.into(),
            prompt_tokens: 12,
            completion_tokens: 2,
            reasoning_tokens: 0,
            finish: jevons_engine::FinishReason::Stop,
        }
    }

    async fn raw(app: Router, path: &str, body: &str) -> (StatusCode, String, String) {
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_owned()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let kind = response
            .headers()
            .get("content-type")
            .map(|v| v.to_str().unwrap().to_string())
            .unwrap_or_default();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, kind, String::from_utf8(bytes.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn chat_completions_reach_the_worker_and_return_the_openai_shape() {
        let (app, receiver) = app(None);
        let worker = generation_worker(receiver, &["Hel", "lo"], Ok(answer("Hello")));
        let (status, body) = send(
            app,
            "/v1/chat/completions",
            r#"{"model":"jev-latest","messages":[{"role":"system","content":"Brief."},{"role":"user","content":"Hi"}],"max_tokens":5,"stop":"\n"}"#,
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "local");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello");
        assert_eq!(body["usage"]["prompt_tokens"], 12);
        let request = worker.await.unwrap();
        assert_eq!(request.max_tokens, Some(5));
        assert_eq!(request.stop, ["\n"]);
    }

    #[tokio::test]
    async fn streamed_chat_completions_are_server_sent_events_ending_in_done() {
        let (app, receiver) = app(None);
        let worker = generation_worker(receiver, &["Hel", "lo"], Ok(answer("Hello")));
        let (status, kind, text) = raw(
            app,
            "/v1/chat/completions",
            r#"{"model":"local","messages":[{"role":"user","content":"Hi"}],"stream":true}"#,
        )
        .await;
        worker.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert!(kind.starts_with("text/event-stream"), "{kind}");
        let data: Vec<&str> = text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .collect();
        assert_eq!(*data.last().unwrap(), "[DONE]");
        let content: String = data[..data.len() - 1]
            .iter()
            .map(|d| serde_json::from_str::<Value>(d).unwrap())
            .filter_map(|c| {
                c["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(String::from)
            })
            .collect();
        assert_eq!(content, "Hello");
    }

    #[tokio::test]
    async fn streamed_responses_name_their_events() {
        let (app, receiver) = app(None);
        let worker = generation_worker(receiver, &["Hi"], Ok(answer("Hi")));
        let (_, _, text) = raw(
            app,
            "/v1/responses",
            r#"{"model":"local","input":"Hello","stream":true}"#,
        )
        .await;
        worker.await.unwrap();
        let events: Vec<&str> = text
            .lines()
            .filter_map(|l| l.strip_prefix("event: "))
            .collect();
        assert_eq!(events.first(), Some(&"response.created"));
        assert!(events.contains(&"response.output_text.delta"));
        assert_eq!(events.last(), Some(&"response.completed"));
    }

    #[tokio::test]
    async fn openai_errors_use_the_openai_shape() {
        let (app, receiver) = app(Some("secret"));
        let (status, body) = send(app.clone(), "/v1/completions", "{}", None).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert!(
            body["detail"].is_object(),
            "authentication is shared: {body}"
        );
        let auth = Some("Bearer secret");
        let (status, body) = send(
            app.clone(),
            "/v1/completions",
            r#"{"model":"other","prompt":"x"}"#,
            auth,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");
        let (status, body) = send(
            app.clone(),
            "/v1/chat/completions",
            r#"{"model":"local","messages":[{"role":"user","content":"x"}],"n":3}"#,
            auth,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "n");
        let (status, body) = send(app.clone(), "/v1/responses", "not json", auth).await;
        assert!(status.is_client_error());
        assert_eq!(body["error"]["type"], "invalid_request_error");
        // Engine validation failures (a prompt too long for the context) are client errors.
        let worker = generation_worker(
            receiver,
            &[],
            Err(jevons_engine::Error::InvalidInput("too long".into())),
        );
        let (status, body) = send(
            app,
            "/v1/chat/completions",
            r#"{"model":"local","messages":[{"role":"user","content":"x"}],"stream":true}"#,
            auth,
        )
        .await;
        worker.await.unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["message"], "too long");
    }
}
