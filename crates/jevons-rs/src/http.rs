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
            let job = receiver.recv().await.unwrap();
            let options = job.request.options();
            assert_eq!(
                (
                    options.steps,
                    options.samples,
                    options.think,
                    options.sequential
                ),
                (3, 2, 8, true)
            );
            job.reply
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
        let responder = tokio::spawn(async move {
            let job = receiver.recv().await.unwrap();
            assert_eq!(job.request.model(), "jev-latest");
            let response = jevons_system_one::Response {
                model: "local".into(),
                answers: [("q".into(), jevons_system_one::Answer::Noul { noul: 0.75 })].into(),
                usage: jevons_system_one::Usage {
                    input_tokens: 10,
                    output_tokens: 0,
                },
            };
            job.reply.send(Ok(response)).unwrap();
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
}
