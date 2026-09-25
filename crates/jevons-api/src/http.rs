//! Compose routes, middleware, and body limits around shared application state.

use crate::workers::{diffusion as worker, speech};
use crate::{error::ApiError, handlers, middleware::request_context};
use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::StatusCode,
    middleware,
    routing::{get, post},
};
use jevons_core::SpeechInfo;
use std::sync::Arc;

pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
/// The largest audio upload, as OpenAI allows.
pub const MAX_AUDIO_BYTES: usize = 25 * 1024 * 1024;
/// Room for the other multipart fields of a transcription request.
const FORM_OVERHEAD_BYTES: usize = 1024 * 1024;

/// A diffusion language model on its worker and the names it answers to. When Generative and
/// Decision use the same model, they share one of these (one engine, one queue).
#[derive(Clone)]
pub struct DiffusionService {
    pub worker: worker::Client,
    pub model_id: String,
    /// Routing aliases accepted in place of `model_id`.
    pub aliases: Arc<[String]>,
    /// Model listing description.
    pub description: String,
}

/// The speech-to-text model and the names it answers to.
#[derive(Clone)]
pub struct SpeechService {
    pub worker: speech::Client,
    pub model_id: String,
    pub aliases: Arc<[String]>,
    pub description: String,
    pub info: SpeechInfo,
    /// Longest upload or Realtime buffer, in seconds.
    pub max_audio_seconds: f64,
    /// Whether `/v1/realtime` is served.
    pub realtime: bool,
}

impl DiffusionService {
    pub fn serves(&self, model: &str) -> bool {
        model == self.model_id || self.aliases.iter().any(|a| a == model)
    }
}

impl SpeechService {
    pub fn serves(&self, model: &str) -> bool {
        model == self.model_id || self.aliases.iter().any(|a| a == model)
    }

    /// The model ID, then its aliases.
    pub fn names(&self) -> Vec<String> {
        std::iter::once(self.model_id.clone())
            .chain(self.aliases.iter().cloned())
            .collect()
    }
}

/// The enabled services; at least one is present.
#[derive(Clone)]
pub struct AppState {
    /// OpenAI Chat Completions, Completions and Responses.
    pub generative: Option<DiffusionService>,
    /// The System One API.
    pub decision: Option<DiffusionService>,
    /// OpenAI audio transcriptions and Realtime transcription sessions.
    pub speech: Option<SpeechService>,
    pub api_key: Option<Arc<str>>,
}

pub fn router(state: AppState) -> Router {
    let mut router = Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::models))
        .route("/v1/systemone", post(handlers::system_one))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/responses", post(handlers::responses))
        .route(
            "/v1/audio/transcriptions",
            post(handlers::transcriptions)
                .layer(DefaultBodyLimit::max(MAX_AUDIO_BYTES + FORM_OVERHEAD_BYTES)),
        );
    if state.speech.as_ref().is_some_and(|s| s.realtime) {
        router = router.route("/v1/realtime", get(handlers::realtime));
    }
    router
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
        // Generative and Decision share one model, as when the settings point both at it.
        let local = DiffusionService {
            worker: worker::Client { sender },
            model_id: "local".into(),
            aliases: ["jev-latest".to_string()].into(),
            description: "Local test model.".into(),
        };
        (
            router(AppState {
                generative: Some(local.clone()),
                decision: Some(local),
                speech: None,
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
                .send(Ok(crate::system_one::Response {
                    model: "local".into(),
                    answers: Default::default(),
                    usage: crate::system_one::Usage {
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
            let response = crate::system_one::Response {
                model: "local".into(),
                answers: [("q".into(), crate::system_one::Answer::Noul { noul: 0.75 })].into(),
                usage: crate::system_one::Usage {
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
        done: jevons_core::Result<jevons_generative::Generation>,
    ) -> tokio::task::JoinHandle<jevons_generative::GenerationRequest> {
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

    fn answer(text: &str) -> jevons_generative::Generation {
        jevons_generative::Generation {
            text: text.into(),
            prompt_tokens: 12,
            completion_tokens: 2,
            reasoning_tokens: 0,
            finish: jevons_generative::FinishReason::Stop,
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
            Err(jevons_core::Error::InvalidInput("too long".into())),
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

    /// A 16-bit mono WAV file.
    fn wav(rate: u32, samples: &[f32]) -> Vec<u8> {
        let data = samples.len() as u32 * 2;
        let mut bytes = Vec::new();
        bytes.extend(b"RIFF");
        bytes.extend((36 + data).to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16u32.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(rate.to_le_bytes());
        bytes.extend((rate * 2).to_le_bytes());
        bytes.extend(2u16.to_le_bytes());
        bytes.extend(16u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend(data.to_le_bytes());
        for s in samples {
            bytes.extend(((s * 32768.0).round() as i16).to_le_bytes());
        }
        bytes
    }

    async fn speech_app() -> Router {
        router(AppState {
            generative: None,
            decision: None,
            speech: Some(crate::workers::speech::scripted_service(20.0).await),
            api_key: None,
        })
    }

    /// Posts a multipart form; `file` is (file name, bytes).
    async fn post_form(
        app: Router,
        fields: &[(&str, &str)],
        file: Option<(&str, Vec<u8>)>,
    ) -> (StatusCode, String, String) {
        let boundary = "jevons-test-boundary";
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .bytes(),
            );
        }
        if let Some((file_name, bytes)) = file {
            body.extend(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; \
                     filename=\"{file_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                )
                .bytes(),
            );
            body.extend(bytes);
            body.extend(b"\r\n");
        }
        body.extend(format!("--{boundary}--\r\n").bytes());
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/v1/audio/transcriptions")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let content_type = response
            .headers()
            .get("content-type")
            .map(|v| v.to_str().unwrap().to_string())
            .unwrap_or_default();
        let bytes = to_bytes(response.into_body(), MAX_BODY_BYTES)
            .await
            .unwrap();
        (
            status,
            content_type,
            String::from_utf8(bytes.to_vec()).unwrap(),
        )
    }

    fn speech_audio(seconds: usize) -> Vec<u8> {
        wav(100, &crate::workers::speech::spoken(seconds, 100))
    }

    #[tokio::test]
    async fn transcriptions_answer_in_every_response_format() {
        let app = speech_app().await;
        let audio = || Some(("speech.wav", speech_audio(7)));
        let (status, content_type, body) =
            post_form(app.clone(), &[("model", "scripted-asr")], audio()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(content_type.starts_with("application/json"));
        let json: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["text"], "w1 w2 w3 w4. w5 w6 w7");
        assert_eq!(json["usage"], json!({"type": "duration", "seconds": 7}));

        let (_, content_type, body) = post_form(
            app.clone(),
            &[("model", "asr-latest"), ("response_format", "text")],
            audio(),
        )
        .await;
        assert!(content_type.starts_with("text/plain"));
        assert_eq!(body, "w1 w2 w3 w4. w5 w6 w7");

        let (_, _, body) = post_form(
            app.clone(),
            &[("model", "scripted-asr"), ("response_format", "srt")],
            audio(),
        )
        .await;
        assert!(
            body.starts_with("1\n00:00:00,000 --> 00:00:03,500\nw1 w2 w3 w4.\n\n2\n"),
            "{body}"
        );

        let (_, _, body) = post_form(
            app,
            &[
                ("model", "scripted-asr"),
                ("language", "es"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
            ],
            audio(),
        )
        .await;
        let json: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["language"], "es");
        assert_eq!(json["duration"], 7.0);
        assert_eq!(json["segments"].as_array().unwrap().len(), 2);
        assert_eq!(
            json["words"][4],
            json!({"word": "w5", "start": 4.0, "end": 4.5})
        );
    }

    #[tokio::test]
    async fn streamed_transcriptions_send_segment_deltas_then_the_text() {
        let (status, content_type, body) = post_form(
            speech_app().await,
            &[("model", "scripted-asr"), ("stream", "true")],
            Some(("speech.wav", speech_audio(7))),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(content_type.starts_with("text/event-stream"));
        let events: Vec<Value> = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).unwrap())
            .collect();
        let kinds: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(
            kinds,
            [
                "transcript.text.delta",
                "transcript.text.delta",
                "transcript.text.done"
            ]
        );
        assert_eq!(events[0]["delta"], "w1 w2 w3 w4.");
        assert_eq!(events[1]["delta"], " w5 w6 w7");
        assert_eq!(events[2]["text"], "w1 w2 w3 w4. w5 w6 w7");
    }

    #[tokio::test]
    async fn transcription_requests_fail_before_inference_with_openai_errors() {
        let app = speech_app().await;
        let error =
            |body: &str| -> Value { serde_json::from_str::<Value>(body).unwrap()["error"].clone() };
        let (status, _, body) = post_form(app.clone(), &[("model", "scripted-asr")], None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error(&body)["param"], "file");
        let (status, _, body) = post_form(
            app.clone(),
            &[("model", "whisper-1")],
            Some(("a.wav", speech_audio(1))),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(error(&body)["code"], "model_not_found");
        let (status, _, body) = post_form(
            app.clone(),
            &[("model", "scripted-asr"), ("prompt", "names")],
            Some(("a.wav", speech_audio(1))),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error(&body)["code"], "unsupported_parameter");
        let (status, _, body) = post_form(
            app.clone(),
            &[("model", "scripted-asr")],
            Some(("a.webm", b"\x1aE\xdf\xa3 not really webm".to_vec())),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            error(&body)["message"]
                .as_str()
                .unwrap()
                .contains("Supported audio")
        );
        let (status, _, _) = post_form(
            app.clone(),
            &[("model", "scripted-asr")],
            Some(("big.wav", vec![0; MAX_AUDIO_BYTES + FORM_OVERHEAD_BYTES])),
        )
        .await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        // Without a language model, its routes do not know any model.
        let (status, body) = send(
            app,
            "/v1/chat/completions",
            r#"{"model":"local","messages":[{"role":"user","content":"x"}]}"#,
            None,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");
    }

    #[tokio::test]
    async fn listings_and_health_include_the_speech_model() {
        let (sender, _receiver) = mpsc::channel(1);
        let local = DiffusionService {
            worker: worker::Client { sender },
            model_id: "local".into(),
            aliases: Arc::from([]),
            description: "Local test model.".into(),
        };
        let app = router(AppState {
            generative: Some(local.clone()),
            decision: Some(local),
            speech: Some(crate::workers::speech::scripted_service(20.0).await),
            api_key: None,
        });
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
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let models: Value = serde_json::from_slice(&bytes).unwrap();
        let ids: Vec<&str> = models["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["local", "scripted-asr", "asr-latest"]);
        assert_eq!(models["models"][1]["description"], "Scripted speech.");
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let health: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            health,
            json!({"status": "ok", "services": {"generative": "local", "decision": "local", "speech": "scripted-asr"}})
        );
    }
}
