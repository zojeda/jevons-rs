//! Endpoint behavior and conversion from HTTP input to validated requests.

use crate::openai::{
    Api, Body, OpenAiError, OpenAiRequest, TranscriptStream, TranscriptionRequest,
};
use crate::system_one::ValidationError;
use crate::workers::{diffusion::Update, speech};
use crate::{AppState, DiffusionService, MAX_AUDIO_BYTES, SpeechService, error::ApiError};
use axum::{
    Json,
    extract::{
        Multipart, Query, State,
        multipart::{MultipartError, MultipartRejection},
        rejection::JsonRejection,
        ws::{WebSocketUpgrade, rejection::WebSocketUpgradeRejection},
    },
    http::{StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use jevons_audio::{AudioLimits, decode_audio};
use jevons_core::Error;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::convert::Infallible;
use tokio::sync::mpsc::UnboundedReceiver;

pub(super) async fn health(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let alive = state
        .generative
        .as_ref()
        .is_none_or(|s| s.worker.is_alive())
        && state.decision.as_ref().is_none_or(|s| s.worker.is_alive())
        && state.speech.as_ref().is_none_or(|s| s.worker.is_alive());
    if !alive {
        return Err(ApiError::unavailable());
    }
    Ok(Json(json!({
        "status": "ok",
        "services": {
            "generative": state.generative.as_ref().map(|s| &s.model_id),
            "decision": state.decision.as_ref().map(|s| &s.model_id),
            "speech": state.speech.as_ref().map(|s| &s.model_id),
        },
    })))
}

/// The System One listing (`models`) and the OpenAI one (`object`, `data`) in one body. A
/// model that serves several services is listed once.
pub(super) async fn models(State(state): State<AppState>) -> Json<Value> {
    let mut models: Vec<(&str, &[String], &str)> = Vec::new();
    for service in [&state.generative, &state.decision].into_iter().flatten() {
        if !models.iter().any(|(id, _, _)| *id == service.model_id) {
            models.push((&service.model_id, &service.aliases, &service.description));
        }
    }
    if let Some(speech) = &state.speech {
        models.push((&speech.model_id, &speech.aliases, &speech.description));
    }
    let names: Vec<(&str, &str)> = models
        .iter()
        .flat_map(|(id, aliases, description)| {
            std::iter::once((*id, *description))
                .chain(aliases.iter().map(move |a| (a.as_str(), *description)))
        })
        .collect();
    Json(json!({
        "models": names.iter().map(|(name, description)| json!({
            "name": name,
            "description": description,
            "release_date": "2026-09-19"
        })).collect::<Vec<_>>(),
        "object": "list",
        "data": names.iter().map(|(name, _)| json!({
            "id": name,
            "object": "model",
            "created": 1_790_294_400,
            "owned_by": "jevons",
        })).collect::<Vec<_>>(),
    }))
}

/// OpenAI's error for a model this server does not serve on the route.
fn model_not_found(model: &str) -> OpenAiError {
    let mut error = OpenAiError::new(
        404,
        "invalid_request_error",
        format!("The model {model:?} does not exist"),
    );
    error.code = Some("model_not_found");
    error.param = Some("model".into());
    error
}

pub(super) async fn system_one(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<crate::system_one::Response>, ApiError> {
    let Json(value) = body.map_err(|rejection| {
        if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
                "Request body exceeds 64 MiB",
            )
        } else {
            ValidationError::new(&["body"], rejection.body_text(), "json_invalid").into()
        }
    })?;
    let request = crate::system_one::Request::parse(value)?;
    let text = state
        .decision
        .as_ref()
        .filter(|t| t.serves(request.model()))
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, "not_found_error", "Unknown model"))?;
    Ok(Json(text.worker.read(request).await?))
}

pub(super) async fn chat_completions(
    state: State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    openai(Api::ChatCompletions, state, body).await
}

pub(super) async fn completions(
    state: State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    openai(Api::Completions, state, body).await
}

pub(super) async fn responses(
    state: State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    openai(Api::Responses, state, body).await
}

async fn openai(
    api: Api,
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    match generate(api, &state, body).await {
        Ok(response) => response,
        Err(error) => error.openai().into_response(),
    }
}

fn engine_error(error: Error) -> OpenAiError {
    failure(error, "Generation failed")
}

/// Invalid input is the client's error; anything else is logged and reported as `what`.
fn failure(error: Error, what: &'static str) -> OpenAiError {
    match error {
        Error::InvalidInput(message) => OpenAiError::invalid(message, None),
        other => {
            tracing::error!(error = %other, "{what}");
            OpenAiError::new(500, "server_error", what)
        }
    }
}

async fn generate(
    api: Api,
    state: &AppState,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(value) = body.map_err(|rejection| {
        OpenAiError::new(
            rejection.status().as_u16(),
            "invalid_request_error",
            rejection.body_text(),
        )
    })?;
    let request = OpenAiRequest::parse(api, &value)?;
    let text: &DiffusionService = state
        .generative
        .as_ref()
        .filter(|t| t.serves(&request.model))
        .ok_or_else(|| model_not_found(&request.model))?;
    let mut updates = text
        .worker
        .generate(request.generation.clone(), request.seed)?;
    let id = uuid::Uuid::new_v4().simple().to_string();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    // Wait for the first update so that invalid prompts fail with a status, not mid-stream.
    let first = updates.recv().await.ok_or_else(ApiError::unavailable)?;
    if !request.stream {
        let mut update = first;
        loop {
            match update {
                Update::Text(_) => {}
                Update::Done(result) => {
                    let generation = result.map_err(engine_error)?;
                    let body = request.response(&id, created, &text.model_id, &generation);
                    return Ok(Json(body).into_response());
                }
            }
            update = updates.recv().await.ok_or_else(ApiError::unavailable)?;
        }
    }
    if let Update::Done(Err(error)) = first {
        return Err(engine_error(error).into());
    }
    let mut renderer = request.stream(&id, created, &text.model_id);
    let opening = renderer.start();
    let events = futures_util::stream::unfold(
        (updates, renderer, Some(first), opening, false),
        |(mut updates, mut renderer, mut next, mut pending, mut done): StreamState| async move {
            loop {
                if !pending.is_empty() {
                    let event = pending.remove(0);
                    let sse = Event::default().data(event.data);
                    let sse = match event.name {
                        Some(name) => sse.event(name),
                        None => sse,
                    };
                    return Some((
                        Ok::<_, Infallible>(sse),
                        (updates, renderer, next, pending, done),
                    ));
                }
                if done {
                    return None;
                }
                let update = match next.take() {
                    Some(update) => update,
                    None => match updates.recv().await {
                        Some(update) => update,
                        None => Update::Done(Err(Error::Backend("the worker stopped".into()))),
                    },
                };
                pending = match update {
                    Update::Text(text) if text.is_empty() => Vec::new(),
                    Update::Text(text) => renderer.delta(&text),
                    Update::Done(Ok(generation)) => {
                        done = true;
                        renderer.finish(&generation)
                    }
                    Update::Done(Err(error)) => {
                        done = true;
                        renderer.error(&engine_error(error))
                    }
                };
            }
        },
    );
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}

type StreamState = (
    UnboundedReceiver<Update>,
    crate::openai::Stream,
    Option<Update>,
    Vec<crate::openai::Event>,
    bool,
);

fn multipart_error(error: MultipartError) -> OpenAiError {
    OpenAiError::new(
        error.status().as_u16(),
        "invalid_request_error",
        error.body_text(),
    )
}

pub(super) async fn transcriptions(
    State(state): State<AppState>,
    form: Result<Multipart, MultipartRejection>,
) -> Response {
    match transcribe(&state, form).await {
        Ok(response) => response,
        Err(error) => error.openai().into_response(),
    }
}

/// The uploaded audio and the other form fields, in order.
async fn read_form(
    form: Result<Multipart, MultipartRejection>,
) -> Result<(Option<(Vec<u8>, Option<String>)>, Vec<(String, String)>), OpenAiError> {
    let mut form = form.map_err(|rejection| {
        OpenAiError::new(
            rejection.status().as_u16(),
            "invalid_request_error",
            rejection.body_text(),
        )
    })?;
    let mut file = None;
    let mut fields = Vec::new();
    while let Some(field) = form.next_field().await.map_err(multipart_error)? {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" {
            if file.is_some() {
                return Err(OpenAiError::invalid(
                    "file is given more than once",
                    Some("file"),
                ));
            }
            let file_name = field.file_name().map(String::from);
            let bytes = field.bytes().await.map_err(multipart_error)?;
            file = Some((bytes.to_vec(), file_name));
        } else {
            fields.push((name, field.text().await.map_err(multipart_error)?));
        }
    }
    Ok((file, fields))
}

async fn transcribe(
    state: &AppState,
    form: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    let (file, fields) = read_form(form).await?;
    let languages = state.speech.as_ref().map_or(&[][..], |s| s.info.languages);
    let request = TranscriptionRequest::parse(&fields, languages)?;
    let speech: &SpeechService = state
        .speech
        .as_ref()
        .filter(|s| s.serves(&request.model))
        .ok_or_else(|| model_not_found(&request.model))?;
    let (bytes, file_name) =
        file.ok_or_else(|| OpenAiError::invalid("file is required", Some("file")))?;
    let extension = file_name
        .as_deref()
        .and_then(|name| std::path::Path::new(name).extension())
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    let limits = AudioLimits {
        max_bytes: MAX_AUDIO_BYTES,
        max_seconds: speech.max_audio_seconds,
    };
    let rate = speech.info.sample_rate;
    // Decoding and resampling an hour of MP3 takes seconds of CPU: keep it off the executor.
    let samples = tokio::task::spawn_blocking(move || {
        decode_audio(bytes, extension.as_deref(), rate, limits)
    })
    .await
    .map_err(|_| ApiError::unavailable())?
    .map_err(|e| failure(e, "Transcription failed"))?;
    let mut updates = speech.worker.transcribe(samples, false)?;
    if !request.stream {
        loop {
            match updates.recv().await.ok_or_else(ApiError::unavailable)? {
                speech::Update::Segment(..) => {}
                speech::Update::Done(result) => {
                    let transcript = result.map_err(|e| failure(e, "Transcription failed"))?;
                    return Ok(match request.response(&transcript) {
                        Body::Json(value) => Json(value).into_response(),
                        Body::Text { content_type, text } => {
                            ([(header::CONTENT_TYPE, content_type)], text).into_response()
                        }
                    });
                }
            }
        }
    }
    let renderer = request.stream();
    let events = futures_util::stream::unfold(
        (updates, renderer, Vec::new(), false),
        |(mut updates, mut renderer, mut pending, mut done): TranscriptState| async move {
            loop {
                if !pending.is_empty() {
                    let event: crate::openai::Event = pending.remove(0);
                    return Some((
                        Ok::<_, Infallible>(Event::default().data(event.data)),
                        (updates, renderer, pending, done),
                    ));
                }
                if done {
                    return None;
                }
                let update = updates.recv().await.unwrap_or_else(|| {
                    speech::Update::Done(Err(Error::Backend("the worker stopped".into())))
                });
                pending = match update {
                    speech::Update::Segment(segment, words) => {
                        let tokens: Vec<_> = words
                            .iter()
                            .flat_map(|w| w.tokens.iter().cloned())
                            .collect();
                        renderer.delta(&segment, &tokens)
                    }
                    speech::Update::Done(Ok(transcript)) => {
                        done = true;
                        renderer.done(&transcript)
                    }
                    speech::Update::Done(Err(error)) => {
                        done = true;
                        renderer.error(&failure(error, "Transcription failed"))
                    }
                };
            }
        },
    );
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}

type TranscriptState = (
    UnboundedReceiver<speech::Update>,
    TranscriptStream,
    Vec<crate::openai::Event>,
    bool,
);

/// `GET /v1/realtime`: a transcription session over a WebSocket. `model`, if given, must be
/// the speech model; `intent`, if given, must be `transcription`.
pub(super) async fn realtime(
    State(state): State<AppState>,
    Query(query): Query<HashMap<String, String>>,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
) -> Response {
    let Some(speech) = state.speech.clone() else {
        return ApiError::new(StatusCode::NOT_FOUND, "not_found_error", "Unknown path")
            .openai()
            .into_response();
    };
    if let Some(model) = query.get("model")
        && !speech.serves(model)
    {
        return ApiError::from(model_not_found(model)).into_response();
    }
    if let Some(intent) = query.get("intent")
        && intent != "transcription"
    {
        return ApiError::from(OpenAiError::unsupported(
            "intent",
            "only transcription sessions are available",
        ))
        .into_response();
    }
    match upgrade {
        Ok(upgrade) => upgrade
            .protocols(["realtime"])
            .on_upgrade(move |socket| crate::realtime::serve(socket, speech)),
        Err(rejection) => ApiError::from(OpenAiError::new(
            rejection.status().as_u16(),
            "invalid_request_error",
            rejection.body_text(),
        ))
        .into_response(),
    }
}
