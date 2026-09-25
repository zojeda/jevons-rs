//! Endpoint behavior and conversion from HTTP input to validated requests.

use crate::{AppState, error::ApiError, worker::Update};
use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use jevons_engine::Error;
use jevons_openai::{Api, OpenAiError, OpenAiRequest};
use jevons_system_one::ValidationError;
use serde_json::{Value, json};
use std::convert::Infallible;
use tokio::sync::mpsc::UnboundedReceiver;

pub(super) async fn health(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    if !state.worker.is_alive() {
        return Err(ApiError::unavailable());
    }
    Ok(Json(json!({"status":"ok", "model":state.model_id})))
}

/// The System One listing (`models`) and the OpenAI one (`object`, `data`) in one body.
pub(super) async fn models(State(state): State<AppState>) -> Json<Value> {
    let mut names = vec![state.model_id.as_str()];
    names.extend(state.aliases.iter().map(String::as_str));
    Json(json!({
        "models": names.iter().map(|name| json!({
            "name":name,
            "description":state.description,
            "release_date":"2026-09-19"
        })).collect::<Vec<_>>(),
        "object": "list",
        "data": names.iter().map(|name| json!({
            "id": name,
            "object": "model",
            "created": 1_790_294_400,
            "owned_by": "jevons",
        })).collect::<Vec<_>>(),
    }))
}

pub(super) async fn system_one(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<jevons_system_one::Response>, ApiError> {
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
    let request = jevons_system_one::Request::parse(value)?;
    if request.model() != state.model_id && !state.aliases.iter().any(|a| a == request.model()) {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found_error",
            "Unknown model",
        ));
    }
    Ok(Json(state.worker.read(request).await?))
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
    match error {
        Error::InvalidInput(message) => OpenAiError::invalid(message, None),
        other => {
            tracing::error!(error = %other, "Generation failed");
            OpenAiError::new(500, "server_error", "Generation failed")
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
    if request.model != state.model_id && !state.aliases.contains(&request.model) {
        let mut error = OpenAiError::new(
            404,
            "invalid_request_error",
            format!("The model {:?} does not exist", request.model),
        );
        error.code = Some("model_not_found");
        error.param = Some("model".into());
        return Err(error.into());
    }
    let mut updates = state
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
                    let body = request.response(&id, created, &state.model_id, &generation);
                    return Ok(Json(body).into_response());
                }
            }
            update = updates.recv().await.ok_or_else(ApiError::unavailable)?;
        }
    }
    if let Update::Done(Err(error)) = first {
        return Err(engine_error(error).into());
    }
    let mut renderer = request.stream(&id, created, &state.model_id);
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
    jevons_openai::Stream,
    Option<Update>,
    Vec<jevons_openai::Event>,
    bool,
);
