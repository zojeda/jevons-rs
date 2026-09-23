//! Endpoint behavior and conversion from HTTP input to validated requests.

use crate::{AppState, error::ApiError};
use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
};
use jevons_system_one::ValidationError;
use serde_json::{Value, json};

const ALIASES: &[&str] = &["gemmadiffusion-latest", "openjev-latest", "jev-latest"];

pub(super) async fn health(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    if !state.worker.is_alive() {
        return Err(ApiError::unavailable());
    }
    Ok(Json(json!({"status":"ok", "model":state.model_id})))
}

pub(super) async fn models(State(state): State<AppState>) -> Json<Value> {
    let mut names = vec![state.model_id.as_str()];
    names.extend(
        ALIASES
            .iter()
            .copied()
            .filter(|name| *name != state.model_id),
    );
    Json(json!({"models": names.iter().map(|name| json!({
        "name":name,
        "description":format!("Local DiffusionGemma GGUF, structured diffusion reads. Served as {}.", state.model_id),
        "release_date":"2026-09-19"
    })).collect::<Vec<_>>()}))
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
    if request.model() != state.model_id && !ALIASES.contains(&request.model()) {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found_error",
            "Unknown model",
        ));
    }
    Ok(Json(state.worker.read(request).await?))
}
