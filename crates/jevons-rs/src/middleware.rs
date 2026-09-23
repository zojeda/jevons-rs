//! Bearer authentication and request IDs for every HTTP response.

use crate::{AppState, error::ApiError};
use axum::{
    extract::{Request, State},
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use uuid::Uuid;

pub(super) async fn request_context(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let auth_error = if request.uri().path() == "/health" {
        None
    } else if let Some(expected) = &state.api_key {
        match request.headers().get("authorization") {
            None => Some(ApiError::new(
                StatusCode::FORBIDDEN,
                "authentication_error",
                "No API key provided",
            )),
            Some(header)
                if header.to_str().ok().and_then(|v| v.strip_prefix("Bearer "))
                    == Some(expected.as_ref()) =>
            {
                None
            }
            Some(_) => Some(ApiError::new(
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "Invalid API key",
            )),
        }
    } else {
        None
    };
    let mut response = match auth_error {
        Some(error) => error.into_response(),
        None => next.run(request).await,
    };
    response.headers_mut().insert(
        "x-typesafe-request-id",
        HeaderValue::from_str(&request_id).expect("UUID is an ASCII header"),
    );
    response
}
