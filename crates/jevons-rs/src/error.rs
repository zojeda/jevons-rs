use axum::{
    Json,
    http::{HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use jevons_system_one::ValidationError;
use serde_json::{Value, json};

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub body: Value,
}

impl ApiError {
    pub fn new(status: StatusCode, kind: &str, message: impl AsRef<str>) -> Self {
        Self {
            status,
            body: json!({"detail":{"error_type":kind,"message":message.as_ref()}}),
        }
    }

    pub fn overloaded() -> Self {
        Self::new(
            StatusCode::from_u16(529).expect("Valid status"),
            "overloaded_error",
            "The inference queue is full. Retry later.",
        )
    }

    pub fn unavailable() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded_error",
            "The inference worker is unavailable",
        )
    }
}

impl From<ValidationError> for ApiError {
    fn from(error: ValidationError) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            body: json!({"detail":[error]}),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.status, Json(self.body)).into_response();
        if self.status.as_u16() == 529 {
            response
                .headers_mut()
                .insert("retry-after", HeaderValue::from_static("1"));
        }
        response
    }
}
