use serde_json::{Value, json};

/// An OpenAI-style error: `{"error": {"message", "type", "param", "code"}}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenAiError {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
    pub param: Option<String>,
    pub code: Option<&'static str>,
}

impl OpenAiError {
    pub fn invalid(message: impl Into<String>, param: Option<&str>) -> Self {
        Self {
            status: 400,
            kind: "invalid_request_error",
            message: message.into(),
            param: param.map(String::from),
            code: None,
        }
    }

    pub fn unsupported(param: &str, detail: &str) -> Self {
        let mut error = Self::invalid(format!("{param} is not supported: {detail}"), Some(param));
        error.code = Some("unsupported_parameter");
        error
    }

    pub fn new(status: u16, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
            param: None,
            code: None,
        }
    }

    pub fn body(&self) -> Value {
        json!({"error": {
            "message": self.message,
            "type": self.kind,
            "param": self.param,
            "code": self.code,
        }})
    }
}

impl std::fmt::Display for OpenAiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for OpenAiError {}
