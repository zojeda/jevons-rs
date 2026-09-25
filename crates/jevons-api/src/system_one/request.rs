//! Parse and validate requests before they reach inference.

use crate::system_one::ValidationError;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use jevons_core::ImageInput;
use jevons_decision::ReadOptions;
use serde_json::{Map, Value, json};

#[derive(Clone, Debug)]
pub struct Request {
    pub(crate) model: String,
    pub(crate) state: Value,
    pub(crate) questions: Vec<(String, Question)>,
    options: ReadOptions,
    images: Vec<ImageInput>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum QuestionKind {
    Noul,
    Choice,
    Score,
}

#[derive(Clone, Debug)]
pub(crate) struct Question {
    pub kind: QuestionKind,
    pub instructions: Option<Value>,
    pub labels: Vec<String>,
    pub descriptions: Vec<Value>,
}

fn required<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    parent: &[&str],
) -> Result<&'a Value, ValidationError> {
    let mut path = parent.to_vec();
    path.push(key);
    object
        .get(key)
        .ok_or_else(|| ValidationError::new(&path, "Field required", "missing"))
}

fn content(value: &Value) -> bool {
    value.is_string() || value.is_object() || value.is_array()
}

impl Request {
    pub fn options(&self) -> ReadOptions {
        self.options
    }

    pub fn images(&self) -> &[ImageInput] {
        &self.images
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn parse(value: Value) -> Result<Self, ValidationError> {
        let object = value.as_object().ok_or_else(|| {
            ValidationError::new(&["body"], "Expected a JSON object", "model_type")
        })?;
        let model = required(object, "model", &["body"])?
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                ValidationError::new(
                    &["body", "model"],
                    "Expected a nonempty model name",
                    "string_type",
                )
            })?
            .to_owned();
        let state = required(object, "state", &["body"])?;
        if !content(state) {
            return Err(ValidationError::new(
                &["body", "state"],
                "Expected a string, object, or array",
                "value_error",
            ));
        }
        let questions = required(object, "questions", &["body"])?
            .as_object()
            .filter(|q| !q.is_empty())
            .ok_or_else(|| {
                ValidationError::new(
                    &["body", "questions"],
                    "At least one question is required",
                    "value_error",
                )
            })?;
        for key in object.keys() {
            if ![
                "state",
                "model",
                "questions",
                "steps",
                "samples",
                "think",
                "sequential",
                "images",
            ]
            .contains(&key.as_str())
            {
                return Err(ValidationError::new(
                    &["body", key],
                    "Unknown field",
                    "extra_forbidden",
                ));
            }
        }
        let options = ReadOptions {
            steps: integer_option(object, "steps", 1, 1, 8)?,
            samples: integer_option(object, "samples", 1, 1, 32)?,
            think: integer_option(object, "think", 0, 0, 4096)?,
            sequential: match object.get("sequential").filter(|v| !v.is_null()) {
                None => false,
                Some(value) => value.as_bool().ok_or_else(|| {
                    ValidationError::new(&["body", "sequential"], "Expected a boolean", "bool_type")
                })?,
            },
        };
        let images = parse_images(object.get("images"))?;
        if !images.is_empty() && (options.think > 0 || options.sequential) {
            let key = if options.think > 0 {
                "think"
            } else {
                "sequential"
            };
            return Err(ValidationError::new(
                &["body", key],
                "Cannot combine this option with images",
                "value_error",
            ));
        }
        let questions = questions
            .iter()
            .map(|(id, value)| Ok((id.clone(), Question::parse(id, value)?)))
            .collect::<Result<_, ValidationError>>()?;
        Ok(Self {
            model,
            state: state.clone(),
            questions,
            options,
            images,
        })
    }
}

fn integer_option(
    object: &Map<String, Value>,
    key: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize, ValidationError> {
    match object.get(key).filter(|v| !v.is_null()) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .filter(|v| (*v >= min as u64) && (*v <= max as u64))
            .map(|v| v as usize)
            .ok_or_else(|| {
                ValidationError::new(
                    &["body", key],
                    format!("Expected an integer from {min} to {max}"),
                    "value_error",
                )
            }),
    }
}

fn parse_images(value: Option<&Value>) -> Result<Vec<ImageInput>, ValidationError> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(Vec::new());
    };
    let error = |message: &str| ValidationError::new(&["body", "images"], message, "value_error");
    let images = value
        .as_array()
        .filter(|v| v.len() <= 8)
        .ok_or_else(|| error("Expected at most 8 images"))?;
    images
        .iter()
        .map(|value| {
            let (mime, encoded) = if let Some(url) = value.as_str() {
                url.strip_prefix("data:")
                    .and_then(|s| s.split_once(";base64,"))
                    .ok_or_else(|| error("Expected a base64 image data URL"))?
            } else {
                let object = value
                    .as_object()
                    .ok_or_else(|| error("Expected an image data URL or object"))?;
                if object.keys().any(|k| k != "content_type" && k != "base64") {
                    return Err(error("Unknown image field"));
                }
                (
                    object
                        .get("content_type")
                        .and_then(Value::as_str)
                        .ok_or_else(|| error("Missing image content_type"))?,
                    object
                        .get("base64")
                        .and_then(Value::as_str)
                        .ok_or_else(|| error("Missing image base64"))?,
                )
            };
            if !["image/jpeg", "image/png", "image/webp", "image/gif"].contains(&mime) {
                return Err(error("Supported image types: JPEG, PNG, WebP, GIF"));
            }
            const MAX_BYTES: usize = 5 * 1024 * 1024;
            if encoded.len() > MAX_BYTES.div_ceil(3) * 4 {
                return Err(error("Image exceeds 5 MiB"));
            }
            let bytes = STANDARD
                .decode(encoded)
                .map_err(|_| error("Invalid image base64"))?;
            if bytes.is_empty() || bytes.len() > MAX_BYTES {
                return Err(error("Image must contain 1 to 5 MiB of data"));
            }
            let matches = match mime {
                "image/png" => bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
                "image/jpeg" => bytes.starts_with(b"\xff\xd8\xff"),
                "image/gif" => bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a"),
                "image/webp" => bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP"),
                _ => false,
            };
            if !matches {
                return Err(error("Image content does not match its content_type"));
            }
            Ok(ImageInput { bytes })
        })
        .collect()
}

impl Question {
    fn parse(id: &str, value: &Value) -> Result<Self, ValidationError> {
        let path = ["body", "questions", id];
        let criteria_path = ["body", "questions", id, "criteria"];
        let object = value.as_object().ok_or_else(|| {
            ValidationError::new(&path, "Expected a question object", "model_type")
        })?;
        for key in object.keys() {
            if !["type", "instructions", "criteria"].contains(&key.as_str()) {
                return Err(ValidationError::new(
                    &["body", "questions", id, key],
                    "Unknown field",
                    "extra_forbidden",
                ));
            }
        }
        let instructions = object.get("instructions").filter(|v| !v.is_null()).cloned();
        if instructions.as_ref().is_some_and(|v| !content(v)) {
            return Err(ValidationError::new(
                &["body", "questions", id, "instructions"],
                "Expected a string, object, or array",
                "value_error",
            ));
        }
        let question_type = required(object, "type", &path)?.as_str();
        let (kind, labels, descriptions) = match question_type {
            Some("noul") => {
                let mut descriptions = vec![json!("yes"), json!("no")];
                if let Some(criteria) = object.get("criteria").filter(|v| !v.is_null()) {
                    let criteria = criteria.as_object().ok_or_else(|| {
                        ValidationError::new(
                            &criteria_path,
                            "Expected an object with true and false descriptions",
                            "dict_type",
                        )
                    })?;
                    if criteria.keys().any(|k| k != "true" && k != "false") {
                        return Err(ValidationError::new(
                            &criteria_path,
                            "Only true and false criteria are allowed",
                            "value_error",
                        ));
                    }
                    for (i, key) in ["true", "false"].iter().enumerate() {
                        if let Some(value) = criteria.get(*key) {
                            descriptions[i] = value.clone();
                        }
                    }
                }
                (
                    QuestionKind::Noul,
                    vec!["yes".into(), "no".into()],
                    descriptions,
                )
            }
            Some("choice") => {
                let criteria = required(object, "criteria", &path)?
                    .as_object()
                    .filter(|v| (1..=128).contains(&v.len()))
                    .ok_or_else(|| {
                        ValidationError::new(
                            &criteria_path,
                            "Expected 1 to 128 options",
                            "value_error",
                        )
                    })?;
                (
                    QuestionKind::Choice,
                    criteria.keys().cloned().collect(),
                    criteria.values().cloned().collect(),
                )
            }
            Some("score") => {
                let criteria = required(object, "criteria", &path)?
                    .as_array()
                    .filter(|v| (2..=10).contains(&v.len()) && v.iter().all(Value::is_string))
                    .ok_or_else(|| {
                        ValidationError::new(
                            &criteria_path,
                            "Expected 2 to 10 string levels, ordered lowest to highest",
                            "value_error",
                        )
                    })?;
                (
                    QuestionKind::Score,
                    (0..criteria.len()).map(|i| i.to_string()).collect(),
                    criteria.clone(),
                )
            }
            _ => {
                return Err(ValidationError::new(
                    &["body", "questions", id, "type"],
                    "Expected noul, choice, or score",
                    "literal_error",
                ));
            }
        };
        Ok(Self {
            kind,
            instructions,
            labels,
            descriptions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Value {
        json!({"model":"jev-latest", "state":"test", "questions":{"q":{"type":"noul"}}})
    }

    #[test]
    fn required_fields_and_invalid_types_have_precise_locations() {
        let error = Request::parse(json!({"model":"x"})).unwrap_err();
        assert_eq!(error.loc, vec![json!("body"), json!("state")]);
        assert_eq!(error.kind, "missing");
        let mut value = base();
        value["questions"]["q"]["type"] = json!("text");
        assert_eq!(
            Request::parse(value).unwrap_err().loc,
            vec![json!("body"), json!("questions"), json!("q"), json!("type")]
        );
        for state in [Value::Null, json!(42), json!(true)] {
            let mut value = base();
            value["state"] = state;
            assert!(Request::parse(value).is_err());
        }
    }

    #[test]
    fn extension_ranges_and_types_are_validated() {
        for (key, valid, invalid) in [
            (
                "steps",
                vec![json!(1), json!(8)],
                vec![json!(0), json!(9), json!(1.5)],
            ),
            (
                "samples",
                vec![json!(1), json!(32)],
                vec![json!(0), json!(33)],
            ),
            (
                "think",
                vec![json!(0), json!(4096)],
                vec![json!(-1), json!(4097)],
            ),
            (
                "sequential",
                vec![json!(true), json!(false)],
                vec![json!(1)],
            ),
        ] {
            for option in valid.into_iter().chain([Value::Null]) {
                let mut value = base();
                value[key] = option;
                assert!(Request::parse(value).is_ok());
            }
            for option in invalid.into_iter().chain([json!("1")]) {
                let mut value = base();
                value[key] = option;
                assert_eq!(
                    Request::parse(value).unwrap_err().loc,
                    vec![json!("body"), json!(key)]
                );
            }
        }
        let defaults = Request::parse(base()).unwrap().options();
        assert_eq!(
            (
                defaults.steps,
                defaults.samples,
                defaults.think,
                defaults.sequential
            ),
            (1, 1, 0, false)
        );
    }

    #[test]
    fn images_accept_both_wire_formats_and_reject_bad_payloads_and_combinations() {
        let png = STANDARD.encode(b"\x89PNG\r\n\x1a\n");
        for image in [
            json!(format!("data:image/png;base64,{png}")),
            json!({"content_type":"image/png", "base64":png}),
        ] {
            let mut value = base();
            value["images"] = json!([image]);
            assert_eq!(
                Request::parse(value.clone()).unwrap().images()[0].bytes,
                b"\x89PNG\r\n\x1a\n"
            );
            for (key, option) in [("think", json!(1)), ("sequential", json!(true))] {
                let mut bad = value.clone();
                bad[key] = option;
                assert!(Request::parse(bad).is_err());
            }
        }
        for images in [
            json!(["https://example.com/a.png"]),
            json!(["data:image/png;base64,!!!"]),
            json!(["data:image/svg+xml;base64,QQ=="]),
            json!(["data:image/png;base64,QQ=="]),
            json!(["data:image/png;base64,"]),
            json!(vec!["x"; 9]),
            json!(false),
        ] {
            let mut value = base();
            value["images"] = images;
            assert!(Request::parse(value).is_err());
        }
        let mut value = base();
        value["images"] = json!([format!("data:image/png;base64,{}", "A".repeat(7_000_000))]);
        assert!(Request::parse(value).is_err());
    }

    #[test]
    fn choice_and_score_enforce_documented_cardinality() {
        for count in [0, 1, 128, 129] {
            let mut value = base();
            let criteria: Map<String, Value> =
                (0..count).map(|i| (i.to_string(), Value::Null)).collect();
            value["questions"]["q"] = json!({"type":"choice", "criteria":criteria});
            assert_eq!(Request::parse(value).is_ok(), (1..=128).contains(&count));
        }
        for count in [1, 2, 10, 11] {
            let mut value = base();
            value["questions"]["q"] = json!({"type":"score", "criteria":vec!["level";count]});
            assert_eq!(Request::parse(value).is_ok(), (2..=10).contains(&count));
        }
    }
}
