//! `POST /v1/audio/transcriptions`: validation of the multipart form fields, the response in
//! each `response_format`, and the transcription stream events.
use crate::openai::{Event, OpenAiError};
use jevons_core::{Segment, Transcript};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseFormat {
    Json,
    Text,
    Srt,
    Vtt,
    VerboseJson,
}

/// A validated transcription request (every form field except `file`).
#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptionRequest {
    pub model: String,
    /// ISO-639-1, lowercase. The model detects the language itself; this is validated and
    /// echoed.
    pub language: Option<String>,
    pub format: ResponseFormat,
    /// `verbose_json` word and segment timestamps.
    pub words: bool,
    pub segments: bool,
    pub stream: bool,
    /// `include[]=logprobs`: token log probabilities (`json` only).
    pub logprobs: bool,
}

/// A response body and its content type.
#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    Json(Value),
    Text {
        content_type: &'static str,
        text: String,
    },
}

fn invalid(message: impl Into<String>, param: &str) -> OpenAiError {
    OpenAiError::invalid(message, Some(param))
}

fn boolean(key: &str, value: &str) -> Result<bool, OpenAiError> {
    match value {
        "true" | "True" | "1" => Ok(true),
        "false" | "False" | "0" => Ok(false),
        _ => Err(invalid(format!("{key} must be true or false"), key)),
    }
}

impl TranscriptionRequest {
    /// Validates the form fields other than `file`, in order. `languages` lists the ISO-639-1
    /// codes the model transcribes. Array fields may be sent as `name[]` or `name`, repeated.
    pub fn parse(fields: &[(String, String)], languages: &[&str]) -> Result<Self, OpenAiError> {
        let mut model = None;
        let mut language = None;
        let mut format = None;
        let mut stream = None;
        let mut granularities = Vec::new();
        let mut logprobs = false;
        let mut seen: Vec<&str> = Vec::new();
        for (name, value) in fields {
            let key = name.strip_suffix("[]").unwrap_or(name);
            let array = matches!(key, "timestamp_granularities" | "include");
            if !array {
                if seen.contains(&key) {
                    return Err(invalid(format!("{key} is given more than once"), key));
                }
                seen.push(key);
            }
            match key {
                "model" => model = Some(value.trim().to_string()),
                "language" => {
                    let code = value.trim().to_ascii_lowercase();
                    if !code.is_empty() {
                        if !languages.contains(&code.as_str()) {
                            return Err(invalid(
                                format!(
                                    "language must be an ISO-639-1 code the model transcribes: {}",
                                    languages.join(", ")
                                ),
                                key,
                            ));
                        }
                        language = Some(code);
                    }
                }
                "prompt" if value.trim().is_empty() => {}
                "prompt" => {
                    return Err(OpenAiError::unsupported(
                        key,
                        "this model does not take prompts",
                    ));
                }
                "response_format" => {
                    format = Some(match value.as_str() {
                        "json" => ResponseFormat::Json,
                        "text" => ResponseFormat::Text,
                        "srt" => ResponseFormat::Srt,
                        "vtt" => ResponseFormat::Vtt,
                        "verbose_json" => ResponseFormat::VerboseJson,
                        "diarized_json" => {
                            return Err(OpenAiError::unsupported(
                                key,
                                "diarization is not available",
                            ));
                        }
                        _ => {
                            return Err(invalid(
                                "response_format must be json, text, srt, vtt or verbose_json",
                                key,
                            ));
                        }
                    })
                }
                // Decoding is greedy; the value is checked for compatibility only.
                "temperature" => match value.parse::<f64>() {
                    Ok(t) if (0.0..=1.0).contains(&t) => {}
                    _ => return Err(invalid("temperature must be a number from 0 to 1", key)),
                },
                "stream" => stream = Some(boolean(key, value)?),
                "timestamp_granularities" => match value.as_str() {
                    "word" | "segment" => granularities.push(value.clone()),
                    _ => {
                        return Err(invalid(
                            "timestamp_granularities must be word or segment",
                            key,
                        ));
                    }
                },
                "include" => match value.as_str() {
                    "logprobs" => logprobs = true,
                    _ => {
                        return Err(OpenAiError::unsupported(
                            key,
                            "only logprobs can be included",
                        ));
                    }
                },
                "chunking_strategy" if value == "auto" => {}
                "chunking_strategy" => {
                    return Err(OpenAiError::unsupported(
                        key,
                        "only auto chunking is available",
                    ));
                }
                _ => return Err(OpenAiError::unsupported(key, "unknown parameter")),
            }
        }
        let model = model
            .filter(|m| !m.is_empty())
            .ok_or_else(|| invalid("model is required", "model"))?;
        let format = format.unwrap_or(ResponseFormat::Json);
        let stream = stream.unwrap_or(false);
        if !granularities.is_empty() && format != ResponseFormat::VerboseJson {
            return Err(invalid(
                "timestamp_granularities requires response_format verbose_json",
                "timestamp_granularities",
            ));
        }
        if logprobs && format != ResponseFormat::Json {
            return Err(invalid(
                "include[]=logprobs requires response_format json",
                "include",
            ));
        }
        if stream && !matches!(format, ResponseFormat::Json | ResponseFormat::Text) {
            return Err(invalid(
                "stream requires response_format json or text",
                "stream",
            ));
        }
        let words = granularities.iter().any(|g| g == "word");
        let segments = format == ResponseFormat::VerboseJson
            && (granularities.is_empty() || granularities.iter().any(|g| g == "segment"));
        Ok(Self {
            model,
            language,
            format,
            words,
            segments,
            stream,
            logprobs,
        })
    }

    /// The complete (non-streaming) response.
    pub fn response(&self, transcript: &Transcript) -> Body {
        match self.format {
            ResponseFormat::Json => {
                let mut body = json!({"text": transcript.text, "usage": usage(transcript)});
                if self.logprobs {
                    body["logprobs"] = logprobs(transcript);
                }
                Body::Json(body)
            }
            ResponseFormat::Text => Body::Text {
                content_type: "text/plain; charset=utf-8",
                text: transcript.text.clone(),
            },
            ResponseFormat::Srt => Body::Text {
                content_type: "application/x-subrip; charset=utf-8",
                text: subtitles(&transcript.segments, false),
            },
            ResponseFormat::Vtt => Body::Text {
                content_type: "text/vtt; charset=utf-8",
                text: subtitles(&transcript.segments, true),
            },
            ResponseFormat::VerboseJson => {
                let mut body = json!({
                    "task": "transcribe",
                    "language": self.language,
                    "duration": transcript.duration,
                    "text": transcript.text,
                    "usage": usage(transcript),
                });
                if self.segments {
                    body["segments"] = transcript
                        .segments
                        .iter()
                        .map(|s| {
                            json!({
                                "id": s.id,
                                "seek": 0,
                                "start": s.start,
                                "end": s.end,
                                "text": s.text,
                                "tokens": s.tokens,
                                "temperature": 0.0,
                                "avg_logprob": s.avg_logprob,
                            })
                        })
                        .collect();
                }
                if self.words {
                    body["words"] = transcript
                        .words
                        .iter()
                        .map(|w| json!({"word": w.text, "start": w.start, "end": w.end}))
                        .collect();
                }
                Body::Json(body)
            }
        }
    }

    /// Renders a streamed transcription.
    pub fn stream(&self) -> TranscriptStream {
        TranscriptStream {
            logprobs: self.logprobs,
            started: false,
        }
    }
}

fn usage(transcript: &Transcript) -> Value {
    json!({"type": "duration", "seconds": transcript.duration.ceil() as u64})
}

fn token_logprobs<'a>(tokens: impl Iterator<Item = &'a jevons_core::SpeechToken>) -> Value {
    tokens
        .map(|t| {
            let text = t.piece.replace('▁', " ");
            json!({"token": text, "logprob": t.logprob, "bytes": text.as_bytes()})
        })
        .collect()
}

fn logprobs(transcript: &Transcript) -> Value {
    token_logprobs(transcript.tokens())
}

/// `HH:MM:SS,mmm` (SRT) or `HH:MM:SS.mmm` (WebVTT).
fn timestamp(seconds: f64, vtt: bool) -> String {
    let millis = (seconds.max(0.0) * 1000.0).round() as u64;
    let (h, m, s, ms) = (
        millis / 3_600_000,
        millis / 60_000 % 60,
        millis / 1000 % 60,
        millis % 1000,
    );
    let separator = if vtt { '.' } else { ',' };
    format!("{h:02}:{m:02}:{s:02}{separator}{ms:03}")
}

fn subtitles(segments: &[Segment], vtt: bool) -> String {
    let mut out = String::new();
    if vtt {
        out.push_str("WEBVTT\n\n");
    }
    for (i, s) in segments.iter().enumerate() {
        if !vtt {
            out.push_str(&format!("{}\n", i + 1));
        }
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            timestamp(s.start, vtt),
            timestamp(s.end, vtt),
            s.text
        ));
    }
    out
}

/// Stream renderer: `transcript.text.delta` per segment, then `transcript.text.done`.
pub struct TranscriptStream {
    logprobs: bool,
    started: bool,
}

impl TranscriptStream {
    /// The delta for a finished segment of `transcript` so far.
    pub fn delta(&mut self, segment: &Segment, tokens: &[jevons_core::SpeechToken]) -> Vec<Event> {
        let delta = if self.started {
            format!(" {}", segment.text)
        } else {
            segment.text.clone()
        };
        self.started = true;
        let mut value = json!({"type": "transcript.text.delta", "delta": delta});
        if self.logprobs {
            value["logprobs"] = token_logprobs(tokens.iter());
        }
        vec![Event::data(value)]
    }

    pub fn done(&mut self, transcript: &Transcript) -> Vec<Event> {
        let mut value = json!({
            "type": "transcript.text.done",
            "text": transcript.text,
            "usage": usage(transcript),
        });
        if self.logprobs {
            value["logprobs"] = logprobs(transcript);
        }
        vec![Event::data(value)]
    }

    pub fn error(&mut self, error: &OpenAiError) -> Vec<Event> {
        let mut value = error.body();
        value["type"] = json!("error");
        vec![Event::data(value)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jevons_core::{SpeechToken, Word};

    const LANGUAGES: &[&str] = &["en", "es"];

    fn fields(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn parse(pairs: &[(&str, &str)]) -> Result<TranscriptionRequest, OpenAiError> {
        TranscriptionRequest::parse(&fields(pairs), LANGUAGES)
    }

    fn transcript() -> Transcript {
        let token = |id, piece: &str, start, end| SpeechToken {
            id,
            piece: piece.into(),
            start,
            end,
            logprob: -0.25,
        };
        let hola = token(1, "▁Hola", 0.0, 0.4);
        let mundo = token(2, "▁mundo.", 0.5, 1.25);
        let otra = token(3, "▁Otra", 3661.5, 3662.0);
        Transcript {
            text: "Hola mundo. Otra".into(),
            duration: 3662.2,
            words: vec![
                Word {
                    text: "Hola".into(),
                    start: 0.0,
                    end: 0.4,
                    tokens: vec![hola],
                },
                Word {
                    text: "mundo.".into(),
                    start: 0.5,
                    end: 1.25,
                    tokens: vec![mundo],
                },
                Word {
                    text: "Otra".into(),
                    start: 3661.5,
                    end: 3662.0,
                    tokens: vec![otra],
                },
            ],
            segments: vec![
                Segment {
                    id: 0,
                    start: 0.0,
                    end: 1.25,
                    text: "Hola mundo.".into(),
                    tokens: vec![1, 2],
                    avg_logprob: -0.25,
                },
                Segment {
                    id: 1,
                    start: 3661.5,
                    end: 3662.0,
                    text: "Otra".into(),
                    tokens: vec![3],
                    avg_logprob: -0.25,
                },
            ],
        }
    }

    #[test]
    fn defaults_are_json_without_timestamps() {
        let request = parse(&[
            ("model", "parakeet"),
            ("temperature", "0.2"),
            ("prompt", ""),
        ])
        .unwrap();
        assert_eq!(request.format, ResponseFormat::Json);
        assert!(!request.stream && !request.words && !request.segments && !request.logprobs);
        assert_eq!(
            request.response(&transcript()),
            Body::Json(
                json!({"text": "Hola mundo. Otra", "usage": {"type": "duration", "seconds": 3663}})
            )
        );
    }

    #[test]
    fn invalid_and_unsupported_fields_are_rejected() {
        type Case<'a> = (&'a [(&'a str, &'a str)], &'a str, Option<&'a str>);
        let cases: &[Case] = &[
            (&[], "model", None),
            (&[("model", "m"), ("language", "xx")], "language", None),
            (
                &[("model", "m"), ("prompt", "context")],
                "prompt",
                Some("unsupported_parameter"),
            ),
            (
                &[("model", "m"), ("response_format", "yaml")],
                "response_format",
                None,
            ),
            (
                &[("model", "m"), ("response_format", "diarized_json")],
                "response_format",
                Some("unsupported_parameter"),
            ),
            (&[("model", "m"), ("temperature", "2")], "temperature", None),
            (&[("model", "m"), ("stream", "maybe")], "stream", None),
            (
                &[
                    ("model", "m"),
                    ("stream", "true"),
                    ("response_format", "srt"),
                ],
                "stream",
                None,
            ),
            (
                &[("model", "m"), ("timestamp_granularities[]", "word")],
                "timestamp_granularities",
                None,
            ),
            (
                &[
                    ("model", "m"),
                    ("include[]", "logprobs"),
                    ("response_format", "text"),
                ],
                "include",
                None,
            ),
            (
                &[("model", "m"), ("known_speaker_names[]", "Ana")],
                "known_speaker_names",
                Some("unsupported_parameter"),
            ),
            (&[("model", "m"), ("model", "n")], "model", None),
        ];
        for (pairs, param, code) in cases {
            let error = parse(pairs).unwrap_err();
            assert_eq!(error.status, 400, "{pairs:?}");
            assert_eq!(error.param.as_deref(), Some(*param), "{pairs:?}");
            assert_eq!(error.code, *code, "{pairs:?}");
        }
    }

    #[test]
    fn verbose_json_has_segments_by_default_and_words_on_request() {
        let request = parse(&[
            ("model", "m"),
            ("language", "ES"),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
        ])
        .unwrap();
        assert_eq!(request.language.as_deref(), Some("es"));
        let Body::Json(body) = request.response(&transcript()) else {
            panic!("verbose_json is JSON");
        };
        assert_eq!(body["language"], "es");
        assert_eq!(body["task"], "transcribe");
        assert!(body.get("segments").is_none(), "only words were requested");
        assert_eq!(
            body["words"][1],
            json!({"word": "mundo.", "start": 0.5, "end": 1.25})
        );

        let request = parse(&[("model", "m"), ("response_format", "verbose_json")]).unwrap();
        let Body::Json(body) = request.response(&transcript()) else {
            panic!("verbose_json is JSON");
        };
        assert_eq!(body["language"], Value::Null);
        assert_eq!(body["segments"][0]["text"], "Hola mundo.");
        assert_eq!(body["segments"][0]["tokens"], json!([1, 2]));
        assert!(body.get("words").is_none());
    }

    #[test]
    fn subtitles_number_cues_and_format_hours() {
        let srt = parse(&[("model", "m"), ("response_format", "srt")]).unwrap();
        assert_eq!(
            srt.response(&transcript()),
            Body::Text {
                content_type: "application/x-subrip; charset=utf-8",
                text: "1\n00:00:00,000 --> 00:00:01,250\nHola mundo.\n\n2\n01:01:01,500 --> 01:01:02,000\nOtra\n\n".into(),
            }
        );
        let vtt = parse(&[("model", "m"), ("response_format", "vtt")]).unwrap();
        let Body::Text { text, .. } = vtt.response(&transcript()) else {
            panic!("vtt is text");
        };
        assert!(text.starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nHola mundo.\n\n"));
    }

    #[test]
    fn streams_space_segments_and_finish_with_the_full_text() {
        let request = parse(&[
            ("model", "m"),
            ("stream", "true"),
            ("include[]", "logprobs"),
        ])
        .unwrap();
        let t = transcript();
        let mut stream = request.stream();
        let first = stream.delta(&t.segments[0], &[t.words[0].tokens[0].clone()]);
        let second = stream.delta(&t.segments[1], &[]);
        let done = stream.done(&t);
        let value = |events: Vec<Event>| serde_json::from_str::<Value>(&events[0].data).unwrap();
        let first = value(first);
        assert_eq!(first["type"], "transcript.text.delta");
        assert_eq!(first["delta"], "Hola mundo.");
        assert_eq!(first["logprobs"][0]["token"], " Hola");
        assert_eq!(value(second)["delta"], " Otra");
        let done = value(done);
        assert_eq!(done["type"], "transcript.text.done");
        assert_eq!(done["text"], "Hola mundo. Otra");
        assert_eq!(done["logprobs"].as_array().unwrap().len(), 3);
    }
}
