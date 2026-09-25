//! The Realtime API for transcription sessions: client events parsed into commands, and server
//! events rendered as JSON text frames.
//!
//! Sessions follow the GA shape (`session.update` with `session.type = "transcription"`,
//! `conversation.item.added` / `.done`). A client that sends the beta
//! `transcription_session.update` gets beta replies (`transcription_session.updated`,
//! `conversation.item.created`) from then on.
use crate::openai::OpenAiError;
use base64::Engine as _;
use serde_json::{Map, Value, json};

/// Input sample formats: 16-bit PCM (mono, little-endian) at a rate, or 8 kHz G.711.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioFormat {
    Pcm { rate: u32 },
    Mulaw,
    Alaw,
}

impl AudioFormat {
    pub fn rate(self) -> u32 {
        match self {
            Self::Pcm { rate } => rate,
            Self::Mulaw | Self::Alaw => 8000,
        }
    }
}

/// `server_vad` settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurnDetection {
    pub threshold: f32,
    pub prefix_padding_ms: u32,
    pub silence_duration_ms: u32,
}

impl Default for TurnDetection {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
        }
    }
}

/// The effective configuration of a transcription session.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionConfig {
    pub format: AudioFormat,
    pub model: String,
    pub language: Option<String>,
    /// `None`: the client commits turns itself.
    pub turn_detection: Option<TurnDetection>,
    pub logprobs: bool,
}

/// What a client event asks the session to do.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// The configuration changed; reply with [`Session::updated`].
    Updated,
    Append(Vec<u8>),
    Commit,
    Clear,
}

/// A rejected client event, rendered with [`Session::error`].
#[derive(Clone, Debug, PartialEq)]
pub struct EventError {
    pub event_id: Option<String>,
    pub code: &'static str,
    pub message: String,
    pub param: Option<String>,
}

impl EventError {
    fn new(code: &'static str, message: impl Into<String>, param: Option<&str>) -> Self {
        Self {
            event_id: None,
            code,
            message: message.into(),
            param: param.map(String::from),
        }
    }

    fn invalid(message: impl Into<String>, param: &str) -> Self {
        Self::new("invalid_value", message, Some(param))
    }

    fn unsupported(param: &str, detail: &str) -> Self {
        Self::new(
            "unsupported_parameter",
            format!("{param} is not supported: {detail}"),
            Some(param),
        )
    }
}

pub struct Session {
    id: String,
    config: SessionConfig,
    /// Names the session's `model` may take.
    models: Vec<String>,
    languages: &'static [&'static str],
    beta: bool,
    events: u64,
}

fn object<'a>(value: &'a Value, param: &str) -> Result<&'a Map<String, Value>, EventError> {
    value
        .as_object()
        .ok_or_else(|| EventError::invalid(format!("{param} must be an object"), param))
}

impl Session {
    /// A session answering as `models[0]` (the other names are aliases).
    pub fn new(id: String, models: Vec<String>, languages: &'static [&'static str]) -> Self {
        Self {
            id,
            config: SessionConfig {
                format: AudioFormat::Pcm { rate: 24000 },
                model: models[0].clone(),
                language: None,
                turn_detection: Some(TurnDetection::default()),
                logprobs: false,
            },
            models,
            languages,
            beta: false,
            events: 0,
        }
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    fn event(&mut self, kind: &str, fields: Value) -> String {
        self.events += 1;
        let mut value = json!({"type": kind, "event_id": format!("event_{}", self.events)});
        if let (Value::Object(target), Value::Object(fields)) = (&mut value, fields) {
            target.extend(fields);
        }
        value.to_string()
    }

    /// Parses one client text frame.
    pub fn handle(&mut self, text: &str) -> Result<Command, EventError> {
        let value: Value = serde_json::from_str(text)
            .map_err(|_| EventError::new("invalid_json", "The event is not valid JSON", None))?;
        let event_id = value["event_id"].as_str().map(String::from);
        self.command(&value).map_err(|mut e| {
            e.event_id = event_id;
            e
        })
    }

    fn command(&mut self, value: &Value) -> Result<Command, EventError> {
        let kind = value["type"].as_str().ok_or_else(|| {
            EventError::new(
                "invalid_event",
                "The 'type' field is missing.",
                Some("type"),
            )
        })?;
        match kind {
            "session.update" => {
                let session = object(&value["session"], "session")?;
                self.update(session)?;
                Ok(Command::Updated)
            }
            "transcription_session.update" => {
                let session = object(&value["session"], "session")?;
                self.update_beta(session)?;
                self.beta = true;
                Ok(Command::Updated)
            }
            "input_audio_buffer.append" => {
                let audio = value["audio"]
                    .as_str()
                    .ok_or_else(|| EventError::invalid("audio must be a base64 string", "audio"))?;
                base64::engine::general_purpose::STANDARD
                    .decode(audio)
                    .map(Command::Append)
                    .map_err(|_| EventError::invalid("audio must be valid base64", "audio"))
            }
            "input_audio_buffer.commit" => Ok(Command::Commit),
            "input_audio_buffer.clear" => Ok(Command::Clear),
            "response.create" | "response.cancel" | "conversation.item.create" => {
                Err(EventError::new(
                    "unsupported_event",
                    format!("{kind} is not available in a transcription session"),
                    Some("type"),
                ))
            }
            _ => Err(EventError::new(
                "invalid_event",
                format!("Unknown event type {kind:?}"),
                Some("type"),
            )),
        }
    }

    fn model(&mut self, model: Option<&Value>) -> Result<(), EventError> {
        match model {
            None | Some(Value::Null) => Ok(()),
            Some(Value::String(m)) if self.models.contains(m) => Ok(()),
            Some(_) => Err(EventError::new(
                "model_not_found",
                format!(
                    "The transcription model must be {}",
                    self.models.join(" or ")
                ),
                Some("model"),
            )),
        }
    }

    fn language(&mut self, language: Option<&Value>) -> Result<Option<String>, EventError> {
        match language
            .and_then(Value::as_str)
            .map(str::to_ascii_lowercase)
        {
            None => Ok(None),
            Some(code) if code.is_empty() => Ok(None),
            Some(code) if self.languages.contains(&code.as_str()) => Ok(Some(code)),
            Some(_) => Err(EventError::invalid(
                format!(
                    "language must be an ISO-639-1 code the model transcribes: {}",
                    self.languages.join(", ")
                ),
                "language",
            )),
        }
    }

    fn transcription(&mut self, t: &Value, config: &mut SessionConfig) -> Result<(), EventError> {
        if t.is_null() {
            return Ok(());
        }
        let t = object(t, "transcription")?;
        self.model(t.get("model"))?;
        if t.get("language").is_some() {
            config.language = self.language(t.get("language"))?;
        }
        if t.get("prompt")
            .and_then(Value::as_str)
            .is_some_and(|p| !p.trim().is_empty())
        {
            return Err(EventError::unsupported(
                "prompt",
                "this model does not take prompts",
            ));
        }
        Ok(())
    }

    fn turn_detection(value: &Value) -> Result<Option<TurnDetection>, EventError> {
        if value.is_null() {
            return Ok(None);
        }
        let t = object(value, "turn_detection")?;
        match t.get("type").and_then(Value::as_str) {
            Some("server_vad") => {}
            Some("semantic_vad") => {
                return Err(EventError::unsupported(
                    "turn_detection.type",
                    "only server_vad is available",
                ));
            }
            _ => {
                return Err(EventError::invalid(
                    "turn_detection.type must be server_vad",
                    "turn_detection.type",
                ));
            }
        }
        let mut detection = TurnDetection::default();
        if let Some(threshold) = t.get("threshold").filter(|v| !v.is_null()) {
            detection.threshold = threshold
                .as_f64()
                .filter(|x| (0.0..=1.0).contains(x))
                .ok_or_else(|| {
                    EventError::invalid(
                        "turn_detection.threshold must be from 0 to 1",
                        "turn_detection.threshold",
                    )
                })? as f32;
        }
        let ms = |key: &str, default: u32| -> Result<u32, EventError> {
            match t.get(key).filter(|v| !v.is_null()) {
                None => Ok(default),
                Some(v) => v
                    .as_u64()
                    .filter(|&n| n <= 10_000)
                    .map(|n| n as u32)
                    .ok_or_else(|| {
                        EventError::invalid(
                            format!("turn_detection.{key} must be 0 to 10000 milliseconds"),
                            "turn_detection",
                        )
                    }),
            }
        };
        detection.prefix_padding_ms = ms("prefix_padding_ms", detection.prefix_padding_ms)?;
        detection.silence_duration_ms = ms("silence_duration_ms", detection.silence_duration_ms)?;
        Ok(Some(detection))
    }

    fn include(value: Option<&Value>, config: &mut SessionConfig) -> Result<(), EventError> {
        let Some(value) = value.filter(|v| !v.is_null()) else {
            return Ok(());
        };
        let items = value
            .as_array()
            .ok_or_else(|| EventError::invalid("include must be an array", "include"))?;
        config.logprobs = false;
        for item in items {
            match item.as_str() {
                Some("item.input_audio_transcription.logprobs") => config.logprobs = true,
                _ => {
                    return Err(EventError::unsupported(
                        "include",
                        "only item.input_audio_transcription.logprobs can be included",
                    ));
                }
            }
        }
        Ok(())
    }

    fn noise_reduction(value: Option<&Value>, param: &str) -> Result<(), EventError> {
        match value {
            None | Some(Value::Null) => Ok(()),
            Some(_) => Err(EventError::unsupported(
                param,
                "noise reduction is not available",
            )),
        }
    }

    /// Applies a GA `session.update`; fields that are absent keep their values.
    fn update(&mut self, session: &Map<String, Value>) -> Result<(), EventError> {
        let mut config = self.config.clone();
        if let Some(kind) = session.get("type")
            && kind != "transcription"
        {
            return Err(EventError::unsupported(
                "session.type",
                "only transcription sessions are available",
            ));
        }
        if let Some(input) = session.get("audio").and_then(|a| a.get("input")) {
            let input = object(input, "audio.input")?;
            if let Some(format) = input.get("format").filter(|v| !v.is_null()) {
                config.format = match format.get("type").and_then(Value::as_str) {
                    Some("audio/pcm") | None => {
                        let rate = format.get("rate").and_then(Value::as_u64).unwrap_or(24000);
                        if !matches!(rate, 8000 | 16000 | 24000 | 48000) {
                            return Err(EventError::invalid(
                                "audio.input.format.rate must be 24000 (or 8000, 16000, 48000)",
                                "audio.input.format.rate",
                            ));
                        }
                        AudioFormat::Pcm { rate: rate as u32 }
                    }
                    Some("audio/pcmu") => AudioFormat::Mulaw,
                    Some("audio/pcma") => AudioFormat::Alaw,
                    Some(_) => {
                        return Err(EventError::invalid(
                            "audio.input.format.type must be audio/pcm, audio/pcmu or audio/pcma",
                            "audio.input.format.type",
                        ));
                    }
                };
            }
            if let Some(t) = input.get("transcription") {
                self.transcription(t, &mut config)?;
            }
            if let Some(t) = input.get("turn_detection") {
                config.turn_detection = Self::turn_detection(t)?;
            }
            Self::noise_reduction(input.get("noise_reduction"), "audio.input.noise_reduction")?;
        }
        Self::include(session.get("include"), &mut config)?;
        self.config = config;
        Ok(())
    }

    /// Applies a beta `transcription_session.update`.
    fn update_beta(&mut self, session: &Map<String, Value>) -> Result<(), EventError> {
        let mut config = self.config.clone();
        if let Some(format) = session.get("input_audio_format").filter(|v| !v.is_null()) {
            config.format = match format.as_str() {
                Some("pcm16") => AudioFormat::Pcm { rate: 24000 },
                Some("g711_ulaw") => AudioFormat::Mulaw,
                Some("g711_alaw") => AudioFormat::Alaw,
                _ => {
                    return Err(EventError::invalid(
                        "input_audio_format must be pcm16, g711_ulaw or g711_alaw",
                        "input_audio_format",
                    ));
                }
            };
        }
        if let Some(t) = session.get("input_audio_transcription") {
            self.transcription(t, &mut config)?;
        }
        if let Some(t) = session.get("turn_detection") {
            config.turn_detection = Self::turn_detection(t)?;
        }
        Self::noise_reduction(
            session.get("input_audio_noise_reduction"),
            "input_audio_noise_reduction",
        )?;
        Self::include(session.get("include"), &mut config)?;
        self.config = config;
        Ok(())
    }

    fn turn_detection_value(&self) -> Value {
        match self.config.turn_detection {
            None => Value::Null,
            Some(t) => json!({
                "type": "server_vad",
                "threshold": t.threshold,
                "prefix_padding_ms": t.prefix_padding_ms,
                "silence_duration_ms": t.silence_duration_ms,
            }),
        }
    }

    fn include_value(&self) -> Value {
        if self.config.logprobs {
            json!(["item.input_audio_transcription.logprobs"])
        } else {
            json!([])
        }
    }

    fn session_value(&self) -> Value {
        let c = &self.config;
        if self.beta {
            let format = match c.format {
                AudioFormat::Pcm { .. } => "pcm16",
                AudioFormat::Mulaw => "g711_ulaw",
                AudioFormat::Alaw => "g711_alaw",
            };
            return json!({
                "id": self.id,
                "object": "realtime.transcription_session",
                "input_audio_format": format,
                "input_audio_transcription": {"model": c.model, "language": c.language, "prompt": ""},
                "turn_detection": self.turn_detection_value(),
                "input_audio_noise_reduction": null,
                "include": self.include_value(),
            });
        }
        let format = match c.format {
            AudioFormat::Pcm { rate } => json!({"type": "audio/pcm", "rate": rate}),
            AudioFormat::Mulaw => json!({"type": "audio/pcmu"}),
            AudioFormat::Alaw => json!({"type": "audio/pcma"}),
        };
        json!({
            "type": "transcription",
            "id": self.id,
            "object": "realtime.transcription_session",
            "audio": {"input": {
                "format": format,
                "transcription": {"model": c.model, "language": c.language, "prompt": ""},
                "turn_detection": self.turn_detection_value(),
                "noise_reduction": null,
            }},
            "include": self.include_value(),
        })
    }

    /// The first event of a connection.
    pub fn created(&mut self) -> String {
        let session = self.session_value();
        self.event("session.created", json!({"session": session}))
    }

    pub fn updated(&mut self) -> String {
        let kind = if self.beta {
            "transcription_session.updated"
        } else {
            "session.updated"
        };
        let session = self.session_value();
        self.event(kind, json!({"session": session}))
    }

    pub fn error(&mut self, error: &EventError) -> String {
        self.event(
            "error",
            json!({"error": {
                "type": "invalid_request_error",
                "code": error.code,
                "message": error.message,
                "param": error.param,
                "event_id": error.event_id,
            }}),
        )
    }

    /// A server-side failure (such as a full queue) as an `error` event.
    pub fn server_error(&mut self, error: &OpenAiError) -> String {
        self.event(
            "error",
            json!({"error": {
                "type": error.kind,
                "code": error.code,
                "message": error.message,
                "param": error.param,
                "event_id": null,
            }}),
        )
    }

    pub fn speech_started(&mut self, item_id: &str, audio_start_ms: u64) -> String {
        self.event(
            "input_audio_buffer.speech_started",
            json!({"audio_start_ms": audio_start_ms, "item_id": item_id}),
        )
    }

    pub fn speech_stopped(&mut self, item_id: &str, audio_end_ms: u64) -> String {
        self.event(
            "input_audio_buffer.speech_stopped",
            json!({"audio_end_ms": audio_end_ms, "item_id": item_id}),
        )
    }

    pub fn cleared(&mut self) -> String {
        self.event("input_audio_buffer.cleared", json!({}))
    }

    /// `input_audio_buffer.committed` followed by the new user item.
    pub fn committed(&mut self, item_id: &str, previous: Option<&str>) -> Vec<String> {
        let committed = self.event(
            "input_audio_buffer.committed",
            json!({"previous_item_id": previous, "item_id": item_id}),
        );
        let item = json!({
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{"type": "input_audio", "transcript": null}],
        });
        let kind = if self.beta {
            "conversation.item.created"
        } else {
            "conversation.item.added"
        };
        let added = self.event(kind, json!({"previous_item_id": previous, "item": item}));
        vec![committed, added]
    }

    /// `logprobs` pairs each token's text with its log probability.
    pub fn delta(&mut self, item_id: &str, delta: &str, logprobs: &[(String, f32)]) -> String {
        let mut fields = json!({"item_id": item_id, "content_index": 0, "delta": delta});
        if self.config.logprobs {
            fields["logprobs"] = logprob_values(logprobs);
        }
        self.event("conversation.item.input_audio_transcription.delta", fields)
    }

    /// The final transcript of an item, then (GA) `conversation.item.done`.
    pub fn completed(
        &mut self,
        item_id: &str,
        previous: Option<&str>,
        transcript: &str,
        seconds: f64,
        logprobs: &[(String, f32)],
    ) -> Vec<String> {
        let mut fields = json!({
            "item_id": item_id,
            "content_index": 0,
            "transcript": transcript,
            "usage": {"type": "duration", "seconds": seconds},
        });
        if self.config.logprobs {
            fields["logprobs"] = logprob_values(logprobs);
        }
        let mut events = vec![self.event(
            "conversation.item.input_audio_transcription.completed",
            fields,
        )];
        if !self.beta {
            let item = json!({
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "user",
                "content": [{"type": "input_audio", "transcript": transcript}],
            });
            events.push(self.event(
                "conversation.item.done",
                json!({"previous_item_id": previous, "item": item}),
            ));
        }
        events
    }

    pub fn failed(&mut self, item_id: &str, message: &str) -> String {
        self.event(
            "conversation.item.input_audio_transcription.failed",
            json!({
                "item_id": item_id,
                "content_index": 0,
                "error": {"type": "transcription_error", "code": "transcription_failed", "message": message, "param": null},
            }),
        )
    }
}

fn logprob_values(logprobs: &[(String, f32)]) -> Value {
    logprobs
        .iter()
        .map(|(token, logprob)| json!({"token": token, "logprob": logprob, "bytes": token.as_bytes()}))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        Session::new(
            "sess_1".into(),
            vec!["parakeet-tdt-0.6b-v3".into(), "parakeet-latest".into()],
            &["en", "es"],
        )
    }

    fn parse(event: &str) -> Value {
        serde_json::from_str(event).unwrap()
    }

    #[test]
    fn the_first_event_describes_the_default_ga_session() {
        let created = parse(&session().created());
        assert_eq!(created["type"], "session.created");
        assert_eq!(created["event_id"], "event_1");
        let input = &created["session"]["audio"]["input"];
        assert_eq!(created["session"]["type"], "transcription");
        assert_eq!(input["format"], json!({"type": "audio/pcm", "rate": 24000}));
        assert_eq!(input["transcription"]["model"], "parakeet-tdt-0.6b-v3");
        assert_eq!(input["turn_detection"]["type"], "server_vad");
    }

    #[test]
    fn ga_updates_patch_the_configuration() {
        let mut s = session();
        let update = json!({"type": "session.update", "session": {
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcmu"},
                "transcription": {"model": "parakeet-latest", "language": "ES"},
                "turn_detection": null,
            }},
            "include": ["item.input_audio_transcription.logprobs"],
        }});
        assert_eq!(s.handle(&update.to_string()), Ok(Command::Updated));
        let c = s.config();
        assert_eq!(c.format, AudioFormat::Mulaw);
        assert_eq!(c.language.as_deref(), Some("es"));
        assert_eq!(c.turn_detection, None);
        assert!(c.logprobs);
        let updated = parse(&s.updated());
        assert_eq!(updated["type"], "session.updated");
        assert_eq!(
            updated["session"]["audio"]["input"]["turn_detection"],
            Value::Null
        );
        // Absent fields keep their values.
        let update = json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 800}}}}});
        s.handle(&update.to_string()).unwrap();
        assert_eq!(s.config().format, AudioFormat::Mulaw);
        assert_eq!(s.config().turn_detection.unwrap().silence_duration_ms, 800);
    }

    #[test]
    fn beta_updates_switch_replies_to_the_beta_shape() {
        let mut s = session();
        let update = json!({"type": "transcription_session.update", "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {"model": "parakeet-tdt-0.6b-v3", "language": "", "prompt": ""},
            "turn_detection": {"type": "server_vad", "threshold": 0.7},
        }});
        s.handle(&update.to_string()).unwrap();
        let updated = parse(&s.updated());
        assert_eq!(updated["type"], "transcription_session.updated");
        assert_eq!(updated["session"]["input_audio_format"], "pcm16");
        assert_eq!(
            updated["session"]["turn_detection"]["threshold"],
            json!(0.7f32)
        );
        let events = s.committed("item_1", None);
        assert_eq!(parse(&events[1])["type"], "conversation.item.created");
        let done = s.completed("item_1", None, "hola", 1.0, &[]);
        assert_eq!(
            done.len(),
            1,
            "beta sessions have no conversation.item.done"
        );
    }

    #[test]
    fn bad_events_are_rejected_with_their_event_id() {
        let mut s = session();
        let cases = [
            (
                r#"{"type":"session.update","event_id":"e1","session":{"type":"realtime"}}"#,
                "unsupported_parameter",
            ),
            (
                r#"{"type":"session.update","event_id":"e1","session":{"audio":{"input":{"transcription":{"model":"gpt-4o-transcribe"}}}}}"#,
                "model_not_found",
            ),
            (
                r#"{"type":"session.update","event_id":"e1","session":{"audio":{"input":{"transcription":{"prompt":"names"}}}}}"#,
                "unsupported_parameter",
            ),
            (
                r#"{"type":"session.update","event_id":"e1","session":{"audio":{"input":{"turn_detection":{"type":"semantic_vad"}}}}}"#,
                "unsupported_parameter",
            ),
            (
                r#"{"type":"session.update","event_id":"e1","session":{"audio":{"input":{"noise_reduction":{"type":"near_field"}}}}}"#,
                "unsupported_parameter",
            ),
            (
                r#"{"type":"input_audio_buffer.append","event_id":"e1","audio":"***"}"#,
                "invalid_value",
            ),
            (
                r#"{"type":"response.create","event_id":"e1"}"#,
                "unsupported_event",
            ),
            (r#"{"type":"nonsense","event_id":"e1"}"#, "invalid_event"),
        ];
        for (event, code) in cases {
            let error = s.handle(event).unwrap_err();
            assert_eq!(
                (error.code, error.event_id.as_deref()),
                (code, Some("e1")),
                "{event}"
            );
        }
        let error = s.handle("not json").unwrap_err();
        assert_eq!(error.code, "invalid_json");
        let rendered = parse(&s.error(&error));
        assert_eq!(rendered["type"], "error");
        assert_eq!(rendered["error"]["code"], "invalid_json");
        // A rejected update leaves the configuration unchanged.
        assert_eq!(s.config().format, AudioFormat::Pcm { rate: 24000 });
    }

    #[test]
    fn audio_commands_decode_and_items_follow_the_ga_order() {
        let mut s = session();
        assert_eq!(
            s.handle(r#"{"type":"input_audio_buffer.append","audio":"AAEC"}"#),
            Ok(Command::Append(vec![0, 1, 2]))
        );
        assert_eq!(
            s.handle(r#"{"type":"input_audio_buffer.commit"}"#),
            Ok(Command::Commit)
        );
        assert_eq!(
            s.handle(r#"{"type":"input_audio_buffer.clear"}"#),
            Ok(Command::Clear)
        );
        let events: Vec<Value> = s
            .committed("item_2", Some("item_1"))
            .iter()
            .map(|e| parse(e))
            .collect();
        assert_eq!(events[0]["type"], "input_audio_buffer.committed");
        assert_eq!(events[0]["previous_item_id"], "item_1");
        assert_eq!(events[1]["type"], "conversation.item.added");
        assert_eq!(events[1]["item"]["role"], "user");
        let delta = parse(&s.delta("item_2", "Hola", &[]));
        assert_eq!(delta["delta"], "Hola");
        assert!(delta.get("logprobs").is_none());
        let done: Vec<Value> = s
            .completed("item_2", Some("item_1"), "Hola mundo.", 1.5, &[])
            .iter()
            .map(|e| parse(e))
            .collect();
        assert_eq!(done[0]["transcript"], "Hola mundo.");
        assert_eq!(
            done[0]["usage"],
            json!({"type": "duration", "seconds": 1.5})
        );
        assert_eq!(done[1]["type"], "conversation.item.done");
        assert_eq!(done[1]["item"]["content"][0]["transcript"], "Hola mundo.");
    }
}
