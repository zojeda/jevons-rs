//! `POST /v1/audio/speech`: validation of the JSON body, and the audio body in each
//! `response_format`.
use crate::openai::OpenAiError;
use crate::openai::request::Fields;
use jevons_audio::{Resampler, pcm16_le_bytes, wav_header};
use jevons_core::VoiceInfo;
use serde_json::{Value, json};

/// The sample rate of a `pcm` body, as OpenAI defines it.
pub const PCM_SAMPLE_RATE: u32 = 24_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeechFormat {
    /// Mono PCM16 at the model's rate, after a WAV header of unknown length.
    Wav,
    /// Mono PCM16 little-endian at [`PCM_SAMPLE_RATE`], with no header.
    Pcm,
}

impl SpeechFormat {
    pub fn content_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/pcm",
        }
    }
}

/// A validated speech request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechRequest {
    pub model: String,
    /// The text to speak.
    pub input: String,
    /// `None` speaks in the service's default voice.
    pub voice: Option<String>,
    /// ISO-639-1, lowercase, for the whole text. `None` leaves each sentence's language to
    /// its words.
    pub language: Option<String>,
    pub format: SpeechFormat,
}

fn invalid(message: impl Into<String>, param: &str) -> OpenAiError {
    OpenAiError::invalid(message, Some(param))
}

impl SpeechRequest {
    /// Validates the body's fields. What depends on the model that serves the request (its
    /// voices, its languages, the input limit) is left to [`SpeechRequest::check`].
    pub fn parse(body: &Value) -> Result<Self, OpenAiError> {
        let mut f = Fields::new(body)?;
        let model = f
            .string("model")?
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .ok_or_else(|| invalid("model is required", "model"))?
            .to_string();
        let input = f
            .string("input")?
            .ok_or_else(|| invalid("input is required", "input"))?;
        if input.trim().is_empty() {
            return Err(invalid("input must not be empty", "input"));
        }
        let voice = f.string("voice")?.map(String::from);
        let format = match f.string("response_format")? {
            None | Some("wav") => SpeechFormat::Wav,
            Some("pcm") => SpeechFormat::Pcm,
            Some(_) => {
                return Err(OpenAiError::unsupported(
                    "response_format",
                    "only wav and pcm are available",
                ));
            }
        };
        let language = f
            .string("language")?
            .map(|code| code.trim().to_ascii_lowercase())
            .filter(|code| !code.is_empty());
        // JSON `1` and `1.0` are the same speed.
        if f.get("speed")
            .is_some_and(|speed| speed.as_f64() != Some(1.0))
        {
            return Err(OpenAiError::unsupported("speed", "only 1.0 is available"));
        }
        f.only(
            "stream_format",
            &[json!("audio")],
            "only audio is available",
        )?;
        f.only("instructions", &[], "this model takes no instructions")?;
        f.finish()?;
        Ok(Self {
            model,
            input: input.to_string(),
            voice,
            language,
            format,
        })
    }

    /// Checks the request against the model that serves it: the input within
    /// `max_input_chars`, and a voice and a language the model has.
    pub fn check(&self, info: &VoiceInfo, max_input_chars: usize) -> Result<(), OpenAiError> {
        if self.input.chars().count() > max_input_chars {
            return Err(invalid(
                format!("input must be at most {max_input_chars} characters"),
                "input",
            ));
        }
        if let Some(voice) = &self.voice
            && !info.voices.iter().any(|known| known == voice)
        {
            return Err(invalid(
                format!(
                    "voice must be one of the model's: {}",
                    info.voices.join(", ")
                ),
                "voice",
            ));
        }
        if let Some(language) = &self.language
            && !info.languages.contains(&language.as_str())
        {
            return Err(invalid(
                format!(
                    "language must be an ISO-639-1 code the model speaks: {}",
                    info.languages.join(", ")
                ),
                "language",
            ));
        }
        Ok(())
    }

    /// The renderer of this request's body, for a model that speaks at `sample_rate`.
    pub fn audio(&self, sample_rate: u32) -> SpeechAudio {
        SpeechAudio::new(self.format, sample_rate)
    }
}

/// Renders a speech as its response body, piece by piece: [`SpeechAudio::start`], a
/// [`SpeechAudio::chunk`] for each stretch of samples, then [`SpeechAudio::finish`]. A piece
/// may be empty.
pub struct SpeechAudio {
    format: SpeechFormat,
    /// The rate of the body's samples.
    sample_rate: u32,
    /// `pcm` from a model that speaks at another rate than [`PCM_SAMPLE_RATE`].
    resampler: Option<Resampler>,
}

impl SpeechAudio {
    /// `sample_rate` is the model's.
    pub fn new(format: SpeechFormat, sample_rate: u32) -> Self {
        let resample = format == SpeechFormat::Pcm && sample_rate != PCM_SAMPLE_RATE;
        Self {
            format,
            sample_rate: match format {
                SpeechFormat::Wav => sample_rate,
                SpeechFormat::Pcm => PCM_SAMPLE_RATE,
            },
            resampler: resample.then(|| Resampler::new(sample_rate, PCM_SAMPLE_RATE)),
        }
    }

    pub fn content_type(&self) -> &'static str {
        self.format.content_type()
    }

    /// The rate of the body's samples: the model's for `wav`, 24 kHz for `pcm`.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// What opens the body: the header of a WAV stream of unknown length, or nothing.
    pub fn start(&self) -> Vec<u8> {
        match self.format {
            SpeechFormat::Wav => wav_header(self.sample_rate, None).to_vec(),
            SpeechFormat::Pcm => Vec::new(),
        }
    }

    /// The bytes of `samples`, mono at the model's rate.
    pub fn chunk(&mut self, samples: &[f32]) -> Vec<u8> {
        match &mut self.resampler {
            Some(resampler) => {
                let mut resampled = Vec::new();
                resampler.process(samples, &mut resampled);
                pcm16_le_bytes(&resampled)
            }
            None => pcm16_le_bytes(samples),
        }
    }

    /// What closes the body: the samples the resampler still holds.
    pub fn finish(&mut self) -> Vec<u8> {
        let mut tail = Vec::new();
        if let Some(resampler) = &mut self.resampler {
            resampler.finish(&mut tail);
        }
        pcm16_le_bytes(&tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jevons_audio::{pcm16_le, resample};

    fn info() -> VoiceInfo {
        VoiceInfo {
            architecture: "fake",
            display_name: "Fake voice".into(),
            sample_rate: 24000,
            languages: &["es", "en"],
            voices: vec!["lector".into(), "reader".into()],
        }
    }

    fn parse(body: Value) -> Result<SpeechRequest, OpenAiError> {
        SpeechRequest::parse(&body)
    }

    #[test]
    fn a_speech_request_defaults_to_wav_in_the_default_voice() {
        let request = parse(json!({"model": "tts", "input": "Hola."})).unwrap();
        assert_eq!(
            request,
            SpeechRequest {
                model: "tts".into(),
                input: "Hola.".into(),
                voice: None,
                language: None,
                format: SpeechFormat::Wav,
            }
        );
        request.check(&info(), 4096).unwrap();
        // Every field at a value this server takes; `null` counts as absent.
        let request = parse(json!({
            "model": " tts ", "input": "Hello.", "voice": "reader", "response_format": "pcm",
            "language": " EN ", "speed": 1, "stream_format": "audio", "instructions": null,
        }))
        .unwrap();
        assert_eq!(
            (request.model.as_str(), request.voice.as_deref()),
            ("tts", Some("reader"))
        );
        assert_eq!(request.language.as_deref(), Some("en"));
        assert_eq!(request.format, SpeechFormat::Pcm);
        request.check(&info(), 4096).unwrap();
        assert!(parse(json!({"model": "tts", "input": "Hola.", "speed": 1.0})).is_ok());
        // An empty language is none.
        let request = parse(json!({"model": "tts", "input": "Hola.", "language": ""})).unwrap();
        assert_eq!(request.language, None);
    }

    #[test]
    fn unsupported_speech_parameters_are_rejected_not_ignored() {
        for (field, value) in [
            ("response_format", json!("mp3")),
            ("response_format", json!("opus")),
            ("speed", json!(1.25)),
            ("speed", json!("fast")),
            ("stream_format", json!("sse")),
            ("instructions", json!("Speak cheerfully.")),
            ("instructions", json!("")),
            ("seed", json!(7)),
        ] {
            let mut body = json!({"model": "tts", "input": "Hola."});
            body[field] = value.clone();
            let error = parse(body).unwrap_err();
            assert_eq!(error.status, 400, "{field}");
            assert_eq!(
                error.code,
                Some("unsupported_parameter"),
                "{field}: {value}"
            );
            assert_eq!(error.param.as_deref(), Some(field));
        }
    }

    #[test]
    fn invalid_speech_fields_name_their_parameter() {
        for (body, param) in [
            (json!({"input": "Hola."}), "model"),
            (json!({"model": " ", "input": "Hola."}), "model"),
            (json!({"model": "tts"}), "input"),
            (json!({"model": "tts", "input": ""}), "input"),
            (json!({"model": "tts", "input": " \n "}), "input"),
            (json!({"model": "tts", "input": ["Hola."]}), "input"),
            (
                json!({"model": "tts", "input": "Hola.", "voice": {"id": "v1"}}),
                "voice",
            ),
            (
                json!({"model": "tts", "input": "Hola.", "response_format": 1}),
                "response_format",
            ),
            (
                json!({"model": "tts", "input": "Hola.", "language": 5}),
                "language",
            ),
        ] {
            let error = parse(body.clone()).unwrap_err();
            assert_eq!(error.status, 400, "{body}");
            assert_eq!(error.code, None, "{body}");
            assert_eq!(error.param.as_deref(), Some(param), "{body}");
        }
        let error = parse(json!(["not", "an", "object"])).unwrap_err();
        assert_eq!((error.status, error.param), (400, None));
    }

    #[test]
    fn a_request_is_checked_against_the_model_that_serves_it() {
        let request = |body: Value| parse(body).unwrap();
        let error = request(json!({"model": "tts", "input": "Hola.", "voice": "alloy"}))
            .check(&info(), 4096)
            .unwrap_err();
        assert_eq!((error.status, error.param.as_deref()), (400, Some("voice")));
        assert!(
            error.message.contains("lector, reader"),
            "{}",
            error.message
        );
        let error = request(json!({"model": "tts", "input": "Bonjour.", "language": "fr"}))
            .check(&info(), 4096)
            .unwrap_err();
        assert_eq!(error.param.as_deref(), Some("language"));
        assert!(error.message.contains("es, en"), "{}", error.message);
        // The limit counts characters, not bytes.
        let text = "ñ".repeat(10);
        let request = request(json!({"model": "tts", "input": text}));
        request.check(&info(), 10).unwrap();
        let error = request.check(&info(), 9).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("input"));
        assert!(error.message.contains("9 characters"), "{}", error.message);
    }

    fn tone(rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * 220.0 * i as f64 / f64::from(rate);
                0.1 * phase.sin() as f32
            })
            .collect()
    }

    #[test]
    fn a_wav_body_is_a_streamed_header_then_pcm16_at_the_models_rate() {
        let samples = tone(16000, 3200);
        let mut audio = SpeechAudio::new(SpeechFormat::Wav, 16000);
        assert_eq!(audio.content_type(), "audio/wav");
        assert_eq!(audio.sample_rate(), 16000);
        assert_eq!(audio.start(), wav_header(16000, None));
        let mut body = audio.chunk(&samples[..1280]);
        body.extend(audio.chunk(&samples[1280..]));
        assert_eq!(body, pcm16_le_bytes(&samples));
        assert!(audio.finish().is_empty());
    }

    #[test]
    fn a_pcm_body_is_headerless_pcm16_at_24_khz() {
        let samples = tone(24000, 4800);
        let mut audio = SpeechAudio::new(SpeechFormat::Pcm, 24000);
        assert_eq!(audio.content_type(), "audio/pcm");
        assert_eq!(audio.sample_rate(), PCM_SAMPLE_RATE);
        assert!(audio.start().is_empty());
        assert_eq!(audio.chunk(&samples), pcm16_le_bytes(&samples));
        assert!(audio.finish().is_empty());
    }

    #[test]
    fn a_pcm_body_is_resampled_from_another_model_rate() {
        // One second at 16 kHz, in the fake's chunks of 80 ms.
        let samples = tone(16000, 16000);
        let mut audio = SpeechAudio::new(SpeechFormat::Pcm, 16000);
        assert!(audio.start().is_empty());
        let mut body = Vec::new();
        for chunk in samples.chunks(1280) {
            body.extend(audio.chunk(chunk));
        }
        body.extend(audio.finish());
        // One second at 24 kHz, the same as resampling the whole second at once.
        assert_eq!(body.len(), 24000 * 2);
        assert_eq!(body, pcm16_le_bytes(&resample(&samples, 16000, 24000)));
        let peak = pcm16_le(&body)
            .iter()
            .fold(0f32, |peak, s| peak.max(s.abs()));
        assert!((0.09..0.11).contains(&peak), "{peak}");
    }
}
