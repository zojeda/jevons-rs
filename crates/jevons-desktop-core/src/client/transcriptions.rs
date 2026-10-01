//! `POST /v1/audio/transcriptions`: the fallback when Realtime is not served.

use super::{Client, ClientError, checked};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Transcription {
    pub text: String,
}

impl Client {
    /// Transcribes mono PCM16 at `rate`, uploaded as WAV.
    pub async fn transcribe(
        &self,
        samples: &[i16],
        rate: u32,
        model: &str,
        language: Option<&str>,
    ) -> Result<Transcription, ClientError> {
        let file = reqwest::multipart::Part::bytes(wav(samples, rate))
            .file_name("take.wav")
            .mime_str("audio/wav")?;
        let mut form = reqwest::multipart::Form::new()
            .part("file", file)
            .text("model", model.to_string())
            .text("response_format", "json");
        if let Some(language) = language {
            form = form.text("language", language.to_string());
        }
        let response = self
            .request(reqwest::Method::POST, "/v1/audio/transcriptions")
            .multipart(form)
            .send()
            .await?;
        Ok(checked(response).await?.json().await?)
    }
}

/// A mono PCM16 WAV file.
pub(crate) fn wav(samples: &[i16], rate: u32) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_uploads_decode_back_to_the_same_audio() {
        let samples: Vec<i16> = (0..2400)
            .map(|i| ((i % 100) * 300 - 15000) as i16)
            .collect();
        let limits = jevons_audio::AudioLimits {
            max_bytes: 1 << 20,
            max_seconds: 10.0,
        };
        let decoded = jevons_audio::decode_audio(wav(&samples, 24000), Some("wav"), 24000, limits);
        let decoded = decoded.expect("the WAV decodes");
        assert_eq!(decoded.len(), samples.len());
        assert!((decoded[10] - f32::from(samples[10]) / 32768.0).abs() < 1e-3);
    }
}
