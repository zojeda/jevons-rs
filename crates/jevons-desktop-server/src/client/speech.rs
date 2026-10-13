//! `POST /v1/audio/speech`: text to speech. The answer is mono PCM16 at 24 kHz, and its bytes
//! come back as the provider makes them, so the first words play while the rest is spoken.
//!
//! These calls never reach the API log: neither the text nor the audio is written anywhere.

use super::{Client, ClientError, checked};
use serde::Serialize;

/// What to say, and how.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpeechRequest {
    pub model: String,
    /// The text to speak.
    pub input: String,
    /// The voice, as the provider names it; its default one when left out.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// The text's language as an ISO-639-1 code, which the jevons API takes beyond OpenAI's
    /// request. Left out, it tells each sentence's language by its words.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

/// The request as it goes: always for raw PCM, the one format that needs no decoder.
#[derive(Serialize)]
struct AsPcm<'a> {
    #[serde(flatten)]
    request: &'a SpeechRequest,
    response_format: &'static str,
}

/// Speech on its way: the answer's body, as it arrives. Dropping it hangs up, which is how
/// the provider learns to stop speaking.
pub struct SpeechStream(reqwest::Response);

impl SpeechStream {
    /// The next bytes of the audio, in pieces of any size; `None` once it is whole.
    pub async fn next(&mut self) -> Result<Option<Vec<u8>>, ClientError> {
        Ok(self.0.chunk().await?.map(|bytes| bytes.to_vec()))
    }
}

impl Client {
    /// Asks for `request` spoken, as mono PCM16 little-endian at 24 kHz. It answers once the
    /// provider starts to: the audio follows on the stream.
    pub async fn speak(&self, request: &SpeechRequest) -> Result<SpeechStream, ClientError> {
        let body = AsPcm {
            request,
            response_format: "pcm",
        };
        let response = self
            .request(reqwest::Method::POST, "/v1/audio/speech")
            .json(&body)
            .send()
            .await?;
        let response = checked(response).await?;
        // A provider that answers in another format would be played as noise.
        let kind = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if kind.starts_with("audio/") && !kind.starts_with("audio/pcm") {
            return Err(ClientError::Protocol(format!(
                "the speech came as {kind}, where raw PCM was asked for"
            )));
        }
        Ok(SpeechStream(response))
    }
}

/// Reads PCM16 little-endian samples out of bytes that arrive in pieces of any size: a sample
/// a piece's end cuts in two is joined with its other half.
#[derive(Debug, Default)]
pub struct Pcm16 {
    half: Option<u8>,
}

impl Pcm16 {
    /// Adds the samples in `bytes` to `samples`.
    pub fn push(&mut self, bytes: &[u8], samples: &mut Vec<i16>) {
        let mut bytes = bytes;
        if let (Some(low), Some((high, rest))) = (self.half, bytes.split_first()) {
            samples.push(i16::from_le_bytes([low, *high]));
            self.half = None;
            bytes = rest;
        }
        let (pairs, rest) = bytes.as_chunks::<2>();
        samples.extend(pairs.iter().map(|pair| i16::from_le_bytes(*pair)));
        if let [low] = rest {
            self.half = Some(*low);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Json;
    use axum::http::{StatusCode, header};
    use axum::response::IntoResponse;
    use axum::routing::post;
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};

    /// A provider that speaks every request as the samples 1, 2, 3 in two pieces, the second
    /// sample cut in half; `"model": "mp3"` gets another format and `"model": "none"` an error.
    async fn provider() -> (Client, Arc<Mutex<Vec<(String, Value)>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let kept = seen.clone();
        let speak = move |headers: axum::http::HeaderMap, Json(body): Json<Value>| {
            let kept = kept.clone();
            async move {
                let key = headers
                    .get(header::AUTHORIZATION)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_string();
                kept.lock().unwrap().push((key, body.clone()));
                match body["model"].as_str() {
                    Some("none") => (
                        StatusCode::NOT_FOUND,
                        Json(json!({"error": {"message": "The model `none` does not exist"}})),
                    )
                        .into_response(),
                    Some("mp3") => {
                        ([(header::CONTENT_TYPE, "audio/mpeg")], vec![0u8; 8]).into_response()
                    }
                    _ => {
                        let pieces: [&[u8]; 2] = [&[1, 0, 2], &[0, 3, 0]];
                        let stream = futures_util::stream::iter(
                            pieces.map(|piece| Ok::<_, std::io::Error>(piece.to_vec())),
                        );
                        (
                            [(header::CONTENT_TYPE, "audio/pcm")],
                            axum::body::Body::from_stream(stream),
                        )
                            .into_response()
                    }
                }
            }
        };
        let app = axum::Router::new().route("/v1/audio/speech", post(speak));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (Client::new(&base, Some("key".into())), seen)
    }

    fn request(model: &str) -> SpeechRequest {
        SpeechRequest {
            model: model.into(),
            input: "Son las cinco.".into(),
            voice: None,
            language: None,
        }
    }

    #[tokio::test]
    async fn speech_is_asked_for_as_pcm_and_its_bytes_come_as_they_arrive() {
        let (client, seen) = provider().await;
        let mut stream = client.speak(&request("qwen3-tts")).await.unwrap();
        let (mut pcm, mut samples) = (Pcm16::default(), Vec::new());
        let mut bytes = 0;
        while let Some(piece) = stream.next().await.unwrap() {
            bytes += piece.len();
            pcm.push(&piece, &mut samples);
        }
        assert_eq!((bytes, samples), (6, vec![1, 2, 3]));
        // What went: the model and the text, the format, the key, and nothing left out as null.
        let (key, body) = seen.lock().unwrap()[0].clone();
        assert_eq!(key, "Bearer key");
        assert_eq!(
            body,
            json!({"model": "qwen3-tts", "input": "Son las cinco.", "response_format": "pcm"})
        );
        // A voice and a language go when they are set.
        let named = SpeechRequest {
            voice: Some("lucia".into()),
            language: Some("es".into()),
            ..request("qwen3-tts")
        };
        client.speak(&named).await.unwrap();
        let (_, body) = seen.lock().unwrap()[1].clone();
        assert_eq!(
            (&body["voice"], &body["language"]),
            (&json!("lucia"), &json!("es"))
        );
        assert_eq!(body["response_format"], "pcm");
    }

    #[tokio::test]
    async fn a_speech_request_that_fails_keeps_the_provider_s_message() {
        let (client, _) = provider().await;
        let refused = client.speak(&request("none")).await;
        assert!(
            matches!(&refused, Err(ClientError::Api { status: 404, message })
                if message == "The model `none` does not exist"),
            "{:?}",
            refused.err()
        );
        // Another format is not played as if it were PCM.
        let other = client.speak(&request("mp3")).await;
        assert!(
            matches!(&other, Err(ClientError::Protocol(why)) if why.contains("audio/mpeg")),
            "{:?}",
            other.err()
        );
        // Nobody there.
        let nobody = Client::new("http://127.0.0.1:9", None);
        assert!(matches!(
            nobody.speak(&request("qwen3-tts")).await,
            Err(ClientError::Http(_))
        ));
    }

    #[test]
    fn samples_cut_by_a_piece_s_end_are_joined() {
        let whole: Vec<i16> = vec![0, 1, -1, 300, -300, i16::MAX, i16::MIN];
        let bytes: Vec<u8> = whole.iter().flat_map(|s| s.to_le_bytes()).collect();
        // Whatever the size of the pieces, the samples are the same.
        for size in [1, 2, 3, 5, 14] {
            let (mut pcm, mut samples) = (Pcm16::default(), Vec::new());
            for piece in bytes.chunks(size) {
                pcm.push(piece, &mut samples);
            }
            assert_eq!(samples, whole, "pieces of {size}");
        }
        // An odd byte at the very end is half a sample: it is never played.
        let (mut pcm, mut samples) = (Pcm16::default(), Vec::new());
        pcm.push(&[7, 0, 9], &mut samples);
        pcm.push(&[], &mut samples);
        assert_eq!(samples, [7]);
    }
}
