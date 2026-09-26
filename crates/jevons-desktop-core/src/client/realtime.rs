//! `GET /v1/realtime`: a transcription session over WebSocket. The client streams PCM16 at
//! [`SAMPLE_RATE`] and commits the turn itself when the hotkey is released (no server VAD).

use super::{Client, ClientError};
use crate::platform::SAMPLE_RATE;
use base64::Engine;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::{self, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Sends audio and commits the turn.
pub struct RealtimeWriter {
    sink: SplitSink<Socket, Message>,
}

/// Receives transcription events.
pub struct RealtimeReader {
    stream: SplitStream<Socket>,
}

/// A server event the pipeline cares about.
#[derive(Clone, Debug, PartialEq)]
pub enum RealtimeEvent {
    /// Live text for the turn so far.
    Delta { item_id: String, delta: String },
    /// The turn's final transcript.
    Completed { item_id: String, transcript: String },
    /// Transcription or protocol failure.
    Error { message: String },
    /// Any other event, by type.
    Other(String),
}

/// Who ends a turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turns {
    /// The client commits the turn (push-to-talk).
    Client,
    /// The server ends a turn after this much silence (live dictation).
    ServerVad { silence_ms: u32 },
}

impl Client {
    /// Opens a transcription session with PCM16 at [`SAMPLE_RATE`].
    /// Returns [`ClientError::NotServed`] when the runtime has no Realtime endpoint.
    pub async fn realtime(
        &self,
        model: Option<&str>,
        language: Option<&str>,
        turns: Turns,
    ) -> Result<(RealtimeWriter, RealtimeReader), ClientError> {
        let base = self
            .base
            .replacen("http://", "ws://", 1)
            .replacen("https://", "wss://", 1);
        let mut request = format!("{base}/v1/realtime?intent=transcription")
            .into_client_request()
            .map_err(|e| ClientError::Realtime(e.to_string()))?;
        let headers = request.headers_mut();
        headers.insert(
            "Sec-WebSocket-Protocol",
            HeaderValue::from_static("realtime"),
        );
        if let Some(key) = &self.key {
            let value = HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|_| ClientError::Realtime("the API key is not a valid header".into()))?;
            headers.insert("Authorization", value);
        }
        let (socket, _) = match tokio_tungstenite::connect_async(request).await {
            Ok(connected) => connected,
            Err(tungstenite::Error::Http(response)) if response.status() == 404 => {
                return Err(ClientError::NotServed("Realtime transcription"));
            }
            Err(tungstenite::Error::Http(response)) => {
                let message = response
                    .body()
                    .as_deref()
                    .map(|b| super::error_message(&String::from_utf8_lossy(b)))
                    .unwrap_or_default();
                return Err(ClientError::Api {
                    status: response.status().as_u16(),
                    message,
                });
            }
            Err(e) => return Err(ClientError::Realtime(e.to_string())),
        };
        let (sink, stream) = socket.split();
        let mut writer = RealtimeWriter { sink };
        let turn_detection = match turns {
            Turns::Client => serde_json::Value::Null,
            Turns::ServerVad { silence_ms } => json!({
                "type": "server_vad",
                "silence_duration_ms": silence_ms,
            }),
        };
        let session = json!({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transcription": {"model": model, "language": language},
                    "turn_detection": turn_detection,
                }},
            },
        });
        writer.send(session).await?;
        Ok((writer, RealtimeReader { stream }))
    }
}

impl RealtimeWriter {
    async fn send(&mut self, event: serde_json::Value) -> Result<(), ClientError> {
        self.sink
            .send(Message::text(event.to_string()))
            .await
            .map_err(|e| ClientError::Realtime(e.to_string()))
    }

    pub async fn append(&mut self, samples: &[i16]) -> Result<(), ClientError> {
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let audio = base64::engine::general_purpose::STANDARD.encode(bytes);
        self.send(json!({"type": "input_audio_buffer.append", "audio": audio}))
            .await
    }

    pub async fn commit(&mut self) -> Result<(), ClientError> {
        self.send(json!({"type": "input_audio_buffer.commit"}))
            .await
    }

    pub async fn close(mut self) {
        let _ = self.sink.close().await;
    }
}

impl RealtimeReader {
    /// The next event; `None` once the server closed the session.
    pub async fn next(&mut self) -> Option<RealtimeEvent> {
        loop {
            let message = match self.stream.next().await? {
                Ok(message) => message,
                Err(e) => {
                    return Some(RealtimeEvent::Error {
                        message: e.to_string(),
                    });
                }
            };
            match message {
                Message::Text(text) => return Some(parse(&text)),
                Message::Close(_) => return None,
                _ => continue,
            }
        }
    }
}

fn parse(text: &str) -> RealtimeEvent {
    let event: serde_json::Value = serde_json::from_str(text).unwrap_or_default();
    let string = |key: &str| event[key].as_str().unwrap_or_default().to_string();
    match event["type"].as_str().unwrap_or_default() {
        "conversation.item.input_audio_transcription.delta" => RealtimeEvent::Delta {
            item_id: string("item_id"),
            delta: string("delta"),
        },
        "conversation.item.input_audio_transcription.completed" => RealtimeEvent::Completed {
            item_id: string("item_id"),
            transcript: string("transcript"),
        },
        "conversation.item.input_audio_transcription.failed" => RealtimeEvent::Error {
            message: event["error"]["message"]
                .as_str()
                .unwrap_or("transcription failed")
                .into(),
        },
        "error" => RealtimeEvent::Error {
            message: event["error"]["message"]
                .as_str()
                .unwrap_or("Realtime error")
                .into(),
        },
        other => RealtimeEvent::Other(other.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcription_events_parse() {
        assert_eq!(
            parse(
                r#"{"type":"conversation.item.input_audio_transcription.completed","item_id":"i","content_index":0,"transcript":"hola"}"#
            ),
            RealtimeEvent::Completed {
                item_id: "i".into(),
                transcript: "hola".into()
            }
        );
        assert_eq!(
            parse(r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"#),
            RealtimeEvent::Error {
                message: "bad".into()
            }
        );
        assert_eq!(
            parse(r#"{"type":"session.updated"}"#),
            RealtimeEvent::Other("session.updated".into())
        );
    }
}
