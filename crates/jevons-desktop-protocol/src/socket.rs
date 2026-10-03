//! The stream between processes: a WebSocket carrying each message as JSON text. Either end
//! gets the same [`Link`] the in-process stream gives, so nothing above it knows which it is.
//!
//! A message one end does not know is skipped, and the stream ends when the socket closes or
//! when the link's sender is dropped.

use crate::wire::Link;
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::mpsc;

/// Connects to a server's desktop endpoint, such as `ws://127.0.0.1:8080/desktop`.
#[cfg(feature = "client")]
pub async fn connect(url: &str) -> Result<crate::wire::ClientLink, String> {
    use tokio_tungstenite::tungstenite::Message;
    let (socket, _) = tokio_tungstenite::connect_async(url)
        .await
        .map_err(|e| format!("cannot reach {url}: {e}"))?;
    let (mut sink, mut stream) = socket.split();
    let (tx, mut out) = mpsc::unbounded_channel();
    let (into, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(message) = out.recv().await {
            if sink.send(Message::text(text(&message))).await.is_err() {
                break;
            }
        }
        let _ = sink.close().await;
    });
    tokio::spawn(async move {
        while let Some(Ok(message)) = stream.next().await {
            let text = match &message {
                Message::Text(text) => text.as_str(),
                Message::Close(_) => break,
                _ => continue,
            };
            if !pass(text, &into) {
                break;
            }
        }
    });
    Ok(Link { tx, rx })
}

/// The server's end of a client's socket.
#[cfg(feature = "server")]
pub fn accept(socket: axum::extract::ws::WebSocket) -> crate::wire::ServerLink {
    use axum::extract::ws::Message;
    let (mut sink, mut stream) = socket.split();
    let (tx, mut out) = mpsc::unbounded_channel();
    let (into, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(message) = out.recv().await {
            if sink
                .send(Message::Text(text(&message).into()))
                .await
                .is_err()
            {
                break;
            }
        }
        let _ = sink.close().await;
    });
    tokio::spawn(async move {
        while let Some(Ok(message)) = stream.next().await {
            let text = match &message {
                Message::Text(text) => text.as_str(),
                Message::Close(_) => break,
                _ => continue,
            };
            if !pass(text, &into) {
                break;
            }
        }
    });
    Link { tx, rx }
}

fn text(message: &impl Serialize) -> String {
    serde_json::to_string(message).expect("a message serializes")
}

/// Hands a received message on; `false` once nobody listens.
fn pass<T: DeserializeOwned>(text: &str, into: &mpsc::UnboundedSender<T>) -> bool {
    match serde_json::from_str(text) {
        Ok(message) => into.send(message).is_ok(),
        // A message this end does not know.
        Err(_) => true,
    }
}
