//! One Realtime transcription session per WebSocket.
//!
//! The session task holds the input audio buffer (resampled to the model rate), server-side
//! turn detection and item bookkeeping; the speech thread only runs model passes. While a turn
//! is live, a pass over the utterance so far runs every [`PARTIAL_INTERVAL_SECONDS`] of audio,
//! and the words two consecutive passes agree on are sent as transcription deltas (local
//! agreement), so a delta is rarely revised. A committed turn is transcribed in full, windowed
//! like an upload, and its `completed` transcript is authoritative.
use crate::SpeechService;
use crate::speech::{Pass, Update};
use axum::extract::ws::{Message, WebSocket};
use jevons_audio::{Resampler, Vad, VadConfig, VadEvent, alaw, mulaw, pcm16_le};
use jevons_engine::{Error, Word};
use jevons_openai::{AudioFormat, Command, EventError, Session, TurnDetection};
use std::collections::VecDeque;
use tokio::sync::{mpsc, oneshot};

/// Audio between live passes over a turn.
const PARTIAL_INTERVAL_SECONDS: f64 = 0.7;
/// Turns longer than this get no more live deltas until they are committed.
const PARTIAL_LIMIT_SECONDS: f64 = 30.0;
/// The shortest buffer a client may commit.
const MIN_COMMIT_SECONDS: f64 = 0.1;
/// Audio kept before a detected turn, beyond its prefix padding.
const IDLE_KEEP_SECONDS: f64 = 1.0;

/// A live turn: speech detected, not yet committed.
struct Turn {
    item_id: String,
    /// Absolute sample where the turn starts.
    start: usize,
    /// Absolute sample of the last live pass.
    passed_at: usize,
    /// The words of the last live pass.
    hypothesis: Vec<String>,
    /// Words already sent as deltas, and their text.
    agreed: usize,
    sent: String,
}

/// A committed turn waiting for, or in, its full transcription.
struct Committed {
    item_id: String,
    previous: Option<String>,
    samples: Vec<f32>,
    /// Delta text already sent for the item.
    sent: String,
    /// Whether segments are streamed as deltas (no live deltas were sent).
    stream_segments: bool,
}

struct Live {
    session: Session,
    speech: SpeechService,
    rate: u32,
    format: AudioFormat,
    turn_detection: Option<TurnDetection>,
    resampler: Resampler,
    vad: Option<Vad>,
    /// Absolute sample where the detector started counting.
    vad_base: usize,
    /// The input audio buffer at the model rate; `buffer[0]` is absolute sample `buffer_start`.
    buffer: Vec<f32>,
    buffer_start: usize,
    turn: Option<Turn>,
    previous_item: Option<String>,
    items: u64,
    partial: Option<(String, oneshot::Receiver<Result<Pass, Error>>)>,
    committed: VecDeque<Committed>,
    active: Option<(Committed, mpsc::UnboundedReceiver<Update>)>,
}

enum Wake {
    Client(Option<Result<Message, axum::Error>>),
    Partial(Result<Result<Pass, Error>, oneshot::error::RecvError>),
    Final(Option<Update>),
}

fn vad_config(t: TurnDetection) -> VadConfig {
    VadConfig {
        threshold: t.threshold,
        prefix_padding_ms: t.prefix_padding_ms,
        silence_duration_ms: t.silence_duration_ms,
    }
}

fn logprobs(words: &[Word]) -> Vec<(String, f32)> {
    words
        .iter()
        .flat_map(|w| &w.tokens)
        .map(|t| (t.piece.replace('▁', " "), t.logprob))
        .collect()
}

impl Live {
    fn new(speech: SpeechService) -> Self {
        let id = format!("sess_{}", uuid::Uuid::new_v4().simple());
        let session = Session::new(id, speech.names(), speech.info.languages);
        let rate = speech.info.sample_rate;
        let format = session.config().format;
        let turn_detection = session.config().turn_detection;
        Self {
            resampler: Resampler::new(format.rate(), rate),
            vad: turn_detection.map(|t| Vad::new(vad_config(t), rate)),
            vad_base: 0,
            session,
            speech,
            rate,
            format,
            turn_detection,
            buffer: Vec::new(),
            buffer_start: 0,
            turn: None,
            previous_item: None,
            items: 0,
            partial: None,
            committed: VecDeque::new(),
            active: None,
        }
    }

    fn seconds(&self, samples: usize) -> f64 {
        samples as f64 / f64::from(self.rate)
    }

    fn samples(&self, seconds: f64) -> usize {
        (seconds * f64::from(self.rate)) as usize
    }

    fn millis(&self, sample: usize) -> u64 {
        (sample as u64 * 1000) / u64::from(self.rate)
    }

    fn new_item(&mut self) -> String {
        self.items += 1;
        format!(
            "item_{}_{}",
            self.items,
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        )
    }

    fn handle(&mut self, text: &str) -> Vec<String> {
        match self.session.handle(text) {
            Err(error) => vec![self.session.error(&error)],
            Ok(Command::Updated) => {
                self.reconfigure();
                vec![self.session.updated()]
            }
            Ok(Command::Append(bytes)) => self.append(&bytes),
            Ok(Command::Commit) => self.commit_manually(),
            Ok(Command::Clear) => {
                self.clear();
                vec![self.session.cleared()]
            }
        }
    }

    fn reconfigure(&mut self) {
        let config = self.session.config();
        if config.format != self.format {
            self.format = config.format;
            self.resampler = Resampler::new(self.format.rate(), self.rate);
        }
        if config.turn_detection != self.turn_detection {
            self.turn_detection = config.turn_detection;
            // Detection restarts with the audio that follows.
            self.vad = self
                .turn_detection
                .map(|t| Vad::new(vad_config(t), self.rate));
            self.vad_base = self.buffer_start + self.buffer.len();
            self.turn = None;
            self.partial = None;
        }
    }

    fn clear(&mut self) {
        self.buffer_start += self.buffer.len();
        self.buffer.clear();
        self.turn = None;
        self.partial = None;
        if let Some(vad) = &mut self.vad {
            vad.reset_turn();
        }
    }

    fn append(&mut self, bytes: &[u8]) -> Vec<String> {
        let samples = match self.format {
            AudioFormat::Pcm { .. } => pcm16_le(bytes),
            AudioFormat::Mulaw => mulaw(bytes),
            AudioFormat::Alaw => alaw(bytes),
        };
        let mut resampled = Vec::with_capacity(samples.len());
        self.resampler.process(&samples, &mut resampled);
        self.buffer.extend_from_slice(&resampled);
        let mut out = Vec::new();
        if self.seconds(self.buffer.len()) > self.speech.max_audio_seconds {
            self.clear();
            out.push(self.session.error(&EventError {
                event_id: None,
                code: "input_audio_buffer_too_long",
                message: format!(
                    "The input audio buffer exceeded {} seconds and was cleared",
                    self.speech.max_audio_seconds
                ),
                param: None,
            }));
            return out;
        }
        let events = match &mut self.vad {
            Some(vad) => vad.push(&resampled),
            None => return out,
        };
        for event in events {
            match event {
                VadEvent::SpeechStarted { sample } => {
                    let sample = (self.vad_base + sample).max(self.buffer_start);
                    let item_id = self.new_item();
                    out.push(self.session.speech_started(&item_id, self.millis(sample)));
                    self.turn = Some(Turn {
                        item_id,
                        start: sample,
                        passed_at: sample,
                        hypothesis: Vec::new(),
                        agreed: 0,
                        sent: String::new(),
                    });
                }
                VadEvent::SpeechStopped { sample } => {
                    let sample = self.vad_base + sample;
                    if let Some(turn) = self.turn.take() {
                        out.push(
                            self.session
                                .speech_stopped(&turn.item_id, self.millis(sample)),
                        );
                        out.extend(self.commit(turn, sample));
                    }
                }
            }
        }
        if self.turn.is_none()
            && let Some(t) = self.turn_detection
        {
            // Between turns keep only what a new turn's prefix padding could reach back to.
            let keep = self.samples(f64::from(t.prefix_padding_ms) / 1000.0 + IDLE_KEEP_SECONDS);
            let drop = self.buffer.len().saturating_sub(keep);
            self.buffer.drain(..drop);
            self.buffer_start += drop;
        }
        out
    }

    /// Commits `turn` up to absolute sample `end`.
    fn commit(&mut self, turn: Turn, end: usize) -> Vec<String> {
        let from = turn
            .start
            .saturating_sub(self.buffer_start)
            .min(self.buffer.len());
        let to = end
            .saturating_sub(self.buffer_start)
            .clamp(from, self.buffer.len());
        let samples = self.buffer[from..to].to_vec();
        self.buffer.drain(..to);
        self.buffer_start += to;
        if self
            .partial
            .as_ref()
            .is_some_and(|(item, _)| *item == turn.item_id)
        {
            self.partial = None;
        }
        let previous = self.previous_item.replace(turn.item_id.clone());
        let events = self.session.committed(&turn.item_id, previous.as_deref());
        self.committed.push_back(Committed {
            item_id: turn.item_id,
            previous,
            samples,
            stream_segments: turn.sent.is_empty(),
            sent: turn.sent,
        });
        events
    }

    fn commit_manually(&mut self) -> Vec<String> {
        let end = self.buffer_start + self.buffer.len();
        let start = self.turn.as_ref().map_or(self.buffer_start, |t| t.start);
        if self.seconds(end.saturating_sub(start)) < MIN_COMMIT_SECONDS {
            return vec![self.session.error(&EventError {
                event_id: None,
                code: "input_audio_buffer_commit_empty",
                message: format!(
                    "Error committing input audio buffer: buffer too small. Expected at least \
                     {:.0}ms of audio.",
                    MIN_COMMIT_SECONDS * 1000.0
                ),
                param: None,
            })];
        }
        if let Some(vad) = &mut self.vad {
            vad.reset_turn();
        }
        let turn = match self.turn.take() {
            Some(turn) => turn,
            None => Turn {
                item_id: self.new_item(),
                start,
                passed_at: start,
                hypothesis: Vec::new(),
                agreed: 0,
                sent: String::new(),
            },
        };
        self.commit(turn, end)
    }

    /// Starts the next full transcription, or a live pass when nothing is committed.
    fn schedule(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if self.active.is_none()
            && let Some(committed) = self.committed.pop_front()
        {
            match self
                .speech
                .worker
                .transcribe(committed.samples.clone(), true)
            {
                Ok(updates) => self.active = Some((committed, updates)),
                Err(_) => {
                    out.push(self.session.failed(
                        &committed.item_id,
                        "The transcription queue is full. Retry later.",
                    ));
                }
            }
        }
        let end = self.buffer_start + self.buffer.len();
        let interval = self.samples(PARTIAL_INTERVAL_SECONDS);
        let limit = self.samples(PARTIAL_LIMIT_SECONDS);
        if self.partial.is_none()
            && self.active.is_none()
            && let Some(turn) = &mut self.turn
            && end - turn.passed_at >= interval
            && end - turn.start <= limit
        {
            let from = turn.start.saturating_sub(self.buffer_start);
            turn.passed_at = end;
            if let Ok(reply) = self.speech.worker.pass(self.buffer[from..].to_vec()) {
                self.partial = Some((turn.item_id.clone(), reply));
            }
        }
        out
    }

    fn on_partial(&mut self, item_id: &str, pass: Pass) -> Vec<String> {
        let Some(turn) = self.turn.as_mut().filter(|t| t.item_id == item_id) else {
            return Vec::new();
        };
        let words: Vec<String> = pass.words.iter().map(|w| w.text.clone()).collect();
        let agreed = agreement(&turn.hypothesis, &words, turn.agreed);
        turn.hypothesis = words;
        if agreed <= turn.agreed {
            return Vec::new();
        }
        let text = turn.hypothesis[turn.agreed..agreed].join(" ");
        let delta = if turn.sent.is_empty() {
            text
        } else {
            format!(" {text}")
        };
        let logprobs = logprobs(&pass.words[turn.agreed..agreed]);
        turn.sent.push_str(&delta);
        turn.agreed = agreed;
        let item_id = turn.item_id.clone();
        vec![self.session.delta(&item_id, &delta, &logprobs)]
    }

    fn on_final(&mut self, update: Option<Update>) -> Vec<String> {
        let Some((committed, _)) = self.active.as_mut() else {
            return Vec::new();
        };
        match update {
            Some(Update::Segment(segment, words)) => {
                if !committed.stream_segments {
                    return Vec::new();
                }
                let delta = if committed.sent.is_empty() {
                    segment.text
                } else {
                    format!(" {}", segment.text)
                };
                committed.sent.push_str(&delta);
                let item_id = committed.item_id.clone();
                vec![self.session.delta(&item_id, &delta, &logprobs(&words))]
            }
            Some(Update::Done(Ok(transcript))) => {
                let (committed, _) = self.active.take().expect("an active transcription");
                let mut out = Vec::new();
                // Finish the deltas when they are a prefix of the final text.
                if let Some(rest) = transcript.text.strip_prefix(committed.sent.as_str())
                    && !rest.trim().is_empty()
                {
                    let rest = if committed.sent.is_empty() {
                        rest.trim_start()
                    } else {
                        rest
                    };
                    out.push(self.session.delta(&committed.item_id, rest, &[]));
                }
                let tokens = logprobs(&transcript.words);
                out.extend(self.session.completed(
                    &committed.item_id,
                    committed.previous.as_deref(),
                    &transcript.text,
                    transcript.duration,
                    &tokens,
                ));
                out
            }
            Some(Update::Done(Err(error))) => {
                let (committed, _) = self.active.take().expect("an active transcription");
                tracing::warn!(%error, "Realtime transcription failed");
                vec![
                    self.session
                        .failed(&committed.item_id, "The audio could not be transcribed."),
                ]
            }
            None => {
                let (committed, _) = self.active.take().expect("an active transcription");
                vec![
                    self.session
                        .failed(&committed.item_id, "The speech worker stopped."),
                ]
            }
        }
    }
}

/// A word compared across passes: case and surrounding punctuation change as context grows.
fn normalized(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

/// Words agreed by two consecutive passes, counting the `sent` words as agreed (a revision of
/// sent text cannot be taken back). The newest pass's last word is never agreed: it is still
/// being spoken, and its punctuation depends on what follows.
fn agreement(previous: &[String], current: &[String], sent: usize) -> usize {
    let stable = current.len().saturating_sub(1);
    let more = previous
        .iter()
        .zip(&current[..stable])
        .skip(sent)
        .take_while(|(a, b)| normalized(a) == normalized(b))
        .count();
    (sent + more).min(stable.max(sent))
}

async fn send(socket: &mut WebSocket, events: Vec<String>) -> bool {
    for event in events {
        if socket.send(Message::Text(event.into())).await.is_err() {
            return false;
        }
    }
    true
}

/// Serves one session until the client disconnects.
pub(crate) async fn serve(mut socket: WebSocket, speech: SpeechService) {
    let mut live = Live::new(speech);
    let created = live.session.created();
    if !send(&mut socket, vec![created]).await {
        return;
    }
    loop {
        let scheduled = live.schedule();
        if !send(&mut socket, scheduled).await {
            return;
        }
        let wake = tokio::select! {
            message = socket.recv() => Wake::Client(message),
            result = async { (&mut live.partial.as_mut().expect("a live pass").1).await },
                if live.partial.is_some() => Wake::Partial(result),
            update = async { live.active.as_mut().expect("a transcription").1.recv().await },
                if live.active.is_some() => Wake::Final(update),
        };
        let events = match wake {
            Wake::Client(Some(Ok(Message::Text(text)))) => live.handle(text.as_str()),
            Wake::Client(Some(Ok(Message::Binary(_)))) => vec![live.session.error(&EventError {
                event_id: None,
                code: "invalid_event",
                message: "Events are JSON text frames".into(),
                param: None,
            })],
            Wake::Client(Some(Ok(Message::Ping(_) | Message::Pong(_)))) => Vec::new(),
            Wake::Client(Some(Ok(Message::Close(_)) | Err(_)) | None) => return,
            Wake::Partial(result) => {
                let (item_id, _) = live.partial.take().expect("a live pass");
                match result {
                    Ok(Ok(pass)) => live.on_partial(&item_id, pass),
                    Ok(Err(error)) => {
                        tracing::warn!(%error, "Live transcription pass failed");
                        Vec::new()
                    }
                    Err(_) => Vec::new(),
                }
            }
            Wake::Final(update) => live.on_final(update),
        };
        if !send(&mut socket, events).await {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::agreement;
    use crate::speech::{scripted_service, spoken};
    use crate::{AppState, router};
    use base64::Engine as _;
    use futures_util::{SinkExt, StreamExt};
    use serde_json::{Value, json};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio_tungstenite::tungstenite::{Message, client::IntoClientRequest};

    type Socket = tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >;

    /// Serves a scripted speech model; returns the Realtime URL.
    async fn server(key: Option<&str>) -> String {
        let app = router(AppState {
            text: None,
            speech: Some(scripted_service(20.0).await),
            api_key: key.map(Arc::from),
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("ws://{address}/v1/realtime?intent=transcription")
    }

    async fn connect(
        url: &str,
        protocols: &str,
    ) -> Result<Socket, tokio_tungstenite::tungstenite::Error> {
        let mut request = url.into_client_request().unwrap();
        request
            .headers_mut()
            .insert("sec-websocket-protocol", protocols.parse().unwrap());
        let (socket, response) = tokio_tungstenite::connect_async(request).await?;
        assert_eq!(response.headers()["sec-websocket-protocol"], "realtime");
        Ok(socket)
    }

    async fn send(socket: &mut Socket, event: Value) {
        socket
            .send(Message::Text(event.to_string().into()))
            .await
            .unwrap();
    }

    async fn next(socket: &mut Socket) -> Value {
        loop {
            let message = tokio::time::timeout(Duration::from_secs(10), socket.next())
                .await
                .expect("a server event within 10 s")
                .expect("an open socket")
                .unwrap();
            if let Message::Text(text) = message {
                return serde_json::from_str(&text).unwrap();
            }
        }
    }

    /// Events up to and including the first of type `kind`.
    async fn until(socket: &mut Socket, kind: &str) -> Vec<Value> {
        let mut events = Vec::new();
        loop {
            let event = next(socket).await;
            let done = event["type"] == kind;
            events.push(event);
            if done {
                return events;
            }
        }
    }

    /// Appends 24 kHz samples as PCM16 in 100 ms chunks.
    async fn append(socket: &mut Socket, samples: &[f32]) {
        for chunk in samples.chunks(2400) {
            let bytes: Vec<u8> = chunk
                .iter()
                .flat_map(|s| ((s * 32768.0).round() as i16).to_le_bytes())
                .collect();
            let audio = base64::engine::general_purpose::STANDARD.encode(bytes);
            send(
                socket,
                json!({"type": "input_audio_buffer.append", "audio": audio}),
            )
            .await;
        }
    }

    fn kinds(events: &[Value]) -> Vec<&str> {
        events
            .iter()
            .map(|e| e["type"].as_str().unwrap())
            .filter(|kind| !kind.ends_with("transcription.delta"))
            .collect()
    }

    fn deltas(events: &[Value]) -> String {
        events
            .iter()
            .filter(|e| e["type"] == "conversation.item.input_audio_transcription.delta")
            .map(|e| e["delta"].as_str().unwrap())
            .collect()
    }

    #[tokio::test]
    async fn server_vad_turns_are_detected_committed_and_transcribed() {
        let mut socket = connect(&server(None).await, "realtime").await.unwrap();
        let created = next(&mut socket).await;
        assert_eq!(created["type"], "session.created");
        assert_eq!(
            created["session"]["audio"]["input"]["turn_detection"]["type"],
            "server_vad"
        );

        let mut audio = vec![0.0; 24000];
        audio.extend(spoken(3, 24000));
        audio.extend(vec![0.0; 36000]);
        append(&mut socket, &audio).await;
        let events = until(&mut socket, "conversation.item.done").await;
        assert_eq!(
            kinds(&events),
            [
                "input_audio_buffer.speech_started",
                "input_audio_buffer.speech_stopped",
                "input_audio_buffer.committed",
                "conversation.item.added",
                "conversation.item.input_audio_transcription.completed",
                "conversation.item.done",
            ]
        );
        let item = &events[0]["item_id"];
        assert!(
            events
                .iter()
                .all(|e| e.get("item_id").is_none_or(|id| id == item))
        );
        let started = events[0]["audio_start_ms"].as_u64().unwrap();
        assert!(
            (500..=1000).contains(&started),
            "speech started at {started} ms"
        );
        let completed = events
            .iter()
            .find(|e| e["type"] == "conversation.item.input_audio_transcription.completed")
            .unwrap();
        assert_eq!(completed["transcript"], "w1 w2 w3");
        // Live deltas and the final remainder add up to the transcript.
        assert_eq!(deltas(&events), "w1 w2 w3");
    }

    #[tokio::test]
    async fn manual_turns_commit_on_request_and_bad_events_get_errors() {
        let mut socket = connect(&server(None).await, "realtime").await.unwrap();
        next(&mut socket).await;
        send(
            &mut socket,
            json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"turn_detection": null}}}}),
        )
        .await;
        let updated = next(&mut socket).await;
        assert_eq!(updated["type"], "session.updated");
        assert_eq!(
            updated["session"]["audio"]["input"]["turn_detection"],
            Value::Null
        );

        send(
            &mut socket,
            json!({"type": "input_audio_buffer.commit", "event_id": "c1"}),
        )
        .await;
        let error = next(&mut socket).await;
        assert_eq!(error["error"]["code"], "input_audio_buffer_commit_empty");

        send(
            &mut socket,
            json!({"type": "response.create", "event_id": "r1"}),
        )
        .await;
        let error = next(&mut socket).await;
        assert_eq!(
            (error["type"].as_str(), error["error"]["event_id"].as_str()),
            (Some("error"), Some("r1"))
        );

        append(&mut socket, &spoken(2, 24000)).await;
        send(&mut socket, json!({"type": "input_audio_buffer.commit"})).await;
        let events = until(&mut socket, "conversation.item.done").await;
        assert_eq!(
            kinds(&events),
            [
                "input_audio_buffer.committed",
                "conversation.item.added",
                "conversation.item.input_audio_transcription.completed",
                "conversation.item.done",
            ]
        );
        let completed = events
            .iter()
            .find(|e| e["type"] == "conversation.item.input_audio_transcription.completed")
            .unwrap();
        assert_eq!(completed["transcript"], "w1 w2");
        assert_eq!(deltas(&events), "w1 w2");

        append(&mut socket, &spoken(1, 24000)).await;
        send(&mut socket, json!({"type": "input_audio_buffer.clear"})).await;
        assert_eq!(
            next(&mut socket).await["type"],
            "input_audio_buffer.cleared"
        );
        send(&mut socket, json!({"type": "input_audio_buffer.commit"})).await;
        assert_eq!(
            next(&mut socket).await["error"]["code"],
            "input_audio_buffer_commit_empty"
        );
    }

    #[tokio::test]
    async fn the_api_key_may_arrive_as_a_subprotocol() {
        let url = server(Some("secret")).await;
        assert!(
            connect(&url, "realtime, openai-insecure-api-key.secret")
                .await
                .is_ok()
        );
        let Err(tokio_tungstenite::tungstenite::Error::Http(response)) =
            connect(&url, "realtime, openai-insecure-api-key.wrong").await
        else {
            panic!("a wrong key was accepted");
        };
        assert_eq!(response.status(), 401);
        let Err(tokio_tungstenite::tungstenite::Error::Http(response)) = connect(
            &url.replace("intent=transcription", "model=gpt-4o-transcribe"),
            "realtime, openai-insecure-api-key.secret",
        )
        .await
        else {
            panic!("an unknown model was accepted");
        };
        assert_eq!(response.status(), 404);
    }

    #[test]
    fn agreement_skips_sent_revisions_and_holds_back_the_last_word() {
        let words = |text: &str| text.split(' ').map(String::from).collect::<Vec<_>>();
        let a = words("Capítulo 6 de José Santie Esteban.");
        let b = words("Capítulo 6 de José Santi Esteban Esta grabación");
        let c = words("Capítulo 6 de José Santi Esteban. Esta grabación de");
        // Nothing sent: the prefix up to the revision.
        assert_eq!(agreement(&a, &b, 0), 4);
        // "Santie" was sent; its revision does not block later words, and punctuation or
        // case changes are the same word. The last word of `c` waits.
        assert_eq!(agreement(&b, &c, 5), 8);
        assert_eq!(agreement(&b, &c, 8), 8);
        assert_eq!(agreement(&[], &words("Hola"), 0), 0);
    }
}
