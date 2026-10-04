//! One take: stream the microphone to the transcriber, walk the flow tree to a leaf, and send
//! the leaf's text where it says: into the application, the bubble or the clipboard.
//!
//! Every step is recorded in a [`Trace`] for the inspector.

use crate::client::{
    ClientError, DecisionRequest, DecisionResponse, RealtimeEvent, ResponseRequest, Routes, Turns,
};
use crate::context::ContextSnapshot;
use crate::flow::frame::Frame;
use crate::flow::investigate::Investigate;
use crate::flow::machine::runtime::{Runtime, Step};
use crate::flow::spec::Output;
use crate::flow::tools::ToolHost;
use crate::flow::walk::{self, FlowStep, Leaf, ToolTrace, Walked};
use crate::flow::{FlowTree, Kind};
use jevons_desktop_protocol::delivery::{
    Action, AudioEvent, DeliveryMethod, DeliveryOutcome, DeliveryRequest, SAMPLE_RATE,
};
use jevons_desktop_protocol::desk::{Delivery, Desk};
pub use jevons_desktop_protocol::take::{Stage, StageKind, Update};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, oneshot};

/// The shortest audio worth transcribing (the server's minimum commit).
const MIN_SAMPLES: usize = SAMPLE_RATE as usize / 10;
/// A push-to-talk press shorter than this was not meant as speech: it is dropped quietly.
const MIN_TAKE_SECONDS: f64 = 0.4;
/// Live dictation: speech in a phrase before a pause can end it.
const PHRASE_SPEECH_MS: u32 = 300;
/// Live dictation: the pause that ends a phrase.
const PHRASE_PAUSE_MS: u32 = 700;
/// Live dictation: the longest phrase; nonstop speech is committed this often.
const PHRASE_LONGEST_MS: u32 = 20_000;
/// Live dictation: audio without speech kept before it is dropped.
const PHRASE_IDLE_MS: u32 = 3_000;

/// What live dictation does after a chunk of microphone audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phrase {
    Continue,
    /// A phrase ended: commit it.
    Commit,
    /// Only background noise so far: drop it, so it is never transcribed into words.
    Clear,
}

/// Where live dictation ends a phrase, from the microphone level. The app commits the turns
/// itself, so no audio is ever dropped as silence: a missed pause only makes a phrase longer.
#[derive(Debug, Default)]
struct Phrases {
    speech_ms: u32,
    quiet_ms: u32,
    total_ms: u32,
    /// The noise floor, as an RMS level from 0 to 1.
    floor: Option<f32>,
}

impl Phrases {
    fn push(&mut self, samples: &[i16]) -> Phrase {
        if samples.is_empty() {
            return Phrase::Continue;
        }
        let ms = (samples.len() as u64 * 1000 / u64::from(SAMPLE_RATE)) as u32;
        let rms = (samples
            .iter()
            .map(|s| (f32::from(*s) / 32768.0).powi(2))
            .sum::<f32>()
            / samples.len() as f32)
            .sqrt();
        // The floor follows quiet chunks down at once and louder ones up slowly, so the
        // pauses between words keep it at the room's level through nonstop speech.
        let floor = match self.floor {
            None => rms.min(0.002),
            Some(f) if rms < f => rms,
            Some(f) => f + (rms - f) * 0.001,
        };
        self.floor = Some(floor);
        let speech = rms > (floor * 4.0).max(0.006);
        self.total_ms += ms;
        if speech {
            self.speech_ms += ms;
            self.quiet_ms = 0;
        } else {
            self.quiet_ms += ms;
        }
        let phrase = if self.heard() {
            if self.quiet_ms >= PHRASE_PAUSE_MS || self.total_ms >= PHRASE_LONGEST_MS {
                Phrase::Commit
            } else {
                Phrase::Continue
            }
        } else if self.total_ms >= PHRASE_IDLE_MS && self.quiet_ms >= 1000 {
            Phrase::Clear
        } else {
            Phrase::Continue
        };
        if phrase != Phrase::Continue {
            self.speech_ms = 0;
            self.quiet_ms = 0;
            self.total_ms = 0;
        }
        phrase
    }

    /// Whether the audio since the last commit held speech.
    fn heard(&self) -> bool {
        self.speech_ms >= PHRASE_SPEECH_MS
    }
}
/// Live dictation: how long to wait for the last turn after stopping.
const LIVE_DRAIN: Duration = Duration::from_secs(5);
/// How long to wait for the final transcript after the take ends.
const TRANSCRIPT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub language: Option<String>,
    /// Ask the decision model at model decisions; when off, they take their fallback.
    pub decide: bool,
    /// The most tokens a generation writes, unless a flow node sets its own.
    pub max_output_tokens: u32,
    /// How long the decision may take before the transcript is typed as heard.
    pub decision_timeout: Duration,
    /// How long generation may take before the transcript is typed as heard.
    pub generation_timeout: Duration,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            language: None,
            decide: true,
            max_output_tokens: 1024,
            decision_timeout: Duration::from_secs(60),
            generation_timeout: Duration::from_secs(120),
        }
    }
}

/// What the pipeline needs from the app.
#[derive(Clone)]
pub struct Env {
    /// Each capability's provider and model: speech (Realtime first, uploads as the
    /// fallback), decisions and generation.
    pub routes: Routes,
    /// The flow tree takes walk (the last one that loaded without errors).
    pub flows: Arc<FlowTree>,
    pub settings: Settings,
    /// The client: it delivers the text, asks the user, reads the screen and runs its own
    /// tools. With no one at it (`Nobody`), the text is only recorded, tool calls that need
    /// confirmation are denied, and nothing is read.
    pub desk: Arc<dyn Desk>,
    /// Answers `[investigate]` questions; without it their answers are empty.
    pub investigator: Option<Arc<dyn Investigate>>,
    /// The tools the settings register; without them tool and agent nodes fail.
    pub tools: Option<Arc<ToolHost>>,
    /// The machines that run across takes: the flows root's and the tasks nested in it.
    pub machines: Arc<Runtime>,
}

/// A take as it starts.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TakeStart {
    pub id: u64,
    pub context: ContextSnapshot,
    /// The branch of the flow tree to start at, such as `ask`; `None` starts at the root.
    pub entry: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionPath {
    Realtime,
    Upload,
}

#[derive(Clone, Debug, Serialize)]
pub struct GenerationTrace {
    pub request: ResponseRequest,
    pub output: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct DecisionTrace {
    pub request: DecisionRequest,
    pub response: Option<DecisionResponse>,
}

/// Everything that happened in a take.
#[derive(Clone, Debug, Serialize)]
pub struct Trace {
    pub take: u64,
    /// The turn within a live dictation take, from 1.
    pub turn: Option<u32>,
    /// Unix milliseconds.
    pub started_at_ms: u64,
    pub context: ContextSnapshot,
    pub audio_seconds: f64,
    pub transcription: Option<TranscriptionPath>,
    pub transcript: String,
    /// Where the walk started: `/` for the root, or a branch a hotkey starts at.
    pub entry: String,
    /// Every node the walk went through, with the guards, decisions and investigations.
    pub flow: Vec<FlowStep>,
    /// Where the walk ended and where its text goes.
    pub leaf: Option<Leaf>,
    /// The tool calls of tool and agent nodes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<ToolTrace>,
    /// The transitions the machines took.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub machine: Vec<Step>,
    pub generation: Option<GenerationTrace>,
    /// The text delivered (or that would be, in a dry run).
    pub output: String,
    pub delivery: Option<DeliveryOutcome>,
    /// Step name and milliseconds, in order.
    pub timings: Vec<(String, u64)>,
    /// Notices that did not stop the take, such as a fallback.
    pub notes: Vec<String>,
    pub error: Option<String>,
}

impl Trace {
    pub fn new(start: &TakeStart) -> Self {
        Self {
            take: start.id,
            turn: None,
            started_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as u64),
            context: start.context.clone(),
            audio_seconds: 0.0,
            transcription: None,
            transcript: String::new(),
            entry: start.entry.clone().unwrap_or_else(|| "/".into()),
            flow: Vec::new(),
            leaf: None,
            calls: Vec::new(),
            machine: Vec::new(),
            generation: None,
            output: String::new(),
            delivery: None,
            timings: Vec::new(),
            notes: Vec::new(),
            error: None,
        }
    }

    fn time(&mut self, step: &str, since: Instant) {
        self.timings
            .push((step.into(), since.elapsed().as_millis() as u64));
    }

    /// The branches taken, such as `dictation/dictate/notes → _actions/rewrite`.
    pub fn route(&self) -> String {
        self.flow
            .iter()
            .filter(|s| s.node != "/")
            .map(|s| s.node.as_str())
            .collect::<Vec<_>>()
            .join(" → ")
    }
}

/// Runs a whole take. Audio arrives on `audio` until `finish` fires (or the source ends);
/// `updates` receives live progress.
pub async fn run_take(
    env: &Env,
    start: TakeStart,
    audio: mpsc::UnboundedReceiver<AudioEvent>,
    finish: oneshot::Receiver<()>,
    updates: &mpsc::UnboundedSender<Update>,
) -> Trace {
    let mut trace = Trace::new(&start);
    let take = start.id;
    tracing::info!(take, app = %start.context.app.process_name, "Take started");
    let begun = Instant::now();
    let transcript = transcribe(env, audio, finish, updates, &mut trace).await;
    trace.time("transcribe", begun);
    tracing::info!(
        take,
        ms = begun.elapsed().as_millis() as u64,
        audio_seconds = trace.audio_seconds,
        path = ?trace.transcription,
        ok = transcript.is_ok(),
        "Transcribed"
    );
    match transcript {
        Ok(_) if trace.audio_seconds < MIN_TAKE_SECONDS => {
            trace
                .notes
                .push("Too short to hold speech: nothing was sent".into());
            return trace;
        }
        Ok(text) if text.trim().is_empty() => {
            trace.error = Some("No speech was recognized".into());
            return trace;
        }
        Ok(text) => trace.transcript = text.trim().to_string(),
        Err(e) => {
            trace.error = Some(e.to_string());
            return trace;
        }
    }
    let _ = updates.send(Update::Thinking);
    finish_take(env, &start, updates, trace).await
}

/// Transcribes a take and nothing more: no flow tree, no delivery. For what the user says
/// while recording a demonstration.
pub async fn transcribe_only(
    env: &Env,
    start: TakeStart,
    audio: mpsc::UnboundedReceiver<AudioEvent>,
    finish: oneshot::Receiver<()>,
    updates: &mpsc::UnboundedSender<Update>,
) -> Result<String, String> {
    let mut trace = Trace::new(&start);
    let text = transcribe(env, audio, finish, updates, &mut trace)
        .await
        .map_err(|e| e.to_string())?;
    if trace.audio_seconds < MIN_TAKE_SECONDS || text.trim().is_empty() {
        return Err("no speech was recognized".into());
    }
    Ok(text.trim().to_string())
}

/// Walks the flow tree for a transcribed take and sends the leaf's text where it goes. Under a
/// machine root the take is `said` for the machines instead.
async fn finish_take(
    env: &Env,
    start: &TakeStart,
    updates: &mpsc::UnboundedSender<Update>,
    mut trace: Trace,
) -> Trace {
    let take = start.id;
    let tree = env.flows.clone();
    let entry = start.entry.as_deref().and_then(|path| {
        let found = tree.find(path);
        if found.is_none() {
            trace.notes.push(format!(
                "The flow tree has no branch {path}: starting at the root"
            ));
        }
        found
    });
    trace.entry = tree.node(entry.unwrap_or(tree.root())).label().to_string();
    let walked = Instant::now();
    if tree.node(tree.root()).kind() == Kind::Machine {
        env.machines
            .take(env, start, entry, updates, &mut trace)
            .await;
        trace.time("walk", walked);
        tracing::info!(
            take,
            steps = trace.machine.len(),
            error = trace.error.is_some(),
            "Moved the machines"
        );
        return trace;
    }
    let frame = Frame::new(start.context.clone(), trace.transcript.clone());
    let walk = walk::run(
        env,
        &tree,
        entry.unwrap_or(tree.root()),
        frame,
        HashMap::new(),
        updates,
        &mut trace,
    )
    .await;
    trace.time("walk", walked);
    match walk {
        Ok((Walked::Leaf(leaf), _)) => deliver_leaf(env, start, leaf, &mut trace).await,
        Ok((Walked::Machine(_), _)) => {
            trace.error = Some("A machine runs only below a machine at the flows root".into())
        }
        Err(e) => trace.error = Some(e.to_string()),
    }
    trace
}

/// Sends a leaf's text where it goes, and records it on the trace.
pub(crate) async fn deliver_leaf(env: &Env, start: &TakeStart, leaf: Leaf, trace: &mut Trace) {
    let take = start.id;
    tracing::info!(take, leaf = %leaf.node, output = ?leaf.output, "Walked the flow tree");
    trace.output = leaf.text.clone();
    let delivered = Instant::now();
    if leaf.text.trim().is_empty() {
        trace
            .notes
            .push("The text is empty: nothing was delivered".into());
    } else {
        match leaf.output {
            Output::Target => match deliver(env, start, &leaf).await {
                Ok(outcome) => trace.delivery = outcome,
                Err(e) => trace.error = Some(e),
            },
            Output::Clipboard => {
                let copy = Leaf {
                    delivery: DeliveryMethod::Clipboard,
                    ..leaf.clone()
                };
                match deliver(env, start, &copy).await {
                    Ok(Some(_)) => {
                        trace.delivery = Some(DeliveryOutcome::OnClipboard {
                            reason: "the flow sends it to the clipboard".into(),
                        });
                    }
                    Ok(None) => {}
                    Err(e) => trace.error = Some(e),
                }
            }
            Output::Bubble => trace.delivery = Some(DeliveryOutcome::Shown),
            Output::None | Output::Next => {}
        }
    }
    trace.leaf = Some(leaf);
    trace.time("deliver", delivered);
    tracing::info!(
        take,
        ms = delivered.elapsed().as_millis() as u64,
        chars = trace.output.chars().count(),
        outcome = ?trace.delivery.as_ref().map(|d| match d {
            DeliveryOutcome::Delivered { method } => format!("{method:?}"),
            DeliveryOutcome::OnClipboard { .. } => "clipboard".into(),
            DeliveryOutcome::Shown => "bubble".into(),
        }),
        error = trace.error.is_some(),
        "Delivered"
    );
}

/// Streams the take to Realtime when possible and returns the transcript; falls back to an
/// upload of the buffered audio.
async fn transcribe(
    env: &Env,
    mut audio: mpsc::UnboundedReceiver<AudioEvent>,
    mut finish: oneshot::Receiver<()>,
    updates: &mpsc::UnboundedSender<Update>,
    trace: &mut Trace,
) -> Result<String, ClientError> {
    let settings = &env.settings;
    let mut session = None;
    if let Some(route) = &env.routes.realtime {
        match route
            .client
            .realtime(
                Some(&route.model),
                settings.language.as_deref(),
                Turns::Client,
            )
            .await
        {
            Ok(opened) => session = Some(opened),
            Err(e) => trace
                .notes
                .push(format!("Uploading instead of streaming: {e}")),
        }
    }
    let mut buffer: Vec<i16> = Vec::new();
    let mut live = String::new();
    loop {
        tokio::select! {
            event = audio.recv() => match event {
                Some(AudioEvent::Chunk(samples)) => {
                    if let Some((writer, _)) = &mut session
                        && let Err(e) = writer.append(&samples).await
                    {
                        trace.notes.push(format!("Streaming stopped: {e}"));
                        session = None;
                    }
                    buffer.extend_from_slice(&samples);
                }
                Some(AudioEvent::Level(bands)) => {
                    let _ = updates.send(Update::Level(bands));
                }
                Some(AudioEvent::Failed(message)) => {
                    return Err(ClientError::Protocol(format!("Microphone: {message}")));
                }
                Some(AudioEvent::Ended) | None => break,
            },
            _ = &mut finish => {
                // Drain what the source already captured.
                while let Ok(event) = audio.try_recv() {
                    if let AudioEvent::Chunk(samples) = event {
                        if let Some((writer, _)) = &mut session {
                            let _ = writer.append(&samples).await;
                        }
                        buffer.extend_from_slice(&samples);
                    }
                }
                break;
            }
            Some(event) = async {
                match &mut session {
                    Some((_, reader)) => reader.next().await,
                    None => std::future::pending().await,
                }
            } => match event {
                RealtimeEvent::Delta { delta, .. } => {
                    live.push_str(&delta);
                    let _ = updates.send(Update::Delta(delta));
                }
                RealtimeEvent::Error { message } => {
                    trace.notes.push(format!("Streaming failed: {message}"));
                    session = None;
                }
                RealtimeEvent::Completed { .. } | RealtimeEvent::Other(_) => {}
            },
        }
    }
    trace.audio_seconds = buffer.len() as f64 / f64::from(SAMPLE_RATE);
    if buffer.len() < MIN_SAMPLES || trace.audio_seconds < MIN_TAKE_SECONDS {
        return Ok(String::new());
    }
    let _ = updates.send(Update::Transcribing);
    if let Some((mut writer, mut reader)) = session.take() {
        let completed = async {
            writer.commit().await?;
            loop {
                match reader.next().await {
                    Some(RealtimeEvent::Completed { transcript, .. }) => return Ok(transcript),
                    Some(RealtimeEvent::Delta { delta, .. }) => {
                        let _ = updates.send(Update::Delta(delta));
                    }
                    Some(RealtimeEvent::Error { message }) => {
                        return Err(ClientError::Realtime(message));
                    }
                    Some(RealtimeEvent::Other(_)) => {}
                    None => return Err(ClientError::Realtime("the session closed".into())),
                }
            }
        };
        match tokio::time::timeout(TRANSCRIPT_TIMEOUT, completed).await {
            Ok(Ok(transcript)) => {
                writer.close().await;
                trace.transcription = Some(TranscriptionPath::Realtime);
                return Ok(transcript);
            }
            Ok(Err(e)) => trace
                .notes
                .push(format!("Uploading after streaming failed: {e}")),
            Err(_) => trace
                .notes
                .push("Uploading after the streamed transcript timed out".into()),
        }
    }
    let route = env
        .routes
        .speech
        .as_ref()
        .ok_or(ClientError::NotServed("Speech to text"))?;
    trace.transcription = Some(TranscriptionPath::Upload);
    let transcription = route
        .client
        .transcribe(
            &buffer,
            SAMPLE_RATE,
            &route.model,
            settings.language.as_deref(),
        )
        .await?;
    Ok(transcription.text)
}

/// Runs a take from text instead of audio, as if it had been said: the headless `--transcript`
/// mode, and a way to test a flow tree without speaking.
pub async fn run_transcript(
    env: &Env,
    start: TakeStart,
    transcript: &str,
    updates: &mpsc::UnboundedSender<Update>,
) -> Trace {
    let mut trace = Trace::new(&start);
    trace.transcript = transcript.trim().to_string();
    if trace.transcript.is_empty() {
        trace.error = Some("The transcript is empty".into());
        return trace;
    }
    let _ = updates.send(Update::Thinking);
    finish_take(env, &start, updates, trace).await
}

/// Live dictation: the audio streams to Realtime until `stop` fires, the app ending a phrase
/// at each pause, and `updates` shows what is heard as it is recognized. Nothing is typed while
/// speaking: once stopped, the whole transcript goes through the same decision, generation and
/// delivery as a take.
pub async fn run_live(
    env: &Env,
    start: TakeStart,
    mut audio: mpsc::UnboundedReceiver<AudioEvent>,
    mut stop: oneshot::Receiver<()>,
    updates: &mpsc::UnboundedSender<Update>,
) -> Trace {
    let mut trace = Trace::new(&start);
    trace.transcription = Some(TranscriptionPath::Realtime);
    let take = start.id;
    tracing::info!(take, app = %start.context.app.process_name, "Live dictation started");
    let settings = &env.settings;
    let session = match &env.routes.realtime {
        Some(route) => {
            route
                .client
                .realtime(
                    Some(&route.model),
                    settings.language.as_deref(),
                    Turns::Client,
                )
                .await
        }
        None => Err(ClientError::NotServed("Realtime transcription")),
    };
    let (mut writer, mut reader) = match session {
        Ok(session) => session,
        Err(e) => {
            trace.error = Some(format!("Live dictation needs Realtime transcription: {e}"));
            return trace;
        }
    };
    let began = Instant::now();
    let mut heard = String::new();
    let mut samples_sent = 0usize;
    let mut deadline: Option<tokio::time::Instant> = None;
    let mut phrases = Phrases::default();
    // Commits whose transcript has not arrived yet.
    let mut in_flight = 0usize;
    let result: Result<(), String> = loop {
        tokio::select! {
            event = audio.recv(), if deadline.is_none() => match event {
                Some(AudioEvent::Chunk(samples)) => {
                    samples_sent += samples.len();
                    if let Err(e) = writer.append(&samples).await {
                        break Err(e.to_string());
                    }
                    let sent = match phrases.push(&samples) {
                        Phrase::Commit => {
                            in_flight += 1;
                            writer.commit().await
                        }
                        Phrase::Clear => writer.clear().await,
                        Phrase::Continue => Ok(()),
                    };
                    if let Err(e) = sent {
                        break Err(e.to_string());
                    }
                }
                Some(AudioEvent::Level(bands)) => {
                    let _ = updates.send(Update::Level(bands));
                }
                Some(AudioEvent::Failed(message)) => break Err(format!("Microphone: {message}")),
                Some(AudioEvent::Ended) | None => {
                    // Whatever speech is still buffered is the last phrase.
                    if phrases.heard() && writer.commit().await.is_ok() {
                        in_flight += 1;
                    }
                    if in_flight == 0 {
                        break Ok(());
                    }
                    let _ = updates.send(Update::Transcribing);
                    deadline = Some(tokio::time::Instant::now() + LIVE_DRAIN);
                }
            },
            _ = &mut stop, if deadline.is_none() => {
                while let Ok(event) = audio.try_recv() {
                    if let AudioEvent::Chunk(samples) = event {
                        samples_sent += samples.len();
                        let _ = writer.append(&samples).await;
                        phrases.push(&samples);
                    }
                }
                if phrases.heard() && writer.commit().await.is_ok() {
                    in_flight += 1;
                }
                if in_flight == 0 {
                    break Ok(());
                }
                let _ = updates.send(Update::Transcribing);
                deadline = Some(tokio::time::Instant::now() + LIVE_DRAIN);
            }
            event = reader.next() => {
                // After stopping, wait for the transcripts as long as they keep coming.
                if deadline.is_some() {
                    deadline = Some(tokio::time::Instant::now() + LIVE_DRAIN);
                }
                match event {
                    Some(RealtimeEvent::Delta { delta, .. }) => {
                        let _ = updates.send(Update::Delta(delta));
                    }
                    Some(RealtimeEvent::Completed { transcript, .. }) => {
                        let words = transcript.trim();
                        if !words.is_empty() {
                            if !heard.is_empty() && !starts_with_punctuation(words) {
                                heard.push(' ');
                            }
                            heard.push_str(words);
                        }
                        let _ = updates.send(Update::Heard(heard.clone()));
                        in_flight = in_flight.saturating_sub(1);
                        if deadline.is_some() && in_flight == 0 {
                            break Ok(());
                        }
                    }
                    // A commit the server found too short: nothing will come for it.
                    Some(RealtimeEvent::Error { message }) if message.contains("buffer too small") => {
                        in_flight = in_flight.saturating_sub(1);
                        if deadline.is_some() && in_flight == 0 {
                            break Ok(());
                        }
                    }
                    Some(RealtimeEvent::Error { message }) if deadline.is_some() => {
                        trace.notes.push(format!("Transcription ended early: {message}"));
                        break Ok(());
                    }
                    Some(RealtimeEvent::Error { message }) => break Err(message),
                    Some(RealtimeEvent::Other(_)) => {}
                    None => break Ok(()),
                }
            }
            () = async {
                match deadline {
                    Some(at) => tokio::time::sleep_until(at).await,
                    None => std::future::pending().await,
                }
            } => {
                trace.notes.push(format!("{in_flight} phrases were still being transcribed"));
                break Ok(());
            }
        }
    };
    writer.close().await;
    trace.time("transcribe", began);
    trace.audio_seconds = samples_sent as f64 / f64::from(SAMPLE_RATE);
    tracing::info!(
        take,
        audio_seconds = trace.audio_seconds,
        ok = result.is_ok(),
        "Live dictation transcribed"
    );
    if let Err(e) = result {
        trace.error = Some(e);
        // Keep what was heard: it still goes to the application below.
        if heard.is_empty() {
            return trace;
        }
    }
    if heard.is_empty() {
        trace.error = Some("No speech was recognized".into());
        return trace;
    }
    trace.transcript = heard;
    let _ = updates.send(Update::Thinking);
    finish_take(env, &start, updates, trace).await
}

fn starts_with_punctuation(text: &str) -> bool {
    text.starts_with(|c: char| ",.;:!?)".contains(c))
}

/// Asks the desk to put a leaf's text into the window the take started in: it types it once
/// that is safe, or leaves it on the clipboard.
async fn deliver(
    env: &Env,
    start: &TakeStart,
    leaf: &Leaf,
) -> Result<Option<DeliveryOutcome>, String> {
    let request = DeliveryRequest {
        action: leaf.action,
        text: leaf.text.clone(),
        method: leaf.delivery,
        select_all: leaf.action == Action::Rewrite && start.context.selection().is_none(),
        erase: 0,
    };
    env.desk
        .deliver(Delivery {
            take: start.id,
            window: start.context.window.handle.unwrap_or(0),
            request,
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{Client, Profile, Route};
    use crate::context::{AppInfo, Element, WindowInfo};
    use crate::flow::{Catalog, Memory};
    use crate::serve::Host;
    use crate::session::Session;
    use axum::Json;
    use axum::routing::{get, post};
    use jevons_desktop_core::desk::LocalDesk;
    use jevons_desktop_core::fake::RecordingSink;
    use jevons_desktop_protocol::desk::Nobody;
    use jevons_desktop_protocol::take::TakeSettings;
    use jevons_desktop_protocol::wire::{ToClient, ToServer, VERSION, attend, in_process};
    use serde_json::{Value, json};
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct Seen {
        decisions: Arc<Mutex<Vec<Value>>>,
        generations: Arc<Mutex<Vec<Value>>>,
        uploads: Arc<Mutex<usize>>,
    }

    type Decider = Arc<dyn Fn(&Value) -> Value + Send + Sync>;

    /// A decision model that answers every question with the first of `labels` it offers (else
    /// its first choice), at 0.9.
    fn prefer(labels: &'static [&'static str]) -> Decider {
        prefer_with(labels, 0.9)
    }

    fn prefer_with(labels: &'static [&'static str], confidence: f64) -> Decider {
        Arc::new(move |request: &Value| {
            let answers: serde_json::Map<String, Value> = request["questions"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, question)| {
                    if question["type"] == "noul" {
                        return (key.clone(), json!({"type": "noul", "noul": confidence}));
                    }
                    let offered: Vec<&String> =
                        question["criteria"].as_object().unwrap().keys().collect();
                    let choice = labels
                        .iter()
                        .find(|l| offered.iter().any(|o| o == *l))
                        .map(|l| l.to_string())
                        .unwrap_or_else(|| offered[0].clone());
                    let answer = json!({"type": "choice", "choice": choice,
                        "probabilities": {choice.clone(): confidence}, "confidence": confidence});
                    (key.clone(), answer)
                })
                .collect();
            json!({"model": "jev", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": answers})
        })
    }

    /// A fake jevons server: no Realtime route, a scripted decision and generation.
    async fn server(decide: Decider, output: &'static str) -> (Client, Seen) {
        server_with(decide, output, Vec::new()).await
    }

    /// The same, with chat turns for agents: text, or `{"call": name, "arguments": {...}}`.
    async fn server_with(
        decide: Decider,
        output: &'static str,
        chat: Vec<Value>,
    ) -> (Client, Seen) {
        let seen = Seen::default();
        let turns = Arc::new(Mutex::new(std::collections::VecDeque::from(chat)));
        let chat = move |Json(_body): Json<Value>| {
            let turns = turns.clone();
            async move {
                let reply = turns
                    .lock()
                    .unwrap()
                    .pop_front()
                    .unwrap_or(json!("(no reply)"));
                let chunk = match &reply {
                    Value::String(text) => json!({"choices": [{"delta": {"content": text}}]}),
                    call => json!({"choices": [{"delta": {"tool_calls": [{"index": 0,
                        "id": "call_1", "type": "function",
                        "function": {"name": call["call"], "arguments": call["arguments"].to_string()}}]}}]}),
                };
                (
                    [("content-type", "text/event-stream")],
                    format!("data: {chunk}\n\ndata: [DONE]\n\n"),
                )
            }
        };
        let s = seen.clone();
        let decide = move |Json(body): Json<Value>| {
            let s = s.clone();
            let decide = decide.clone();
            async move {
                let answer = decide(&body);
                s.decisions.lock().unwrap().push(body);
                Json(answer)
            }
        };
        let s = seen.clone();
        let respond = move |Json(body): Json<Value>| {
            let s = s.clone();
            async move {
                s.generations.lock().unwrap().push(body);
                let events = [
                    json!({"type": "response.created"}),
                    json!({"type": "response.output_text.delta", "delta": output}),
                    json!({"type": "response.output_text.done", "text": output}),
                ];
                let body: String = events
                    .iter()
                    .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
                    .collect();
                ([("content-type", "text/event-stream")], body)
            }
        };
        let s = seen.clone();
        // Reading the upload keeps the connection reusable, as a real server does.
        let transcribe = move |_upload: axum::body::Bytes| {
            let s = s.clone();
            async move {
                *s.uploads.lock().unwrap() += 1;
                Json(json!({"text": "hello world"}))
            }
        };
        let app = axum::Router::new()
            .route(
                "/health",
                get(|| async { Json(json!({"status": "ok", "services": {}})) }),
            )
            .route("/v1/systemone", post(decide))
            .route("/v1/responses", post(respond))
            .route("/v1/chat/completions", post(chat))
            .route("/v1/audio/transcriptions", post(transcribe));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (Client::new(&base, None), seen)
    }

    /// Every capability on one server of ours.
    fn routes(client: Client) -> Routes {
        let route = |model: &str| Some(Route::ours(client.clone(), "embedded", model));
        Routes {
            speech: route("parakeet"),
            realtime: route("parakeet"),
            decision: route("jev"),
            generation: route("jev"),
        }
    }

    fn builtin() -> Arc<FlowTree> {
        let tree = FlowTree::load(&crate::flow::defaults::builtin(), &Catalog::default());
        assert!(tree.is_valid(), "{:?}", tree.errors);
        Arc::new(tree)
    }

    /// The client's desk over fakes: text goes to `sink`, when there is one, and there is no
    /// one to ask and nothing to read.
    fn desk(sink: Option<&RecordingSink>) -> Arc<dyn Desk> {
        let desk = LocalDesk::default();
        Arc::new(match sink {
            Some(sink) => desk.with_sink(sink.shared()),
            None => desk,
        })
    }

    fn env(client: Client, sink: Option<&RecordingSink>) -> Env {
        Env {
            routes: routes(client),
            flows: builtin(),
            settings: Settings::default(),
            desk: desk(sink),
            investigator: None,
            tools: None,
            machines: Arc::new(Runtime::new()),
        }
    }

    fn context(selection: Option<&str>) -> ContextSnapshot {
        ContextSnapshot {
            app: AppInfo {
                process_name: "notepad.exe".into(),
                ..AppInfo::default()
            },
            window: WindowInfo {
                title: "notes.txt - Notepad".into(),
                handle: Some(7),
                ..WindowInfo::default()
            },
            focused: Some(Element {
                role: "Document".into(),
                is_editable: true,
                selection: selection.map(String::from),
                ..Element::default()
            }),
            ..ContextSnapshot::default()
        }
    }

    fn one_second_of_audio() -> (mpsc::UnboundedReceiver<AudioEvent>, oneshot::Receiver<()>) {
        let (audio, receiver) = mpsc::unbounded_channel();
        audio.send(AudioEvent::Level([3; 5])).unwrap();
        audio
            .send(AudioEvent::Chunk(vec![1000; SAMPLE_RATE as usize]))
            .unwrap();
        audio.send(AudioEvent::Ended).unwrap();
        let (_finish, finished) = oneshot::channel();
        (receiver, finished)
    }

    async fn take_at(env: &Env, selection: Option<&str>, entry: Option<&str>) -> Trace {
        let (audio, finish) = one_second_of_audio();
        let (updates, _) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: context(selection),
            entry: entry.map(String::from),
        };
        run_take(env, start, audio, finish, &updates).await
    }

    async fn take(env: &Env, selection: Option<&str>) -> Trace {
        take_at(env, selection, None).await
    }

    #[tokio::test]
    async fn realtime_404_falls_back_to_batch_upload() {
        let (client, seen) = server(prefer(&["dictation", "verbatim"]), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let trace = take(&env(client, Some(&sink)), None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.transcription, Some(TranscriptionPath::Upload));
        assert_eq!(*seen.uploads.lock().unwrap(), 1);
        assert!(trace.notes[0].contains("Realtime"), "{:?}", trace.notes);
        assert_eq!(trace.transcript, "hello world");
    }

    /// A route to a System One that is not ours, as Jev on OpenRouter is.
    fn external(client: Client) -> Route {
        Route {
            client,
            model: "typesafe/jev-1.13".into(),
            provider: "openrouter".into(),
            profile: Profile::external(),
        }
    }

    #[tokio::test]
    async fn each_capability_goes_to_its_own_provider() {
        // Speech on one provider, decisions on one that is not ours, generation on a third.
        let (speech, heard) = server(prefer(&[]), "unused").await;
        let (decision, decided) = server(prefer(&["dictation", "rewrite"]), "unused").await;
        let (generation, written) = server(prefer(&[]), "Dear team, hello world.").await;
        let sink = RecordingSink::new(Some(7));
        let env = Env {
            routes: Routes {
                speech: Some(Route::ours(speech, "box", "parakeet")),
                realtime: None,
                decision: Some(external(decision)),
                generation: Some(Route::ours(generation, "embedded", "gemma")),
            },
            ..env(Client::new("http://127.0.0.1:9", None), Some(&sink))
        };
        let trace = take(&env, Some("hi all")).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.output, "Dear team, hello world.");
        // Each provider got its own capability, asked for its own model, and nothing else.
        let count = |seen: &Seen| {
            (
                *seen.uploads.lock().unwrap(),
                seen.decisions.lock().unwrap().len(),
                seen.generations.lock().unwrap().len(),
            )
        };
        assert_eq!(count(&heard), (1, 0, 0));
        assert_eq!(count(&decided), (0, 1, 0));
        assert_eq!(count(&written), (0, 0, 1));
        assert_eq!(
            decided.decisions.lock().unwrap()[0]["model"],
            "typesafe/jev-1.13"
        );
        assert_eq!(written.generations.lock().unwrap()[0]["model"], "gemma");
        // With no Realtime route the take uploads, without trying to stream first.
        assert_eq!(trace.transcription, Some(TranscriptionPath::Upload));
        assert_eq!(trace.notes, Vec::<String>::new());
    }

    /// A root that chooses between two agents by the model, with our System One extensions set.
    const TWO_AGENTS: &[(&str, &str)] = &[
        ("root.toml", "steps = 4\nsamples = 2"),
        (
            "root.fsm",
            "fsm App {\n[*] --> idle\nidle --> typing : said [else]\nidle --> asking : said\ntyping --> idle\nasking --> idle\n}",
        ),
        ("typing/agent.toml", "description = \"Dictation\""),
        (
            "typing/agent.fsm",
            "fsm Typing {\n[*] --> idle\nidle --> type : said\ntype --> idle\n}",
        ),
        ("typing/type/transcript.toml", ""),
        (
            "asking/agent.toml",
            "description = \"A question for the assistant\"",
        ),
        (
            "asking/agent.fsm",
            "fsm Asking {\n[*] --> idle\nidle --> answer : said\nanswer --> idle\n}",
        ),
        ("asking/answer/transcript.toml", "output = \"bubble\""),
    ];

    #[tokio::test]
    async fn a_provider_without_our_extensions_gets_none_and_the_trace_says_so() {
        let (client, seen) = server(prefer(&["asking"]), "unused").await;
        let elsewhere = Env {
            routes: Routes {
                decision: Some(external(client.clone())),
                ..routes(client.clone())
            },
            flows: tree_of(TWO_AGENTS),
            ..env(client, None)
        };
        let trace = say(&elsewhere, 1, "what time is it").await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(moves(&trace)[0], "idle said → asking");
        assert_eq!(
            trace.notes,
            [
                "steps dropped: openrouter/typesafe/jev-1.13 does not support it",
                "samples dropped: openrouter/typesafe/jev-1.13 does not support it",
            ]
        );
        let sent = seen.decisions.lock().unwrap()[0].clone();
        assert!(sent.get("steps").is_none() && sent.get("samples").is_none());
        // The trace keeps the request as it went.
        let asked = trace.flow.iter().find_map(|s| s.decision.as_ref()).unwrap();
        assert_eq!((asked.request.steps, asked.request.samples), (None, None));
        // Our own System One takes them, and nothing is noted.
        let (client, seen) = server(prefer(&["asking"]), "unused").await;
        let ours = Env {
            flows: tree_of(TWO_AGENTS),
            ..env(client, None)
        };
        let trace = say(&ours, 1, "what time is it").await;
        assert_eq!(trace.notes, Vec::<String>::new());
        let sent = seen.decisions.lock().unwrap()[0].clone();
        assert_eq!(
            (sent["steps"].clone(), sent["samples"].clone()),
            (json!(4), json!(2))
        );
    }

    #[tokio::test]
    async fn questions_over_the_provider_s_limit_are_asked_in_several_requests() {
        // The built-in tree asks the root's question and the action's in one request; a
        // provider that takes one a request gets two, and the take ends the same.
        let (client, seen) = server(prefer(&["dictation", "verbatim"]), "unused").await;
        let mut one = external(client.clone());
        one.profile.max_questions = Some(1);
        let sink = RecordingSink::new(Some(7));
        let env = Env {
            routes: Routes {
                decision: Some(one),
                ..routes(client.clone())
            },
            ..env(client, Some(&sink))
        };
        let trace = say(&env, 1, "hello world").await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(
            trace.route(),
            "dictation/dictate → dictation/dictate/notes → _actions/verbatim"
        );
        assert_eq!(sink.requests()[0].text, "hello world");
        assert_eq!(
            trace.notes,
            ["2 questions asked in 2 requests: openrouter/typesafe/jev-1.13 takes 1 in one"]
        );
        let decisions = seen.decisions.lock().unwrap();
        let asked: Vec<Vec<&String>> = decisions
            .iter()
            .map(|d| d["questions"].as_object().unwrap().keys().collect())
            .collect();
        assert_eq!(asked, [["q00"], ["q01"]]);
    }

    #[tokio::test]
    async fn the_decision_provider_says_how_sure_its_model_must_be() {
        // The model chooses `asking` at 0.65. Our models are sure from 0.7, so the root takes
        // its `[else]`.
        let (client, _) = server(prefer_with(&["asking"], 0.65), "unused").await;
        let unsure = Env {
            flows: tree_of(TWO_AGENTS),
            ..env(client.clone(), None)
        };
        let trace = say(&unsure, 1, "what time is it").await;
        assert_eq!(moves(&trace)[0], "idle said → typing");
        assert_eq!(trace.machine[0].how, "unsure (asking 0.65): the fallback");
        // A provider whose model is sure from 0.6 has its choice taken.
        let mut calibrated = Route::ours(client.clone(), "box", "jev");
        calibrated.profile.min_probability = 0.6;
        let routes = Routes {
            decision: Some(calibrated),
            ..routes(client.clone())
        };
        let sure = Env {
            routes: routes.clone(),
            flows: tree_of(TWO_AGENTS),
            ..env(client.clone(), None)
        };
        let trace = say(&sure, 1, "what time is it").await;
        assert_eq!(moves(&trace)[0], "idle said → asking");
        assert_eq!(trace.machine[0].how, "model 0.65");
        // A machine that sets its own is not moved by the provider's.
        let mut files = TWO_AGENTS.to_vec();
        files[0] = ("root.toml", "min_probability = 0.9");
        let own = Env {
            routes,
            flows: tree_of(&files),
            ..env(client, None)
        };
        let trace = say(&own, 1, "what time is it").await;
        assert_eq!(moves(&trace)[0], "idle said → typing");
    }

    #[tokio::test]
    async fn words_needing_no_edits_are_typed_after_one_merged_decision() {
        let (client, seen) = server(prefer(&["dictation", "verbatim"]), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let trace = take(&env(client, Some(&sink)), None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert!(seen.generations.lock().unwrap().is_empty());
        assert_eq!(trace.output, "hello world");
        let delivered = sink.requests();
        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].action, Action::Insert);
        assert_eq!(delivered[0].text, "hello world");
        assert_eq!(
            trace.route(),
            "dictation/dictate → dictation/dictate/notes → _actions/verbatim"
        );
        // The root and the action are asked together: one decision call for the take.
        let decisions = seen.decisions.lock().unwrap();
        assert_eq!(decisions.len(), 1);
        let questions = decisions[0]["questions"].as_object().unwrap();
        assert_eq!(questions.len(), 2);
        let actions = &questions["q01"]["criteria"];
        assert!(actions.get("verbatim").is_some() && actions.get("insert").is_some());
        assert!(
            actions.get("replace").is_none(),
            "nothing is selected: {actions}"
        );
        let notes = trace
            .flow
            .iter()
            .find(|s| s.node == "dictation/dictate/notes")
            .unwrap();
        assert!(
            notes.how.as_deref().unwrap().starts_with("asked ahead"),
            "{:?}",
            notes.how
        );
    }

    #[tokio::test]
    async fn a_rewrite_of_the_selection_generates_with_the_branch_instructions() {
        let (client, seen) =
            server(prefer(&["dictation", "rewrite"]), "Dear team, hello world.").await;
        let sink = RecordingSink::new(Some(7));
        let trace = take(&env(client, Some(&sink)), Some("hi all")).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.leaf.as_ref().unwrap().action, Action::Rewrite);
        assert_eq!(trace.output, "Dear team, hello world.");
        let generation = &seen.generations.lock().unwrap()[0];
        let instructions = generation["instructions"].as_str().unwrap();
        assert!(instructions.contains("Tidy prose"), "{instructions}");
        assert!(
            instructions.contains("exact text to type"),
            "{instructions}"
        );
        assert!(
            instructions.contains("rewrite the given text"),
            "{instructions}"
        );
        assert!(generation["input"].as_str().unwrap().contains("hi all"));
        assert_eq!(generation["stream"], true);
        let delivered = sink.requests();
        assert_eq!(delivered[0].action, Action::Rewrite);
        assert!(!delivered[0].select_all);
    }

    #[tokio::test]
    async fn every_decision_and_the_generation_report_their_stages_in_order() {
        let (client, _) = server(prefer(&["dictation", "rewrite"]), "Dear team.").await;
        let sink = RecordingSink::new(Some(7));
        let env = env(client, Some(&sink));
        let (audio, finish) = one_second_of_audio();
        let (updates, mut shown) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: context(Some("hi all")),
            entry: None,
        };
        let trace = run_take(&env, start, audio, finish, &updates).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        let mut stages = Vec::new();
        while let Ok(update) = shown.try_recv() {
            match update {
                Update::Stage(stage) => stages.push(format!(
                    "{:?} {} [{}]",
                    stage.kind,
                    stage.label,
                    stage.choices.join(" ")
                )),
                Update::StageDone { detail, chosen, ok } => {
                    stages.push(format!("→ {} {detail} {ok}", chosen.unwrap_or_default()))
                }
                _ => {}
            }
        }
        assert_eq!(
            stages,
            [
                "Deciding what to do [assistant dictation]",
                "→ dictation 0.90 true",
                "Deciding dictate [any notes]",
                "→ notes rules true",
                "Deciding notes [insert replace rewrite verbatim]",
                "→ rewrite 0.90 true",
                "Writing text []",
                "→  2 words true",
            ]
        );
    }

    #[tokio::test]
    async fn a_question_is_answered_in_the_bubble_and_never_typed() {
        let (client, seen) = server(prefer(&["assistant"]), "It is five o'clock.").await;
        let sink = RecordingSink::new(Some(7));
        let trace = take(&env(client, Some(&sink)), None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        let leaf = trace.leaf.as_ref().unwrap();
        assert_eq!(
            (leaf.node.as_str(), leaf.output),
            ("assistant/ask/any", Output::Bubble)
        );
        assert_eq!(trace.delivery, Some(DeliveryOutcome::Shown));
        assert!(sink.requests().is_empty());
        assert!(sink.clipboard().is_none());
        let generation = &seen.generations.lock().unwrap()[0];
        assert!(
            generation["instructions"]
                .as_str()
                .unwrap()
                .contains("small bubble")
        );
        assert!(
            generation["input"]
                .as_str()
                .unwrap()
                .contains("The user asked: hello world")
        );
    }

    #[tokio::test]
    async fn a_hotkey_entry_starts_below_the_root_without_its_decision() {
        let (client, seen) = server(prefer(&["dictation"]), "An answer.").await;
        let trace = take_at(&env(client, None), None, Some("assistant")).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.entry, "assistant");
        assert_eq!(trace.leaf.as_ref().unwrap().node, "assistant/ask/any");
        assert!(
            seen.decisions.lock().unwrap().is_empty(),
            "ask chooses by rules"
        );
    }

    #[tokio::test]
    async fn an_unsure_root_decision_takes_the_fallback() {
        let (client, _) = server(prefer_with(&["assistant", "verbatim"], 0.3), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let trace = take(&env(client, Some(&sink)), None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        let root = &trace.flow[0];
        assert_eq!(root.chosen.as_deref(), Some("dictation"));
        assert!(
            root.how.as_deref().unwrap().contains("unsure"),
            "{:?}",
            root.how
        );
        assert_eq!(sink.requests()[0].text, "hello world");
    }

    #[tokio::test]
    async fn a_changed_window_leaves_the_text_on_the_clipboard() {
        let (client, _) = server(prefer(&["dictation", "verbatim"]), "unused").await;
        let sink = RecordingSink::new(Some(99));
        let trace = take(&env(client, Some(&sink)), None).await;
        assert!(matches!(
            trace.delivery,
            Some(DeliveryOutcome::OnClipboard { .. })
        ));
        assert!(sink.requests().is_empty());
        assert_eq!(sink.clipboard().as_deref(), Some("hello world"));
    }

    #[tokio::test]
    async fn a_press_too_short_for_speech_is_dropped_without_requests() {
        let (client, seen) = server(prefer(&["dictation"]), "unused").await;
        let env = env(client, None);
        let (audio, receiver) = mpsc::unbounded_channel();
        audio.send(AudioEvent::Chunk(vec![0; 100])).unwrap();
        audio.send(AudioEvent::Ended).unwrap();
        let (_finish, finished) = oneshot::channel();
        let (updates, _) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: context(None),
            entry: None,
        };
        let trace = run_take(&env, start, receiver, finished, &updates).await;
        assert_eq!(trace.error, None);
        assert!(
            trace.notes.iter().any(|n| n.contains("Too short")),
            "{:?}",
            trace.notes
        );
        assert_eq!(*seen.uploads.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn a_guarded_branch_catches_its_keyword_and_lazy_values_stay_unasked() {
        let tree = FlowTree::load(
            &Memory::new(
                "test",
                [
                    ("decide.toml", "fallback = \"type\""),
                    ("type/transcript.toml", "description = \"Dictation\""),
                    (
                        "note/transcript.toml",
                        "description = \"Notes\"\noutput = \"clipboard\"\n[when]\ntranscript = \"(?i)^note\"",
                    ),
                ],
            ),
            &Catalog::default(),
        );
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let (client, seen) = server(prefer(&["note"]), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let env = Env {
            flows: Arc::new(tree),
            ..env(client, Some(&sink))
        };
        // "hello world" does not start with "note": only one branch applies, so no model call.
        let trace = take(&env, None).await;
        assert_eq!(trace.leaf.as_ref().unwrap().node, "type");
        assert!(seen.decisions.lock().unwrap().is_empty());
        assert_eq!(
            trace.flow[0].how.as_deref(),
            Some("the only branch that applies")
        );
        let note = trace.flow[0]
            .branches
            .iter()
            .find(|b| b.name == "note")
            .unwrap();
        assert!(!note.passed);
        assert_eq!(note.checks[0].value.as_deref(), Some("hello world"));
    }

    #[tokio::test]
    async fn extracts_read_the_interface_with_no_model_and_feed_decisions_and_prompts() {
        let tree = FlowTree::load(
            &Memory::new(
                "test",
                [
                    (
                        "decide.toml",
                        "fallback = \"other\"\n\
                         [extract.channels]\n\
                         xpath = \"//TreeItem[.//Group[has-class(@class, 'p-channel_sidebar__channel')]]/@name\"\n\
                         as = \"list\"\n\
                         [extract.last]\n\
                         xpath = \"string((//ListItem[.//Text])[last()]//Text)\"\n\
                         lazy = true\n\
                         [extract.unused]\n\
                         xpath = \"//Slider\"\n\
                         lazy = true",
                    ),
                    (
                        "reply/generate.toml",
                        "description = \"Replies\"\noutput = \"bubble\"\n\
                         prompt = \"Last: {last}. Channels: {channels}. Said: {transcript}\"\n\
                         [extract.channels]\n\
                         xpath = \"//TreeItem[.//Group[has-class(@class, 'p-channel_sidebar__channel')]]/@name\"\n\
                         as = \"list\"",
                    ),
                    ("other/transcript.toml", "description = \"Anything else\""),
                ],
            ),
            &Catalog::default(),
        );
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let (client, seen) = server(prefer(&["reply"]), "Sure.").await;
        let recorded: jevons_desktop_core::recorded::RecordedTree =
            serde_json::from_str(include_str!("../../../examples/desktop/trees/slack.json"))
                .unwrap();
        let env = Env {
            flows: Arc::new(tree),
            desk: Arc::new(LocalDesk::default().with_reader(Arc::new(
                jevons_desktop_core::reader::Reader::new(
                    Arc::new(jevons_desktop_core::recorded::RecordedInspector::new(
                        recorded,
                    )),
                    crate::context::Privacy::default(),
                ),
            ))),
            ..env(client, None)
        };
        let (audio, finish) = one_second_of_audio();
        let (updates, _) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: ContextSnapshot {
                app: AppInfo {
                    process_name: "slack.exe".into(),
                    ..AppInfo::default()
                },
                window: WindowInfo {
                    title: "general (Channel) - Acme - Slack".into(),
                    handle: Some(7),
                    ..WindowInfo::default()
                },
                ..ContextSnapshot::default()
            },
            entry: None,
        };
        let trace = run_take(&env, start, audio, finish, &updates).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.leaf.as_ref().unwrap().node, "reply");
        let root = &trace.flow[0].extracts;
        assert_eq!(
            root.len(),
            1,
            "lazy extracts wait until something uses them"
        );
        assert_eq!(
            root[0].answer,
            json!([
                "general",
                "launch 3 unread messages",
                "random",
                "Ana Silva",
                "Bo Chen"
            ])
        );
        let decision = serde_json::to_string(&seen.decisions.lock().unwrap()[0]).unwrap();
        assert!(
            decision.contains("Ana Silva"),
            "the decision reads the extract: {decision}"
        );
        let reply = &trace.flow[1].extracts;
        assert_eq!(
            reply
                .iter()
                .map(|e| (e.name.as_str(), e.reused))
                .collect::<Vec<_>>(),
            [("channels", true), ("last", false)]
        );
        assert_eq!(
            reply[1].answer,
            json!("Can someone review the release notes?")
        );
        let generation = serde_json::to_string(&seen.generations.lock().unwrap()[0]).unwrap();
        assert!(
            generation.contains("Last: Can someone review the release notes?. Channels: [")
                && generation.contains("Said: hello world"),
            "{generation}"
        );
        assert!(
            !trace
                .flow
                .iter()
                .any(|f| f.extracts.iter().any(|e| e.name == "unused"))
        );
    }

    /// A take of `words` in `app`, from text, through the built-in tree.
    async fn said(env: &Env, app: &str, words: &str) -> Trace {
        let (updates, _) = mpsc::unbounded_channel();
        let mut context = context(None);
        context.app.process_name = app.into();
        let start = TakeStart {
            id: 1,
            context,
            entry: None,
        };
        run_transcript(env, start, words, &updates).await
    }

    #[tokio::test]
    async fn words_starting_with_pregunta_are_a_question_with_no_root_decision() {
        let (client, seen) = server(prefer(&["dictation"]), "Paul means Friday.").await;
        let trace = said(
            &env(client, None),
            "slack.exe",
            "Pregunta, ¿qué quiere decir Paul?",
        )
        .await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.flow[0].chosen.as_deref(), Some("assistant"));
        assert_eq!(
            trace.flow[0].how.as_deref(),
            Some("preferred: its transcript rule passed")
        );
        let ask = trace.flow[0]
            .branches
            .iter()
            .find(|b| b.name == "assistant")
            .unwrap();
        assert!(ask.preferred && ask.prefer[0].passed);
        assert!(
            seen.decisions
                .lock()
                .unwrap()
                .iter()
                .all(|d| !d.to_string().contains("\"dictation\"")),
            "the root was not asked"
        );
        assert_eq!(trace.output, "Paul means Friday.");
    }

    #[tokio::test]
    async fn in_a_terminal_the_words_are_dictated_and_never_rewritten() {
        let (client, seen) = server(prefer(&["assistant", "rewrite", "verbatim"]), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let trace = said(
            &env(client, Some(&sink)),
            "WindowsTerminal.exe",
            "can you check why the build fails?",
        )
        .await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(
            trace.flow[0].how.as_deref(),
            Some("preferred: its app rule passed")
        );
        assert_eq!(
            trace.route(),
            "dictation/dictate → dictation/dictate/terminal → _actions/verbatim"
        );
        let decisions = seen.decisions.lock().unwrap();
        assert_eq!(decisions.len(), 1, "only the action is asked");
        let actions = decisions[0]["questions"]["q00"]["criteria"]
            .as_object()
            .unwrap();
        assert_eq!(actions.keys().collect::<Vec<_>>(), ["insert", "verbatim"]);
        assert_eq!(
            sink.requests()[0].text,
            "can you check why the build fails?"
        );
    }

    #[tokio::test]
    async fn the_root_takes_the_model_s_choice_from_seventy_percent_and_dictates_below() {
        let (client, seen) = server(prefer_with(&["assistant"], 0.86), "It means yes.").await;
        let trace = said(&env(client, None), "notepad.exe", "what does this mean?").await;
        assert_eq!(trace.flow[0].chosen.as_deref(), Some("assistant"));
        assert_eq!(trace.flow[0].how.as_deref(), Some("model 0.86"));
        let state = seen.decisions.lock().unwrap()[0]["state"].clone();
        assert!(
            state.as_str().unwrap().contains("(accepts typing)"),
            "{state}"
        );
        let (client, _) = server(prefer_with(&["assistant"], 0.65), "unused").await;
        let trace = said(&env(client, None), "notepad.exe", "what does this mean?").await;
        assert_eq!(trace.flow[0].chosen.as_deref(), Some("dictation"));
        assert_eq!(
            trace.flow[0].how.as_deref(),
            Some("unsure (assistant 0.65): the fallback")
        );
    }

    #[tokio::test]
    async fn in_slack_the_built_in_ask_branch_answers_from_the_root_s_slack_extracts() {
        let tree = FlowTree::load(&crate::flow::defaults::builtin(), &Catalog::default());
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let (client, seen) = server(prefer(&["assistant"]), "Five.").await;
        let recorded: jevons_desktop_core::recorded::RecordedTree =
            serde_json::from_str(include_str!("../../../examples/desktop/trees/slack.json"))
                .unwrap();
        let env = Env {
            flows: Arc::new(tree),
            desk: Arc::new(LocalDesk::default().with_reader(Arc::new(
                jevons_desktop_core::reader::Reader::new(
                    Arc::new(jevons_desktop_core::recorded::RecordedInspector::new(
                        recorded,
                    )),
                    crate::context::Privacy::default(),
                ),
            ))),
            ..env(client, None)
        };
        let (audio, finish) = one_second_of_audio();
        let (updates, _) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: ContextSnapshot {
                app: AppInfo {
                    process_name: "slack.exe".into(),
                    ..AppInfo::default()
                },
                window: WindowInfo {
                    title: "general (Channel) - Acme - Slack".into(),
                    handle: Some(7),
                    ..WindowInfo::default()
                },
                ..ContextSnapshot::default()
            },
            entry: None,
        };
        let trace = run_take(&env, start, audio, finish, &updates).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.leaf.as_ref().unwrap().node, "assistant/ask/slack");
        let extracts: Vec<_> = trace.flow.iter().flat_map(|f| &f.extracts).collect();
        let mut names: Vec<&str> = extracts.iter().map(|e| e.name.as_str()).collect();
        names.sort();
        assert_eq!(
            names,
            ["slack_channels", "slack_conversation", "slack_messages"]
        );
        assert!(extracts.iter().all(|e| e.note.is_none()), "{extracts:?}");
        let generation = serde_json::to_string(&seen.generations.lock().unwrap()[0]).unwrap();
        assert!(
            generation.contains("in the conversation general.")
                && generation.contains("Bo Chen")
                && generation.contains("Can someone review the release notes?"),
            "{generation}"
        );
    }

    #[tokio::test]
    async fn without_approved_automations_the_run_branch_is_no_candidate() {
        let (client, seen) =
            server(prefer(&["automations", "dictation", "verbatim"]), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let trace = take(&env(client, Some(&sink)), None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        let run = trace.flow[0]
            .branches
            .iter()
            .find(|b| b.name == "automations")
            .unwrap();
        // The agent's only state runs an automation, and none is approved: it has nothing to
        // do with the take, so the root does not offer it.
        assert!(!run.passed);
        assert_eq!(
            (run.checks[0].rule, run.checks[0].value.as_deref()),
            ("agent", Some("nothing"))
        );
        let asked = serde_json::to_string(&seen.decisions.lock().unwrap()[0]).unwrap();
        assert!(
            !asked.contains("\"automations\""),
            "the model is never offered it: {asked}"
        );
        assert_ne!(trace.leaf.unwrap().node, "automations/run");
    }

    #[tokio::test]
    async fn a_run_node_fills_an_automation_s_arguments_from_the_words_and_runs_it() {
        let dir = std::env::temp_dir().join(format!("jevons-run-node-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("open-channel")).unwrap();
        std::fs::write(
            dir.join("open-channel/automation.toml"),
            "description = \"Opens a Slack channel\"\napps = [\"slack.exe\"]\n[args.channel]\ndescription = \"The channel's name\"\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("open-channel/script.rhai"),
            "find(\"//TreeItem[.//Text[@name = $channel]]\").invoke();\n`opened ${args.channel}`\n",
        )
        .unwrap();
        let (demonstration, _, _) = jevons_desktop_core::fake::slack_demonstration();
        let replay = Arc::new(jevons_desktop_core::recorded::ReplayActor::new(
            demonstration,
        ));
        let host = Arc::new(jevons_desktop_core::automation::host::AutomationHost::new(
            &dir,
            jevons_desktop_core::config::AutomationSettings::default(),
            replay.clone(),
            replay.clone(),
        ));
        let version = host.list()[0].version.clone();
        let mut settings = jevons_desktop_core::config::AutomationSettings::default();
        settings.approved.insert("open-channel".into(), version);
        settings.unconfirmed.push("open-channel".into());
        host.set_settings(settings);
        let none = std::collections::BTreeMap::new();
        let desk: Arc<dyn Desk> = Arc::new(LocalDesk::default().with_automations(host));
        let tools = Arc::new(
            ToolHost::new(&none, &std::collections::BTreeMap::new()).with_desk(desk.clone()),
        );
        let tree = FlowTree::load(
            &Memory::new("test", [("run.toml", "description = \"Runs automations\"")]),
            &tools.catalog(),
        );
        assert!(tree.is_valid(), "{:?}", tree.errors);
        // The generation writes the argument from what the user said.
        let (client, _) = server(prefer(&[]), "random").await;
        let env = Env {
            flows: Arc::new(tree),
            tools: Some(tools),
            desk,
            ..env(client, None)
        };
        let trace = take(&env, None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.calls[0].tool, "script:open-channel");
        assert_eq!(trace.calls[0].arguments, json!({"channel": "random"}));
        assert!(trace.output.contains("opened random"), "{}", trace.output);
        assert_eq!(replay.done(), 1, "the automation acted");
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn tool_host() -> Arc<ToolHost> {
        let config: crate::config::ServerConfig = toml::from_str(
            r#"
[tools.note]
kind = "command"
description = "Saves a note"
program = "notes"
args = ["{title}", "{folder}", "{body}"]
arguments = { title = "A title", folder = "Where", body = "The note" }

[tools.search]
kind = "open"
description = "Searches the web"
url = "https://duckduckgo.com/?q={query}"
arguments = { query = "What to find" }
confirm = false
"#,
        )
        .unwrap();
        Arc::new(ToolHost::new(&config.tools, &config.mcp).dry_run())
    }

    fn tree_of(files: &[(&str, &str)]) -> Arc<FlowTree> {
        let tree = FlowTree::load(
            &Memory::new("test", files.iter().copied()),
            &tool_host().catalog(),
        );
        assert!(tree.is_valid(), "{:?}", tree.errors);
        Arc::new(tree)
    }

    #[tokio::test]
    async fn a_tool_node_fills_its_arguments_asks_and_answers_with_the_result() {
        let (client, seen) = server(prefer(&["work"]), "Launch moved").await;
        let (confirm, mut asked) =
            mpsc::unbounded_channel::<jevons_desktop_core::confirm::Confirmation>();
        let approver = tokio::spawn(async move {
            let call = asked.recv().await.unwrap();
            let tool = call.tool.clone();
            call.reply.send(true).unwrap();
            tool
        });
        let env = Env {
            flows: tree_of(&[(
                "tool.toml",
                "tool = \"note\"\n[args.title]\ngenerate = \"A short title\"\n[args.folder]\nchoose = { inbox = \"Unsorted\", work = \"About work\" }\n[args.body]\nvalue = \"{transcript}\"",
            )]),
            tools: Some(tool_host()),
            desk: Arc::new(LocalDesk::default().with_confirmer(Arc::new(
                jevons_desktop_core::confirm::ChannelConfirmer::new(confirm),
            ))),
            ..env(client, None)
        };
        let trace = take(&env, None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(approver.await.unwrap(), "note");
        let call = &trace.calls[0];
        assert_eq!(call.tool, "note");
        assert_eq!(call.confirmed, Some(true));
        assert_eq!(
            call.arguments,
            json!({"title": "Launch moved", "folder": "work", "body": "hello world"})
        );
        assert!(trace.output.contains("dry_run"), "{}", trace.output);
        assert_eq!(trace.leaf.as_ref().unwrap().output, Output::Bubble);
        // The labels were one System One read; the title one generation.
        assert_eq!(seen.decisions.lock().unwrap().len(), 1);
        assert_eq!(seen.generations.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn an_unconfirmed_tool_call_does_not_run() {
        let (client, _) = server(prefer(&["work"]), "Title").await;
        let env = Env {
            flows: tree_of(&[(
                "tool.toml",
                "tool = \"note\"\n[args.title]\nvalue = \"x\"\n[args.folder]\nvalue = \"y\"\n[args.body]\nvalue = \"z\"",
            )]),
            tools: Some(tool_host()),
            ..env(client, None)
        };
        let trace = take(&env, None).await;
        assert!(
            trace.error.as_deref().unwrap().contains("not confirmed"),
            "{:?}",
            trace.error
        );
        assert_eq!(trace.calls[0].confirmed, Some(false));
    }

    /// A root machine that dictates, or starts a search task that waits for what the user says
    /// about its results.
    /// A root with two agents: `typing` types the words as heard, and `research` starts a
    /// search, a task that waits for what the user says about its results.
    const SEARCH_TASK: &[(&str, &str)] = &[
        ("root.toml", "tools = [\"search\"]"),
        (
            "root.fsm",
            "fsm App {\n[*] --> idle\nidle --> typing : said [else]\nidle --> research : said\ntyping --> idle\nresearch --> idle\n}",
        ),
        ("typing/agent.toml", "description = \"Dictation\""),
        (
            "typing/agent.fsm",
            "fsm Typing {\n[*] --> idle\nidle --> type : said\ntype --> idle\n}",
        ),
        ("typing/type/transcript.toml", ""),
        (
            "research/agent.toml",
            "description = \"The user wants to search the web, or says something about a search\"\ntools = [\"search\"]",
        ),
        (
            "research/agent.fsm",
            "fsm Research {\n[*] --> idle\nidle --> find : said\nfind --> idle\n}",
        ),
        (
            "research/find/task.toml",
            "description = \"The user wants to search the web\"\ntools = [\"search\"]",
        ),
        (
            "research/find/task.fsm",
            "fsm Find {\ntimer idle = 40 -> quiet\n[*] --> searching\nstate searching: \"Searching\"\nstate answering: \"The results are in the bubble\"\nsearching --> answering\nsearching --> [*] : failed\nanswering --> opening : said [the user wants a result opened]\nanswering --> [*] : said [the user is done with the results]\nanswering --> [*] : quiet\nopening --> [*]\n}",
        ),
        (
            "research/find/searching/tool.toml",
            "tool = \"search\"\noutput = \"none\"\n[args.query]\nvalue = \"{transcript}\"",
        ),
        (
            "research/find/answering/generate.toml",
            "output = \"bubble\"\nprompt = \"Results: {searching}. Question: {transcript}\"",
        ),
        (
            "research/find/opening/tool.toml",
            "tool = \"search\"\noutput = \"none\"\n[args.query]\nvalue = \"{transcript}\"",
        ),
    ];

    /// The files of one machine with its states' work, as the agent `main` of a root that hands
    /// it every take: for tests of what an agent and its tasks do.
    fn agent_tree(files: &[(&str, &str)]) -> Arc<FlowTree> {
        let tools = files
            .iter()
            .find(|(path, _)| *path == "root.toml")
            .and_then(|(_, text)| text.lines().find(|l| l.starts_with("tools")))
            .unwrap_or_default();
        let mut wrapped = vec![
            ("root.toml".to_string(), tools.to_string()),
            (
                "root.fsm".to_string(),
                "fsm Root {\n[*] --> idle\nidle --> main : said\nmain --> idle\n}".to_string(),
            ),
        ];
        for (path, text) in files {
            let path = match *path {
                "root.toml" => "main/agent.toml".to_string(),
                "root.fsm" => "main/agent.fsm".to_string(),
                other => format!("main/{other}"),
            };
            wrapped.push((path, text.to_string()));
        }
        let wrapped: Vec<(&str, &str)> = wrapped
            .iter()
            .map(|(p, t)| (p.as_str(), t.as_str()))
            .collect();
        tree_of(&wrapped)
    }

    /// The steps below the root: the agents' and the tasks'.
    fn inner(trace: &Trace) -> Vec<String> {
        trace
            .machine
            .iter()
            .filter(|s| s.machine != "/")
            .map(|s| format!("{} {} → {}", s.from, s.event, s.to))
            .collect()
    }

    fn task_env(client: Client, machines: Arc<Runtime>) -> Env {
        Env {
            flows: tree_of(SEARCH_TASK),
            tools: Some(tool_host()),
            machines,
            ..env(client, None)
        }
    }

    async fn say(env: &Env, id: u64, words: &str) -> Trace {
        let (updates, _) = mpsc::unbounded_channel();
        let start = TakeStart {
            id,
            context: context(None),
            entry: None,
        };
        run_transcript(env, start, words, &updates).await
    }

    fn moves(trace: &Trace) -> Vec<String> {
        trace
            .machine
            .iter()
            .map(|s| format!("{} {} → {}", s.from, s.event, s.to))
            .collect()
    }

    #[tokio::test]
    async fn a_task_waits_across_takes_and_the_model_takes_its_transitions() {
        let labels = &["research", "find-1", "opening"];
        let (client, seen) = server(prefer(labels), "Two crates fit.").await;
        let machines = Arc::new(Runtime::new());
        let env = task_env(client, machines.clone());
        let first = say(&env, 1, "search for state machine crates").await;
        assert_eq!(first.error, None, "{:?}", first.notes);
        // The root hands the take to the agent and is back in `idle` at once; the agent starts
        // the task and is back in `idle` too, while the task goes on.
        assert_eq!(
            moves(&first),
            [
                "idle said → research",
                "research done → idle",
                "idle said → find",
                "find done → idle",
                "[*] start → searching",
                "searching done → answering"
            ]
        );
        assert_eq!(first.machine[0].how, "model 0.90");
        assert_eq!(first.machine[2].how, "the only transition that applies");
        assert_eq!(
            first.calls[0].arguments,
            json!({"query": "search for state machine crates"})
        );
        // The answer read the search's result, and waits in the bubble.
        assert_eq!(first.delivery, Some(DeliveryOutcome::Shown));
        let prompt = seen.generations.lock().unwrap()[0]["input"].to_string();
        assert!(
            prompt.contains("Results: ") && prompt.contains("dry_run"),
            "{prompt}"
        );
        assert_eq!(machines.view().path(), "research › find › answering");
        assert!(machines.view().in_task());

        // What the user says next goes the same way: the root chooses the agent, the agent the
        // task that waits, the task its transition. One request asks all three.
        let second = say(&env, 2, "open the second one").await;
        assert_eq!(second.error, None, "{:?}", second.notes);
        assert_eq!(
            moves(&second),
            [
                "idle said → research",
                "research done → idle",
                "idle said → task find-1",
                "answering said → opening",
                "opening done → [*]"
            ]
        );
        let decisions = seen.decisions.lock().unwrap();
        assert_eq!(decisions.len(), 2, "one request per take");
        let questions = decisions[1]["questions"].as_object().unwrap();
        let keys = |q: &str| -> Vec<String> {
            questions[q]["criteria"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect()
        };
        assert_eq!(keys("q00"), ["research", "typing"]);
        // The agent: start another search, or the one that waits, by where it is.
        assert_eq!(keys("q01"), ["find", "find-1"]);
        let waiting = questions["q01"]["criteria"]["find-1"].as_str().unwrap();
        assert_eq!(
            waiting,
            "For the running task find (Find), now at answering: The results are in the bubble"
        );
        // The task: its own transitions.
        assert_eq!(keys("q02"), ["end", "opening"]);
        let asked = questions["q02"]["instructions"].as_str().unwrap();
        assert!(asked.contains("in the middle of a task, Find"), "{asked}");
        assert_eq!(
            second.calls[0].arguments,
            json!({"query": "open the second one"})
        );
        assert_eq!(machines.view().path(), "idle");
        assert!(!machines.view().in_task());
        let history: Vec<String> = machines
            .view()
            .history
            .iter()
            .map(|s| format!("{}:{}", s.machine, s.to))
            .collect();
        assert_eq!(history.len(), 11, "{history:?}");
        assert_eq!(history[4], "research/find:searching");
    }

    #[tokio::test]
    async fn an_unsure_take_leaves_a_waiting_task_where_it_was() {
        let machines = Arc::new(Runtime::new());
        let (client, _) = server(prefer(&["research"]), "Results.").await;
        let first = say(&task_env(client, machines.clone()), 1, "search for crates").await;
        assert_eq!(first.error, None, "{:?}", first.notes);
        // Sure of the agent and of the task, unsure of what the task should do.
        let unsure: Decider = Arc::new(|request: &Value| {
            let answers: serde_json::Map<String, Value> = request["questions"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, question)| {
                    let offered = question["criteria"].as_object().unwrap();
                    let (choice, sure) = ["research", "find-1"]
                        .iter()
                        .find(|l| offered.contains_key(**l))
                        .map_or(("opening", 0.4), |l| (*l, 0.9));
                    let answer = json!({"type": "choice", "choice": choice,
                        "probabilities": {choice: sure}, "confidence": sure});
                    (key.clone(), answer)
                })
                .collect();
            json!({"model": "jev", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": answers})
        });
        let (client, seen) = server(unsure, "unused").await;
        let env = Env {
            flows: machines.view().tree.unwrap(),
            ..task_env(client, machines.clone())
        };
        let second = say(&env, 2, "hmm, maybe").await;
        assert_eq!(
            inner(&second),
            ["idle said → task find-1", "answering said → answering"]
        );
        let stay = second.machine.last().unwrap();
        assert!(stay.how.contains("stayed"), "{:?}", second.machine);
        assert!(second.calls.is_empty(), "nothing ran");
        assert!(seen.generations.lock().unwrap().is_empty());
        assert_eq!(machines.view().path(), "research › find › answering");
        // Cancelling ends the task and runs nothing.
        assert_eq!(
            machines.cancel().await.as_deref(),
            Some("research › find › answering")
        );
        assert_eq!(machines.view().path(), "idle");
        assert_eq!(machines.cancel().await, None);
    }

    #[tokio::test]
    async fn a_timer_ends_a_task_that_waits_and_stale_timers_do_nothing() {
        let machines = Arc::new(Runtime::new());
        let (due, mut timers) = mpsc::unbounded_channel();
        machines.set_timers(due);
        let (client, _) = server(prefer(&["research"]), "Results.").await;
        let env = task_env(client, machines.clone());
        say(&env, 1, "search for crates").await;
        let fired = tokio::time::timeout(Duration::from_secs(5), timers.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fired.event, "quiet");
        assert!(machines.view().in_task());
        assert!(machines.view().waits_for(&fired));
        let (updates, _) = mpsc::unbounded_channel();
        let trace = machines
            .timer(&env, fired.clone(), 9, &updates)
            .await
            .expect("the task still waited in answering");
        // The task ends; its agent, which has no transition on `task_done`, stays in `idle`.
        assert_eq!(moves(&trace), ["answering quiet → [*]"]);
        assert_eq!(trace.take, 9);
        assert_eq!(machines.view().path(), "idle");
        // The state it was armed in is gone: the view says so before anything runs.
        assert!(!machines.view().in_task());
        assert!(!machines.view().waits_for(&fired));
        assert!(machines.timer(&env, fired, 10, &updates).await.is_none());
    }

    #[tokio::test]
    async fn a_session_runs_a_timer_s_take_by_itself_and_tells_the_client() {
        use crate::session::{Event, Session, TIMER_TAKES};
        let (client, _) = server(prefer(&["research"]), "Results.").await;
        let sink = RecordingSink::new(Some(7));
        let (session, mut events) =
            Session::open(desk(Some(&sink)), tree_of(SEARCH_TASK), Settings::default());
        // With no provider answering yet, nothing can run.
        assert!(!session.ready());
        session.set_routes(Some(routes(client)));
        session.set_tools(Some(tool_host()));
        assert!(session.ready());
        let (updates, _) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: context(None),
            entry: None,
        };
        let first = session
            .transcript(start, "search for crates", &updates)
            .await;
        assert_eq!(first.error, None, "{:?}", first.notes);
        assert!(session.view().in_task());
        // The search waits, and its timer runs out: the session runs that take itself, numbered
        // apart from the client's, and says so.
        async fn next(events: &mut mpsc::UnboundedReceiver<Event>) -> Event {
            tokio::time::timeout(Duration::from_secs(5), events.recv())
                .await
                .expect("the session reports it")
                .unwrap()
        }
        let Event::Timer { take, due } = next(&mut events).await else {
            panic!("the timer's take starts first");
        };
        assert!(take >= TIMER_TAKES);
        assert_eq!(due.event, "quiet");
        let trace = loop {
            match next(&mut events).await {
                Event::Update { take: of, .. } => assert_eq!(of, take),
                Event::Finished(trace) => break trace,
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(trace.take, take);
        assert_eq!(moves(&trace), ["answering quiet → [*]"]);
        assert!(session.view().at_rest());
    }

    /// The search task's tree, read again, with a search that ends after `ms` of quiet.
    fn search_quiet_after(ms: u64) -> Arc<FlowTree> {
        let files: Vec<(&str, String)> = SEARCH_TASK
            .iter()
            .map(|(path, text)| {
                let text = text.replace("timer idle = 40", &format!("timer idle = {ms}"));
                (*path, text)
            })
            .collect();
        let files = files.iter().map(|(p, t)| (*p, t.as_str()));
        Arc::new(FlowTree::load(
            &Memory::new("test", files),
            &tool_host().catalog(),
        ))
    }

    #[tokio::test]
    async fn the_user_answers_an_unsure_decision_and_the_take_goes_on() {
        use crate::session::{Event, Session, TIMER_TAKES};
        // A model sure of everything but of opening a result.
        let decide: Decider = Arc::new(|request: &Value| {
            let labels = ["research", "find-1", "opening"];
            let answers: serde_json::Map<String, Value> = request["questions"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, question)| {
                    if question["type"] == "noul" {
                        return (key.clone(), json!({"type": "noul", "noul": 0.9}));
                    }
                    let offered: Vec<&String> =
                        question["criteria"].as_object().unwrap().keys().collect();
                    let choice = labels
                        .iter()
                        .find(|l| offered.iter().any(|o| o == *l))
                        .map_or_else(|| offered[0].clone(), |l| l.to_string());
                    let sure = if choice == "opening" { 0.4 } else { 0.9 };
                    let answer = json!({"type": "choice", "choice": choice,
                        "probabilities": {choice.clone(): sure}, "confidence": sure});
                    (key.clone(), answer)
                })
                .collect();
            json!({"model": "jev", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": answers})
        });
        let (client, seen) = server(decide, "Results.").await;
        let file = kept_in("answer");
        let sink = RecordingSink::new(Some(7));
        let (session, mut events) = Session::open(
            desk(Some(&sink)),
            search_quiet_after(600_000),
            Settings::default(),
        );
        session.keep_machines_in(file.clone());
        session.set_routes(Some(routes(client)));
        session.set_tools(Some(tool_host()));
        let (updates, _) = mpsc::unbounded_channel();
        let said = |id: u64, text: &'static str| {
            let start = TakeStart {
                id,
                context: context(None),
                entry: None,
            };
            let (session, updates) = (session.clone(), updates.clone());
            async move { session.transcript(start, text, &updates).await }
        };
        let first = said(1, "search for crates").await;
        assert_eq!(first.error, None, "{:?}", first.notes);
        // The follow-up reaches the search, and the model is unsure which of its transitions
        // it is for: the search stays, and keeps what it was asked.
        let second = said(2, "open the first one").await;
        assert_eq!(
            inner(&second),
            ["idle said → task find-1", "answering said → answering"]
        );
        let view = session.view();
        let task = view.focus.unwrap();
        let unsure = view.running(task).unwrap().unsure.clone().unwrap();
        assert_eq!(unsure.said, "open the first one");
        assert_eq!(unsure.candidates, ["opening", "end"]);
        assert_eq!(
            unsure.probabilities,
            std::collections::BTreeMap::from([("opening".to_string(), 0.4)])
        );
        assert_eq!(unsure.how, "unsure (opening 0.40): stayed");
        let asked = seen.decisions.lock().unwrap().len();
        // What is no candidate, or no machine, starts nothing.
        session.answer(task, "elsewhere".into()).await;
        session.answer(task + 100, "opening".into()).await;
        assert!(events.try_recv().is_err());

        // The user says which: the take goes on with what was said, as a take of the
        // session's own, and no model is asked for that decision.
        session.answer(task, "opening".into()).await;
        let Some(Event::Answered {
            take,
            instance,
            label,
        }) = events.recv().await
        else {
            panic!("the answer's take starts first");
        };
        assert!(take >= TIMER_TAKES);
        assert_eq!((instance, label.as_str()), (task, "opening"));
        let trace = loop {
            match events.recv().await.unwrap() {
                Event::Update { take: of, .. } => assert_eq!(of, take),
                Event::Finished(trace) => break trace,
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(trace.transcript, "open the first one");
        assert_eq!(
            moves(&trace),
            ["answering said → opening", "opening done → [*]"]
        );
        assert_eq!(trace.machine[0].how, "the user chose");
        assert_eq!(trace.calls[0].tool, "search");
        assert_eq!(seen.decisions.lock().unwrap().len(), asked);
        assert!(session.view().at_rest());
        // The answer is kept as a labelled example, beside the kept machines.
        let examples = std::fs::read_to_string(file.with_file_name("examples.jsonl")).unwrap();
        let example: Value = serde_json::from_str(examples.trim()).unwrap();
        assert_eq!(
            (
                &example["machine"],
                &example["state"],
                &example["said"],
                &example["chosen"],
                &example["probabilities"]["opening"]
            ),
            (
                &json!("research/find"),
                &json!("answering"),
                &json!("open the first one"),
                &json!("opening"),
                &json!(0.4)
            )
        );

        // An unsure decision is the user's to answer only until its machine gets another
        // event: what is said next takes its place.
        said(3, "search for crates").await;
        said(4, "open the first one").await;
        let task = session.view().focus.unwrap();
        assert!(session.view().running(task).unwrap().unsure.is_some());
        let (client, _) = server(prefer(&["research", "find-1", "end"]), "unused").await;
        session.set_routes(Some(routes(client)));
        let last = said(5, "that is all, thanks").await;
        assert_eq!(
            inner(&last),
            ["idle said → task find-1", "answering said → [*]"]
        );
        session.answer(task, "opening".into()).await;
        assert!(events.try_recv().is_err());
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }

    /// The search task's tree, where the agent starts a search only on "search …", as the
    /// example does, and the task, the agent or both hand what they are unsure about to the
    /// machine above.
    fn handing_up(agent: bool, task: bool) -> Arc<FlowTree> {
        handing_up_from(agent, task, "idle --> find : said [search]")
    }

    /// The same, with the agent's transition into a search as `starts` says.
    fn handing_up_from(agent: bool, task: bool, starts: &str) -> Arc<FlowTree> {
        const SEARCH: &str = "\n[guards.search]\nwhen = { transcript = \"(?i)^search\" }\n\
            prefer = { transcript = \"(?i)^search\" }";
        let by_rule = starts.ends_with("[search]");
        let files: Vec<(&str, String)> = SEARCH_TASK
            .iter()
            .map(|(path, text)| {
                let up = "\nunsure = \"parent\"";
                let text = text
                    .replace("timer idle = 40", "timer idle = 600000")
                    .replace("idle --> find : said", starts);
                let rule = if by_rule { SEARCH } else { "" };
                let text = match *path {
                    "research/agent.toml" if agent => format!("{text}{up}{rule}"),
                    "research/agent.toml" => format!("{text}{rule}"),
                    "research/find/task.toml" if task => format!("{text}{up}"),
                    _ => text,
                };
                (*path, text)
            })
            .collect();
        let files = files.iter().map(|(p, t)| (*p, t.as_str()));
        let tree = FlowTree::load(&Memory::new("test", files), &tool_host().catalog());
        assert!(tree.is_valid(), "{:?}", tree.errors);
        Arc::new(tree)
    }

    /// A model sure that what is said is for the search that waits, and unsure which of the
    /// search's transitions it is for.
    fn sure_of_the_search_alone() -> Decider {
        sure_of(&["research", "find-1"])
    }

    /// A model sure of the candidates in `known`, and of any yes-or-no question, and unsure of
    /// every other choice.
    fn sure_of(known: &'static [&'static str]) -> Decider {
        Arc::new(move |request: &Value| {
            let answers: serde_json::Map<String, Value> = request["questions"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, question)| {
                    if question["type"] == "noul" {
                        return (key.clone(), json!({"type": "noul", "noul": 0.9}));
                    }
                    let offered: Vec<&String> =
                        question["criteria"].as_object().unwrap().keys().collect();
                    let choice = known
                        .iter()
                        .find(|l| offered.iter().any(|o| o == *l))
                        .map_or_else(|| offered[0].clone(), |l| l.to_string());
                    let sure = if known.contains(&choice.as_str()) {
                        0.9
                    } else {
                        0.4
                    };
                    let answer = json!({"type": "choice", "choice": choice,
                        "probabilities": {choice.clone(): sure}, "confidence": sure});
                    (key.clone(), answer)
                })
                .collect();
            json!({"model": "jev", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": answers})
        })
    }

    #[tokio::test]
    async fn an_unsure_take_goes_up_a_level_where_the_machine_says_so() {
        // Dictation while a search waits. The model takes the words for the search, and the
        // search does not know what to do with them.
        let words = "hello team, the build is green";
        let run = |agent: bool, task: bool, entry: Option<&'static str>| {
            run_on(handing_up(agent, task), entry)
        };
        async fn run_on(
            flows: Arc<FlowTree>,
            entry: Option<&'static str>,
        ) -> (
            Trace,
            Vec<String>,
            Vec<usize>,
            crate::flow::machine::runtime::View,
        ) {
            let words = "hello team, the build is green";
            let (client, _) = server(prefer(&["research"]), "Results.").await;
            let sink = RecordingSink::new(Some(7));
            let machines = Arc::new(Runtime::new());
            let env = Env {
                flows,
                tools: Some(tool_host()),
                machines: machines.clone(),
                ..env(client, Some(&sink))
            };
            let first = say(&env, 1, "search for crates").await;
            assert_eq!(first.error, None, "{:?}", first.notes);
            assert_eq!(machines.view().path(), "research › find › answering");
            let (client, seen) = server(sure_of_the_search_alone(), "unused").await;
            let env = Env {
                routes: routes(client),
                ..env
            };
            let (updates, _) = mpsc::unbounded_channel();
            let start = TakeStart {
                id: 2,
                context: context(None),
                entry: entry.map(String::from),
            };
            let trace = run_transcript(&env, start, words, &updates).await;
            let typed: Vec<String> = sink.requests().iter().map(|r| r.text.clone()).collect();
            // How many questions each decision call asked.
            let asked: Vec<usize> = seen
                .decisions
                .lock()
                .unwrap()
                .iter()
                .map(|request| request["questions"].as_object().map_or(0, |q| q.len()))
                .collect();
            (trace, typed, asked, machines.view())
        }

        // By default it stays there, and nothing is typed.
        let (trace, typed, asked, view) = run(false, false, None).await;
        assert_eq!(
            inner(&trace),
            ["idle said → task find-1", "answering said → answering"]
        );
        assert_eq!((typed.len(), asked.len()), (0, 1));
        assert!(view.in_task());

        // The search hands it up: its agent takes it as if the search were not there, has no
        // transition for words that do not ask for a search, and stays, having no leave to
        // hand it on.
        let (trace, typed, asked, _) = run(false, true, None).await;
        assert_eq!(
            inner(&trace),
            [
                "idle said → task find-1",
                "answering said → answering",
                "idle said → idle"
            ]
        );
        assert_eq!((typed.len(), asked.len()), (0, 1), "{:?}", trace.notes);

        // Both hand up: the root takes it without the agent, and its `[else]` types the
        // words. One request asked the search, its agent without it, and the root.
        let (trace, typed, asked, view) = run(true, true, None).await;
        assert_eq!(
            moves(&trace),
            [
                "idle said → research",
                "research done → idle",
                "idle said → task find-1",
                "answering said → answering",
                "idle said → idle",
                "idle said → typing",
                "typing done → idle",
                "idle said → type",
                "type done → idle"
            ],
            "{:?}",
            trace.notes
        );
        assert_eq!(typed, [words]);
        // One decision call, of three questions: the root's, the agent's and the search's.
        // What the agent and the root would ask on the way back, rules settle.
        assert_eq!(asked, [3], "one decision call");
        assert_eq!(trace.error, None);
        assert!(
            trace.notes.contains(
                &"find-1 did not take what you said (unsure (end 0.40): stayed): research takes it"
                    .to_string()
            ) && trace.notes.contains(
                &"research did not take what you said (no transition applies: stayed): the \
                  root takes it"
                    .to_string()
            ),
            "{:?}",
            trace.notes
        );
        // The search still waits, and the bubble no longer follows it: the words were typed.
        assert_eq!(view.focus, None);
        assert_eq!(view.stack.last().unwrap().state, "answering");
        // What it was unsure about is still the user's to answer.
        assert!(view.stack.last().unwrap().unsure.is_some());

        // The same from a hotkey that starts at the agent, where the root asks nothing: the
        // agent's request carries the root's question too.
        let (trace, typed, asked, _) = run(true, true, Some("research")).await;
        assert_eq!(typed, [words], "{:?}", trace.notes);
        assert_eq!(asked.len(), 1, "one decision call: {asked:?}");

        // An agent whose own transition is the model's to judge: that question, without the
        // search, rode in the request that asked the search, so no second one is made. This
        // model says yes to it, and another search starts.
        let by_model = "idle --> find : said [the user asks to search the web]";
        let (trace, _, asked, view) = run_on(handing_up_from(false, true, by_model), None).await;
        assert_eq!(
            inner(&trace),
            [
                "idle said → task find-1",
                "answering said → answering",
                "idle said → find",
                "find done → idle",
                "[*] start → searching",
                "searching done → answering"
            ],
            "{:?}",
            trace.notes
        );
        // The root's question, the agent's, the search's, and the agent's without the search.
        assert_eq!(asked, [4], "one decision call");
        assert_eq!(view.path(), "research › find › answering");
        assert_eq!(view.stack.last().unwrap().number, 2);
    }

    #[tokio::test]
    async fn words_two_waiting_tasks_both_claim_are_for_their_agent_and_the_model_says_which() {
        // A search that searches again by rule, started by an agent that starts one when
        // the model says so.
        let files: Vec<(&str, String)> = SEARCH_TASK
            .iter()
            .map(|(path, text)| {
                let text = text
                    .replace("timer idle = 40", "timer idle = 600000")
                    .replace(
                        "idle --> find : said",
                        "idle --> find : said [the user asks to look something up]",
                    )
                    .replace(
                        "answering --> [*] : quiet",
                        "answering --> searching : said [more]\nanswering --> [*] : quiet",
                    );
                let text = match *path {
                    "research/find/task.toml" => format!(
                        "{text}\n[guards.more]\nprefer = {{ transcript = \"(?i)^search\" }}\n\
                         criterion = \"The user asks to search for something else\""
                    ),
                    _ => text,
                };
                (*path, text)
            })
            .collect();
        let files = files.iter().map(|(p, t)| (*p, t.as_str()));
        let tree = FlowTree::load(&Memory::new("test", files), &tool_host().catalog());
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let machines = Arc::new(Runtime::new());
        let (client, _) = server(prefer(&["research", "find"]), "Results.").await;
        let env = Env {
            flows: Arc::new(tree),
            tools: Some(tool_host()),
            machines: machines.clone(),
            ..env(client, None)
        };
        // Two searches side by side: the model starts the second. A take that starts a task
        // is no follow-up to one.
        let (updates, mut said) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: context(None),
            entry: None,
        };
        run_transcript(&env, start, "look up rust", &updates).await;
        drop(updates);
        while let Some(update) = said.recv().await {
            assert!(!matches!(update, Update::Task { .. }), "{update:?}");
        }
        let second = say(&env, 2, "look up tokio").await;
        assert_eq!(inner(&second)[0], "idle said → find", "{:?}", second.notes);
        assert_eq!(
            machines.view().tasks(machines.view().stack[1].id).count(),
            2
        );
        // Both search again on "search …", by their own rule. The root needs no model to
        // know the words are the agent's; the model says which search, and reads which of
        // them the bubble shows.
        let (client, seen) = server(prefer(&["find-2"]), "Results.").await;
        let env = Env {
            routes: routes(client),
            ..env
        };
        let (updates, mut said) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 3,
            context: context(None),
            entry: None,
        };
        let third = run_transcript(&env, start, "search for async", &updates).await;
        assert_eq!(third.error, None, "{:?}", third.notes);
        // The client is told which task the words were for, once: from then on the take is
        // part of that task's conversation.
        drop(updates);
        let mut reached = Vec::new();
        while let Some(update) = said.recv().await {
            if let Update::Task { instance } = update {
                reached.push(instance);
            }
        }
        assert_eq!(reached, [machines.view().focus.unwrap()]);
        assert_eq!(
            moves(&third)[..4],
            [
                "idle said → research",
                "research done → idle",
                "idle said → task find-2",
                "answering said → searching"
            ]
        );
        assert_eq!(
            third.machine[0].how,
            "preferred: its transcript rule passed"
        );
        let requests = seen.decisions.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let criteria = &requests[0]["questions"]["q00"]["criteria"];
        let ends =
            |label: &str, with: &str| criteria[label].as_str().is_some_and(|c| c.ends_with(with));
        assert!(
            ends("find-1", "(an earlier one, no longer in the bubble)")
                && ends("find-2", "(the one the bubble shows now)")
                && criteria.as_object().unwrap().len() == 2,
            "{criteria}"
        );
    }

    /// A file to keep the machines in, in a folder of this test's own.
    fn kept_in(test: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("jevons-kept-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("machines.json")
    }

    #[tokio::test]
    async fn a_search_that_waits_is_still_there_after_a_restart() {
        let (confirm, mut asked) =
            mpsc::unbounded_channel::<jevons_desktop_core::confirm::Confirmation>();
        let (flows, tools, desk, searched) = with_search_example(confirm).await;
        let (client, seen) =
            server(prefer(&["research", "search-1", "opening", "end"]), "1.").await;
        tokio::spawn(async move {
            while let Some(call) = asked.recv().await {
                let _ = call.reply.send(true);
            }
        });
        let file = kept_in("search");
        let machines = Arc::new(Runtime::new());
        machines.keep_in(file.clone());
        let env = Env {
            flows,
            tools: Some(tools),
            desk,
            machines: machines.clone(),
            ..env(client, None)
        };
        // Nothing is kept while nothing runs.
        assert_eq!(machines.restore(&env).await, Vec::<String>::new());
        assert!(!file.exists());
        let first = say(&env, 1, "Buscar crates de máquinas de estado para Rust").await;
        assert_eq!(first.error, None, "{:?}", first.notes);
        assert_eq!(machines.view().path(), "research › search › results");
        let kept = std::fs::read_to_string(&file).unwrap();
        assert!(kept.contains("\"state\":\"results\""), "{kept}");

        // The server stops and starts: other machines, the same file, the flows read again.
        let again = Arc::new(Runtime::new());
        again.keep_in(file.clone());
        let env = Env {
            machines: again.clone(),
            ..env
        };
        assert_eq!(again.restore(&env).await, Vec::<String>::new());
        let view = again.view();
        assert_eq!(view.path(), "research › search › results");
        let task = view.stack.last().unwrap();
        assert_eq!((task.folder.as_str(), task.number), ("research/search", 1));
        assert!(
            task.waiting.contains(&"said".to_string()),
            "{:?}",
            task.waiting
        );
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            kept,
            "kept as it was"
        );
        // It goes on where it was: what its search found is still there to open a result
        // from, and nothing searched again.
        let second = say(&env, 2, "open the first one").await;
        assert_eq!(second.error, None, "{:?}", second.notes);
        assert_eq!(
            inner(&second),
            [
                "idle said → task search-1",
                "results said → opening",
                "opening done → results"
            ]
        );
        assert_eq!(second.calls[0].tool, "open_url");
        let asked_for = seen.generations.lock().unwrap().last().unwrap().to_string();
        assert!(asked_for.contains("https://x/1"), "{asked_for}");
        assert_eq!(searched.lock().unwrap().len(), 1);
        // Its end leaves nothing to keep.
        let (client, _) = server(prefer(&["research", "search-1", "end"]), "unused").await;
        let env = Env {
            routes: routes(client),
            ..env
        };
        let third = say(&env, 3, "thanks, that's all").await;
        assert_eq!(
            inner(&third),
            ["idle said → task search-1", "results said → [*]"]
        );
        assert!(!file.exists());
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }

    #[tokio::test]
    async fn work_cut_short_by_a_restart_fails_and_is_not_run_again() {
        // A search whose tool asks first: while the question is out, its state's work runs.
        let (confirm, mut asked) =
            mpsc::unbounded_channel::<jevons_desktop_core::confirm::Confirmation>();
        let (tools, desk, hits) = two_sides("", Some(confirm)).await;
        let files: Vec<(&str, String)> = SEARCH_TASK
            .iter()
            .map(|(path, text)| (*path, text.replace("\"search\"", "\"lookup\"")))
            .collect();
        let tree = FlowTree::load(
            &Memory::new("test", files.iter().map(|(p, t)| (*p, t.as_str()))),
            &tools.catalog(),
        );
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let file = kept_in("cut-short");
        // What is kept at that moment is what a server that stops then leaves behind.
        let (left, at) = (file.with_file_name("left.json"), file.clone());
        let behind = left.clone();
        tokio::spawn(async move {
            while let Some(call) = asked.recv().await {
                std::fs::copy(&at, &behind).unwrap();
                let _ = call.reply.send(true);
            }
        });
        let (client, _) = server(prefer(&["research"]), "Results.").await;
        let machines = Arc::new(Runtime::new());
        machines.keep_in(file.clone());
        let env = Env {
            flows: Arc::new(tree),
            tools: Some(tools),
            desk,
            machines,
            ..env(client, None)
        };
        let first = say(&env, 1, "search for crates").await;
        assert_eq!(first.error, None, "{:?}", first.notes);
        assert_eq!(*hits.lock().unwrap(), 1);
        let kept = std::fs::read_to_string(&left).unwrap();
        assert!(
            kept.contains(r#""engine":{"state":"searching","generation":1,"working":true}"#),
            "{kept}"
        );

        // The server starts again from what was left: the search does not run again. Its
        // machine takes `failed`, which ends it.
        let again = Arc::new(Runtime::new());
        again.keep_in(left.clone());
        let env = Env {
            machines: again.clone(),
            ..env
        };
        let notes = again.restore(&env).await;
        assert!(
            notes.contains(
                &"find-1 was at searching, and jevons stopped while it ran: that is not run \
                  again, and it failed"
                    .to_string()
            ),
            "{notes:?}"
        );
        assert_eq!(*hits.lock().unwrap(), 1, "not run again");
        let view = again.view();
        assert!(view.at_rest(), "{}", view.path());
        let took: Vec<String> = view
            .history
            .iter()
            .filter(|s| s.machine != "/" && s.machine != "research")
            .map(|s| format!("{} {} → {}", s.from, s.event, s.to))
            .collect();
        assert_eq!(took, ["searching failed → [*]"]);
        assert!(!left.exists(), "nothing runs: nothing is kept");
        // Bringing back happens once.
        assert_eq!(again.restore(&env).await, Vec::<String>::new());
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }

    #[tokio::test]
    async fn a_task_whose_files_changed_is_ended_instead_of_resumed() {
        let (client, _) = server(prefer(&["research"]), "Results.").await;
        let file = kept_in("changed");
        let machines = Arc::new(Runtime::new());
        machines.keep_in(file.clone());
        let env = task_env(client, machines);
        let first = say(&env, 1, "search for crates").await;
        assert_eq!(first.error, None, "{:?}", first.notes);
        let kept = std::fs::read_to_string(&file).unwrap();
        let changed = |path: &str, text: &str| {
            let files: Vec<(&str, &str)> = SEARCH_TASK
                .iter()
                .map(|(p, t)| if *p == path { (*p, text) } else { (*p, *t) })
                .collect();
            tree_of(&files)
        };
        let restart = |flows: Arc<FlowTree>| {
            std::fs::write(&file, &kept).unwrap();
            let machines = Arc::new(Runtime::new());
            machines.keep_in(file.clone());
            Env {
                flows,
                machines,
                ..env.clone()
            }
        };
        // Another machine's files changed: the search is not touched by it.
        let other = restart(changed(
            "typing/type/transcript.toml",
            "output = \"clipboard\"",
        ));
        assert_eq!(other.machines.restore(&other).await, Vec::<String>::new());
        assert_eq!(other.machines.view().path(), "research › find › answering");
        // One of its own states' files changed: it is ended, and the Machines tab says why.
        let own = restart(changed(
            "research/find/answering/generate.toml",
            "output = \"bubble\"\nprompt = \"Answer from: {searching}\"",
        ));
        assert_eq!(
            own.machines.restore(&own).await,
            ["find-1 was at answering: its files changed while jevons did not run: it ended"]
        );
        let view = own.machines.view();
        assert!(view.at_rest(), "{}", view.path());
        let last = view.history.back().unwrap();
        assert_eq!(
            (last.machine.as_str(), last.event.as_str(), last.to.as_str()),
            ("research/find", "changed", "[*]")
        );
        assert!(!file.exists());
        // Its folder gone: ended too.
        let files: Vec<(&str, &str)> = SEARCH_TASK
            .iter()
            .filter(|(p, _)| !p.starts_with("research/find/"))
            .map(|(p, t)| {
                if *p == "research/agent.fsm" {
                    (*p, "fsm Research {\n[*] --> idle\n}")
                } else {
                    (*p, *t)
                }
            })
            .collect();
        let gone =
            restart(FlowTree::load(&Memory::new("test", files), &tool_host().catalog()).into());
        assert_eq!(
            gone.machines.restore(&gone).await,
            ["find-1 was at answering: its machine is not in the flows folder any more: it ended"]
        );
        // A file that cannot be read is dropped, and the first take says so.
        std::fs::write(&file, "{").unwrap();
        let machines = Arc::new(Runtime::new());
        machines.keep_in(file.clone());
        let broken = Env {
            machines,
            ..env.clone()
        };
        let take = say(&broken, 2, "search for crates").await;
        assert!(
            take.notes[0].starts_with("The tasks kept when jevons stopped could not be read"),
            "{:?}",
            take.notes
        );
        assert_eq!(broken.machines.view().path(), "research › find › answering");
        // A cancel before anything was brought back ends what was kept, unseen.
        std::fs::write(&file, &kept).unwrap();
        let machines = Runtime::new();
        machines.keep_in(file.clone());
        assert_eq!(machines.cancel().await, None);
        assert!(!file.exists());
        let cancelled = Env {
            machines: Arc::new(machines),
            ..env.clone()
        };
        assert_eq!(
            cancelled.machines.restore(&cancelled).await,
            Vec::<String>::new()
        );
        assert!(cancelled.machines.view().at_rest());
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }

    #[tokio::test]
    async fn a_session_brings_its_tasks_back_and_their_timers_start_over() {
        use crate::session::{Event, Session};
        let (client, _) = server(prefer(&["research"]), "Results.").await;
        let file = kept_in("session");
        // A long timer, so that the search waits when the session goes.
        let open = || {
            let sink = RecordingSink::new(Some(7));
            let (session, events) = Session::open(
                desk(Some(&sink)),
                search_quiet_after(400),
                Settings::default(),
            );
            session.keep_machines_in(file.clone());
            session.set_routes(Some(routes(client.clone())));
            session.set_tools(Some(tool_host()));
            (session, events)
        };
        let (session, events) = open();
        let (updates, _) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: context(None),
            entry: None,
        };
        let first = session
            .transcript(start, "search for crates", &updates)
            .await;
        assert_eq!(first.error, None, "{:?}", first.notes);
        assert!(session.view().in_task());
        drop((session, events));

        // Another session, as after a restart: the search is there, and its timer runs its
        // whole time again before it ends the search.
        let (session, mut events) = open();
        assert!(session.view().at_rest(), "nothing until it is brought back");
        let before = Instant::now();
        assert_eq!(session.restore().await, Vec::<String>::new());
        assert_eq!(session.view().path(), "research › find › answering");
        let trace = loop {
            let event = tokio::time::timeout(Duration::from_secs(5), events.recv());
            match event.await.expect("the timer runs out").unwrap() {
                Event::Finished(trace) => break trace,
                Event::Timer { due, .. } => assert_eq!(due.event, "quiet"),
                Event::Update { .. } => {}
                other => panic!("{other:?}"),
            }
        };
        assert!(before.elapsed() >= Duration::from_millis(400));
        assert_eq!(moves(&trace), ["answering quiet → [*]"]);
        assert!(session.view().at_rest());
        assert!(!file.exists());
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }

    /// A client of a host over a stream: its desk attends to the server's effects.
    struct Connected {
        say: mpsc::UnboundedSender<ToServer>,
        hear: mpsc::UnboundedReceiver<ToClient>,
    }

    /// Connects a client at `desk` to `host`, in this process or over a WebSocket.
    async fn connected(host: &Arc<Host>, desk: Arc<dyn Desk>, websocket: bool) -> Connected {
        let link = if websocket {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let app = crate::serve::router(host.clone());
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            jevons_desktop_protocol::socket::connect(&format!("ws://{address}/desktop"))
                .await
                .unwrap()
        } else {
            let (client, server) = in_process();
            let host = host.clone();
            tokio::spawn(async move { host.serve(server).await });
            client
        };
        let tools = desk.tools();
        let (say, hear) = attend(desk, link);
        say.send(ToServer::Hello {
            version: VERSION,
            key: None,
            tools,
            settings: TakeSettings::default(),
        })
        .unwrap();
        Connected { say, hear }
    }

    impl Connected {
        /// The next thing the server says that is not a take's update.
        async fn next(&mut self) -> ToClient {
            loop {
                let heard = tokio::time::timeout(Duration::from_secs(5), self.hear.recv());
                match heard.await.expect("the server says something") {
                    Some(ToClient::Update { .. }) => continue,
                    Some(other) => return other,
                    None => panic!("the stream ended"),
                }
            }
        }

        /// The session opens: a welcome, then where the machines are.
        async fn welcomed(&mut self) -> Value {
            assert_eq!(self.next().await, ToClient::Welcome { version: VERSION });
            match self.next().await {
                ToClient::Machines { view } => view,
                other => panic!("{other:?}"),
            }
        }

        /// Says `text` as take `take` from `context`; its trace and the machines after it.
        async fn say(&mut self, take: u64, context: ContextSnapshot, text: &str) -> (Value, Value) {
            self.say
                .send(ToServer::Transcript {
                    take,
                    context,
                    entry: None,
                    text: text.into(),
                })
                .unwrap();
            let ToClient::Trace { take: of, trace } = self.next().await else {
                panic!("the take's trace comes first");
            };
            assert_eq!(of, take);
            let ToClient::Machines { view } = self.next().await else {
                panic!("then where the machines are");
            };
            (trace, view)
        }
    }

    /// A trace without what differs between two runs of the same take: when, and how long.
    fn comparable(mut trace: Value) -> Value {
        fn strip(value: &mut Value) {
            match value {
                Value::Object(map) => {
                    for key in ["ms", "at_ms", "started_at_ms", "timings", "since_ms"] {
                        map.remove(key);
                    }
                    map.values_mut().for_each(strip);
                }
                Value::Array(items) => items.iter_mut().for_each(strip),
                _ => {}
            }
        }
        strip(&mut trace);
        trace
    }

    #[tokio::test]
    async fn a_take_over_a_stream_is_the_take_run_directly() {
        // A rewrite of the selection through the built-in tree: decisions, a generation and
        // a delivery. First with the client's desk in the take's own environment.
        let decide = || prefer(&["dictation", "rewrite"]);
        let (client, _) = server(decide(), "Dear team, hello world.").await;
        let sink = RecordingSink::new(Some(7));
        let (updates, _) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: context(Some("hi all")),
            entry: None,
        };
        let direct =
            run_transcript(&env(client, Some(&sink)), start, "hello world", &updates).await;
        assert_eq!(direct.error, None, "{:?}", direct.notes);
        assert_eq!(sink.requests().len(), 1);
        let direct = comparable(serde_json::to_value(&direct).unwrap());
        // Then with the server and the client each at an end of a stream, in this process and
        // over a WebSocket: the same trace, and the same text delivered.
        for websocket in [false, true] {
            let (client, _) = server(decide(), "Dear team, hello world.").await;
            let (session, events) = Session::open(Arc::new(Nobody), builtin(), Settings::default());
            session.set_routes(Some(routes(client)));
            let host = Host::new(session, events, None);
            let sink = RecordingSink::new(Some(7));
            let mut client = connected(&host, desk(Some(&sink)), websocket).await;
            let machines = client.welcomed().await;
            assert_eq!(machines["busy"], false);
            let (trace, machines) = client.say(1, context(Some("hi all")), "hello world").await;
            assert_eq!(comparable(trace), direct, "websocket: {websocket}");
            assert_eq!(sink.requests()[0].text, "Dear team, hello world.");
            assert_eq!(sink.requests()[0].action, Action::Rewrite);
            assert_eq!(machines["focus"], Value::Null);
        }
    }

    #[tokio::test]
    async fn a_client_finds_its_tasks_when_the_server_has_restarted() {
        let labels = &["research", "find-1", "opening"];
        let (client, _) = server(prefer(labels), "Two crates fit.").await;
        let file = kept_in("host");
        let serve = || {
            let (session, events) = Session::open(
                Arc::new(Nobody),
                search_quiet_after(600_000),
                Settings::default(),
            );
            session.keep_machines_in(file.clone());
            session.set_routes(Some(routes(client.clone())));
            session.set_tools(Some(tool_host()));
            Host::new(session, events, None)
        };
        let desk = || Arc::new(LocalDesk::default()) as Arc<dyn Desk>;
        let host = serve();
        let mut first = connected(&host, desk(), false).await;
        first.welcomed().await;
        let (trace, view) = first.say(1, context(None), "search for crates").await;
        assert_eq!(trace["error"], Value::Null, "{trace}");
        let task = view["focus"].clone();
        assert!(task.is_u64(), "{view}");
        drop((first, host));

        // Another server on the same file. Its first client is told where the machines are
        // once they are back: its search, waiting where it was.
        let host = serve();
        assert!(host.session().view().at_rest());
        let mut back = connected(&host, desk(), false).await;
        let view = back.welcomed().await;
        assert_eq!(view["focus"], task, "{view}");
        let running = view["stack"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == task)
            .unwrap();
        assert_eq!(
            (running["state"].as_str(), running["number"].as_u64()),
            (Some("answering"), Some(1))
        );
        // And the follow-up goes to it.
        let (trace, _) = back.say(2, context(None), "open the first one").await;
        let took: Vec<&str> = trace["machine"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["to"].as_str().unwrap())
            .collect();
        assert_eq!(took[2..], ["task find-1", "opening", "[*]"], "{took:?}");
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }

    #[tokio::test]
    async fn a_cancel_while_a_take_waits_on_a_question_leaves_the_stream_open() {
        let (client, _) = server(prefer(&[]), "unused").await;
        let tree = agent_tree(&[
            ("root.toml", "tools = [\"note\"]"),
            (
                "root.fsm",
                "fsm App {\n[*] --> idle\nidle --> saving : said\nsaving --> idle\nsaving --> idle : denied\n}",
            ),
            (
                "saving/tool.toml",
                "description = \"Saves a note\"\ntool = \"note\"\n[args.title]\nvalue = \"x\"\n[args.folder]\nvalue = \"y\"\n[args.body]\nvalue = \"{transcript}\"",
            ),
        ]);
        let (session, events) = Session::open(Arc::new(Nobody), tree, Settings::default());
        session.set_routes(Some(routes(client)));
        session.set_tools(Some(tool_host()));
        let host = Host::new(session, events, None);
        let (confirm, mut asked) =
            mpsc::unbounded_channel::<jevons_desktop_core::confirm::Confirmation>();
        let desk: Arc<dyn Desk> = Arc::new(LocalDesk::default().with_confirmer(Arc::new(
            jevons_desktop_core::confirm::ChannelConfirmer::new(confirm),
        )));
        let mut client = connected(&host, desk, false).await;
        client.welcomed().await;
        client
            .say
            .send(ToServer::Transcript {
                take: 1,
                context: context(None),
                entry: None,
                text: "note that the build is green".into(),
            })
            .unwrap();
        let question = tokio::time::timeout(Duration::from_secs(5), asked.recv())
            .await
            .unwrap()
            .unwrap();
        // The user cancels while the question is out. The cancel waits for the take, and the
        // take for the answer: the server goes on reading, so the answer reaches it.
        client.say.send(ToServer::Cancel).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        question.reply.send(false).unwrap();
        // The take ends declined, and the client is told where the machines are after it and
        // after the cancel, in whichever order the two finish.
        let (mut traces, mut views) = (Vec::new(), 0);
        for _ in 0..3 {
            match client.next().await {
                ToClient::Trace { trace, .. } => traces.push(trace),
                ToClient::Machines { .. } => views += 1,
                other => panic!("{other:?}"),
            }
        }
        assert_eq!((traces.len(), views), (1, 2));
        assert_eq!(traces[0]["calls"][0]["confirmed"], false, "{}", traces[0]);
        assert!(host.session().view().at_rest());
    }

    #[tokio::test]
    async fn a_client_that_leaves_is_refused_and_finds_its_tasks_when_it_returns() {
        // A client whose user never answers is asked to confirm a tool, and leaves.
        let (client, _) = server(prefer(&[]), "unused").await;
        let tree = agent_tree(&[
            ("root.toml", "tools = [\"note\"]"),
            (
                "root.fsm",
                "fsm App {\n[*] --> idle\nidle --> saving : said\nsaving --> idle\nsaving --> told : denied\nstate told\ntold --> idle\n}",
            ),
            (
                "saving/tool.toml",
                "description = \"Saves a note\"\ntool = \"note\"\n[args.title]\nvalue = \"x\"\n[args.folder]\nvalue = \"y\"\n[args.body]\nvalue = \"{transcript}\"",
            ),
            ("told/transcript.toml", "output = \"clipboard\""),
        ]);
        let (session, events) = Session::open(Arc::new(Nobody), tree, Settings::default());
        session.set_routes(Some(routes(client)));
        session.set_tools(Some(tool_host()));
        let host = Host::new(session.clone(), events, None);
        let (confirm, mut asked) =
            mpsc::unbounded_channel::<jevons_desktop_core::confirm::Confirmation>();
        let silent: Arc<dyn Desk> = Arc::new(LocalDesk::default().with_confirmer(Arc::new(
            jevons_desktop_core::confirm::ChannelConfirmer::new(confirm),
        )));
        let mut client = connected(&host, silent, false).await;
        client.welcomed().await;
        let take = {
            let session = session.clone();
            tokio::spawn(async move {
                let (updates, _) = mpsc::unbounded_channel();
                let start = TakeStart {
                    id: 1,
                    context: context(None),
                    entry: None,
                };
                session
                    .transcript(start, "note that the build is green", &updates)
                    .await
            })
        };
        // The question reached the client's desk. Then the client goes away.
        let question = tokio::time::timeout(Duration::from_secs(5), asked.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(question.tool, "note");
        drop(client);
        // The take in flight ends with the call declined, which the machine handles.
        let trace = take.await.unwrap();
        assert_eq!(trace.calls[0].confirmed, Some(false));
        assert_eq!(
            inner(&trace),
            [
                "idle said → saving",
                "saving denied → told",
                "told done → idle"
            ]
        );
        drop(question);

        // A search that waits, and a client that leaves: the task stays where it was, and the
        // next client finds it and follows it up.
        let labels = &["research", "find-1", "opening"];
        let (client, _) = server(prefer(labels), "Two crates fit.").await;
        let (session, events) =
            Session::open(Arc::new(Nobody), tree_of(SEARCH_TASK), Settings::default());
        session.set_routes(Some(routes(client)));
        session.set_tools(Some(tool_host()));
        let host = Host::new(session, events, Some("the-key".into()));
        let hello = |key: Option<&str>, version: u32| ToServer::Hello {
            version,
            key: key.map(String::from),
            tools: Default::default(),
            settings: TakeSettings::default(),
        };
        // A client with another key, or of another version, is told why and turned away.
        for (key, version, why) in [
            (Some("another"), VERSION, "the key is not the server's"),
            (
                Some("the-key"),
                VERSION + 1,
                "the client speaks version 2, the server 1",
            ),
        ] {
            let (mut client, server) = in_process();
            let serving = host.clone();
            tokio::spawn(async move { serving.serve(server).await });
            client.tx.send(hello(key, version)).unwrap();
            assert_eq!(
                client.rx.recv().await,
                Some(ToClient::Closed { why: why.into() })
            );
        }
        let join = |host: &Arc<Host>| {
            let (client, server) = in_process();
            let serving = host.clone();
            tokio::spawn(async move { serving.serve(server).await });
            let (say, hear) = attend(desk(None), client);
            say.send(hello(Some("the-key"), VERSION)).unwrap();
            Connected { say, hear }
        };
        let mut first = join(&host);
        assert_eq!(first.welcomed().await["busy"], false);
        let (trace, machines) = first.say(1, context(None), "search for crates").await;
        assert_eq!(trace["error"], Value::Null, "{trace}");
        let waiting = |machines: &Value| {
            let stack = machines["stack"].as_array().unwrap();
            let task = stack.last().unwrap();
            (stack.len(), task["folder"].clone(), task["state"].clone())
        };
        assert_eq!(
            waiting(&machines),
            (4, json!("research/find"), json!("answering"))
        );
        drop(first);
        // The second client is told where the machines are as it connects.
        let mut second = join(&host);
        let machines = second.welcomed().await;
        assert_eq!(
            waiting(&machines),
            (4, json!("research/find"), json!("answering"))
        );
        let (trace, machines) = second.say(2, context(None), "open the first one").await;
        let moves: Vec<&str> = trace["machine"]
            .as_array()
            .unwrap()
            .iter()
            .map(|step| step["to"].as_str().unwrap())
            .collect();
        assert!(moves.contains(&"opening"), "{moves:?}");
        assert_eq!(
            machines["stack"].as_array().unwrap().len(),
            3,
            "the search ended"
        );
    }

    #[tokio::test]
    async fn a_guard_on_a_state_s_result_takes_a_transition_with_no_model() {
        let files = [
            ("root.toml", "tools = [\"search\"]"),
            (
                "root.fsm",
                "fsm App {\n[*] --> idle\nidle --> find : said\nfind --> idle\n}",
            ),
            (
                "find/task.toml",
                "description = \"Search\"\ntools = [\"search\"]\n[guards.nothing]\n\
                 when = { value = \"{look.arguments.query}\", matches = \"(?i)nothing\" }\n\
                 prefer = { value = \"{look.arguments.query}\", matches = \"(?i)nothing\" }",
            ),
            (
                "find/task.fsm",
                "fsm Find {\n[*] --> look\nlook --> none : done [nothing]\n\
                 look --> shown : done [else]\nnone --> [*] : said\nshown --> wait\n\
                 wait --> [*] : said\n}",
            ),
            (
                "find/look/tool.toml",
                "tool = \"search\"\noutput = \"none\"\n[args.query]\nvalue = \"{transcript}\"",
            ),
            (
                "find/shown/generate.toml",
                "output = \"bubble\"\nprompt = \"Asked {look.arguments.query} of {look.would_call}\"",
            ),
        ];
        // The tool's result keeps its fields: the guard reads one, and so does the next state.
        let (client, seen) = server(prefer(&[]), "Here they are.").await;
        let env = Env {
            flows: agent_tree(&files),
            tools: Some(tool_host()),
            machines: Arc::new(Runtime::new()),
            ..env(client, None)
        };
        let found = say(&env, 1, "state machine crates").await;
        assert_eq!(found.error, None, "{:?}", found.notes);
        assert_eq!(
            inner(&found),
            [
                "idle said → find",
                "find done → idle",
                "[*] start → look",
                "look done → shown",
                "shown done → wait"
            ]
        );
        let how = |trace: &Trace, to: &str| {
            let step = trace.machine.iter().find(|s| s.to == to).unwrap();
            step.how.clone()
        };
        assert_eq!(how(&found, "shown"), "the only transition that applies");
        let prompt = seen.generations.lock().unwrap()[0]["input"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(prompt, "Asked state machine crates of search");
        // Another search, whose result the guard's rule matches: rules take it there.
        let env = Env {
            machines: Arc::new(Runtime::new()),
            ..env
        };
        let none = say(&env, 2, "Nothing at all").await;
        assert_eq!(
            inner(&none),
            [
                "idle said → find",
                "find done → idle",
                "[*] start → look",
                "look done → none"
            ]
        );
        assert_eq!(how(&none, "none"), "preferred: its value rule passed");
        assert!(seen.decisions.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_generation_with_a_schema_answers_in_fields_later_states_read() {
        let files = [
            ("root.toml", ""),
            (
                "root.fsm",
                "fsm App {\n[*] --> idle\nidle --> sort : said\nsort --> idle\n}",
            ),
            ("sort/task.toml", "description = \"Sorts a request\""),
            (
                "sort/task.fsm",
                "fsm Sort {\n[*] --> kind\nkind --> urgent : done [urgent]\n\
                 kind --> told : done [else]\nurgent --> [*] : said\ntold --> wait\n\
                 wait --> [*] : said\n}\n",
            ),
            (
                "sort/kind/generate.toml",
                "output = \"none\"\n[schema]\nkind = \"news | web\"\ncount = \"integer\"",
            ),
            (
                "sort/told/generate.toml",
                "output = \"bubble\"\nprompt = \"It is {kind.kind}, {kind.count} of them\"",
            ),
        ];
        let guards = "\n[guards.urgent]\nwhen = { value = \"{kind.count}\", above = 5 }\n\
                      prefer = { value = \"{kind.count}\", above = 5 }";
        let task = format!("{}{guards}", files[2].1);
        let mut files = files.to_vec();
        files[2].1 = &task;
        // The model answers in JSON; what does not fit the schema is dropped or made to fit.
        let answer = r#"{"kind": "News", "count": "3", "extra": true}"#;
        let (client, seen) = server(prefer(&[]), answer).await;
        let env = Env {
            flows: agent_tree(&files),
            machines: Arc::new(Runtime::new()),
            ..env(client, None)
        };
        let trace = say(&env, 1, "the latest on Rust").await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(
            inner(&trace),
            [
                "idle said → sort",
                "sort done → idle",
                "[*] start → kind",
                "kind done → told",
                "told done → wait"
            ]
        );
        let generations = seen.generations.lock().unwrap();
        assert_eq!(generations[0]["text"]["format"]["type"], "json_schema");
        assert_eq!(
            generations[0]["text"]["format"]["schema"]["properties"]["count"]["type"],
            json!(["integer", "null"])
        );
        // The next state reads the fields, typed; and the guard compared the number.
        assert_eq!(generations[1]["input"], "It is news, 3 of them");
        assert!(generations[1].get("text").is_none());
        assert!(seen.decisions.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn two_tasks_of_one_agent_run_side_by_side_and_each_gets_its_own_follow_ups() {
        // The model routes by what was said: a new search for "search", the first search for
        // "first", the second for "second"; and the task then opens a result.
        let router: Decider = Arc::new(|request: &Value| {
            let said = request["state"].as_str().unwrap_or_default().to_string();
            let answers: serde_json::Map<String, Value> = request["questions"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, question)| {
                    let offered = question["criteria"].as_object().unwrap();
                    let wanted = if said.contains("first") {
                        "find-1"
                    } else if said.contains("second") {
                        "find-2"
                    } else {
                        "find"
                    };
                    let choice = ["research", wanted, "opening"]
                        .into_iter()
                        .find(|l| offered.contains_key(*l))
                        .unwrap_or_else(|| offered.keys().next().unwrap());
                    let answer = json!({"type": "choice", "choice": choice,
                        "probabilities": {choice: 0.9}, "confidence": 0.9});
                    (key.clone(), answer)
                })
                .collect();
            json!({"model": "jev", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": answers})
        });
        let (client, seen) = server(router, "Results.").await;
        let machines = Arc::new(Runtime::new());
        let env = task_env(client, machines.clone());
        say(&env, 1, "search for crates").await;
        // A take for the agent itself while a task waits: it starts another search beside it.
        let another = say(&env, 2, "search for books").await;
        assert_eq!(another.error, None, "{:?}", another.notes);
        assert_eq!(
            inner(&another),
            [
                "idle said → find",
                "find done → idle",
                "[*] start → searching",
                "searching done → answering"
            ]
        );
        let view = machines.view();
        let agent = view.stack.iter().find(|r| r.folder == "research").unwrap();
        let tasks: Vec<(String, String)> = view
            .tasks(agent.id)
            .map(|t| (t.label(), t.state.clone()))
            .collect();
        assert_eq!(
            tasks,
            [
                ("find-1".to_string(), "answering".to_string()),
                ("find-2".to_string(), "answering".to_string())
            ]
        );
        let (first, second) = {
            let mut ids = view.tasks(agent.id).map(|t| t.id);
            (ids.next().unwrap(), ids.next().unwrap())
        };
        assert_eq!(view.focus, Some(second), "the bubble follows the latest");
        // Each follow-up reaches the search it is about, and only that one moves.
        let on_second = say(&env, 3, "open the second search's result").await;
        assert_eq!(
            inner(&on_second),
            [
                "idle said → task find-2",
                "answering said → opening",
                "opening done → [*]"
            ]
        );
        assert!(on_second.machine[3..].iter().all(|s| s.instance == second));
        let left: Vec<u64> = machines.view().tasks(agent.id).map(|t| t.id).collect();
        assert_eq!(left, [first]);
        let on_first = say(&env, 4, "open the first search's result").await;
        assert_eq!(
            inner(&on_first),
            [
                "idle said → task find-1",
                "answering said → opening",
                "opening done → [*]"
            ]
        );
        assert!(on_first.machine[3..].iter().all(|s| s.instance == first));
        assert!(machines.view().at_rest());
        assert_eq!(
            seen.decisions.lock().unwrap().len(),
            4,
            "one request per take"
        );
    }

    #[tokio::test]
    async fn a_take_for_another_agent_leaves_a_waiting_task_alone() {
        // The model routes "search …" to research, "open …" to the task that waits, and the
        // rest to typing.
        let router: Decider = Arc::new(|request: &Value| {
            let said = request["state"].as_str().unwrap_or_default().to_string();
            let wanted: &[&str] = if said.contains("search for") {
                &["research", "find"]
            } else if said.contains("open") {
                &["research", "find-1", "opening"]
            } else {
                &["typing"]
            };
            let answers: serde_json::Map<String, Value> = request["questions"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, question)| {
                    let offered = question["criteria"].as_object().unwrap();
                    let choice = wanted
                        .iter()
                        .copied()
                        .find(|l| offered.contains_key(*l))
                        .unwrap_or_else(|| offered.keys().next().unwrap());
                    let answer = json!({"type": "choice", "choice": choice,
                        "probabilities": {choice: 0.9}, "confidence": 0.9});
                    (key.clone(), answer)
                })
                .collect();
            json!({"model": "jev", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": answers})
        });
        let (client, _) = server(router, "Results.").await;
        let machines = Arc::new(Runtime::new());
        let sink = RecordingSink::new(Some(7));
        let env = Env {
            machines: machines.clone(),
            flows: tree_of(SEARCH_TASK),
            tools: Some(tool_host()),
            ..env(client, Some(&sink))
        };
        say(&env, 1, "search for crates").await;
        let task = machines.view().focus.expect("the search waits");
        // Dictation while the search waits is typed, and is no part of the search: the app is
        // not in a task for this take, and the search is where it was.
        let typed = say(&env, 2, "see you at five").await;
        assert_eq!(typed.error, None, "{:?}", typed.notes);
        assert_eq!(
            moves(&typed),
            [
                "idle said → typing",
                "typing done → idle",
                "idle said → type",
                "type done → idle"
            ]
        );
        assert_eq!(sink.requests()[0].text, "see you at five");
        let view = machines.view();
        assert!(!view.in_task());
        assert_eq!((view.focus, view.path().as_str()), (None, "idle"));
        assert_eq!(view.running(task).unwrap().state, "answering");
        // A follow-up reaches it again.
        let opened = say(&env, 3, "open the first one").await;
        assert_eq!(
            inner(&opened),
            [
                "idle said → task find-1",
                "answering said → opening",
                "opening done → [*]"
            ]
        );
        assert!(opened.machine[3..].iter().all(|s| s.instance == task));
    }

    #[tokio::test]
    async fn one_task_can_be_cancelled_and_the_others_go_on() {
        let (client, _) = server(prefer(&["research", "find"]), "Results.").await;
        let machines = Arc::new(Runtime::new());
        let env = task_env(client, machines.clone());
        say(&env, 1, "search for crates").await;
        say(&env, 2, "search for books").await;
        let view = machines.view();
        let agent = view
            .stack
            .iter()
            .find(|r| r.folder == "research")
            .unwrap()
            .id;
        let tasks: Vec<u64> = view.tasks(agent).map(|t| t.id).collect();
        assert_eq!(tasks.len(), 2);
        // The second one is where the app is; cancelling it leaves the first, and its agent
        // where it was.
        assert_eq!(
            machines.cancel_task(tasks[1]).await.as_deref(),
            Some("research › find › answering")
        );
        let view = machines.view();
        let left: Vec<String> = view.tasks(agent).map(|t| t.label()).collect();
        assert_eq!(left, ["find-1"]);
        assert_eq!(view.focus, None);
        assert_eq!(view.history.back().unwrap().how, "the user cancelled");
        // Neither the root nor an agent is a task to cancel, nor one that is gone.
        assert_eq!(machines.cancel_task(agent).await, None);
        assert_eq!(machines.cancel_task(tasks[1]).await, None);
        // The next search of that folder takes the free number above the one that runs.
        say(&env, 3, "search for films").await;
        let labels: Vec<String> = machines.view().tasks(agent).map(|t| t.label()).collect();
        assert_eq!(labels, ["find-1", "find-2"]);
    }

    #[tokio::test]
    async fn a_task_s_end_reaches_its_agent_as_an_event_with_what_it_wrote() {
        let files = [
            ("root.toml", "tools = [\"note\"]"),
            (
                "root.fsm",
                "fsm Helper {\n[*] --> idle\nidle --> job : said\njob --> idle\n\
                 idle --> told : task_done\ntold --> idle\n\
                 idle --> sorry : task_failed\nsorry --> idle\n}",
            ),
            ("job/task.toml", "tools = [\"note\"]"),
            (
                "job/task.fsm",
                "fsm Job {\n[*] --> work\nwork --> save : done [keep]\nwork --> [*] : done [else]\n\
                 save --> [*]\n}",
            ),
            ("job/work/transcript.toml", "output = \"none\""),
            (
                "job/save/tool.toml",
                "tool = \"note\"\n[args.title]\nvalue = \"x\"\n[args.folder]\nvalue = \"y\"\n\
                 [args.body]\nvalue = \"{work}\"",
            ),
            (
                "told/generate.toml",
                "output = \"bubble\"\nprompt = \"{task.name} finished with: {task.result}\"",
            ),
            (
                "sorry/generate.toml",
                "output = \"bubble\"\nprompt = \"{task.name} failed\"",
            ),
        ];
        let guards = "\n[guards.keep]\nwhen = { transcript = \"^keep\" }\n\
                      prefer = { transcript = \"^keep\" }";
        let task = format!("{}{guards}", files[2].1);
        let mut files = files.to_vec();
        files[2].1 = &task;
        let (client, seen) = server(prefer(&[]), "Done.").await;
        let env = Env {
            flows: agent_tree(&files),
            tools: Some(tool_host()),
            machines: Arc::new(Runtime::new()),
            ..env(client, Some(&RecordingSink::new(Some(7))))
        };
        // The task ends in the take that started it: its agent takes `task_done`, and reads
        // which task it was and what it last wrote.
        let done = say(&env, 1, "tidy the notes").await;
        assert_eq!(done.error, None, "{:?}", done.notes);
        assert_eq!(
            inner(&done),
            [
                "idle said → job",
                "job done → idle",
                "[*] start → work",
                "work done → [*]",
                "idle task_done → told",
                "told done → idle"
            ]
        );
        assert_eq!(
            seen.generations.lock().unwrap()[0]["input"],
            "job finished with: tidy the notes"
        );
        // A task that fails with nothing of its own to handle it ends, and its agent takes
        // `task_failed`: the failure is handled, and the trace says why.
        let failed = say(&env, 2, "keep this one").await;
        assert_eq!(
            inner(&failed),
            [
                "idle said → job",
                "job done → idle",
                "[*] start → work",
                "work done → save",
                "save failed → [*]",
                "idle task_failed → sorry",
                "sorry done → idle"
            ]
        );
        assert_eq!(failed.error, None, "the agent handled it");
        assert!(
            failed.notes.iter().any(|n| n.contains("not confirmed")),
            "{:?}",
            failed.notes
        );
        assert_eq!(seen.generations.lock().unwrap()[1]["input"], "job failed");
        assert!(env.machines.view().at_rest());
    }

    #[tokio::test]
    async fn a_task_keeps_the_extracts_read_on_the_way_to_it() {
        let tree = agent_tree(&[
            ("root.toml", ""),
            (
                "root.fsm",
                "fsm App {\n[*] --> idle\nidle --> chat : said\nchat --> idle\n}",
            ),
            (
                "chat/decide.toml",
                "description = \"Chat\"\n\
                         [extract.channels]\n\
                         xpath = \"//TreeItem[.//Group[has-class(@class, 'p-channel_sidebar__channel')]]/@name\"\n\
                         as = \"list\"\n\
                         [extract.last]\n\
                         xpath = \"string((//ListItem[.//Text])[last()]//Text)\"\n\
                         lazy = true",
            ),
            ("chat/task/task.toml", "description = \"A reply\""),
            (
                "chat/task/task.fsm",
                "fsm Reply {\n[*] --> waiting\nwaiting --> reply : said\nreply --> [*]\n}",
            ),
            (
                "chat/task/reply/generate.toml",
                "output = \"bubble\"\nprompt = \"Last: {last}. Channels: {channels}. Said: {transcript}\"",
            ),
        ]);
        assert!(tree.is_valid(), "{:?}", tree.errors);
        // The second take is for the task that waits, not for another one.
        let (client, seen) = server(prefer(&["task-1"]), "Sure.").await;
        let recorded: jevons_desktop_core::recorded::RecordedTree =
            serde_json::from_str(include_str!("../../../examples/desktop/trees/slack.json"))
                .unwrap();
        let machines = Arc::new(Runtime::new());
        let env = Env {
            flows: tree,
            desk: Arc::new(LocalDesk::default().with_reader(Arc::new(
                jevons_desktop_core::reader::Reader::new(
                    Arc::new(jevons_desktop_core::recorded::RecordedInspector::new(
                        recorded,
                    )),
                    crate::context::Privacy::default(),
                ),
            ))),
            machines: machines.clone(),
            ..env(client, None)
        };
        let slack = |id| TakeStart {
            id,
            context: ContextSnapshot {
                app: AppInfo {
                    process_name: "slack.exe".into(),
                    ..AppInfo::default()
                },
                window: WindowInfo {
                    title: "general (Channel) - Acme - Slack".into(),
                    handle: Some(7),
                    ..WindowInfo::default()
                },
                ..ContextSnapshot::default()
            },
            entry: None,
        };
        let (updates, _) = mpsc::unbounded_channel();
        let first = run_transcript(&env, slack(1), "draft a reply", &updates).await;
        assert_eq!(first.error, None, "{:?}", first.notes);
        assert_eq!(machines.view().path(), "main › task › waiting");
        // A later take in the task still has the extract read on the way, and reads the lazy one.
        let second = run_transcript(&env, slack(2), "say yes", &updates).await;
        assert_eq!(second.error, None, "{:?}", second.notes);
        assert_eq!(machines.view().path(), "idle");
        let generation = serde_json::to_string(&seen.generations.lock().unwrap()[0]).unwrap();
        assert!(
            generation.contains("Last: Can someone review the release notes?. Channels: [")
                && generation.contains("Ana Silva")
                && generation.contains("Said: say yes"),
            "{generation}"
        );
    }

    #[tokio::test]
    async fn a_declined_tool_call_takes_the_denied_transition() {
        let (client, _) = server(prefer(&[]), "unused").await;
        let env = Env {
            flows: agent_tree(&[
                ("root.toml", "tools = [\"note\"]"),
                (
                    "root.fsm",
                    "fsm App {\n[*] --> idle\nidle --> saving : said\nsaving --> idle\nsaving --> told : denied\nstate told\ntold --> idle\n}",
                ),
                (
                    "saving/tool.toml",
                    "description = \"Saves a note\"\ntool = \"note\"\n[args.title]\nvalue = \"x\"\n[args.folder]\nvalue = \"y\"\n[args.body]\nvalue = \"{transcript}\"",
                ),
                ("told/transcript.toml", "output = \"clipboard\""),
            ]),
            tools: Some(tool_host()),
            ..env(client, Some(&RecordingSink::new(Some(7))))
        };
        let trace = say(&env, 1, "note that the build is green").await;
        assert_eq!(
            inner(&trace),
            [
                "idle said → saving",
                "saving denied → told",
                "told done → idle"
            ]
        );
        assert_eq!(trace.error, None, "the machine handled it");
        assert!(
            trace.notes.iter().any(|n| n.contains("not confirmed")),
            "{:?}",
            trace.notes
        );
        assert_eq!(trace.calls[0].confirmed, Some(false));
    }

    /// A server whose settings register an HTTP tool (answered here) and a client whose
    /// settings register a program: each asks the user unless its own settings say not to.
    async fn two_sides(
        client: &str,
        confirm: Option<mpsc::UnboundedSender<jevons_desktop_core::confirm::Confirmation>>,
    ) -> (Arc<ToolHost>, Arc<dyn Desk>, Arc<Mutex<usize>>) {
        let hits = Arc::new(Mutex::new(0usize));
        let got = hits.clone();
        let app = axum::Router::new().route(
            "/lookup",
            get(move || {
                *got.lock().unwrap() += 1;
                async { Json(json!({"found": "the answer"})) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let server: crate::config::ServerConfig = toml::from_str(&format!(
            "[tools.lookup]\nkind = \"http\"\nmethod = \"GET\"\ndescription = \"Looks it up\"\n\
             url = \"http://{address}/lookup?q={{query}}\"\narguments = {{ query = \"What\" }}\n"
        ))
        .unwrap();
        let client: jevons_desktop_core::config::ClientConfig = toml::from_str(client).unwrap();
        let mut desk = LocalDesk::default().with_tools(Arc::new(
            jevons_desktop_tools::ToolSet::new(&client.tools, &client.mcp),
        ));
        if let Some(confirm) = confirm {
            desk = desk.with_confirmer(Arc::new(
                jevons_desktop_core::confirm::ChannelConfirmer::new(confirm),
            ));
        }
        let desk: Arc<dyn Desk> = Arc::new(desk);
        let tools = Arc::new(ToolHost::new(&server.tools, &server.mcp).with_desk(desk.clone()));
        (tools, desk, hits)
    }

    /// A program the client runs: it echoes what was said.
    const SAY: &str = "[tools.say]\nkind = \"command\"\ndescription = \"Says it\"\n\
        program = \"echo\"\nargs = [\"{text}\"]\narguments = { text = \"What to say\" }\n";

    #[tokio::test]
    async fn a_server_tool_asks_through_the_client_and_a_client_tool_asks_by_itself() {
        if cfg!(windows) {
            return;
        }
        let (confirm, mut asked) =
            mpsc::unbounded_channel::<jevons_desktop_core::confirm::Confirmation>();
        let (tools, desk, hits) = two_sides(SAY, Some(confirm)).await;
        // The user says yes to each call, and what was asked is kept.
        let questions = Arc::new(Mutex::new(Vec::<String>::new()));
        let kept = questions.clone();
        tokio::spawn(async move {
            while let Some(question) = asked.recv().await {
                kept.lock().unwrap().push(question.tool.clone());
                let _ = question.reply.send(true);
            }
        });
        let catalog = tools.catalog();
        assert!(!catalog.tools["lookup"].client && catalog.tools["say"].client);
        let md = tools.tools_md();
        let about = |name: &str| {
            let at = md.find(&format!("## `{name}`")).unwrap();
            md[at..].lines().nth(4).unwrap().to_string()
        };
        assert_eq!(
            about("lookup"),
            "Runs on the server. Asks in the bubble before it runs."
        );
        assert_eq!(
            about("say"),
            "Runs on the client. Asks in the bubble before it runs."
        );
        let take_with = |tool: &'static str, argument: &'static str| {
            let (tools, desk, catalog) = (tools.clone(), desk.clone(), catalog.clone());
            async move {
                let (client, _) = server(prefer(&[]), "unused").await;
                let node =
                    format!("tool = \"{tool}\"\n[args.{argument}]\nvalue = \"{{transcript}}\"");
                let tree = FlowTree::load(
                    &Memory::new("test", [("tool.toml", node.as_str())]),
                    &catalog,
                );
                assert!(tree.is_valid(), "{:?}", tree.errors);
                let env = Env {
                    flows: Arc::new(tree),
                    tools: Some(tools),
                    desk,
                    ..env(client, None)
                };
                take(&env, None).await
            }
        };
        // The server's tool: the server asks, through the client, and then runs it itself.
        let trace = take_with("lookup", "query").await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.calls[0].confirmed, Some(true));
        assert_eq!(*hits.lock().unwrap(), 1);
        assert!(trace.output.contains("the answer"), "{}", trace.output);
        // The client's tool: the node does not ask, the server does not ask, and the client
        // asks all the same, because its settings say so. Then it runs there.
        let trace = take_with("say", "text").await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.calls[0].confirmed, Some(true));
        assert_eq!(trace.output.trim(), "hello world");
        assert_eq!(*questions.lock().unwrap(), ["lookup", "say"]);
    }

    #[tokio::test]
    async fn a_declined_client_tool_takes_the_denied_transition() {
        // Nobody answers at the client: its tool does not run, and the machine takes `denied`.
        let (tools, desk, _) = two_sides(SAY, None).await;
        let files = [
            ("root.toml", "tools = [\"say\"]"),
            (
                "root.fsm",
                "fsm App {\n[*] --> idle\nidle --> saying : said\nsaying --> idle\nsaying --> told : denied\nstate told\ntold --> idle\n}",
            ),
            (
                "saying/tool.toml",
                "description = \"Says it\"\ntool = \"say\"\n[args.text]\nvalue = \"{transcript}\"",
            ),
            ("told/transcript.toml", "output = \"clipboard\""),
        ];
        let mut wrapped = vec![
            ("root.toml".to_string(), "tools = [\"say\"]".to_string()),
            (
                "root.fsm".to_string(),
                "fsm Root {\n[*] --> idle\nidle --> main : said\nmain --> idle\n}".to_string(),
            ),
        ];
        for (path, text) in files {
            let (path, text) = match path {
                "root.toml" => (
                    "main/agent.toml".to_string(),
                    format!("description = \"Main\"\n{text}"),
                ),
                "root.fsm" => ("main/agent.fsm".to_string(), text.to_string()),
                other => (format!("main/{other}"), text.to_string()),
            };
            wrapped.push((path, text));
        }
        let tree = FlowTree::load(
            &Memory::new(
                "test",
                wrapped.iter().map(|(p, t)| (p.as_str(), t.as_str())),
            ),
            &tools.catalog(),
        );
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let (client, _) = server(prefer(&[]), "unused").await;
        let env = Env {
            flows: Arc::new(tree),
            tools: Some(tools),
            desk,
            ..env(client, None)
        };
        let trace = say(&env, 1, "the build is green").await;
        assert_eq!(
            inner(&trace),
            [
                "idle said → saying",
                "saying denied → told",
                "told done → idle"
            ]
        );
        assert_eq!(trace.calls[0].tool, "say");
        assert_eq!(trace.calls[0].confirmed, Some(false));
        assert!(
            trace.notes.iter().any(|n| n.contains("not confirmed")),
            "{:?}",
            trace.notes
        );
    }

    #[tokio::test]
    async fn a_tool_both_sides_register_is_nobody_s_to_run() {
        let twice = "[tools.lookup]\nkind = \"open\"\ndescription = \"Opens it\"\nurl = \"{query}\"\n\
            arguments = { query = \"What\" }\n";
        let (tools, _, _) = two_sides(twice, None).await;
        assert_eq!(
            tools.clashes(),
            ["the tool lookup is registered by both the server and the client"]
        );
        let refused = tools
            .resolve(&["lookup".into()], "any")
            .await
            .err()
            .unwrap();
        assert_eq!(
            refused,
            "the tool lookup is registered by both the server and the client"
        );
    }

    /// The built-in tree with `examples/desktop/machines/research` added as its README says:
    /// `web_search` in the server's settings, answered by a fake search here, and `open_url` in
    /// the client's, where nothing opens and the user is asked through `confirm`. Returns the
    /// tree, the server's tools, the client's desk and the searches the fake got.
    async fn with_search_example(
        confirm: mpsc::UnboundedSender<jevons_desktop_core::confirm::Confirmation>,
    ) -> (
        Arc<FlowTree>,
        Arc<ToolHost>,
        Arc<dyn Desk>,
        Arc<Mutex<Vec<String>>>,
    ) {
        let example = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/desktop/machines/research");
        let read = |file: &str| std::fs::read_to_string(example.join(file)).unwrap();
        let search: Vec<(String, String)> = [
            "agent.toml",
            "agent.fsm",
            "search/task.toml",
            "search/task.fsm",
            "search/searching/tool.toml",
            "search/answering/generate.toml",
            "search/opening/tool.toml",
        ]
        .iter()
        .map(|f| (format!("research/{f}"), read(f)))
        .collect();
        let mut files: Vec<(String, String)> = crate::flow::defaults::TREE
            .iter()
            .map(|(p, t)| (p.to_string(), t.to_string()))
            .collect();
        for (path, text) in &mut files {
            if path == "root.fsm" {
                *text = text.replace(
                    "    assistant --> idle",
                    "    idle --> research : said\n    research --> idle\n    assistant --> idle",
                );
            }
            if path == "root.toml" {
                *text = text.replace(
                    "tools = [\"script:*\"]",
                    "tools = [\"script:*\", \"web_search\", \"open_url\"]",
                );
            }
        }
        files.extend(search);
        // A search service that answers every query with one result.
        let searched = Arc::new(Mutex::new(Vec::<String>::new()));
        let got = searched.clone();
        let search = axum::Router::new().route(
            "/search",
            get(move |query: axum::extract::RawQuery| {
                got.lock().unwrap().push(query.0.unwrap_or_default());
                async { Json(json!({"results": [{"title": "jevons-fsm", "url": "https://x/1"}]})) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, search).await.unwrap() });
        let server: crate::config::ServerConfig = toml::from_str(&format!(
            r#"
[tools.web_search]
kind = "http"
method = "GET"
description = "Searches the web and returns the top results as JSON"
url = "http://{address}/search?q={{query}}&count=5"
arguments = {{ query = "What to search for" }}
confirm = false
allow = ["research/search/*"]
"#
        ))
        .unwrap();
        let client: jevons_desktop_core::config::ClientConfig = toml::from_str(
            r#"
[tools.open_url]
kind = "open"
description = "Opens an address in the default browser"
url = "{url}"
arguments = { url = "The address to open" }
allow = ["research/search/*"]
"#,
        )
        .unwrap();
        // The client's tools check and ask as always, and then only say what they would do:
        // no browser opens in a test.
        let desk: Arc<dyn Desk> = Arc::new(
            LocalDesk::default()
                .with_confirmer(Arc::new(
                    jevons_desktop_core::confirm::ChannelConfirmer::new(confirm),
                ))
                .with_tools(Arc::new(jevons_desktop_tools::ToolSet::new(
                    &client.tools,
                    &client.mcp,
                )))
                .dry_run(),
        );
        let tools = Arc::new(ToolHost::new(&server.tools, &server.mcp).with_desk(desk.clone()));
        let catalog = tools.catalog();
        assert!(!catalog.tools["web_search"].client && catalog.tools["open_url"].client);
        let tree = FlowTree::load(
            &Memory::new("test", files.iter().map(|(p, t)| (p.as_str(), t.as_str()))),
            &catalog,
        );
        assert!(tree.is_valid(), "{:?}", tree.errors);
        (Arc::new(tree), tools, desk, searched)
    }

    #[tokio::test]
    async fn the_search_example_searches_answers_and_opens_a_result_once_approved() {
        let (confirm, mut asked) =
            mpsc::unbounded_channel::<jevons_desktop_core::confirm::Confirmation>();
        let (flows, tools, desk, searched) = with_search_example(confirm).await;
        let labels = &["research", "search-1", "opening", "end"];
        let (client, seen) = server(prefer(labels), "1. jevons-fsm").await;
        let approver = tokio::spawn(async move {
            let call = asked.recv().await.unwrap();
            call.reply.send(true).unwrap();
            call.tool
        });
        let machines = Arc::new(Runtime::new());
        let env = Env {
            flows,
            tools: Some(tools),
            desk,
            machines: machines.clone(),
            ..env(client, None)
        };
        // "Buscar" (like "Search", "Busca" or "Búscame") starts a search with no decision: the
        // agent's rule prefers it, which makes the root prefer the agent.
        let first = say(&env, 1, "Buscar crates de máquinas de estado para Rust").await;
        assert_eq!(first.error, None, "{:?}", first.notes);
        assert_eq!(
            first.flow[0].how.as_deref(),
            Some("preferred: its transcript rule passed")
        );
        assert_eq!(
            moves(&first),
            [
                "idle said → research",
                "research done → idle",
                "idle said → search",
                "search done → idle",
                "[*] start → searching",
                "searching done → answering",
                "answering done → results"
            ]
        );
        // The search ran with the server: its request reached the search service.
        assert_eq!(first.calls[0].tool, "web_search");
        assert_eq!(searched.lock().unwrap().len(), 1);
        assert!(
            first.calls[0]
                .result
                .as_ref()
                .unwrap()
                .contains("jevons-fsm")
        );
        assert_eq!(first.delivery, Some(DeliveryOutcome::Shown));
        assert_eq!(machines.view().path(), "research › search › results");
        assert!(seen.decisions.lock().unwrap().is_empty());
        // The same words while that search waits: the agent's rule would start another, and
        // the search's own rule searches again. The one that runs has the words, with no
        // model at any level, so searches do not pile up.
        let again = say(&env, 2, "Búscame las asíncronas").await;
        assert_eq!(again.error, None, "{:?}", again.notes);
        assert_eq!(
            inner(&again),
            [
                "idle said → task search-1",
                "results said → searching",
                "searching done → answering",
                "answering done → results"
            ]
        );
        assert!(seen.decisions.lock().unwrap().is_empty());
        for step in again.machine.iter().filter(|s| s.event == "said") {
            assert_eq!(
                step.how, "preferred: its transcript rule passed",
                "{step:?}"
            );
        }
        // A lead-in before the word starts nothing new either, and voseo counts.
        for words in ["Ah, buscame las de tokio", "A ver, buscá otra cosa"] {
            let more = say(&env, 20, words).await;
            assert_eq!(inner(&more)[..2], inner(&again)[..2], "{words}");
        }
        assert!(seen.decisions.lock().unwrap().is_empty());
        // A follow-up is for the agent only because its search waits. It opens a result, once
        // the user approves it in the bubble.
        let second = say(&env, 3, "open the first one").await;
        assert_eq!(second.error, None, "{:?}", second.notes);
        assert_eq!(
            inner(&second),
            [
                "idle said → task search-1",
                "results said → opening",
                "opening done → results"
            ]
        );
        assert_eq!(seen.decisions.lock().unwrap().len(), 1, "one request");
        // The root reads what the agent is in the middle of, not its description alone: these
        // words name no search.
        let asked = seen.decisions.lock().unwrap()[0]["questions"]["q00"].to_string();
        assert!(
            asked.contains("Waiting now for what the user says next: For the running task search (Search), now at results")
                && asked.contains("abrir el segundo"),
            "{asked}"
        );
        // Opening is the client's: it asked the user by itself, and ran there.
        assert_eq!(approver.await.unwrap(), "open_url");
        assert_eq!(second.calls[0].tool, "open_url");
        assert_eq!(second.calls[0].confirmed, Some(true));
        assert!(
            second.calls[0]
                .result
                .as_ref()
                .unwrap()
                .contains("would_call"),
            "{:?}",
            second.calls[0].result
        );
        assert_eq!(searched.lock().unwrap().len(), 4, "four searches, no more");
        // What is dictated while the search waits: the model takes it for the search, which
        // does not know what to do with it. The example hands it up, to the agent and then
        // the root, whose `[else]` types it. The search still waits.
        let known = &["research", "search-1", "verbatim"];
        let (client, seen) = server(sure_of(known), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let dictating = Env {
            routes: routes(client),
            desk: self::desk(Some(&sink)),
            ..env.clone()
        };
        let dictated = say(&dictating, 9, "hello team, the build is green").await;
        assert_eq!(dictated.error, None, "{:?}", dictated.notes);
        let typed: Vec<String> = sink.requests().iter().map(|r| r.text.clone()).collect();
        assert_eq!(
            typed,
            ["hello team, the build is green"],
            "{:?}",
            dictated.notes
        );
        assert!(
            dictated
                .notes
                .iter()
                .any(|n| n.starts_with("search-1 did not take what you said"))
                && dictated
                    .notes
                    .iter()
                    .any(|n| n.starts_with("research did not take")),
            "{:?}",
            dictated.notes
        );
        assert_eq!(seen.decisions.lock().unwrap().len(), 1, "one decision call");
        assert_eq!(machines.view().stack.last().unwrap().state, "results");
        // Done: the search ends, and with none waiting the agent is no choice for the root.
        let (client, _) = server(prefer(&["research", "search-1", "end"]), "unused").await;
        let env = Env {
            routes: routes(client),
            ..env
        };
        let third = say(&env, 4, "thanks, that's all").await;
        assert_eq!(
            inner(&third),
            ["idle said → task search-1", "results said → [*]"]
        );
        assert_eq!(machines.view().path(), "idle");
        let (client, seen) = server(prefer(&["research", "dictation", "verbatim"]), "unused").await;
        let env = Env {
            routes: routes(client),
            ..env
        };
        let later = say(&env, 5, "open the first one").await;
        assert_eq!(later.machine[0].to, "dictation");
        let asked = serde_json::to_string(&seen.decisions.lock().unwrap()[0]).unwrap();
        assert!(!asked.contains("\"research\""), "{asked}");
    }

    #[tokio::test]
    async fn an_agent_node_calls_its_tools_and_its_answer_goes_to_the_bubble() {
        let (client, _) = server_with(
            prefer(&[]),
            "unused",
            vec![
                json!({"call": "search", "arguments": {"query": "launch date"}}),
                json!("It is on Friday."),
            ],
        )
        .await;
        let env = Env {
            flows: tree_of(&[("loop.toml", "tools = [\"search\"]\nmax_steps = 3")]),
            tools: Some(tool_host()),
            ..env(client, None)
        };
        let (audio, finish) = one_second_of_audio();
        let (updates, mut shown) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: context(None),
            entry: None,
        };
        let trace = run_take(&env, start, audio, finish, &updates).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.output, "It is on Friday.");
        assert_eq!(trace.delivery, Some(DeliveryOutcome::Shown));
        assert_eq!(trace.calls[0].tool, "search");
        assert!(
            trace.calls[0]
                .result
                .as_deref()
                .unwrap()
                .contains("dry_run")
        );
        let mut streamed = String::new();
        let mut answering = false;
        while let Ok(update) = shown.try_recv() {
            match update {
                Update::Answering => answering = true,
                Update::Output(text) => streamed.push_str(&text),
                _ => {}
            }
        }
        assert!(answering);
        assert_eq!(streamed, "It is on Friday.");
    }

    /// A live turn in a fake Realtime session: its live deltas, then its final transcript.
    type Turn = (&'static [&'static str], &'static str);

    /// A fake Realtime session: a turn (its deltas, then its transcript) completes after every
    /// second append, and a commit of the (then empty) buffer is an error.
    async fn realtime_server(turns: &'static [Turn]) -> Client {
        use axum::extract::ws::{Message, WebSocketUpgrade};
        let session = move |upgrade: WebSocketUpgrade| async move {
            upgrade.protocols(["realtime"]).on_upgrade(move |mut socket| async move {
                let mut next = turns.iter();
                while let Some(Ok(Message::Text(text))) = socket.recv().await {
                    let event: Value = serde_json::from_str(&text).unwrap();
                    let replies: Vec<Value> = match event["type"].as_str().unwrap() {
                        "session.update" => {
                            // The app commits the phrases itself.
                            assert!(event["session"]["audio"]["input"]["turn_detection"].is_null());
                            Vec::new()
                        }
                        "input_audio_buffer.append" => Vec::new(),
                        // Each phrase the app commits is the next scripted turn.
                        "input_audio_buffer.commit" => {
                            match next.next() {
                                Some((deltas, transcript)) => deltas
                                    .iter()
                                    .map(|d| json!({"type": "conversation.item.input_audio_transcription.delta",
                                                    "item_id": "i", "content_index": 0, "delta": d}))
                                    .chain([json!({"type": "conversation.item.input_audio_transcription.completed",
                                                   "item_id": "i", "content_index": 0, "transcript": transcript})])
                                    .collect(),
                                None => vec![json!({"type": "error", "error": {"message": "buffer too small"}})],
                            }
                        }
                        "input_audio_buffer.clear" => vec![json!({"type": "input_audio_buffer.cleared"})],
                        _ => vec![json!({"type": "error", "error": {"message": "buffer too small"}})],
                    };
                    for reply in replies {
                        socket.send(Message::Text(reply.to_string().into())).await.unwrap();
                    }
                }
            })
        };
        let (client, _) = server(prefer(&["dictation", "verbatim"]), "unused").await;
        let app = axum::Router::new()
            .route("/v1/realtime", get(session))
            .fallback_service(axum::routing::any(
                move |request: axum::extract::Request| {
                    let base = client.base().to_string();
                    async move {
                        // Everything else goes to the scripted HTTP server.
                        let url = format!("{base}{}", request.uri());
                        let (parts, body) = request.into_parts();
                        let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();
                        let response = reqwest::Client::new()
                            .request(parts.method, url)
                            .header("content-type", "application/json")
                            .body(body)
                            .send()
                            .await
                            .unwrap();
                        let status = response.status();
                        (status, response.bytes().await.unwrap())
                    }
                },
            ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Client::new(&base, None)
    }

    /// Runs live dictation over two phrases with a pause between them.
    async fn live(env: &Env, chunks: &[i16]) -> (Trace, Vec<Update>) {
        let (audio, received) = mpsc::unbounded_channel();
        for amplitude in chunks {
            audio.send(AudioEvent::Chunk(chunk_at(*amplitude))).unwrap();
        }
        audio.send(AudioEvent::Ended).unwrap();
        let (_stop, stopped) = oneshot::channel();
        let (updates, mut seen) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 3,
            context: context(Some("old text")),
            entry: None,
        };
        let trace = run_live(env, start, received, stopped, &updates).await;
        let mut shown = Vec::new();
        while let Ok(update) = seen.try_recv() {
            shown.push(update);
        }
        (trace, shown)
    }

    #[tokio::test]
    async fn live_dictation_shows_each_phrase_and_types_the_whole_text_once_stopped() {
        let client = realtime_server(&[
            (&["hello", " there"], "hello there"),
            (&["sekond", " phrase"], "second phrase."),
        ])
        .await;
        let sink = RecordingSink::new(Some(7));
        let env = env(client, Some(&sink));
        let speech = [[6000; 4].as_slice(), &[30; 8], &[6000; 4]].concat();
        let (trace, shown) = live(&env, &speech).await;
        assert_eq!(trace.error, None);
        assert_eq!(trace.transcript, "hello there second phrase.");
        // Typed once, after stopping.
        let typed: Vec<String> = sink.requests().into_iter().map(|r| r.text).collect();
        assert_eq!(typed, ["hello there second phrase."]);
        let heard: Vec<&str> = shown
            .iter()
            .filter_map(|u| match u {
                Update::Heard(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(heard, ["hello there", "hello there second phrase."]);
        assert!(
            shown
                .iter()
                .any(|u| matches!(u, Update::Delta(d) if d == "sekond"))
        );
        assert!(
            shown.iter().any(
                |u| matches!(u, Update::StageDone { chosen: Some(c), .. } if c == "dictation")
            )
        );
    }

    #[tokio::test]
    async fn live_dictation_without_speech_types_nothing() {
        let client = realtime_server(&[]).await;
        let sink = RecordingSink::new(Some(7));
        let env = env(client, Some(&sink));
        let (trace, _) = live(&env, &[30; 12]).await;
        assert_eq!(trace.error.as_deref(), Some("No speech was recognized"));
        assert!(sink.requests().is_empty());
    }

    #[tokio::test]
    async fn a_stalled_decision_types_the_transcript_without_generating() {
        let generations = Arc::new(Mutex::new(0));
        let counted = generations.clone();
        let app = axum::Router::new()
            .route(
                "/v1/audio/transcriptions",
                post(|_upload: axum::body::Bytes| async { Json(json!({"text": "hello world"})) }),
            )
            .route(
                "/v1/systemone",
                post(|| async {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    Json(json!({}))
                }),
            )
            .route(
                "/v1/responses",
                post(move || {
                    *counted.lock().unwrap() += 1;
                    async { "" }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let sink = RecordingSink::new(Some(7));
        let env = Env {
            settings: Settings {
                decision_timeout: Duration::from_millis(200),
                ..Settings::default()
            },
            ..env(Client::new(&base, None), Some(&sink))
        };
        let trace = take(&env, None).await;
        assert_eq!(trace.error, None);
        assert_eq!(trace.output, "hello world");
        assert_eq!(*generations.lock().unwrap(), 0);
        assert!(
            trace.notes.iter().any(|n| n.contains("did not answer")),
            "{:?}",
            trace.notes
        );
        assert_eq!(sink.requests()[0].text, "hello world");
    }

    /// 100 ms of audio at `amplitude`.
    fn chunk_at(amplitude: i16) -> Vec<i16> {
        (0..SAMPLE_RATE as usize / 10)
            .map(|i| if i % 2 == 0 { amplitude } else { -amplitude })
            .collect()
    }

    #[test]
    fn phrases_end_at_pauses_after_speech_and_keep_every_word() {
        let mut phrases = Phrases::default();
        for _ in 0..5 {
            assert_eq!(phrases.push(&chunk_at(30)), Phrase::Continue, "room noise");
        }
        for _ in 0..10 {
            assert_eq!(phrases.push(&chunk_at(6000)), Phrase::Continue, "speech");
        }
        let ends: Vec<Phrase> = (0..7).map(|_| phrases.push(&chunk_at(30))).collect();
        assert_eq!(
            ends[..6],
            [Phrase::Continue; 6],
            "a short pause keeps the phrase"
        );
        assert_eq!(ends[6], Phrase::Commit, "0.7 s of quiet ends it");
    }

    #[test]
    fn nonstop_speech_is_committed_every_twenty_seconds() {
        let mut phrases = Phrases::default();
        // Words, with the short dips between them.
        let commits = (0..400)
            .map(|i| if i % 4 == 3 { 30 } else { 6000 })
            .filter(|a| phrases.push(&chunk_at(*a)) == Phrase::Commit)
            .count();
        assert_eq!(commits, 2);
    }

    #[test]
    fn noise_alone_is_dropped_and_never_committed() {
        let mut phrases = Phrases::default();
        let results: Vec<Phrase> = (0..40).map(|_| phrases.push(&chunk_at(30))).collect();
        assert!(!results.contains(&Phrase::Commit));
        assert_eq!(results.iter().filter(|p| **p == Phrase::Clear).count(), 1);
        assert!(!phrases.heard());
    }
}
