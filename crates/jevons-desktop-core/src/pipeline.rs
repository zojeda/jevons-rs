//! One take: stream the microphone to the transcriber, walk the flow tree to a leaf, and send
//! the leaf's text where it says: into the application, the bubble or the clipboard.
//!
//! Every step is recorded in a [`Trace`] for the inspector.

use crate::client::{
    Client, ClientError, DecisionRequest, DecisionResponse, RealtimeEvent, ResponseRequest, Turns,
};
use crate::context::ContextSnapshot;
use crate::delivery::{Decision, Pending};
use crate::flow::FlowTree;
use crate::flow::frame::Frame;
use crate::flow::investigate::Investigate;
use crate::flow::spec::Output;
use crate::flow::tools::ToolHost;
use crate::flow::walk::{self, FlowStep, Leaf, ToolTrace};
use crate::platform::{
    Action, AudioEvent, DeliveryMethod, DeliveryOutcome, DeliveryRequest, SAMPLE_RATE, TextSink,
};
use serde::Serialize;
use std::sync::{Arc, Mutex};
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

/// The model ids to request; each is `None` when that service is not available.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ServiceModels {
    pub speech: Option<String>,
    pub decision: Option<String>,
    pub generative: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub models: ServiceModels,
    /// Try Realtime first; uploads are the fallback.
    pub realtime: bool,
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
            models: ServiceModels::default(),
            realtime: true,
            language: None,
            decide: true,
            max_output_tokens: 1024,
            decision_timeout: Duration::from_secs(60),
            generation_timeout: Duration::from_secs(120),
        }
    }
}

/// What the pipeline needs from the app.
pub struct Env {
    pub client: Client,
    /// The flow tree takes walk (the last one that loaded without errors).
    pub flows: Arc<FlowTree>,
    pub settings: Settings,
    /// `None` for a dry run: the text is only recorded.
    pub sink: Option<Arc<Mutex<Box<dyn TextSink>>>>,
    /// Answers `[investigate]` questions; without it their answers are empty.
    pub investigator: Option<Arc<dyn Investigate>>,
    /// Approves tool calls that need confirmation; without it they are denied.
    pub confirmer: Option<Arc<crate::flow::confirm::ChannelConfirmer>>,
    /// The tools the settings register; without them tool and agent nodes fail.
    pub tools: Option<Arc<ToolHost>>,
}

/// A take as it starts.
#[derive(Clone, Debug)]
pub struct TakeStart {
    pub id: u64,
    pub context: ContextSnapshot,
    /// The branch of the flow tree to start at, such as `ask`; `None` starts at the root.
    pub entry: Option<String>,
}

/// Live updates for the tray and inspector.
#[derive(Clone, Debug, PartialEq)]
pub enum Update {
    Level([u8; 5]),
    /// Words recognized so far in the phrase being spoken.
    Delta(String),
    /// Live dictation: every phrase finished so far; the phrase being spoken starts over.
    Heard(String),
    Transcribing,
    Thinking,
    /// A step of processing, such as the route through the flow tree.
    Step(String),
    /// Generated text.
    Output(String),
    /// The text being generated is an answer for the bubble, not text for the application.
    Answering,
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

    /// The branches taken, such as `dictate/notes → _actions/rewrite`.
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

/// Walks the flow tree for a transcribed take and sends the leaf's text where it goes.
async fn finish_take(
    env: &Env,
    start: &TakeStart,
    updates: &mpsc::UnboundedSender<Update>,
    mut trace: Trace,
) -> Trace {
    let take = start.id;
    let tree = env.flows.clone();
    let entry = match start.entry.as_deref() {
        Some(path) => tree.find(path).unwrap_or_else(|| {
            trace.notes.push(format!(
                "The flow tree has no branch {path}: starting at the root"
            ));
            tree.root()
        }),
        None => tree.root(),
    };
    trace.entry = tree.node(entry).label().to_string();
    let frame = Frame::new(start.context.clone(), trace.transcript.clone());
    let walked = Instant::now();
    let leaf = walk::run(env, &tree, entry, frame, updates, &mut trace).await;
    trace.time("walk", walked);
    let leaf = match leaf {
        Ok(leaf) => leaf,
        Err(e) => {
            trace.error = Some(e.to_string());
            return trace;
        }
    };
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
                if let Some(sink) = &env.sink {
                    trace.delivery = match sink
                        .lock()
                        .expect("the sink lock is not poisoned")
                        .copy(&leaf.text)
                    {
                        Ok(()) => Some(DeliveryOutcome::OnClipboard {
                            reason: "the flow sends it to the clipboard".into(),
                        }),
                        Err(e) => {
                            trace.error = Some(e.to_string());
                            None
                        }
                    };
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
    trace
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
    let model = settings.models.speech.as_deref();
    let mut session = None;
    if settings.realtime {
        match env
            .client
            .realtime(model, settings.language.as_deref(), Turns::Client)
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
    let model = model.ok_or(ClientError::NotServed("Speech to text"))?;
    trace.transcription = Some(TranscriptionPath::Upload);
    let transcription = env
        .client
        .transcribe(&buffer, SAMPLE_RATE, model, settings.language.as_deref())
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
    let (mut writer, mut reader) = match env
        .client
        .realtime(
            settings.models.speech.as_deref(),
            settings.language.as_deref(),
            Turns::Client,
        )
        .await
    {
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

/// Delivers a leaf's text into the application once it is safe, or leaves it on the clipboard.
async fn deliver(
    env: &Env,
    start: &TakeStart,
    leaf: &Leaf,
) -> Result<Option<DeliveryOutcome>, String> {
    let Some(sink) = &env.sink else {
        return Ok(None);
    };
    let request = DeliveryRequest {
        action: leaf.action,
        text: leaf.text.clone(),
        method: leaf.delivery,
        select_all: leaf.action == Action::Rewrite && start.context.selection().is_none(),
        erase: 0,
    };
    let window = start.context.window.handle.unwrap_or(0);
    deliver_text(sink, start.id, window, request).await
}

/// Types `request` into `window` once every key is released, or leaves it on the clipboard when
/// the window changed or keys stayed down (the bubble's Insert uses it too).
pub async fn deliver_text(
    sink: &Arc<Mutex<Box<dyn TextSink>>>,
    take: u64,
    window: u64,
    request: DeliveryRequest,
) -> Result<Option<DeliveryOutcome>, String> {
    let copy = |reason: String| -> Result<Option<DeliveryOutcome>, String> {
        sink.lock()
            .expect("the sink lock is not poisoned")
            .copy(&request.text)
            .map_err(|e| e.to_string())?;
        Ok(Some(DeliveryOutcome::OnClipboard { reason }))
    };
    if request.method == DeliveryMethod::Clipboard {
        return copy("the flow delivers to the clipboard".into());
    }
    let pending = Pending::new(take, window, Instant::now());
    loop {
        let (foreground, keys_down) = {
            let sink = sink.lock().expect("the sink lock is not poisoned");
            (sink.foreground_window().unwrap_or(0), sink.keys_down())
        };
        match pending.decide(take, foreground, keys_down, Instant::now()) {
            Decision::Deliver => {
                let result = sink
                    .lock()
                    .expect("the sink lock is not poisoned")
                    .deliver(&request);
                return match result {
                    Ok(outcome) => Ok(Some(outcome)),
                    Err(e) => copy(format!("delivery failed: {e}")),
                };
            }
            Decision::Manual => {
                return copy(if foreground != window {
                    "the focused window changed".into()
                } else {
                    "keys were held too long".into()
                });
            }
            Decision::Wait => tokio::time::sleep(Duration::from_millis(25)).await,
            Decision::Cancel => return Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AppInfo, Element, WindowInfo};
    use crate::fake::RecordingSink;
    use crate::flow::{Catalog, Memory};
    use axum::Json;
    use axum::routing::{get, post};
    use serde_json::{Value, json};

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

    fn settings() -> Settings {
        Settings {
            models: ServiceModels {
                speech: Some("parakeet".into()),
                decision: Some("jev".into()),
                generative: Some("jev".into()),
            },
            ..Settings::default()
        }
    }

    fn builtin() -> Arc<FlowTree> {
        let tree = FlowTree::load(&crate::flow::defaults::builtin(), &Catalog::default());
        assert!(tree.is_valid(), "{:?}", tree.errors);
        Arc::new(tree)
    }

    fn env(client: Client, sink: Option<&RecordingSink>) -> Env {
        Env {
            client,
            flows: builtin(),
            settings: settings(),
            sink: sink.map(RecordingSink::shared),
            investigator: None,
            confirmer: None,
            tools: None,
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
        let (client, seen) = server(prefer(&["dictate", "verbatim"]), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let trace = take(&env(client, Some(&sink)), None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.transcription, Some(TranscriptionPath::Upload));
        assert_eq!(*seen.uploads.lock().unwrap(), 1);
        assert!(trace.notes[0].contains("Realtime"), "{:?}", trace.notes);
        assert_eq!(trace.transcript, "hello world");
    }

    #[tokio::test]
    async fn words_needing_no_edits_are_typed_after_one_merged_decision() {
        let (client, seen) = server(prefer(&["dictate", "verbatim"]), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let trace = take(&env(client, Some(&sink)), None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert!(seen.generations.lock().unwrap().is_empty());
        assert_eq!(trace.output, "hello world");
        let delivered = sink.requests();
        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].action, Action::Insert);
        assert_eq!(delivered[0].text, "hello world");
        assert_eq!(trace.route(), "dictate → dictate/notes → _actions/verbatim");
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
            .find(|s| s.node == "dictate/notes")
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
            server(prefer(&["dictate", "rewrite"]), "Dear team, hello world.").await;
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
    async fn a_question_is_answered_in_the_bubble_and_never_typed() {
        let (client, seen) = server(prefer(&["ask"]), "It is five o'clock.").await;
        let sink = RecordingSink::new(Some(7));
        let trace = take(&env(client, Some(&sink)), None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        let leaf = trace.leaf.as_ref().unwrap();
        assert_eq!(
            (leaf.node.as_str(), leaf.output),
            ("ask/any", Output::Bubble)
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
        let (client, seen) = server(prefer(&["dictate"]), "An answer.").await;
        let trace = take_at(&env(client, None), None, Some("ask")).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.entry, "ask");
        assert_eq!(trace.leaf.as_ref().unwrap().node, "ask/any");
        assert!(
            seen.decisions.lock().unwrap().is_empty(),
            "ask chooses by rules"
        );
    }

    #[tokio::test]
    async fn an_unsure_root_decision_takes_the_fallback() {
        let (client, _) = server(prefer_with(&["ask", "verbatim"], 0.3), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let trace = take(&env(client, Some(&sink)), None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        let root = &trace.flow[0];
        assert_eq!(root.chosen.as_deref(), Some("dictate"));
        assert!(
            root.how.as_deref().unwrap().contains("unsure"),
            "{:?}",
            root.how
        );
        assert_eq!(sink.requests()[0].text, "hello world");
    }

    #[tokio::test]
    async fn a_changed_window_leaves_the_text_on_the_clipboard() {
        let (client, _) = server(prefer(&["dictate", "verbatim"]), "unused").await;
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
        let (client, seen) = server(prefer(&["dictate"]), "unused").await;
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

    fn tool_host() -> Arc<ToolHost> {
        let config: crate::config::DesktopConfig = toml::from_str(
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
        let (confirm, mut asked) = mpsc::unbounded_channel::<crate::flow::confirm::Confirmation>();
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
            confirmer: Some(Arc::new(crate::flow::confirm::ChannelConfirmer::new(
                confirm,
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
            confirmer: None,
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
            flows: tree_of(&[("agent.toml", "tools = [\"search\"]\nmax_steps = 3")]),
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
        let (client, _) = server(prefer(&["dictate", "verbatim"]), "unused").await;
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
            shown
                .iter()
                .any(|u| matches!(u, Update::Step(s) if s.starts_with("Route")))
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
                ..settings()
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
