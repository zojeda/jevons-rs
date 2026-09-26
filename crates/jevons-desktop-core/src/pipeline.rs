//! One dictation take: stream the microphone to the transcriber, choose a profile, ask the
//! decision model what to do, generate the text when it needs editing, and deliver it.
//!
//! Every step is recorded in a [`Trace`] for the inspector.

use crate::client::{
    Answer, Client, ClientError, DecisionRequest, DecisionResponse, Question, RealtimeEvent,
    ResponseRequest,
};
use crate::context::ContextSnapshot;
use crate::delivery::{Decision, Pending};
use crate::platform::{
    Action, AudioEvent, DeliveryOutcome, DeliveryRequest, SAMPLE_RATE, TextSink,
};
use crate::profile::{ActionPreference, DeliveryMethod, Effective, Profiles, Resolution};
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, oneshot};

/// The shortest audio worth transcribing (the server's minimum commit).
const MIN_SAMPLES: usize = SAMPLE_RATE as usize / 10;
/// How long to wait for the final transcript after the take ends.
const TRANSCRIPT_TIMEOUT: Duration = Duration::from_secs(30);

const BASE_INSTRUCTIONS: &str = "You turn dictated speech into the exact text to type into the \
focused field of the user's application. Output only that text: no quotes, no preamble, no \
commentary. Fix punctuation, capitalization and obvious recognition errors, and keep the \
user's language and meaning.";

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
    pub decide: bool,
    pub generation_threshold: f64,
    pub max_output_tokens: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            models: ServiceModels::default(),
            realtime: true,
            language: None,
            decide: true,
            generation_threshold: 0.5,
            max_output_tokens: 1024,
        }
    }
}

/// What the pipeline needs from the app.
pub struct Env {
    pub client: Client,
    pub profiles: Arc<Profiles>,
    pub settings: Settings,
    /// `None` for a dry run: the text is only recorded.
    pub sink: Option<Arc<Mutex<Box<dyn TextSink>>>>,
}

/// A take as it starts.
#[derive(Clone, Debug)]
pub struct TakeStart {
    pub id: u64,
    pub context: ContextSnapshot,
    /// The profile forced from the tray.
    pub forced_profile: Option<String>,
}

/// Live updates for the tray and inspector.
#[derive(Clone, Debug, PartialEq)]
pub enum Update {
    Level([u8; 5]),
    /// Live transcript text.
    Delta(String),
    Transcribing,
    Thinking,
    /// Generated text.
    Output(String),
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
    /// Unix milliseconds.
    pub started_at_ms: u64,
    pub context: ContextSnapshot,
    pub audio_seconds: f64,
    pub transcription: Option<TranscriptionPath>,
    pub transcript: String,
    pub resolution: Option<Resolution>,
    pub effective: Option<Effective>,
    pub decision: Option<DecisionTrace>,
    pub action: Option<Action>,
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
    fn new(start: &TakeStart) -> Self {
        Self {
            take: start.id,
            started_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as u64),
            context: start.context.clone(),
            audio_seconds: 0.0,
            transcription: None,
            transcript: String::new(),
            resolution: None,
            effective: None,
            decision: None,
            action: None,
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
    let begun = Instant::now();
    let transcript = transcribe(env, audio, finish, updates, &mut trace).await;
    trace.time("transcribe", begun);
    match transcript {
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
    if let Err(e) = process(env, &start, updates, &mut trace).await {
        trace.error = Some(e.to_string());
        return trace;
    }
    let delivered = Instant::now();
    match deliver(env, &start, &mut trace).await {
        Ok(outcome) => trace.delivery = outcome,
        Err(e) => trace.error = Some(e),
    }
    trace.time("deliver", delivered);
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
            .realtime(model, settings.language.as_deref())
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
    if buffer.len() < MIN_SAMPLES {
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

/// Chooses the profile and action and produces the text, filling `trace`.
async fn process(
    env: &Env,
    start: &TakeStart,
    updates: &mpsc::UnboundedSender<Update>,
    trace: &mut Trace,
) -> Result<(), ClientError> {
    let settings = &env.settings;
    let context = &start.context;
    let mut resolution = env
        .profiles
        .resolve(context, start.forced_profile.as_deref());
    let mut effective = env
        .profiles
        .effective(&resolution.profile, resolution.destination.as_deref());

    // Ask the decision model what only it can tell.
    let has_text = context.has_text();
    let has_selection = context.selection().is_some();
    let ask_action = effective.action == ActionPreference::Auto && has_text;
    let ask_generation =
        settings.models.generative.is_some() && effective.action != ActionPreference::Rewrite;
    let ask_profile = !resolution.tied.is_empty();
    let mut action_answer = None;
    let mut needs_generation = None;
    if let Some(model) = settings.models.decision.clone().filter(|_| settings.decide)
        && (ask_action || ask_generation || ask_profile)
    {
        let request = decision_request(
            model,
            env,
            context,
            &resolution,
            &effective,
            &trace.transcript,
            ask_action.then_some(has_selection),
            ask_generation,
        );
        let began = Instant::now();
        let response = env.client.decide(&request).await;
        trace.time("decide", began);
        match response {
            Ok(response) => {
                if let Some(Answer::Choice { choice, .. }) = response.answers.get("profile") {
                    let destination = env.profiles.resolve(context, Some(choice)).destination;
                    resolution.profile = choice.clone();
                    resolution.destination = destination;
                    effective = env
                        .profiles
                        .effective(&resolution.profile, resolution.destination.as_deref());
                }
                if let Some(Answer::Choice { choice, .. }) = response.answers.get("action") {
                    action_answer = Some(choice.clone());
                }
                if let Some(Answer::Noul { noul }) = response.answers.get("needs_generation") {
                    needs_generation = Some(*noul);
                }
                trace.decision = Some(DecisionTrace {
                    request,
                    response: Some(response),
                });
            }
            Err(e) => {
                trace.notes.push(format!("Skipped the decision: {e}"));
                trace.decision = Some(DecisionTrace {
                    request,
                    response: None,
                });
            }
        }
    }

    let action = choose_action(
        effective.action,
        action_answer.as_deref(),
        has_text,
        has_selection,
    );
    let generate = match settings.models.generative {
        None => false,
        Some(_) if action == Action::Rewrite => true,
        Some(_) => needs_generation.is_none_or(|p| p >= settings.generation_threshold),
    };
    trace.action = Some(action);
    trace.resolution = Some(resolution);
    trace.output = trace.transcript.clone();
    if action == Action::Rewrite && settings.models.generative.is_none() {
        trace
            .notes
            .push("No generative model: inserting the transcript instead of rewriting".into());
        trace.action = Some(Action::Insert);
    } else if generate && let Some(model) = &settings.models.generative {
        let request =
            generation_request(model, env, context, &effective, action, &trace.transcript);
        let began = Instant::now();
        let output = env
            .client
            .respond(&request, |delta| {
                let _ = updates.send(Update::Output(delta.to_string()));
            })
            .await?;
        trace.time("generate", began);
        trace.output = output.trim().to_string();
        trace.generation = Some(GenerationTrace { request, output });
    }
    trace.effective = Some(effective);
    Ok(())
}

fn choose_action(
    preference: ActionPreference,
    answer: Option<&str>,
    has_text: bool,
    has_selection: bool,
) -> Action {
    let wanted = match preference {
        ActionPreference::Insert => Action::Insert,
        ActionPreference::Replace => Action::Replace,
        ActionPreference::Rewrite => Action::Rewrite,
        ActionPreference::Auto => match answer {
            Some("replace") => Action::Replace,
            Some("rewrite") => Action::Rewrite,
            _ => Action::Insert,
        },
    };
    match wanted {
        Action::Replace if !has_selection => Action::Insert,
        Action::Rewrite if !has_text => Action::Insert,
        other => other,
    }
}

#[allow(clippy::too_many_arguments)]
fn decision_request(
    model: String,
    env: &Env,
    context: &ContextSnapshot,
    resolution: &Resolution,
    effective: &Effective,
    transcript: &str,
    action_with_selection: Option<bool>,
    ask_generation: bool,
) -> DecisionRequest {
    let mut state = context.describe();
    if !effective.instructions.is_empty() {
        state.push_str(&format!(
            "Instructions for this field: {}\n",
            effective.instructions.join(" ")
        ));
    }
    state.push_str(&format!("The user dictated: {transcript:?}\n"));
    let mut questions = BTreeMap::new();
    if let Some(has_selection) = action_with_selection {
        let mut criteria = BTreeMap::from([
            (
                "insert".to_string(),
                "Add the dictated text at the cursor, keeping the existing text".to_string(),
            ),
            (
                "rewrite".to_string(),
                "The dictation is an instruction to edit the existing or selected text".to_string(),
            ),
        ]);
        if has_selection {
            criteria.insert(
                "replace".into(),
                "Replace the selected text with the dictated text".into(),
            );
        }
        questions.insert(
            "action".into(),
            Question::Choice {
                instructions: Some("What should happen with the dictation?".into()),
                criteria,
            },
        );
    }
    if ask_generation {
        questions.insert(
            "needs_generation".into(),
            Question::Noul {
                instructions: Some(
                    "Does the dictated text need editing beyond punctuation and capitalization \
                     to fit this field and its instructions (formatting, tone, translation, or \
                     a spoken command)?"
                        .into(),
                ),
                criteria: None,
            },
        );
    }
    if !resolution.tied.is_empty() {
        let criteria = std::iter::once(&resolution.profile)
            .chain(&resolution.tied)
            .filter_map(|id| env.profiles.get(id))
            .map(|p| {
                let description = match &p.spec.instructions {
                    Some(i) => format!("{}: {i}", p.display_name()),
                    None => p.display_name().to_string(),
                };
                (p.spec.id.clone(), description)
            })
            .collect();
        questions.insert(
            "profile".into(),
            Question::Choice {
                instructions: Some("Which profile fits where the user is writing?".into()),
                criteria,
            },
        );
    }
    DecisionRequest {
        model,
        state,
        questions,
        steps: None,
        samples: None,
        think: None,
    }
}

fn generation_request(
    model: &str,
    env: &Env,
    context: &ContextSnapshot,
    effective: &Effective,
    action: Action,
    transcript: &str,
) -> ResponseRequest {
    let mut instructions = vec![BASE_INSTRUCTIONS.to_string()];
    instructions.push(match action {
        Action::Insert => {
            "The text is inserted at the cursor; fit it to the text around it.".into()
        }
        Action::Replace => "The text replaces the selected text.".into(),
        Action::Rewrite => "The dictation is an instruction: rewrite the given text accordingly \
                            and output the complete rewritten text."
            .into(),
    });
    instructions.extend(effective.instructions.iter().cloned());
    let mut input = context.describe();
    if action == Action::Rewrite {
        let text = context
            .selection()
            .or(context
                .focused
                .as_ref()
                .and_then(|e| e.value_excerpt.as_deref()))
            .unwrap_or_default();
        input.push_str(&format!(
            "\nText to rewrite:\n{text}\n\nInstruction: {transcript}"
        ));
    } else {
        input.push_str(&format!("\nDictation: {transcript}"));
    }
    ResponseRequest {
        model: model.into(),
        instructions: Some(instructions.join("\n\n")),
        input,
        max_output_tokens: Some(env.settings.max_output_tokens),
    }
}

/// Delivers the output once it is safe, or leaves it on the clipboard.
async fn deliver(
    env: &Env,
    start: &TakeStart,
    trace: &mut Trace,
) -> Result<Option<DeliveryOutcome>, String> {
    let Some(sink) = &env.sink else {
        return Ok(None);
    };
    let effective = trace
        .effective
        .as_ref()
        .expect("processed takes have settings");
    let action = trace.action.unwrap_or(Action::Insert);
    let request = DeliveryRequest {
        action,
        text: trace.output.clone(),
        method: effective.delivery,
        select_all: action == Action::Rewrite && start.context.selection().is_none(),
    };
    let copy = |reason: String| -> Result<Option<DeliveryOutcome>, String> {
        sink.lock()
            .expect("the sink lock is not poisoned")
            .copy(&request.text)
            .map_err(|e| e.to_string())?;
        Ok(Some(DeliveryOutcome::OnClipboard { reason }))
    };
    if request.method == DeliveryMethod::Clipboard {
        return copy("the profile delivers to the clipboard".into());
    }
    let window = start.context.window.handle.unwrap_or(0);
    let pending = Pending::new(start.id, window, Instant::now());
    loop {
        let (foreground, keys_down) = {
            let sink = sink.lock().expect("the sink lock is not poisoned");
            (sink.foreground_window().unwrap_or(0), sink.keys_down())
        };
        match pending.decide(start.id, foreground, keys_down, Instant::now()) {
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
    use crate::profile::ProfileSpec;
    use axum::Json;
    use axum::routing::{get, post};
    use serde_json::{Value, json};

    #[derive(Clone, Default)]
    struct Seen {
        decisions: Arc<Mutex<Vec<Value>>>,
        generations: Arc<Mutex<Vec<Value>>>,
        uploads: Arc<Mutex<usize>>,
    }

    /// A fake jevons server: no Realtime route, a scripted decision and generation.
    async fn server(decision: Value, output: &'static str) -> (Client, Seen) {
        let seen = Seen::default();
        let s = seen.clone();
        let decide = move |Json(body): Json<Value>| {
            let s = s.clone();
            let decision = decision.clone();
            async move {
                s.decisions.lock().unwrap().push(body);
                Json(decision)
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

    fn noul(p: f64) -> Value {
        json!({"model": "jev", "usage": {"input_tokens": 1, "output_tokens": 1},
               "answers": {"needs_generation": {"type": "noul", "noul": p}}})
    }

    async fn take(env: &Env, selection: Option<&str>) -> Trace {
        let (audio, finish) = one_second_of_audio();
        let (updates, _) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: context(selection),
            forced_profile: None,
        };
        run_take(env, start, audio, finish, &updates).await
    }

    #[tokio::test]
    async fn realtime_404_falls_back_to_batch_upload() {
        let (client, seen) = server(noul(0.1), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let env = Env {
            client,
            profiles: Arc::default(),
            settings: settings(),
            sink: Some(sink.shared()),
        };
        let trace = take(&env, None).await;
        assert_eq!(trace.error, None, "{:?}", trace.notes);
        assert_eq!(trace.transcription, Some(TranscriptionPath::Upload));
        assert_eq!(*seen.uploads.lock().unwrap(), 1);
        assert!(trace.notes[0].contains("Realtime"), "{:?}", trace.notes);
        assert_eq!(trace.transcript, "hello world");
    }

    #[tokio::test]
    async fn clean_dictation_is_typed_without_generating() {
        let (client, seen) = server(noul(0.1), "unused").await;
        let sink = RecordingSink::new(Some(7));
        let env = Env {
            client,
            profiles: Arc::default(),
            settings: settings(),
            sink: Some(sink.shared()),
        };
        let trace = take(&env, None).await;
        assert!(seen.generations.lock().unwrap().is_empty());
        assert_eq!(trace.output, "hello world");
        let delivered = sink.requests();
        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].action, Action::Insert);
        assert_eq!(delivered[0].text, "hello world");
        // No text in the field: the action is not asked, only whether to edit.
        let asked = &seen.decisions.lock().unwrap()[0]["questions"];
        assert!(asked.get("action").is_none());
        assert!(asked.get("needs_generation").is_some());
    }

    #[tokio::test]
    async fn a_rewrite_of_the_selection_generates_with_profile_instructions() {
        let decision = json!({"model": "jev", "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"action": {"type": "choice", "choice": "rewrite",
                                   "probabilities": {"rewrite": 0.9}, "confidence": 0.9}}});
        let (client, seen) = server(decision, "Dear team, hello world.").await;
        let profiles = Profiles::new([(
            toml::from_str::<ProfileSpec>(
                r#"id = "notes"
match = { app = ["notepad.exe"] }
instructions = "Formal tone.""#,
            )
            .unwrap(),
            None,
        )]);
        let sink = RecordingSink::new(Some(7));
        let env = Env {
            client,
            profiles: Arc::new(profiles),
            settings: settings(),
            sink: Some(sink.shared()),
        };
        let trace = take(&env, Some("hi all")).await;
        assert_eq!(trace.error, None);
        assert_eq!(trace.action, Some(Action::Rewrite), "{:?}", trace.notes);
        assert_eq!(trace.output, "Dear team, hello world.");
        let generation = &seen.generations.lock().unwrap()[0];
        assert!(
            generation["instructions"]
                .as_str()
                .unwrap()
                .contains("Formal tone.")
        );
        assert!(generation["input"].as_str().unwrap().contains("hi all"));
        assert_eq!(generation["stream"], true);
        let delivered = sink.requests();
        assert_eq!(delivered[0].action, Action::Rewrite);
        assert!(!delivered[0].select_all);
    }

    #[tokio::test]
    async fn a_changed_window_leaves_the_text_on_the_clipboard() {
        let (client, _) = server(noul(0.1), "unused").await;
        let sink = RecordingSink::new(Some(99));
        let env = Env {
            client,
            profiles: Arc::default(),
            settings: settings(),
            sink: Some(sink.shared()),
        };
        let trace = take(&env, None).await;
        assert!(matches!(
            trace.delivery,
            Some(DeliveryOutcome::OnClipboard { .. })
        ));
        assert!(sink.requests().is_empty());
        assert_eq!(sink.clipboard().as_deref(), Some("hello world"));
    }

    #[tokio::test]
    async fn silence_ends_the_take_without_requests() {
        let (client, seen) = server(noul(0.1), "unused").await;
        let env = Env {
            client,
            profiles: Arc::default(),
            settings: settings(),
            sink: None,
        };
        let (audio, receiver) = mpsc::unbounded_channel();
        audio.send(AudioEvent::Chunk(vec![0; 100])).unwrap();
        audio.send(AudioEvent::Ended).unwrap();
        let (_finish, finished) = oneshot::channel();
        let (updates, _) = mpsc::unbounded_channel();
        let start = TakeStart {
            id: 1,
            context: context(None),
            forced_profile: None,
        };
        let trace = run_take(&env, start, receiver, finished, &updates).await;
        assert!(trace.error.is_some());
        assert_eq!(*seen.uploads.lock().unwrap(), 0);
    }

    #[test]
    fn replace_needs_a_selection_and_rewrite_needs_text() {
        use ActionPreference::*;
        assert_eq!(choose_action(Replace, None, true, false), Action::Insert);
        assert_eq!(
            choose_action(Auto, Some("rewrite"), false, false),
            Action::Insert
        );
        assert_eq!(
            choose_action(Auto, Some("rewrite"), true, false),
            Action::Rewrite
        );
        assert_eq!(
            choose_action(Auto, Some("replace"), true, true),
            Action::Replace
        );
    }
}
