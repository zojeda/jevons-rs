//! Speech-to-text queues and the dedicated thread that owns the speech model.
//!
//! Two bounded queues feed the thread: live (Realtime) jobs and batch uploads. Live jobs run
//! first, and between the windows of a long upload, so an hour of audio never stalls a live
//! session.
use crate::error::ApiError;
use jevons_core::{Error, Segment, SpeechConfig, SpeechInfo, Transcript, Word};
use jevons_speech::{Transcriber, Transcription, words};
use std::thread::JoinHandle;
use tokio::sync::{mpsc, oneshot};

/// Live jobs waiting at once, across every Realtime session.
const LIVE_CAPACITY: usize = 16;

pub(crate) enum Job {
    /// A whole recording, windowed, streamed as [`Update`]s.
    Transcribe {
        samples: Vec<f32>,
        updates: mpsc::UnboundedSender<Update>,
    },
    /// One pass over a live utterance (at most one model window): its words and text.
    Pass {
        samples: Vec<f32>,
        reply: oneshot::Sender<Result<Pass, Error>>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pass {
    pub words: Vec<Word>,
    pub text: String,
}

/// Progress of a transcription: each closed segment with its words, then the result.
#[derive(Debug)]
pub enum Update {
    Segment(Segment, Vec<Word>),
    Done(Result<Transcript, Error>),
}

#[derive(Clone)]
pub struct Client {
    pub(crate) batch: mpsc::Sender<Job>,
    pub(crate) live: mpsc::Sender<Job>,
}

fn submit(queue: &mpsc::Sender<Job>, job: Job) -> Result<(), ApiError> {
    queue.try_send(job).map_err(|error| match error {
        mpsc::error::TrySendError::Full(_) => ApiError::overloaded(),
        mpsc::error::TrySendError::Closed(_) => ApiError::unavailable(),
    })
}

impl Client {
    /// Queues a recording (`live` for a committed Realtime turn). Dropping the receiver
    /// cancels it after the current window.
    pub fn transcribe(
        &self,
        samples: Vec<f32>,
        live: bool,
    ) -> Result<mpsc::UnboundedReceiver<Update>, ApiError> {
        let (updates, receiver) = mpsc::unbounded_channel();
        let queue = if live { &self.live } else { &self.batch };
        submit(queue, Job::Transcribe { samples, updates })?;
        Ok(receiver)
    }

    /// Queues a live pass over at most one model window of audio.
    pub fn pass(
        &self,
        samples: Vec<f32>,
    ) -> Result<oneshot::Receiver<Result<Pass, Error>>, ApiError> {
        let (reply, receiver) = oneshot::channel();
        submit(&self.live, Job::Pass { samples, reply })?;
        Ok(receiver)
    }

    pub fn is_alive(&self) -> bool {
        !self.batch.is_closed()
    }
}

fn run_pass(transcriber: &mut Transcriber, samples: &[f32]) -> Result<Pass, Error> {
    let tokens = transcriber.pass(samples)?;
    let words = words(&tokens);
    let text = transcriber.text(&words)?;
    Ok(Pass { words, text })
}

/// Runs one job; `between` runs between the windows of a recording.
fn run(transcriber: &mut Transcriber, job: Job, between: &mut dyn FnMut(&mut Transcriber)) {
    match job {
        Job::Pass { samples, reply } => {
            if !reply.is_closed() {
                let _ = reply.send(run_pass(transcriber, &samples));
            }
        }
        Job::Transcribe { samples, updates } => {
            let seconds = samples.len() as f64 / f64::from(transcriber.info().sample_rate);
            let mut job = Transcription::new(samples);
            while !job.is_done() {
                if updates.is_closed() {
                    return;
                }
                match transcriber.step(&mut job) {
                    Ok(segments) => {
                        for (segment, words) in segments {
                            let _ = updates.send(Update::Segment(segment, words));
                        }
                    }
                    Err(error) => {
                        let _ = updates.send(Update::Done(Err(error)));
                        return;
                    }
                }
                if !job.is_done() {
                    between(transcriber);
                }
            }
            let result = transcriber.finish(job);
            if result.is_ok() {
                tracing::info!(seconds, "Transcription completed");
            }
            let _ = updates.send(Update::Done(result));
        }
    }
}

/// Loads the speech model on a new thread and serves both queues.
pub async fn start(
    config: SpeechConfig,
    capacity: usize,
) -> Result<(Client, JoinHandle<()>, SpeechInfo), Box<dyn std::error::Error>> {
    start_with(capacity, move || {
        let mut transcriber = Transcriber::load(&config)?;
        // The first pass loads (or compiles) the kernels; do it before serving.
        let second = transcriber.info().sample_rate as usize;
        transcriber.pass(&vec![0.0; second])?;
        Ok(transcriber)
    })
    .await
}

/// [`start`] with any model constructor (tests use a scripted model).
pub(crate) async fn start_with(
    capacity: usize,
    load: impl FnOnce() -> Result<Transcriber, Error> + Send + 'static,
) -> Result<(Client, JoinHandle<()>, SpeechInfo), Box<dyn std::error::Error>> {
    if capacity == 0 {
        return Err("Speech queue capacity must be positive".into());
    }
    let (batch, mut batch_receiver) = mpsc::channel::<Job>(capacity);
    let (live, mut live_receiver) = mpsc::channel::<Job>(LIVE_CAPACITY);
    let (ready_sender, ready_receiver) = oneshot::channel();
    let thread = std::thread::Builder::new()
        .name("speech-inference".into())
        .spawn(move || {
            let mut transcriber = match load() {
                Ok(transcriber) => transcriber,
                Err(error) => {
                    let _ = ready_sender.send(Err(error.to_string()));
                    return;
                }
            };
            if ready_sender.send(Ok(transcriber.info().clone())).is_err() {
                return;
            }
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("a current-thread runtime for the speech queues");
            loop {
                let job = runtime.block_on(async {
                    tokio::select! {
                        biased;
                        Some(job) = live_receiver.recv() => Some(job),
                        Some(job) = batch_receiver.recv() => Some(job),
                        else => None,
                    }
                });
                let Some(job) = job else { break };
                run(&mut transcriber, job, &mut |transcriber| {
                    while let Ok(live_job) = live_receiver.try_recv() {
                        run(transcriber, live_job, &mut |_| {});
                    }
                });
            }
        })?;
    let info = ready_receiver.await??;
    Ok((Client { batch, live }, thread, info))
}

/// A model that speaks by numbers, for tests. At 100 Hz, each run of at least half a second
/// whose samples round to `n / 100` reads as the word `w{n}` at the run's start, and
/// `n % 5 == 4` ends a sentence; quiet samples are silence. Runs, rather than fixed blocks,
/// keep the words intact through resampling and turn trimming.
#[cfg(test)]
pub(crate) struct ScriptedModel {
    pub info: SpeechInfo,
    /// When set, the first pass waits for a signal, so a test can queue work behind it.
    pub gate: Option<std::sync::mpsc::Receiver<()>>,
}

#[cfg(test)]
impl ScriptedModel {
    pub fn new(max_window_seconds: f64) -> Self {
        Self {
            info: SpeechInfo {
                architecture: "scripted",
                display_name: "Scripted speech".into(),
                sample_rate: 100,
                frame_seconds: 0.01,
                max_window_seconds,
                languages: &["en", "es"],
            },
            gate: None,
        }
    }
}

#[cfg(test)]
impl jevons_core::SpeechModel for ScriptedModel {
    fn info(&self) -> &SpeechInfo {
        &self.info
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
    ) -> jevons_core::Result<Vec<jevons_core::SpeechToken>> {
        if let Some(gate) = self.gate.take() {
            let _ = gate.recv();
        }
        let level = |x: f32| {
            if x >= 0.005 {
                (x * 100.0).round() as u32
            } else {
                0
            }
        };
        let mut tokens = Vec::new();
        let mut start = 0;
        for end in 1..=samples.len() {
            if end < samples.len() && level(samples[end]) == level(samples[start]) {
                continue;
            }
            let n = level(samples[start]);
            if n > 0 && end - start >= 50 {
                let dot = if n % 5 == 4 { "." } else { "" };
                let at = start as f64 / 100.0;
                tokens.push(jevons_core::SpeechToken {
                    id: n,
                    piece: format!("▁w{n}{dot}"),
                    start: at,
                    end: at + 0.5,
                    logprob: -0.5,
                });
            }
            start = end;
        }
        Ok(tokens)
    }

    fn detokenize(&self, ids: &[u32]) -> jevons_core::Result<String> {
        Ok(ids
            .iter()
            .map(|n| format!(" w{n}{}", if n % 5 == 4 { "." } else { "" }))
            .collect::<String>()
            .trim()
            .to_string())
    }
}

/// `seconds` of scripted speech at `rate`, spelling w1, w2, ...
#[cfg(test)]
pub(crate) fn spoken(seconds: usize, rate: usize) -> Vec<f32> {
    (0..seconds * rate)
        .map(|i| (i / rate + 1) as f32 / 100.0)
        .collect()
}

/// A speech service on the scripted model, with the given model window.
#[cfg(test)]
pub(crate) async fn scripted_service(window: f64) -> crate::SpeechService {
    let (worker, _thread, info) = start_with(4, move || {
        Ok(Transcriber::new(Box::new(ScriptedModel::new(window))))
    })
    .await
    .unwrap();
    crate::SpeechService {
        worker,
        model_id: "scripted-asr".into(),
        aliases: ["asr-latest".to_string()].into(),
        description: "Scripted speech.".into(),
        info,
        max_audio_seconds: 600.0,
        realtime: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn scripted(window: f64) -> Client {
        let (client, _thread, info) = start_with(4, move || {
            Ok(Transcriber::new(Box::new(ScriptedModel::new(window))))
        })
        .await
        .unwrap();
        assert_eq!(info.architecture, "scripted");
        client
    }

    #[tokio::test]
    async fn recordings_stream_segments_then_the_transcript() {
        let client = scripted(20.0).await;
        let mut updates = client.transcribe(spoken(12, 100), false).unwrap();
        let mut segments = Vec::new();
        let transcript = loop {
            match updates.recv().await.unwrap() {
                Update::Segment(segment, words) => {
                    assert_eq!(segment.tokens.len(), words.len());
                    segments.push(segment.text);
                }
                Update::Done(result) => break result.unwrap(),
            }
        };
        assert_eq!(
            segments,
            vec!["w1 w2 w3 w4.", "w5 w6 w7 w8 w9.", "w10 w11 w12"]
        );
        assert_eq!(transcript.words.len(), 12);
    }

    #[tokio::test]
    async fn live_passes_return_words_and_text() {
        let client = scripted(20.0).await;
        let pass = client.pass(spoken(3, 100)).unwrap().await.unwrap().unwrap();
        assert_eq!(pass.text, "w1 w2 w3");
        assert_eq!(pass.words.len(), 3);
    }

    #[tokio::test]
    async fn live_passes_run_between_the_windows_of_a_long_recording() {
        // The recording's first window waits until the live pass is queued behind it.
        let (open, gate) = std::sync::mpsc::channel();
        let (client, _thread, _) = start_with(4, move || {
            let mut model = ScriptedModel::new(12.0);
            model.gate = Some(gate);
            Ok(Transcriber::new(Box::new(model)))
        })
        .await
        .unwrap();
        let mut long = client.transcribe(spoken(60, 100), false).unwrap();
        let pass = client.pass(spoken(2, 100)).unwrap();
        open.send(()).unwrap();
        let mut segments_before_pass = None;
        let mut segments = 0;
        let mut pass = Some(pass);
        loop {
            tokio::select! {
                biased;
                result = async { pass.as_mut().unwrap().await }, if pass.is_some() => {
                    assert_eq!(result.unwrap().unwrap().text, "w1 w2");
                    segments_before_pass = Some(segments);
                    pass = None;
                }
                update = long.recv() => match update.unwrap() {
                    Update::Segment(..) => segments += 1,
                    Update::Done(result) => {
                        result.unwrap();
                        break;
                    }
                }
            }
        }
        let before = segments_before_pass.expect("the pass finished before the recording");
        assert!(before < segments, "the pass waited for the whole recording");
    }
}
