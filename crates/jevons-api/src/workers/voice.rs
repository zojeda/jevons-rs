//! The text-to-speech queue and the dedicated thread that owns the voice model.
//!
//! One bounded queue feeds the thread. A job's audio leaves in chunks as the model produces
//! them, and a client that stops listening stops the model at the next chunk.
use crate::error::ApiError;
use jevons_core::{Error, VoiceConfig, VoiceInfo};
use jevons_voice::{SpeakRequest, Spoken, Synthesizer};
use std::thread::JoinHandle;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};

/// What the worker speaks before it serves, so the first request does not pay for compiling
/// kernels.
const WARM_UP: &str = "Hola, buenos días.";

/// A text to speak, with the channel its audio leaves on.
pub(crate) struct Job {
    pub text: String,
    pub voice: Option<String>,
    pub language: Option<String>,
    pub updates: mpsc::UnboundedSender<Update>,
}

/// Progress of a speech: mono samples at the model's rate as the model produces them, then
/// the result.
#[derive(Debug)]
pub enum Update {
    Audio(Vec<f32>),
    Done(Result<Spoken, Error>),
}

/// What the worker serves once its model is loaded.
#[derive(Clone, Debug)]
pub struct Ready {
    pub info: VoiceInfo,
    /// The voice of a request that names none.
    pub default_voice: String,
}

#[derive(Clone)]
pub struct Client {
    pub(crate) sender: mpsc::Sender<Job>,
}

impl Client {
    /// Queues `text`, in `voice` (the default voice when `None`) and in `language` (guessed
    /// per sentence when `None`). Dropping the receiver stops the model at its next chunk.
    pub fn speak(
        &self,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<mpsc::UnboundedReceiver<Update>, ApiError> {
        let (updates, receiver) = mpsc::unbounded_channel();
        let job = Job {
            text,
            voice,
            language,
            updates,
        };
        self.sender.try_send(job).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => ApiError::overloaded(),
            mpsc::error::TrySendError::Closed(_) => ApiError::unavailable(),
        })?;
        Ok(receiver)
    }

    pub fn is_alive(&self) -> bool {
        !self.sender.is_closed()
    }
}

/// Speaks one job. The log line has counts and timings, never the text.
fn run(synthesizer: &mut Synthesizer, job: Job) {
    let Job {
        text,
        voice,
        language,
        updates,
    } = job;
    // A client that left while the job waited costs nothing.
    if updates.is_closed() {
        return;
    }
    let request = SpeakRequest {
        text: &text,
        voice: voice.as_deref(),
        language: language.as_deref(),
    };
    let started = Instant::now();
    let mut first_audio_ms = None;
    let result = synthesizer.speak(&request, &mut |samples| {
        if first_audio_ms.is_none() {
            first_audio_ms = Some(started.elapsed().as_millis());
        }
        updates.send(Update::Audio(samples.to_vec())).is_ok()
    });
    if let Ok(spoken) = &result {
        let seconds = spoken.samples as f64 / f64::from(synthesizer.info().sample_rate);
        tracing::info!(
            characters = text.chars().count(),
            sentences = spoken.sentences,
            seconds,
            first_audio_ms = first_audio_ms.unwrap_or_default(),
            elapsed_ms = started.elapsed().as_millis(),
            stopped = spoken.stopped,
            "Speech completed"
        );
    }
    let _ = updates.send(Update::Done(result));
}

/// Loads the voice model on a new thread, warms it up and serves the queue. `default_voice`
/// is the voice of a request that names none: one of the model's, or `None` for its first.
pub async fn start(
    config: VoiceConfig,
    default_voice: Option<String>,
    capacity: usize,
) -> Result<(Client, JoinHandle<()>, Ready), Box<dyn std::error::Error>> {
    start_with(capacity, move || {
        let mut synthesizer = Synthesizer::load(&config, default_voice.as_deref())?;
        synthesizer.speak(&SpeakRequest::new(WARM_UP), &mut |_| true)?;
        Ok(synthesizer)
    })
    .await
}

/// [`start`] with any model constructor (tests use a scripted model).
pub(crate) async fn start_with(
    capacity: usize,
    load: impl FnOnce() -> Result<Synthesizer, Error> + Send + 'static,
) -> Result<(Client, JoinHandle<()>, Ready), Box<dyn std::error::Error>> {
    if capacity == 0 {
        return Err("Voice queue capacity must be positive".into());
    }
    let (sender, mut receiver) = mpsc::channel::<Job>(capacity);
    let (ready_sender, ready_receiver) = oneshot::channel();
    let thread = std::thread::Builder::new()
        .name("voice-inference".into())
        .spawn(move || {
            let mut synthesizer = match load() {
                Ok(synthesizer) => synthesizer,
                Err(error) => {
                    let _ = ready_sender.send(Err(error.to_string()));
                    return;
                }
            };
            let ready = Ready {
                info: synthesizer.info().clone(),
                default_voice: synthesizer.default_voice().to_string(),
            };
            if ready_sender.send(Ok(ready)).is_err() {
                return;
            }
            while let Some(job) = receiver.blocking_recv() {
                run(&mut synthesizer, job);
            }
        })?;
    match ready_receiver.await {
        Ok(Ok(ready)) => Ok((Client { sender }, thread, ready)),
        failed => {
            // The thread is on its way out. The caller may load again, or go on without this
            // model, once nothing of it is left on the device.
            let _ = tokio::task::spawn_blocking(move || thread.join()).await;
            Err(match failed {
                Ok(Err(error)) => error.into(),
                _ => "The voice worker stopped while loading".into(),
            })
        }
    }
}

/// A voice service on the scripted model, and the log of what the model is asked to speak.
/// `adjust` changes the model before it starts: its rate, its delay before each chunk.
#[cfg(test)]
pub(crate) async fn scripted_service(
    capacity: usize,
    adjust: impl FnOnce(&mut jevons_voice::fake::FakeVoice) + Send + 'static,
) -> (crate::VoiceService, jevons_voice::fake::Log) {
    let (log_sender, log_receiver) = std::sync::mpsc::channel();
    let (worker, _thread, ready) = start_with(capacity, move || {
        let mut model = jevons_voice::fake::FakeVoice::new();
        adjust(&mut model);
        let _ = log_sender.send(model.log.clone());
        Synthesizer::new(Box::new(model), None)
    })
    .await
    .unwrap();
    let service = crate::VoiceService {
        worker,
        model_id: "scripted-tts".into(),
        aliases: ["tts-latest".to_string()].into(),
        description: "Scripted voice.".into(),
        info: ready.info,
        default_voice: ready.default_voice,
        max_input_chars: 4096,
    };
    (service, log_receiver.recv().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use jevons_voice::fake::{FakeVoice, Spoke, samples_for, spoken};
    use std::time::Duration;

    /// The audio of a speech and its result.
    async fn collect(mut updates: mpsc::UnboundedReceiver<Update>) -> (Vec<f32>, Spoken) {
        let mut audio = Vec::new();
        loop {
            match updates.recv().await.expect("the worker answers") {
                Update::Audio(samples) => audio.extend(samples),
                Update::Done(result) => return (audio, result.unwrap()),
            }
        }
    }

    #[tokio::test]
    async fn a_speech_streams_audio_chunks_then_what_was_spoken() {
        let (service, log) = scripted_service(4, |_| {}).await;
        assert_eq!(service.info.architecture, "fake");
        assert_eq!(service.default_voice, "lector");
        let mut updates = service
            .worker
            .speak("Hola. Todo bien.".into(), None, None)
            .unwrap();
        let mut chunks = Vec::new();
        let done = loop {
            match updates.recv().await.unwrap() {
                Update::Audio(samples) => chunks.push(samples.len()),
                Update::Done(result) => break result.unwrap(),
            }
        };
        // "Hola." is 200 ms and "Todo bien." 400 ms, in chunks of 80 ms at 24 kHz.
        assert_eq!(chunks, [1920, 1920, 960, 1920, 1920, 1920, 1920, 1920]);
        let samples = samples_for("Hola.", 24000) + samples_for("Todo bien.", 24000);
        assert_eq!(
            done,
            Spoken {
                sentences: 2,
                samples,
                stopped: false
            }
        );
        // After the result, the channel closes.
        assert!(updates.recv().await.is_none());
        // A request's voice and language reach the model.
        let updates = service
            .worker
            .speak(
                "Good morning.".into(),
                Some("reader".into()),
                Some("en".into()),
            )
            .unwrap();
        collect(updates).await;
        assert_eq!(
            spoken(&log).last(),
            Some(&Spoke {
                text: "Good morning.".into(),
                voice: "reader".into(),
                language: Some("en".into()),
            })
        );
    }

    #[tokio::test]
    async fn a_text_the_model_cannot_speak_fails_before_any_audio() {
        let (service, log) = scripted_service(4, |model| {
            model.fail_on = Some("falla".into());
        })
        .await;
        for (text, voice) in [("...", None), ("Hola.", Some("nadie"))] {
            let mut updates = service
                .worker
                .speak(text.into(), voice.map(String::from), None)
                .unwrap();
            let Update::Done(Err(Error::InvalidInput(_))) = updates.recv().await.unwrap() else {
                panic!("{text:?} in {voice:?} is invalid input");
            };
        }
        assert!(spoken(&log).is_empty());
        // A model failure after the first sentence follows that sentence's audio.
        let mut updates = service
            .worker
            .speak("Uno. Esto falla.".into(), None, None)
            .unwrap();
        let mut heard = 0;
        let error = loop {
            match updates.recv().await.unwrap() {
                Update::Audio(samples) => heard += samples.len(),
                Update::Done(result) => break result.unwrap_err(),
            }
        };
        assert!(matches!(error, Error::Backend(_)), "{error:?}");
        assert_eq!(heard, samples_for("Uno.", 24000));
    }

    #[tokio::test]
    async fn a_full_voice_queue_is_overloaded_and_a_stopped_worker_unavailable() {
        let (sender, mut receiver) = mpsc::channel(1);
        let client = Client { sender };
        assert!(client.is_alive());
        // Nothing takes the first job, so the queue of one is full.
        let _waiting = client.speak("Hola.".into(), None, None).unwrap();
        let error = client.speak("Hola.".into(), None, None).unwrap_err();
        assert_eq!(error.status.as_u16(), 529);
        assert_eq!(error.into_response().headers()["retry-after"], "1");
        receiver.close();
        let error = client.speak("Hola.".into(), None, None).unwrap_err();
        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(!client.is_alive());
    }

    #[tokio::test]
    async fn the_voice_worker_speaks_one_job_at_a_time_in_arrival_order() {
        let (service, log) = scripted_service(4, |model| {
            model.chunk_delay = Duration::from_millis(2);
        })
        .await;
        let first = service
            .worker
            .speak("Primera frase. Segunda frase.".into(), None, None)
            .unwrap();
        let second = service.worker.speak("Tercera.".into(), None, None).unwrap();
        let (second, first) = tokio::join!(collect(second), collect(first));
        assert_eq!((first.1.sentences, second.1.sentences), (2, 1));
        let said: Vec<String> = spoken(&log).into_iter().map(|s| s.text).collect();
        assert_eq!(said, ["Primera frase.", "Segunda frase.", "Tercera."]);
    }

    #[tokio::test]
    async fn a_listener_that_leaves_stops_the_model_at_its_next_chunk() {
        let (service, log) = scripted_service(4, |model| {
            model.chunk_delay = Duration::from_millis(10);
        })
        .await;
        let text = "Esta es la primera frase. Esta es la segunda. Y la tercera.";
        let mut updates = service.worker.speak(text.into(), None, None).unwrap();
        // A job that waits behind it, whose listener leaves before its turn.
        let skipped = service.worker.speak("Nadie escucha.".into(), None, None);
        drop(skipped);
        let Update::Audio(_) = updates.recv().await.unwrap() else {
            panic!("audio first");
        };
        drop(updates);
        // The worker is free again: a job behind both is spoken whole.
        let after = service.worker.speak("Ya está.".into(), None, None).unwrap();
        let (_, done) = collect(after).await;
        assert!(!done.stopped);
        let said: Vec<String> = spoken(&log).into_iter().map(|s| s.text).collect();
        assert_eq!(said, ["Esta es la primera frase.", "Ya está."]);
    }

    #[tokio::test]
    async fn a_voice_model_that_fails_to_load_fails_the_start_once_its_thread_ended() {
        let error = start_with(4, || Err(Error::ModelLoad)).await.err().unwrap();
        assert_eq!(error.to_string(), Error::ModelLoad.to_string());
        // A default voice the model lacks is a load failure too.
        let error = start_with(4, || {
            Synthesizer::new(Box::new(FakeVoice::new()), Some("nadie"))
        })
        .await
        .err()
        .unwrap();
        assert!(error.to_string().contains("nadie"), "{error}");
        let error = start_with(0, || Err(Error::ModelLoad)).await.err().unwrap();
        assert!(error.to_string().contains("capacity"), "{error}");
    }
}
