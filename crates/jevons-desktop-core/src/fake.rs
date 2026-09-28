//! Stand-ins for the platform layers, for tests and the headless replay mode.

use crate::context::{ContextSnapshot, Privacy};
use crate::levels::Meter;
use crate::platform::{
    AudioDevice, AudioEvent, AudioSource, CaptureHandle, ContextProvider, DeliveryOutcome,
    DeliveryRequest, PlatformError, SAMPLE_RATE, SinkCapabilities, TextSink,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;

/// Always reports the same context.
pub struct FakeContext(pub ContextSnapshot);

impl ContextProvider for FakeContext {
    fn name(&self) -> &'static str {
        "fixture"
    }

    fn snapshot(&self, privacy: &Privacy) -> Result<ContextSnapshot, PlatformError> {
        Ok(self.0.clone().sanitized(privacy))
    }
}

/// Plays an audio file as if it were a microphone.
pub struct FileAudioSource {
    pub file: PathBuf,
    /// Send chunks at real-time pace, as a microphone would.
    pub paced: bool,
}

struct FileCapture(Arc<AtomicBool>);

impl CaptureHandle for FileCapture {
    fn stop(self: Box<Self>) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl AudioSource for FileAudioSource {
    fn devices(&self) -> Vec<AudioDevice> {
        vec![AudioDevice {
            name: self.file.display().to_string(),
            default: true,
        }]
    }

    fn start(
        &mut self,
        _: Option<&str>,
        events: UnboundedSender<AudioEvent>,
    ) -> Result<Box<dyn CaptureHandle>, PlatformError> {
        let bytes = std::fs::read(&self.file).map_err(|e| PlatformError::Failed(e.to_string()))?;
        let extension = self.file.extension().and_then(|e| e.to_str());
        let limits = jevons_audio::AudioLimits {
            max_bytes: 100 << 20,
            max_seconds: 600.0,
        };
        let samples = jevons_audio::decode_audio(bytes, extension, SAMPLE_RATE, limits)
            .map_err(|e| PlatformError::Failed(e.to_string()))?;
        let pcm: Vec<i16> = samples
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let paced = self.paced;
        std::thread::spawn(move || {
            let mut meter = Meter::default();
            for chunk in pcm.chunks(SAMPLE_RATE as usize / 10) {
                if stopped.load(Ordering::Relaxed) {
                    break;
                }
                let _ = events.send(AudioEvent::Level(meter.update(chunk)));
                if events.send(AudioEvent::Chunk(chunk.to_vec())).is_err() {
                    return;
                }
                if paced {
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
            let _ = events.send(AudioEvent::Ended);
        });
        Ok(Box::new(FileCapture(stop)))
    }
}

#[derive(Default)]
struct Recorded {
    requests: Vec<DeliveryRequest>,
    clipboard: Option<String>,
}

/// Records deliveries instead of typing; reports a fixed foreground window.
#[derive(Clone)]
pub struct RecordingSink {
    foreground: Option<u64>,
    keys_down: bool,
    recorded: Arc<Mutex<Recorded>>,
}

impl RecordingSink {
    pub fn new(foreground: Option<u64>) -> Self {
        Self {
            foreground,
            keys_down: false,
            recorded: Arc::default(),
        }
    }

    /// Reports keys as held, like a hotkey held while dictating.
    pub fn holding_keys(mut self) -> Self {
        self.keys_down = true;
        self
    }

    pub fn shared(&self) -> Arc<Mutex<Box<dyn TextSink>>> {
        Arc::new(Mutex::new(Box::new(self.clone())))
    }

    pub fn requests(&self) -> Vec<DeliveryRequest> {
        self.recorded.lock().unwrap().requests.clone()
    }

    pub fn clipboard(&self) -> Option<String> {
        self.recorded.lock().unwrap().clipboard.clone()
    }
}

impl TextSink for RecordingSink {
    fn name(&self) -> &'static str {
        "recording"
    }

    fn capabilities(&self) -> SinkCapabilities {
        SinkCapabilities {
            paste: true,
            type_text: true,
            set_value: false,
        }
    }

    fn foreground_window(&self) -> Option<u64> {
        self.foreground
    }

    fn keys_down(&self) -> bool {
        self.keys_down
    }

    fn deliver(&mut self, request: &DeliveryRequest) -> Result<DeliveryOutcome, PlatformError> {
        self.recorded.lock().unwrap().requests.push(request.clone());
        Ok(DeliveryOutcome::Delivered {
            method: request.method,
        })
    }

    fn copy(&mut self, text: &str) -> Result<(), PlatformError> {
        self.recorded.lock().unwrap().clipboard = Some(text.into());
        Ok(())
    }
}
