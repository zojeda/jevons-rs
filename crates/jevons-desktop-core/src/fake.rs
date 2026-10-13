//! Stand-ins for the platform layers, for tests and the headless replay mode.

use crate::context::{ContextSnapshot, Privacy};
use crate::levels::Meter;
use crate::platform::{
    AudioDevice, AudioEvent, AudioSource, CaptureHandle, Chord, ContextInspector, ContextProvider,
    DeliveryOutcome, DeliveryRequest, PlatformError, PlaybackHandle, SAMPLE_RATE, SinkCapabilities,
    Speaker, TextSink, UiAction, UiElement, WindowEntry,
};
use crate::recorded::{
    Deed, DemonstratedStep, Demonstration, RecordedElement, RecordedInspector, RecordedTree,
    RecordedWindow,
};
use crate::voice::Voice;
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

/// What a [`RecordingSpeaker`] played in one go, from a start to its end or its stop.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Played {
    /// The device asked for; `None` is the default one.
    pub device: Option<String>,
    /// Every sample queued, in order.
    pub samples: Vec<i16>,
    /// Whether it was cut before all of it was heard.
    pub stopped: bool,
    /// How many of the samples were heard.
    heard: usize,
    /// How many stretches were queued.
    stretches: usize,
}

/// Records speech instead of playing it. What is queued counts as still playing until
/// [`RecordingSpeaker::hear`] says it was heard.
#[derive(Clone, Default)]
pub struct RecordingSpeaker {
    played: Arc<Mutex<Vec<Played>>>,
    no_device: bool,
    unplugged_after: Option<usize>,
}

impl RecordingSpeaker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Has no device to play on: starting fails.
    pub fn without_device(mut self) -> Self {
        self.no_device = true;
        self
    }

    /// Takes this many stretches of a take's speech and then goes away, like a device
    /// unplugged.
    pub fn unplugged_after(mut self, stretches: usize) -> Self {
        self.unplugged_after = Some(stretches);
        self
    }

    /// The desk's voice over this speaker.
    pub fn voice(&self) -> Arc<Voice> {
        Arc::new(Voice::new(Box::new(self.clone())))
    }

    /// Each playback so far, in the order they started.
    pub fn played(&self) -> Vec<Played> {
        self.played.lock().unwrap().clone()
    }

    /// Everything queued so far has been heard.
    pub fn hear(&self) {
        for played in self.played.lock().unwrap().iter_mut() {
            played.heard = played.samples.len();
        }
    }
}

impl Speaker for RecordingSpeaker {
    fn devices(&self) -> Vec<AudioDevice> {
        vec![AudioDevice {
            name: "recording".into(),
            default: true,
        }]
    }

    fn start(&mut self, device: Option<&str>) -> Result<Box<dyn PlaybackHandle>, PlatformError> {
        if self.no_device {
            return Err(PlatformError::Failed("no output device".into()));
        }
        let mut played = self.played.lock().unwrap();
        played.push(Played {
            device: device.map(String::from),
            ..Played::default()
        });
        Ok(Box::new(RecordedPlayback {
            played: self.played.clone(),
            index: played.len() - 1,
            unplugged_after: self.unplugged_after,
        }))
    }
}

struct RecordedPlayback {
    played: Arc<Mutex<Vec<Played>>>,
    index: usize,
    unplugged_after: Option<usize>,
}

impl PlaybackHandle for RecordedPlayback {
    fn play(&mut self, samples: &[i16]) -> bool {
        let played = &mut self.played.lock().unwrap()[self.index];
        if self
            .unplugged_after
            .is_some_and(|most| played.stretches >= most)
        {
            return false;
        }
        played.stretches += 1;
        played.samples.extend_from_slice(samples);
        true
    }

    fn playing(&self) -> bool {
        let played = &self.played.lock().unwrap()[self.index];
        played.heard < played.samples.len()
    }

    fn stop(self: Box<Self>) {}
}

/// Stopped or dropped, what was not heard yet never is.
impl Drop for RecordedPlayback {
    fn drop(&mut self) {
        let played = &mut self.played.lock().unwrap()[self.index];
        played.stopped = played.heard < played.samples.len();
    }
}

fn el(role: &str, name: &str, children: Vec<RecordedElement>) -> RecordedElement {
    RecordedElement {
        element: UiElement {
            role: role.into(),
            name: name.into(),
            ..UiElement::default()
        },
        children,
    }
}

/// A chat app as the context investigator sees it: wrappers, a channel header and a message
/// list, with a mail window it may not read behind it.
pub fn slack_inspector() -> Arc<dyn ContextInspector> {
    let messages = el(
        "List",
        "Messages in general",
        vec![
            el(
                "ListItem",
                "",
                vec![
                    el("Text", "Ana", vec![]),
                    el("Text", "Launch moved to Friday", vec![]),
                ],
            ),
            el(
                "ListItem",
                "",
                vec![el("Text", "Bo", vec![]), el("Text", "Thanks!", vec![])],
            ),
        ],
    );
    let body = el(
        "Group",
        "",
        vec![el(
            "Group",
            "",
            vec![el("Heading", "general", vec![]), messages],
        )],
    );
    let tree = RecordedTree {
        windows: vec![
            RecordedWindow {
                window: WindowEntry {
                    id: "w-slack".into(),
                    app: "slack.exe".into(),
                    title: "general - Acme".into(),
                    front: true,
                },
                children: vec![
                    el("Pane", "", vec![body]),
                    el("Edit", "Message #general", vec![]),
                ],
            },
            RecordedWindow {
                window: WindowEntry {
                    id: "w-mail".into(),
                    app: "outlook.exe".into(),
                    title: "Inbox".into(),
                    front: false,
                },
                children: vec![el("Text", "Secret mail", vec![])],
            },
        ],
    };
    Arc::new(RecordedInspector::new(tree))
}

/// Selecting a channel, typing into the composer, and pressing Enter, over the Slack
/// fixture.
pub fn slack_demonstration() -> (Demonstration, String, String) {
    let tree: RecordedTree =
        serde_json::from_str(include_str!("../../../examples/desktop/trees/slack.json")).unwrap();
    let inspector = RecordedInspector::new(tree.clone());
    let flat = inspector.subtree("w-slack", 64, 10_000).unwrap();
    let id = |role: &str, name: &str| {
        flat.iter()
            .find(|(_, e)| e.role == role && e.name == name)
            .map(|(_, e)| e.id.clone())
            .unwrap()
    };
    let random = id("TreeItem", "random");
    let composer = id("Edit", "Message #general");
    let demonstration = Demonstration {
        steps: vec![
            DemonstratedStep {
                tree: tree.clone(),
                deed: Deed::Act {
                    target: random.clone(),
                    action: UiAction::Click,
                },
            },
            DemonstratedStep {
                tree: tree.clone(),
                deed: Deed::Act {
                    target: composer.clone(),
                    action: UiAction::TypeText("lunch is ready".into()),
                },
            },
            DemonstratedStep {
                tree: tree.clone(),
                deed: Deed::Press {
                    chord: Chord::parse("enter").unwrap(),
                },
            },
        ],
        end: tree,
    };
    (demonstration, random, composer)
}
