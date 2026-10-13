//! The app's hold on the desk's voice. The desk plays what the server sends it under its own
//! rules ([`Voice`]); the app says when the microphone opens and closes, stops the voice, and
//! follows whether it still plays, for the tray, the bubble and the Escape key.
//!
//! The agent opens and closes the microphone only through [`VoiceControl`], so the voice always
//! knows: it stops before the microphone opens, and speaks again once it has closed.

use crate::config::VoiceSettings;
use jevons_desktop_core::platform::{
    AudioDevice, AudioEvent, AudioSource, CaptureHandle, PlatformError, Speaker,
};
use jevons_desktop_core::voice::Voice;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

pub struct VoiceControl {
    voice: Arc<Voice>,
    /// The take that spoke when the app last looked.
    seen: Option<u64>,
}

impl VoiceControl {
    /// The voice over `speaker`, as the settings have it.
    pub fn new(speaker: Box<dyn Speaker>, settings: &VoiceSettings) -> Self {
        let voice = Arc::new(Voice::new(speaker));
        voice.configure(settings);
        Self { voice, seen: None }
    }

    /// The voice the desk speaks through.
    pub fn desk(&self) -> Arc<Voice> {
        self.voice.clone()
    }

    /// The output devices, for the settings.
    pub fn devices(&self) -> Vec<AudioDevice> {
        self.voice.devices()
    }

    /// Follows the settings: whether to speak at all, and on which device.
    pub fn configure(&self, settings: &VoiceSettings) {
        self.voice.configure(settings);
    }

    /// Opens the microphone for a take. The voice stops first, so jevons never hears itself,
    /// and stays silent while the microphone is open. A microphone that does not open leaves
    /// the voice free to speak.
    pub fn listen(
        &self,
        audio: &mut dyn AudioSource,
        device: Option<&str>,
        events: UnboundedSender<AudioEvent>,
    ) -> Result<Box<dyn CaptureHandle>, PlatformError> {
        self.voice.set_listening(true);
        audio.start(device, events).inspect_err(|_| {
            self.voice.set_listening(false);
        })
    }

    /// Closes the microphone a take listened on, if it is still open: the voice may speak
    /// again.
    pub fn close(&self, capture: Option<Box<dyn CaptureHandle>>) {
        if let Some(capture) = capture {
            capture.stop();
            self.voice.set_listening(false);
        }
    }

    /// Stops what plays. The take it stopped speaks no more: the server learns it from the next
    /// stretch it sends, and that take ends by itself. False when nothing played.
    pub fn stop(&self) -> bool {
        self.voice.stop().is_some()
    }

    /// The take whose speech is still to be heard, if any.
    pub fn speaking(&self) -> Option<u64> {
        self.voice.speaking()
    }

    /// Looks at the voice again. `Some` when it differs from the last look: the take that
    /// speaks now, or `None` when the voice has ended.
    pub fn changed(&mut self) -> Option<Option<u64>> {
        let now = self.voice.speaking();
        (now != self.seen).then(|| {
            self.seen = now;
            now
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jevons_desktop_core::desk::LocalDesk;
    use jevons_desktop_core::fake::RecordingSpeaker;
    use jevons_desktop_protocol::desk::{Desk, Speech};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A microphone that counts the captures it has open, or cannot open at all.
    #[derive(Default)]
    struct Microphone {
        open: Arc<AtomicUsize>,
        broken: bool,
    }

    struct Open(Arc<AtomicUsize>);

    impl CaptureHandle for Open {
        fn stop(self: Box<Self>) {
            self.0.fetch_sub(1, Ordering::Relaxed);
        }
    }

    impl AudioSource for Microphone {
        fn devices(&self) -> Vec<AudioDevice> {
            Vec::new()
        }

        fn start(
            &mut self,
            _: Option<&str>,
            _: UnboundedSender<AudioEvent>,
        ) -> Result<Box<dyn CaptureHandle>, PlatformError> {
            if self.broken {
                return Err(PlatformError::Failed("No microphone is available".into()));
            }
            self.open.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(Open(self.open.clone())))
        }
    }

    /// The app's voice over a recording speaker, and the desk a take speaks through.
    fn app() -> (RecordingSpeaker, VoiceControl, LocalDesk) {
        let speaker = RecordingSpeaker::new();
        let control = VoiceControl::new(Box::new(speaker.clone()), &VoiceSettings::default());
        let desk = LocalDesk::default().with_speaker(control.desk());
        (speaker, control, desk)
    }

    fn stretch(take: u64, last: bool) -> Speech {
        Speech {
            take,
            samples: vec![take as i16; 4],
            last,
        }
    }

    fn events() -> UnboundedSender<AudioEvent> {
        tokio::sync::mpsc::unbounded_channel().0
    }

    #[tokio::test]
    async fn starting_a_take_stops_the_voice_before_the_microphone_opens() {
        let (speaker, control, desk) = app();
        let mut microphone = Microphone::default();
        // An answer is read aloud, and its take still makes the rest of it.
        assert_eq!(desk.speak(stretch(1, false)).await, Ok(true));
        assert_eq!(control.speaking(), Some(1));
        // The next take starts: the voice is cut, then the microphone opens.
        let capture = control
            .listen(&mut microphone, None, events())
            .map_err(|e| e.to_string())
            .unwrap();
        assert!(speaker.played()[0].stopped);
        assert_eq!(control.speaking(), None);
        assert_eq!(microphone.open.load(Ordering::Relaxed), 1);
        // The take that was cut learns it from its next stretch, and a timer's take cannot
        // talk over the user.
        assert_eq!(desk.speak(stretch(1, false)).await, Ok(false));
        assert_eq!(desk.speak(stretch(2, true)).await, Ok(false));
        // The microphone closes: the take that follows speaks.
        control.close(Some(capture));
        assert_eq!(microphone.open.load(Ordering::Relaxed), 0);
        assert_eq!(desk.speak(stretch(3, true)).await, Ok(true));
        assert_eq!(control.speaking(), Some(3));
        // A microphone already closed changes nothing.
        control.close(None);
        assert_eq!(control.speaking(), Some(3));
    }

    #[tokio::test]
    async fn a_microphone_that_does_not_open_leaves_the_voice_free_to_speak() {
        let (_, control, desk) = app();
        let mut microphone = Microphone {
            broken: true,
            ..Microphone::default()
        };
        let failed = control.listen(&mut microphone, None, events());
        assert!(failed.is_err());
        // Nothing listens, so the next answer is read aloud.
        assert_eq!(desk.speak(stretch(1, true)).await, Ok(true));
    }

    #[tokio::test]
    async fn stopping_the_voice_ends_that_take_s_speech_and_leaves_the_next_one_s() {
        let (speaker, control, desk) = app();
        assert!(!control.stop(), "nothing played");
        assert_eq!(desk.speak(stretch(1, false)).await, Ok(true));
        // Escape, the bubble's Stop or the tray's Cancel.
        assert!(control.stop());
        assert!(speaker.played()[0].stopped);
        assert_eq!(control.speaking(), None);
        // The server hears no at its next stretch, and stops making the speech.
        assert_eq!(desk.speak(stretch(1, true)).await, Ok(false));
        assert_eq!(desk.speak(stretch(2, true)).await, Ok(true));
    }

    #[tokio::test]
    async fn the_app_sees_the_voice_start_and_end_once_each() {
        let (speaker, mut control, desk) = app();
        assert_eq!(control.changed(), None);
        assert_eq!(desk.speak(stretch(1, false)).await, Ok(true));
        assert_eq!(control.changed(), Some(Some(1)));
        assert_eq!(control.changed(), None, "nothing new since the last look");
        // The take ends long before its speech was heard: the voice still plays.
        assert_eq!(desk.speak(stretch(1, true)).await, Ok(true));
        assert_eq!(control.changed(), None);
        // Heard to its end, it is over.
        speaker.hear();
        assert_eq!(control.changed(), Some(None));
        assert_eq!(control.changed(), None);
        // A newer take's speech replaces the one that plays.
        assert_eq!(desk.speak(stretch(2, false)).await, Ok(true));
        assert_eq!(control.changed(), Some(Some(2)));
        assert_eq!(desk.speak(stretch(3, false)).await, Ok(true));
        assert_eq!(control.changed(), Some(Some(3)));
        assert!(control.stop());
        assert_eq!(control.changed(), Some(None));
    }

    #[tokio::test]
    async fn the_settings_mute_the_voice_and_name_its_device() {
        let speaker = RecordingSpeaker::new();
        let muted = VoiceSettings {
            speak: false,
            output: None,
        };
        let control = VoiceControl::new(Box::new(speaker.clone()), &muted);
        let desk = LocalDesk::default().with_speaker(control.desk());
        assert_eq!(control.devices()[0].name, "recording");
        // Off from the start: nothing is said.
        assert_eq!(desk.speak(stretch(1, true)).await, Ok(false));
        // Saved settings apply without a restart, and the device is the one they name.
        control.configure(&VoiceSettings {
            speak: true,
            output: Some("Headphones".into()),
        });
        assert_eq!(desk.speak(stretch(2, false)).await, Ok(true));
        assert_eq!(speaker.played()[0].device.as_deref(), Some("Headphones"));
        // Turned off while it speaks, it stops.
        control.configure(&muted);
        assert_eq!(control.speaking(), None);
        assert!(speaker.played()[0].stopped);
    }
}
