//! The desk's voice: an answer's speech, played through the platform's speaker under the
//! client's own rules, whatever the server asks.
//!
//! One take speaks at a time, and a newer take's speech replaces the one that plays. Nothing
//! plays while the microphone is open, so jevons never talks over the user or into their take.
//! A take whose voice ended early (stopped, replaced, refused, or lost with its device) speaks
//! no more: what is left of its speech is refused, which tells the server to stop making it.
//!
//! The app keeps the [`Voice`] it gives the desk: it says when the microphone opens and
//! closes, stops the voice, and asks whether it still plays.

use crate::config::VoiceSettings;
use crate::platform::{AudioDevice, PlaybackHandle, Speaker};
use jevons_desktop_protocol::desk::Speech;
use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};

/// How many takes whose voice ended early are remembered. Their speech stops coming once the
/// server hears the first refusal, so a few are plenty.
const REMEMBERED: usize = 16;

/// The take that speaks, and its speech as it plays.
struct Playing {
    take: u64,
    handle: Box<dyn PlaybackHandle>,
    /// Its last stretch came: once the queue plays out, it is over.
    whole: bool,
}

struct Inner {
    speaker: Box<dyn Speaker>,
    playing: Option<Playing>,
    /// Takes whose voice ended early: the rest of their speech is refused.
    over: VecDeque<u64>,
    /// The microphone is open.
    listening: bool,
    /// The user turned the voice off.
    muted: bool,
    /// The output device; the default one when `None`.
    device: Option<String>,
}

impl Inner {
    /// Stops what plays, and refuses the rest of that take's speech.
    fn stop(&mut self) -> Option<u64> {
        let playing = self.playing.take()?;
        playing.handle.stop();
        self.end(playing.take);
        Some(playing.take)
    }

    /// `take` speaks no more.
    fn end(&mut self, take: u64) {
        if !self.over.contains(&take) {
            self.over.push_back(take);
        }
        while self.over.len() > REMEMBERED {
            self.over.pop_front();
        }
    }

    /// Forgets speech that was heard to its end.
    fn settle(&mut self) {
        if self
            .playing
            .as_ref()
            .is_some_and(|playing| playing.whole && !playing.handle.playing())
        {
            self.playing = None;
        }
    }
}

/// What the desk says aloud, one take at a time.
pub struct Voice(Mutex<Inner>);

impl Voice {
    /// A voice that plays through `speaker`, on its default device.
    pub fn new(speaker: Box<dyn Speaker>) -> Self {
        Self(Mutex::new(Inner {
            speaker,
            playing: None,
            over: VecDeque::new(),
            listening: false,
            muted: false,
            device: None,
        }))
    }

    fn inner(&self) -> MutexGuard<'_, Inner> {
        self.0.lock().expect("the voice lock is not poisoned")
    }

    /// The output devices, for the settings.
    pub fn devices(&self) -> Vec<AudioDevice> {
        self.inner().speaker.devices()
    }

    /// Follows the client's `[voice]` settings: whether to speak at all, and on which device.
    /// Turned off, it stops what plays. Another device is used from the next take's speech on.
    pub fn configure(&self, settings: &VoiceSettings) {
        let mut inner = self.inner();
        inner.muted = !settings.speak;
        inner.device = settings.output.clone();
        if inner.muted {
            inner.stop();
        }
    }

    /// The microphone opened or closed. Opening it stops what plays, and nothing plays while
    /// it is open.
    pub fn set_listening(&self, listening: bool) {
        let mut inner = self.inner();
        inner.listening = listening;
        if listening {
            inner.stop();
        }
    }

    /// Stops what plays and refuses the rest of that take's speech. The take it stopped, if
    /// one spoke.
    pub fn stop(&self) -> Option<u64> {
        self.inner().stop()
    }

    /// The take whose speech is still to be heard, if any.
    pub fn speaking(&self) -> Option<u64> {
        let mut inner = self.inner();
        inner.settle();
        inner.playing.as_ref().map(|playing| playing.take)
    }

    /// Queues a stretch of a take's speech, as [`Desk::speak`] asks. False when this take does
    /// not speak: the voice is off, the microphone is open, or its voice ended early. An error
    /// is a speaker that cannot play.
    ///
    /// [`Desk::speak`]: jevons_desktop_protocol::desk::Desk::speak
    pub fn speak(&self, speech: Speech) -> Result<bool, String> {
        let mut inner = self.inner();
        if inner.muted || inner.listening || inner.over.contains(&speech.take) {
            // Refused once, a take stays refused: its speech never starts in the middle.
            inner.end(speech.take);
            return Ok(false);
        }
        inner.settle();
        let plays = inner
            .playing
            .as_ref()
            .is_some_and(|playing| playing.take == speech.take);
        if !plays {
            if speech.samples.is_empty() {
                // An end with no sound before it: it starts nothing, and cuts nobody.
                return Ok(true);
            }
            // A newer take's speech replaces the one that plays.
            inner.stop();
            let device = inner.device.clone();
            let handle = match inner.speaker.start(device.as_deref()) {
                Ok(handle) => handle,
                Err(e) => {
                    inner.end(speech.take);
                    return Err(e.to_string());
                }
            };
            inner.playing = Some(Playing {
                take: speech.take,
                handle,
                whole: false,
            });
        }
        let playing = inner.playing.as_mut().expect("it was just started");
        let queued = speech.samples.is_empty() || playing.handle.play(&speech.samples);
        playing.whole |= speech.last;
        if !queued {
            // The device went away under it.
            inner.playing = None;
            inner.end(speech.take);
        }
        Ok(queued)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desk::LocalDesk;
    use crate::fake::RecordingSpeaker;
    use jevons_desktop_protocol::desk::{Desk, Nobody};

    fn stretch(take: u64, sample: i16, last: bool) -> Speech {
        Speech {
            take,
            samples: vec![sample; 4],
            last,
        }
    }

    #[test]
    fn a_take_s_speech_plays_in_order_until_it_was_heard() {
        let speaker = RecordingSpeaker::new();
        let voice = speaker.voice();
        assert_eq!(voice.speaking(), None);
        assert_eq!(voice.speak(stretch(1, 10, false)), Ok(true));
        assert_eq!(voice.speak(stretch(1, 20, false)), Ok(true));
        // The whole answer is queued long before it is heard.
        assert_eq!(voice.speak(stretch(1, 30, true)), Ok(true));
        assert_eq!(voice.speaking(), Some(1));
        let played = speaker.played();
        assert_eq!(played.len(), 1, "one playback for the take");
        assert_eq!(played[0].device, None);
        assert_eq!(
            played[0].samples,
            [10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30]
        );
        // Heard to its end, it is over, and nobody stopped it.
        speaker.hear();
        assert_eq!(voice.speaking(), None);
        assert!(!speaker.played()[0].stopped);
        // An answer that ends on a boundary closes with a stretch that holds nothing.
        assert_eq!(voice.speak(stretch(2, 1, false)), Ok(true));
        let end = Speech {
            take: 2,
            samples: Vec::new(),
            last: true,
        };
        assert_eq!(voice.speak(end), Ok(true));
        speaker.hear();
        assert_eq!(voice.speaking(), None);
        // Until its last stretch came, a take still speaks, even with nothing left queued.
        assert_eq!(voice.speak(stretch(3, 1, false)), Ok(true));
        speaker.hear();
        assert_eq!(voice.speaking(), Some(3));
    }

    #[test]
    fn a_newer_take_s_speech_replaces_the_one_that_plays() {
        let speaker = RecordingSpeaker::new();
        let voice = speaker.voice();
        assert_eq!(voice.speak(stretch(1, 10, false)), Ok(true));
        assert_eq!(voice.speak(stretch(2, 20, false)), Ok(true));
        assert_eq!(voice.speaking(), Some(2));
        let played = speaker.played();
        assert_eq!(played.len(), 2);
        assert!(played[0].stopped, "the older take's speech was cut");
        assert!(!played[1].stopped);
        // What is left of the older take is refused, and does not cut the newer one.
        assert_eq!(voice.speak(stretch(1, 11, true)), Ok(false));
        assert_eq!(voice.speaking(), Some(2));
        assert_eq!(speaker.played()[1].samples, [20; 4]);
        // A stretch with no samples plays nothing, so it cuts nobody either.
        let empty = Speech {
            take: 3,
            samples: Vec::new(),
            last: true,
        };
        assert_eq!(voice.speak(empty), Ok(true));
        assert_eq!(voice.speaking(), Some(2));
        assert!(!speaker.played()[1].stopped);
    }

    #[test]
    fn nothing_plays_while_the_microphone_is_open() {
        let speaker = RecordingSpeaker::new();
        let voice = speaker.voice();
        assert_eq!(voice.speak(stretch(1, 10, false)), Ok(true));
        // The microphone opens: what plays stops before it hears itself.
        voice.set_listening(true);
        assert_eq!(voice.speaking(), None);
        assert!(speaker.played()[0].stopped);
        // A timer's take cannot talk over the user, nor can the take that was cut.
        assert_eq!(voice.speak(stretch(2, 20, false)), Ok(false));
        assert_eq!(voice.speak(stretch(1, 11, false)), Ok(false));
        assert_eq!(speaker.played().len(), 1);
        // Once it closes, the next take speaks; the one that was cut stays cut.
        voice.set_listening(false);
        assert_eq!(voice.speak(stretch(1, 12, true)), Ok(false));
        assert_eq!(voice.speak(stretch(3, 30, true)), Ok(true));
        assert_eq!(voice.speaking(), Some(3));
    }

    #[test]
    fn a_stopped_take_speaks_no_more() {
        let speaker = RecordingSpeaker::new();
        let voice = speaker.voice();
        assert_eq!(voice.stop(), None, "nothing to stop");
        assert_eq!(voice.speak(stretch(1, 10, false)), Ok(true));
        assert_eq!(voice.stop(), Some(1));
        assert_eq!(voice.speaking(), None);
        assert!(speaker.played()[0].stopped);
        // The server learns it from the next stretch, and stops making the speech.
        assert_eq!(voice.speak(stretch(1, 11, false)), Ok(false));
        assert_eq!(voice.speak(stretch(1, 12, true)), Ok(false));
        assert_eq!(speaker.played()[0].samples, [10; 4]);
        // The next take speaks as usual.
        assert_eq!(voice.speak(stretch(2, 20, true)), Ok(true));
    }

    #[test]
    fn the_settings_turn_the_voice_off_and_choose_its_device() {
        let speaker = RecordingSpeaker::new();
        let voice = speaker.voice();
        assert_eq!(voice.devices()[0].name, "recording");
        let mut settings = VoiceSettings {
            output: Some("Headphones".into()),
            ..VoiceSettings::default()
        };
        voice.configure(&settings);
        assert_eq!(voice.speak(stretch(1, 10, false)), Ok(true));
        assert_eq!(speaker.played()[0].device.as_deref(), Some("Headphones"));
        // Turned off, it stops what plays and speaks nothing.
        settings.speak = false;
        voice.configure(&settings);
        assert_eq!(voice.speaking(), None);
        assert!(speaker.played()[0].stopped);
        assert_eq!(voice.speak(stretch(2, 20, false)), Ok(false));
        assert_eq!(speaker.played().len(), 1);
        // Turned on again, the next take speaks.
        settings.speak = true;
        voice.configure(&settings);
        assert_eq!(voice.speak(stretch(3, 30, false)), Ok(true));
    }

    #[test]
    fn a_speaker_that_cannot_play_says_why_and_one_that_goes_away_ends_the_take() {
        // No device to start on: the error says so, and the take is not tried again.
        let none = RecordingSpeaker::new().without_device();
        let voice = none.voice();
        let failed = voice.speak(stretch(1, 10, false));
        assert_eq!(failed, Err("no output device".into()));
        assert_eq!(voice.speak(stretch(1, 11, false)), Ok(false));
        assert!(none.played().is_empty());
        // A device unplugged while it plays: that take speaks no more.
        let unplugged = RecordingSpeaker::new().unplugged_after(1);
        let voice = unplugged.voice();
        assert_eq!(voice.speak(stretch(1, 10, false)), Ok(true));
        assert_eq!(voice.speak(stretch(1, 11, false)), Ok(false));
        assert_eq!(voice.speaking(), None);
        assert_eq!(voice.speak(stretch(1, 12, true)), Ok(false));
        assert_eq!(unplugged.played()[0].samples, [10; 4]);
    }

    #[tokio::test]
    async fn a_desk_speaks_through_its_speaker_and_one_without_is_silent() {
        let speaker = RecordingSpeaker::new();
        let desk = LocalDesk::default().with_speaker(speaker.voice());
        assert_eq!(desk.speak(stretch(1, 10, true)).await, Ok(true));
        assert_eq!(speaker.played()[0].samples, [10; 4]);
        // Without a speaker the desk says it does not speak, as a desk with no one at it.
        let silent = LocalDesk::default();
        assert_eq!(silent.speak(stretch(1, 10, true)).await, Ok(false));
        assert_eq!(Nobody.speak(stretch(1, 10, true)).await, Ok(false));
    }
}
