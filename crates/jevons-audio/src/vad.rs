//! A streaming energy voice activity detector for server-side turn detection.
//!
//! Each 20 ms frame's level is compared with an adaptive noise floor. Speech starts after
//! [`START_FRAMES`] loud frames and stops after `silence_duration_ms` of quiet frames. Event
//! positions are sample offsets from the start of the stream.

/// Consecutive loud frames that start speech (60 ms), so clicks do not.
const START_FRAMES: usize = 3;
/// Levels below this never count as speech, however quiet the room.
const ABSOLUTE_FLOOR_DB: f32 = -55.0;

/// OpenAI `server_vad` turn-detection settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadConfig {
    /// Activation threshold in `[0, 1]`; higher needs louder speech. 0.5 means 12 dB above
    /// the noise floor.
    pub threshold: f32,
    /// Audio kept before the detected start, reported in the start position.
    pub prefix_padding_ms: u32,
    /// Silence that ends a turn.
    pub silence_duration_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadEvent {
    /// Speech began at this sample (already moved back by the prefix padding).
    SpeechStarted { sample: usize },
    /// Speech ended at this sample (the end of the last loud frame).
    SpeechStopped { sample: usize },
}

pub struct Vad {
    config: VadConfig,
    frame: usize,
    rate: u32,
    pending: Vec<f32>,
    /// Samples fully processed into frames.
    position: usize,
    /// The noise floor, seeded by the first frame.
    floor_db: Option<f32>,
    speaking: bool,
    loud_run: usize,
    quiet_run: usize,
    last_loud_end: usize,
}

impl Vad {
    pub fn new(config: VadConfig, rate: u32) -> Self {
        Self {
            config,
            frame: (rate / 50) as usize,
            rate,
            pending: Vec::new(),
            position: 0,
            floor_db: None,
            speaking: false,
            loud_run: 0,
            quiet_run: 0,
            last_loud_end: 0,
        }
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking
    }

    /// Ends the current turn without waiting for silence, as a manual commit does.
    pub fn reset_turn(&mut self) {
        self.speaking = false;
        self.loud_run = 0;
        self.quiet_run = 0;
    }

    fn samples(&self, ms: u32) -> usize {
        (u64::from(ms) * u64::from(self.rate) / 1000) as usize
    }

    /// Feeds samples and returns the transitions they complete.
    pub fn push(&mut self, samples: &[f32]) -> Vec<VadEvent> {
        self.pending.extend_from_slice(samples);
        let mut events = Vec::new();
        let frames = self.pending.len() / self.frame;
        for index in 0..frames {
            let frame = &self.pending[index * self.frame..(index + 1) * self.frame];
            let energy = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
            let db = 10.0 * (energy + 1e-12).log10();
            let margin = 6.0 + 12.0 * self.config.threshold.clamp(0.0, 1.0);
            let floor = *self.floor_db.get_or_insert(db);
            let loud = db > ABSOLUTE_FLOOR_DB && db > floor + margin;
            let start = self.position;
            let end = start + self.frame;
            self.position = end;
            if loud {
                self.loud_run += 1;
                self.quiet_run = 0;
                self.last_loud_end = end;
                if !self.speaking && self.loud_run >= START_FRAMES {
                    self.speaking = true;
                    let onset = end - START_FRAMES * self.frame;
                    let sample = onset.saturating_sub(self.samples(self.config.prefix_padding_ms));
                    events.push(VadEvent::SpeechStarted { sample });
                }
            } else {
                self.loud_run = 0;
                self.quiet_run += 1;
                // The floor follows quiet frames: quickly down, slowly up.
                let rate = if db < floor { 0.3 } else { 0.05 };
                self.floor_db = Some(floor + rate * (db.max(-100.0) - floor));
                if self.speaking
                    && self.quiet_run * self.frame >= self.samples(self.config.silence_duration_ms)
                {
                    self.speaking = false;
                    events.push(VadEvent::SpeechStopped {
                        sample: self.last_loud_end,
                    });
                }
            }
        }
        self.pending.drain(..frames * self.frame);
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize, amplitude: f32, seed: &mut u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (*seed >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
            })
            .map(|x| x * amplitude)
            .collect()
    }

    fn tone(len: usize, amplitude: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 * 0.2).sin() * amplitude)
            .collect()
    }

    #[test]
    fn a_turn_starts_with_prefix_padding_and_stops_after_the_silence() {
        let mut seed = 7;
        let mut vad = Vad::new(VadConfig::default(), 16000);
        let mut events = Vec::new();
        // 1 s of room noise, 1 s of speech, then 1 s of room noise, fed in 10 ms chunks.
        let mut audio = noise(16000, 0.001, &mut seed);
        audio.extend(tone(16000, 0.2));
        audio.extend(noise(16000, 0.001, &mut seed));
        for chunk in audio.chunks(160) {
            events.extend(vad.push(chunk));
        }
        assert_eq!(
            events,
            vec![
                VadEvent::SpeechStarted {
                    sample: 16000 - 4800
                },
                VadEvent::SpeechStopped { sample: 32000 },
            ]
        );
        assert!(!vad.is_speaking());
    }

    #[test]
    fn short_clicks_and_pauses_shorter_than_the_silence_do_not_split_turns() {
        let mut vad = Vad::new(VadConfig::default(), 16000);
        let mut events = vad.push(&vec![0.0; 16000]);
        events.extend(vad.push(&tone(320, 0.5)));
        events.extend(vad.push(&vec![0.0; 8000]));
        assert!(
            events.is_empty(),
            "a 20 ms click started speech: {events:?}"
        );
        events.extend(vad.push(&tone(8000, 0.2)));
        events.extend(vad.push(&vec![0.0; 4800]));
        events.extend(vad.push(&tone(8000, 0.2)));
        assert_eq!(events.len(), 1, "a 300 ms pause ended the turn: {events:?}");
        assert!(vad.is_speaking());
    }

    #[test]
    fn a_higher_threshold_ignores_quieter_speech() {
        let mut seed = 3;
        let mut audio = noise(16000, 0.01, &mut seed);
        audio.extend(tone(16000, 0.05));
        let lenient = Vad::new(
            VadConfig {
                threshold: 0.2,
                ..VadConfig::default()
            },
            16000,
        );
        let strict = Vad::new(
            VadConfig {
                threshold: 1.0,
                ..VadConfig::default()
            },
            16000,
        );
        for (mut vad, expected) in [(lenient, true), (strict, false)] {
            vad.push(&audio);
            assert_eq!(vad.is_speaking(), expected);
        }
    }
}
