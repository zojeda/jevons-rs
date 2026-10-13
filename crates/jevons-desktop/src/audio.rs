//! Audio through CPAL (WASAPI, ALSA/PulseAudio, CoreAudio). The microphone is converted to the
//! 24 kHz mono PCM16 the Realtime API takes, and an answer's speech, which arrives in that
//! format, is converted to what the output device plays.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use jevons_audio::Resampler;
use jevons_desktop_core::levels::Meter;
use jevons_desktop_core::platform::{
    AudioDevice, AudioEvent, AudioSource, CaptureHandle, PlatformError, PlaybackHandle,
    SAMPLE_RATE, Speaker,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::UnboundedSender;

/// 100 ms at [`SAMPLE_RATE`].
const CHUNK: usize = SAMPLE_RATE as usize / 10;

pub struct CpalSource;

struct Capture {
    stop: mpsc::Sender<()>,
}

impl CaptureHandle for Capture {
    fn stop(self: Box<Self>) {
        let _ = self.stop.send(());
    }
}

/// Downmixes, resamples and cuts the device audio into chunks.
struct Converter {
    resampler: Resampler,
    resampled: Vec<f32>,
    pending: Vec<i16>,
    meter: Meter,
    events: UnboundedSender<AudioEvent>,
}

impl Converter {
    fn push(&mut self, mono: &[f32]) {
        self.resampled.clear();
        self.resampler.process(mono, &mut self.resampled);
        self.emit();
    }

    fn finish(&mut self) {
        self.resampled.clear();
        self.resampler.finish(&mut self.resampled);
        self.emit();
        if !self.pending.is_empty() {
            let chunk = std::mem::take(&mut self.pending);
            let _ = self.events.send(AudioEvent::Chunk(chunk));
        }
    }

    fn emit(&mut self) {
        self.pending.extend(
            self.resampled
                .iter()
                .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16),
        );
        while self.pending.len() >= CHUNK {
            let chunk: Vec<i16> = self.pending.drain(..CHUNK).collect();
            let _ = self
                .events
                .send(AudioEvent::Level(self.meter.update(&chunk)));
            let _ = self.events.send(AudioEvent::Chunk(chunk));
        }
    }
}

fn failed(e: impl std::fmt::Display) -> PlatformError {
    PlatformError::Failed(e.to_string())
}

fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_default()
}

impl AudioSource for CpalSource {
    fn devices(&self) -> Vec<AudioDevice> {
        let host = cpal::default_host();
        let default = host.default_input_device().map(|d| device_name(&d));
        host.input_devices()
            .map(|devices| {
                devices
                    .map(|d| {
                        let name = device_name(&d);
                        AudioDevice {
                            default: Some(&name) == default.as_ref(),
                            name,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn start(
        &mut self,
        device: Option<&str>,
        events: UnboundedSender<AudioEvent>,
    ) -> Result<Box<dyn CaptureHandle>, PlatformError> {
        let wanted = device.map(String::from);
        let (stop, stopped) = mpsc::channel();
        let (ready, started) = mpsc::channel();
        // Streams are not Send on every platform: own the stream on its own thread.
        std::thread::Builder::new()
            .name("microphone".into())
            .spawn(move || {
                let stream = match open(wanted.as_deref(), events.clone()) {
                    Ok(opened) => opened,
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                let _ = ready.send(Ok(()));
                let _ = stopped.recv();
                let (stream, converter) = stream;
                drop(stream);
                converter.lock().expect("the converter lock").finish();
                let _ = events.send(AudioEvent::Ended);
            })
            .map_err(failed)?;
        started
            .recv()
            .map_err(|_| failed("the microphone thread ended"))??;
        Ok(Box::new(Capture { stop }))
    }
}

type Opened = (cpal::Stream, Arc<Mutex<Converter>>);

fn open(
    wanted: Option<&str>,
    events: UnboundedSender<AudioEvent>,
) -> Result<Opened, PlatformError> {
    let host = cpal::default_host();
    let device = match wanted {
        Some(name) => host
            .input_devices()
            .map_err(failed)?
            .find(|d| device_name(d) == name)
            .ok_or_else(|| failed(format!("No microphone named {name:?}")))?,
        None => host
            .default_input_device()
            .ok_or_else(|| failed("No microphone is available"))?,
    };
    let supported = device.default_input_config().map_err(failed)?;
    let channels = usize::from(supported.channels()).max(1);
    let converter = Arc::new(Mutex::new(Converter {
        resampler: Resampler::new(supported.sample_rate(), SAMPLE_RATE),
        resampled: Vec::new(),
        pending: Vec::new(),
        meter: Meter::default(),
        events: events.clone(),
    }));
    let errors = events.clone();
    let on_error = move |e: cpal::Error| {
        if ends_stream(e.kind()) {
            let _ = errors.send(AudioEvent::Failed(e.to_string()));
        } else {
            tracing::warn!(error = %e, "The microphone skipped");
        }
    };
    let config = supported.config();
    let sink = converter.clone();
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream::<f32, _, _>(
            config,
            move |data, _| {
                let mono = downmix(data, channels, |s| s);
                sink.lock().expect("the converter lock").push(&mono);
            },
            on_error,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream::<i16, _, _>(
            config,
            move |data, _| {
                let mono = downmix(data, channels, |s| f32::from(s) / 32768.0);
                sink.lock().expect("the converter lock").push(&mono);
            },
            on_error,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream::<u16, _, _>(
            config,
            move |data, _| {
                let mono = downmix(data, channels, |s| (f32::from(s) - 32768.0) / 32768.0);
                sink.lock().expect("the converter lock").push(&mono);
            },
            on_error,
            None,
        ),
        other => return Err(failed(format!("Unsupported microphone format {other:?}"))),
    }
    .map_err(failed)?;
    stream.play().map_err(failed)?;
    Ok((stream, converter))
}

fn downmix<T: Copy>(data: &[T], channels: usize, to_f32: impl Fn(T) -> f32) -> Vec<f32> {
    data.chunks(channels)
        .map(|frame| frame.iter().map(|&s| to_f32(s)).sum::<f32>() / frame.len() as f32)
        .collect()
}

/// Whether a stream error ends the capture or the playback. A glitch does not (the system was
/// busy for a moment and some samples were lost), nor does a change of route the stream followed
/// by itself: the take goes on with what is heard, and the speech with what is left of it.
fn ends_stream(kind: cpal::ErrorKind) -> bool {
    !matches!(kind, cpal::ErrorKind::Xrun | cpal::ErrorKind::DeviceChanged)
}

/// Plays an answer's speech on an output device.
pub struct CpalSpeaker;

/// What a playback's two sides share: the handle queues, the device's callback plays.
#[derive(Default)]
struct Played {
    /// Frames queued and frames the device took, at the device's rate.
    queued: AtomicU64,
    heard: AtomicU64,
    /// Stopped, or the device went away: nothing more plays.
    over: AtomicBool,
}

/// The handle's side of a playback: speech in, resampled to the device's rate.
struct Feed {
    resampler: Resampler,
    chunks: mpsc::Sender<Vec<f32>>,
    played: Arc<Played>,
}

/// The device's side: its callback takes what is queued without ever waiting for it. The
/// chunks come over a channel the callback only polls, so no lock is shared with the handle.
struct Drain {
    chunks: mpsc::Receiver<Vec<f32>>,
    /// The chunk being played, and how far into it.
    chunk: Vec<f32>,
    at: usize,
    played: Arc<Played>,
}

/// A playback's two sides, for a device that plays at `rate`.
fn playback(rate: u32) -> (Feed, Drain) {
    let (send, receive) = mpsc::channel();
    let played = Arc::new(Played::default());
    (
        Feed {
            resampler: Resampler::new(SAMPLE_RATE, rate),
            chunks: send,
            played: played.clone(),
        },
        Drain {
            chunks: receive,
            chunk: Vec::new(),
            at: 0,
            played,
        },
    )
}

impl Feed {
    /// Queues mono PCM16 at [`SAMPLE_RATE`] behind what is queued. False once nothing plays
    /// any more.
    fn push(&mut self, samples: &[i16]) -> bool {
        if self.played.over.load(Ordering::Acquire) {
            return false;
        }
        let mono: Vec<f32> = samples.iter().map(|s| f32::from(*s) / 32768.0).collect();
        let mut chunk = Vec::new();
        self.resampler.process(&mono, &mut chunk);
        if chunk.is_empty() {
            return true;
        }
        // Counted before it is sent: the callback may take it at once.
        self.played
            .queued
            .fetch_add(chunk.len() as u64, Ordering::AcqRel);
        self.chunks.send(chunk).is_ok()
    }

    /// Whether any queued frame is still to be heard.
    fn playing(&self) -> bool {
        !self.played.over.load(Ordering::Acquire)
            && self.played.heard.load(Ordering::Acquire)
                < self.played.queued.load(Ordering::Acquire)
    }
}

impl Drain {
    /// The next queued sample, if there is one.
    fn next(&mut self) -> Option<f32> {
        while self.at >= self.chunk.len() {
            self.chunk = self.chunks.try_recv().ok()?;
            self.at = 0;
        }
        self.at += 1;
        Some(self.chunk[self.at - 1])
    }

    /// Fills one buffer of the device: every channel of a frame gets the next sample, and
    /// silence when nothing is queued or the playback is over.
    fn fill<T: Copy>(&mut self, out: &mut [T], channels: usize, convert: impl Fn(f32) -> T) {
        let over = self.played.over.load(Ordering::Acquire);
        let mut heard = 0;
        for frame in out.chunks_mut(channels.max(1)) {
            let sample = if over { None } else { self.next() };
            heard += u64::from(sample.is_some());
            frame.fill(convert(sample.unwrap_or(0.0)));
        }
        self.played.heard.fetch_add(heard, Ordering::AcqRel);
    }
}

/// Speech as it plays: its stream lives on a thread of its own until this is dropped.
struct Playback {
    feed: Feed,
    /// Dropped with the handle, which ends the stream's thread.
    _stop: mpsc::Sender<()>,
}

impl PlaybackHandle for Playback {
    fn play(&mut self, samples: &[i16]) -> bool {
        self.feed.push(samples)
    }

    fn playing(&self) -> bool {
        self.feed.playing()
    }

    fn stop(self: Box<Self>) {}
}

/// Stopped or dropped, the device goes silent at once: what is queued is never played.
impl Drop for Playback {
    fn drop(&mut self) {
        self.feed.played.over.store(true, Ordering::Release);
    }
}

impl Speaker for CpalSpeaker {
    fn devices(&self) -> Vec<AudioDevice> {
        let host = cpal::default_host();
        let default = host.default_output_device().map(|d| device_name(&d));
        host.output_devices()
            .map(|devices| {
                devices
                    .map(|d| {
                        let name = device_name(&d);
                        AudioDevice {
                            default: Some(&name) == default.as_ref(),
                            name,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn start(&mut self, device: Option<&str>) -> Result<Box<dyn PlaybackHandle>, PlatformError> {
        let wanted = device.map(String::from);
        let (stop, stopped) = mpsc::channel::<()>();
        let (ready, started) = mpsc::channel();
        // Streams are not Send on every platform: own the stream on its own thread.
        std::thread::Builder::new()
            .name("speaker".into())
            .spawn(move || {
                let (stream, feed) = match open_output(wanted.as_deref()) {
                    Ok(opened) => opened,
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                let _ = ready.send(Ok(feed));
                // Until the handle is stopped or dropped.
                let _ = stopped.recv();
                drop(stream);
            })
            .map_err(failed)?;
        let feed = started
            .recv()
            .map_err(|_| failed("the speaker thread ended"))??;
        Ok(Box::new(Playback { feed, _stop: stop }))
    }
}

/// PCM16 from a sample, as the microphone's chunks are made.
fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * 32767.0) as i16
}

/// Unsigned PCM16, whose silence is the middle of its range.
fn to_u16(sample: f32) -> u16 {
    (i32::from(to_i16(sample)) + 32768) as u16
}

fn open_output(wanted: Option<&str>) -> Result<(cpal::Stream, Feed), PlatformError> {
    let host = cpal::default_host();
    let device = match wanted {
        Some(name) => host
            .output_devices()
            .map_err(failed)?
            .find(|d| device_name(d) == name)
            .ok_or_else(|| failed(format!("No output device named {name:?}")))?,
        None => host
            .default_output_device()
            .ok_or_else(|| failed("No output device is available"))?,
    };
    let supported = device.default_output_config().map_err(failed)?;
    let channels = usize::from(supported.channels()).max(1);
    let (feed, mut drain) = playback(supported.sample_rate());
    let lost = feed.played.clone();
    let on_error = move |e: cpal::Error| {
        if ends_stream(e.kind()) {
            // The handle learns it from its next stretch, and that take speaks no more.
            lost.over.store(true, Ordering::Release);
            tracing::warn!(error = %e, "The output device stopped playing");
        } else {
            tracing::warn!(error = %e, "The output device skipped");
        }
    };
    let config = supported.config();
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream::<f32, _, _>(
            config,
            move |out, _| drain.fill(out, channels, |s| s),
            on_error,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_output_stream::<i16, _, _>(
            config,
            move |out, _| drain.fill(out, channels, to_i16),
            on_error,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_output_stream::<u16, _, _>(
            config,
            move |out, _| drain.fill(out, channels, to_u16),
            on_error,
            None,
        ),
        other => return Err(failed(format!("Unsupported output format {other:?}"))),
    }
    .map_err(failed)?;
    stream.play().map_err(failed)?;
    Ok((stream, feed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_glitch_does_not_end_the_capture_and_a_lost_device_does() {
        assert!(!ends_stream(cpal::ErrorKind::Xrun));
        assert!(!ends_stream(cpal::ErrorKind::DeviceChanged));
        assert!(ends_stream(cpal::ErrorKind::DeviceNotAvailable));
        assert!(ends_stream(cpal::ErrorKind::BackendError));
    }

    /// One buffer of a device with `channels` channels and `frames` frames, as floats.
    fn buffer(drain: &mut Drain, channels: usize, frames: usize) -> Vec<f32> {
        let mut out = vec![9.0f32; channels * frames];
        drain.fill(&mut out, channels, |s| s);
        out
    }

    #[test]
    fn speech_plays_in_order_on_every_channel_and_silence_when_nothing_is_queued() {
        // A device at the speech's own rate: every sample is a frame, as it was given.
        let (mut feed, mut drain) = playback(SAMPLE_RATE);
        // Nothing sounds until the handle takes samples.
        assert!(!feed.playing());
        assert_eq!(buffer(&mut drain, 2, 2), [0.0; 4]);
        assert!(feed.push(&[16384, -16384]));
        assert!(feed.push(&[8192]));
        assert!(feed.playing());
        // Stereo: both channels of a frame carry the sample, across the two stretches.
        assert_eq!(buffer(&mut drain, 2, 2), [0.5, 0.5, -0.5, -0.5]);
        assert!(feed.playing(), "a sample is still to be heard");
        // The queue runs dry in the middle of a buffer: the rest of it is silence.
        assert_eq!(buffer(&mut drain, 2, 3), [0.25, 0.25, 0.0, 0.0, 0.0, 0.0]);
        assert!(!feed.playing(), "everything queued was heard");
        // More speech after a gap plays on, as when the voice is made slower than it is heard.
        assert!(feed.push(&[-8192]));
        assert!(feed.playing());
        assert_eq!(buffer(&mut drain, 1, 2), [-0.25, 0.0]);
        assert!(!feed.playing());
        // A stretch that holds nothing queues nothing.
        assert!(feed.push(&[]));
        assert!(!feed.playing());
    }

    #[test]
    fn speech_is_resampled_to_the_device_s_rate() {
        let (mut feed, mut drain) = playback(48_000);
        // Half a second at 24 kHz, in the stretches the server sends.
        for _ in 0..5 {
            assert!(feed.push(&[8192; 2400]));
        }
        let queued = feed.played.queued.load(Ordering::Acquire) as usize;
        // Twice the frames, less the few the resampler still holds for what follows.
        assert!((23_800..=24_000).contains(&queued), "{queued}");
        let heard = buffer(&mut drain, 2, queued);
        assert!(!feed.playing());
        // The level is kept, on both channels.
        let middle = &heard[queued..queued + 2];
        assert!(middle.iter().all(|s| (s - 0.25).abs() < 0.01), "{middle:?}");
        assert_eq!(buffer(&mut drain, 2, 4), [0.0; 8]);
    }

    #[test]
    fn a_playback_that_was_stopped_or_lost_its_device_plays_no_more() {
        let (mut feed, mut drain) = playback(SAMPLE_RATE);
        assert!(feed.push(&[16384; 8]));
        assert_eq!(buffer(&mut drain, 1, 2), [0.5, 0.5]);
        // The device goes away: what was queued is dropped, and the next stretch is refused.
        feed.played.over.store(true, Ordering::Release);
        assert_eq!(buffer(&mut drain, 1, 4), [0.0; 4]);
        assert!(!feed.playing());
        assert!(!feed.push(&[16384; 8]));
        // A handle that is dropped silences the device the same way.
        let (send, _receive) = mpsc::channel();
        let (feed, mut drain) = playback(SAMPLE_RATE);
        let mut handle = Box::new(Playback { feed, _stop: send });
        assert!(handle.play(&[16384; 8]) && handle.playing());
        handle.stop();
        assert_eq!(buffer(&mut drain, 1, 4), [0.0; 4]);
        // One whose stream ended refuses what comes after.
        let (mut feed, drain) = playback(SAMPLE_RATE);
        drop(drain);
        assert!(!feed.push(&[1; 4]));
    }

    #[test]
    fn integer_devices_get_pcm16_with_silence_where_their_format_has_it() {
        let (mut feed, mut drain) = playback(SAMPLE_RATE);
        assert!(feed.push(&[16384, -32768]));
        let mut signed = [7i16; 3];
        drain.fill(&mut signed, 1, to_i16);
        assert_eq!(signed, [16383, -32767, 0]);
        // Unsigned samples are silent at the middle of their range.
        let mut unsigned = [7u16; 2];
        drain.fill(&mut unsigned, 1, to_u16);
        assert_eq!(unsigned, [32768, 32768]);
        assert_eq!((to_u16(1.0), to_u16(-1.0)), (65535, 1));
    }

    #[test]
    fn stereo_is_averaged_and_output_comes_in_100_ms_chunks() {
        assert_eq!(downmix(&[1.0f32, 0.0, 0.5, 0.5], 2, |s| s), [0.5, 0.5]);
        let (events, mut received) = tokio::sync::mpsc::unbounded_channel();
        let mut converter = Converter {
            resampler: Resampler::new(48_000, SAMPLE_RATE),
            resampled: Vec::new(),
            pending: Vec::new(),
            meter: Meter::default(),
            events,
        };
        converter.push(&vec![0.25; 48_000]);
        converter.finish();
        let mut samples = 0;
        let mut chunks = Vec::new();
        while let Ok(event) = received.try_recv() {
            if let AudioEvent::Chunk(chunk) = event {
                samples += chunk.len();
                chunks.push(chunk.len());
            }
        }
        assert_eq!(samples, SAMPLE_RATE as usize);
        assert!(chunks[..chunks.len() - 1].iter().all(|&n| n == CHUNK));
    }
}
