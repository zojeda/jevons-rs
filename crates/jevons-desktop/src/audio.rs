//! Microphone capture through CPAL (WASAPI, ALSA/PulseAudio, CoreAudio), converted to the
//! 24 kHz mono PCM16 the Realtime API takes.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use jevons_audio::Resampler;
use jevons_desktop_core::levels::Meter;
use jevons_desktop_core::platform::{
    AudioDevice, AudioEvent, AudioSource, CaptureHandle, PlatformError, SAMPLE_RATE,
};
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
        let _ = errors.send(AudioEvent::Failed(e.to_string()));
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

#[cfg(test)]
mod tests {
    use super::*;

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
