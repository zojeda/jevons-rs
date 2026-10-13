//! CPU audio for speech and voice models: bounded decoding of uploaded files (Opus included),
//! resampling, raw PCM and G.711 input, log-mel features, a streaming voice activity detector,
//! and the output half: f32 to PCM16 and WAV.
#![forbid(unsafe_code)]

mod decode;
mod mel;
mod pcm;
mod resample;
mod vad;
mod wav;

pub use decode::{AudioLimits, decode_audio};
pub use mel::{Features, LogMel, MelConfig};
pub use pcm::{alaw, mulaw, pcm16_le, pcm16_le_bytes, to_pcm16};
pub use resample::{Resampler, resample};
pub use vad::{Vad, VadConfig, VadEvent};
pub use wav::{wav, wav_header};
