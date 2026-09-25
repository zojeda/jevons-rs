//! CPU audio for speech models: bounded decoding of uploaded files (Opus included), resampling,
//! raw PCM and G.711 input, log-mel features, and a streaming voice activity detector.
#![forbid(unsafe_code)]

mod decode;
mod mel;
mod pcm;
mod resample;
mod vad;

pub use decode::{AudioLimits, decode_audio};
pub use mel::{Features, LogMel, MelConfig};
pub use pcm::{alaw, mulaw, pcm16_le};
pub use resample::{Resampler, resample};
pub use vad::{Vad, VadConfig, VadEvent};
