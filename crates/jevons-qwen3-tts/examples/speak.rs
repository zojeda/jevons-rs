//! Speaks sentences with a Qwen3-TTS checkpoint, writes each as a WAV file and prints where
//! the time went. Needs a HIP GPU.
//!
//!     cargo run --release -p jevons-qwen3-tts --example speak -- MODEL_DIR OUT_DIR \
//!         [--voice NAME] [--language CODE] [--repeat N] [--engine layers|fused] \
//!         [--texts FILE] [TEXT ...]
//!
//! `--texts` reads a sentence per line of a UTF-8 file, for shells that mangle accents.
//! `--engine` chooses what makes the frames: the fused kernels, which is what the runtime
//! runs, or Burn's operators layer by layer, which tune for minutes at each new shape.
use jevons_core::{Utterance, VoiceConfig, VoiceModel};
use jevons_qwen3_tts::{Engine, Qwen3Tts};
use std::path::PathBuf;
use std::time::Instant;

/// A mono PCM16 WAV file of `samples`.
fn wav(rate: u32, samples: &[f32]) -> Vec<u8> {
    let data = samples.len() as u32 * 2;
    let mut file = Vec::with_capacity(44 + data as usize);
    file.extend(b"RIFF");
    file.extend((36 + data).to_le_bytes());
    file.extend(b"WAVEfmt ");
    file.extend(16u32.to_le_bytes());
    file.extend(1u16.to_le_bytes());
    file.extend(1u16.to_le_bytes());
    file.extend(rate.to_le_bytes());
    file.extend((rate * 2).to_le_bytes());
    file.extend(2u16.to_le_bytes());
    file.extend(16u16.to_le_bytes());
    file.extend(b"data");
    file.extend(data.to_le_bytes());
    for sample in samples {
        file.extend(((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    file
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `RUST_LOG=info` shows what the runtime compiles and tunes.
    let filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "error".into());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .ok_or("usage: speak MODEL_DIR OUT_DIR [options] TEXT...")?;
    let out = PathBuf::from(args.next().ok_or("an output folder")?);
    let (mut voice, mut language, mut repeat) = (None, None, 1usize);
    let mut engine = None;
    let mut texts = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--voice" => voice = args.next(),
            "--language" => language = args.next(),
            "--repeat" => repeat = args.next().ok_or("a count")?.parse()?,
            "--texts" => {
                let file = std::fs::read_to_string(args.next().ok_or("a file of sentences")?)?;
                texts.extend(
                    file.lines()
                        .map(str::trim)
                        .filter(|line| !line.is_empty())
                        .map(String::from),
                );
            }
            "--engine" => {
                engine = Some(match args.next().as_deref() {
                    Some("layers") => Engine::Layers,
                    Some("fused") => Engine::Fused,
                    other => return Err(format!("unknown engine {other:?}").into()),
                })
            }
            _ => texts.push(arg),
        }
    }
    std::fs::create_dir_all(&out)?;
    let started = Instant::now();
    let config = VoiceConfig::new(model_dir);
    let mut model = match engine {
        Some(engine) => Qwen3Tts::load_with(&config, engine)?,
        None => Qwen3Tts::load(&config)?,
    };
    println!("loaded in {:.1} s", started.elapsed().as_secs_f64());
    let rate = model.info().sample_rate;
    let voice = voice.unwrap_or_else(|| model.info().voices[0].clone());
    for round in 0..repeat {
        for (i, text) in texts.iter().enumerate() {
            let utterance = Utterance {
                text,
                voice: &voice,
                language: language.as_deref(),
            };
            let mut samples = Vec::new();
            model.speak(&utterance, &mut |chunk| {
                samples.extend_from_slice(chunk);
                true
            })?;
            let t = model.timings();
            let seconds = samples.len() as f64 / f64::from(rate);
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            let frames = t.frames.max(1) as f64;
            println!(
                "{round}.{i}: {seconds:.2} s of audio in {:.0} ms ({:.2} of real time), first audio {:.0} ms, \
                 prompt {:.0} ms, per frame {:.1} ms: frames {:.1}, codec {:.1}",
                ms(t.total),
                t.total.as_secs_f64() / seconds.max(1e-9),
                ms(t.first_audio),
                ms(t.prompt),
                ms(t.total - t.prompt) / frames,
                ms(t.frames_time) / frames,
                ms(t.codec) / frames,
            );
            std::fs::write(out.join(format!("{round}-{i}.wav")), wav(rate, &samples))?;
        }
    }
    Ok(())
}
