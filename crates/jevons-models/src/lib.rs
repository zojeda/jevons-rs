//! Architecture detection and model loading.
//!
//! Each supported architecture implements [`jevons_core::DiffusionModel`]; [`load`] picks one
//! from the model files, or from an explicit `--arch` choice, and loads it on the calling thread.
#![forbid(unsafe_code)]

mod detect;
#[cfg(feature = "gemma4")]
mod gemma4;

pub use detect::{Architecture, default_model_id, detect};
use jevons_core::{DiffusionModel, Error, ModelConfig, Result};

/// Loads the model named by `config` on the calling thread.
pub fn load(config: &ModelConfig) -> Result<Box<dyn DiffusionModel>> {
    config.validate()?;
    let architecture = resolve(config)?;
    match architecture {
        #[cfg(feature = "gemma4")]
        Architecture::Gemma4Diffusion => Ok(Box::new(gemma4::Gemma4::load(config)?)),
        #[cfg(feature = "nemotron")]
        Architecture::NemotronDiffusion => {
            let mut config = config.clone();
            config.model = detect::checkpoint_dir(&config.model).unwrap_or(config.model);
            Ok(Box::new(jevons_nemotron_diffusion::Nemotron::load(
                &config,
            )?))
        }
        #[allow(unreachable_patterns)]
        other => Err(Error::UnsupportedModel(format!(
            "{} support is not built into this binary",
            other.id()
        ))),
    }
}

/// The architecture `config` selects: the explicit choice, checked against the files, or the
/// detected one.
pub fn resolve(config: &ModelConfig) -> Result<Architecture> {
    let detected = detect(&config.model)?;
    match config.architecture.as_deref() {
        None | Some("auto") => Ok(detected),
        Some(name) => {
            let chosen: Architecture = name.parse()?;
            if chosen != detected {
                return Err(Error::UnsupportedModel(format!(
                    "--arch {} does not match the model files, which contain {}",
                    chosen.id(),
                    detected.id()
                )));
            }
            Ok(chosen)
        }
    }
}
