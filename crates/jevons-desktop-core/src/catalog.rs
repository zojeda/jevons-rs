//! Models the app knows how to download. The built-in entries can be extended with
//! `models.toml` next to the settings file:
//!
//! ```toml
//! [[models]]
//! id = "my-model"
//! name = "My model"
//! services = ["generative", "decision"]
//! repo = "someone/my-model"
//! files = ["*.json", "*.safetensors"]
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Service {
    Generative,
    Decision,
    Speech,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    /// Also the folder name under the models folder.
    pub id: String,
    pub name: String,
    pub services: Vec<Service>,
    /// The Hugging Face repository.
    pub repo: String,
    #[serde(default = "main")]
    pub revision: String,
    /// Globs over the repository's file paths.
    pub files: Vec<String>,
    /// For single-file models (GGUF): the file to load. Directories load as a whole.
    #[serde(default)]
    pub model_file: Option<String>,
    #[serde(default)]
    pub mmproj_file: Option<String>,
    #[serde(default)]
    pub license: String,
    /// Approximate memory while loaded, in GB.
    #[serde(default)]
    pub memory_gb: f32,
}

fn main() -> String {
    "main".into()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogFile {
    #[serde(default)]
    models: Vec<CatalogEntry>,
}

/// The models shipped with the app.
pub fn builtin() -> Vec<CatalogEntry> {
    let nemotron = |id: &str, name: &str, repo: &str, memory_gb| CatalogEntry {
        id: id.into(),
        name: name.into(),
        services: vec![Service::Generative, Service::Decision],
        repo: repo.into(),
        revision: main(),
        files: [
            "*.json",
            "*.jinja",
            "*.py",
            "*.safetensors",
            "linear_spec_lora/**",
        ]
        .map(String::from)
        .to_vec(),
        model_file: None,
        mmproj_file: None,
        license: "NVIDIA Open Model License".into(),
        memory_gb,
    };
    vec![
        nemotron(
            "nemotron-labs-diffusion-3b",
            "Nemotron-Labs-Diffusion 3B",
            "nvidia/Nemotron-Labs-Diffusion-3B",
            8.0,
        ),
        nemotron(
            "nemotron-labs-diffusion-vlm-8b",
            "Nemotron-Labs-Diffusion VLM 8B",
            "nvidia/Nemotron-Labs-Diffusion-VLM-8B",
            18.0,
        ),
        CatalogEntry {
            id: "parakeet-tdt-0.6b-v3".into(),
            name: "Parakeet TDT 0.6B v3 (speech, 25 languages)".into(),
            services: vec![Service::Speech],
            repo: "nvidia/parakeet-tdt-0.6b-v3".into(),
            revision: main(),
            files: [
                "config.json",
                "generation_config.json",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
            ]
            .map(String::from)
            .to_vec(),
            model_file: None,
            mmproj_file: None,
            license: "CC-BY-4.0".into(),
            memory_gb: 1.5,
        },
    ]
}

/// The built-in entries plus those in `file` (which may override a built-in id).
pub fn load(file: &Path) -> (Vec<CatalogEntry>, Option<String>) {
    let mut entries = builtin();
    let extra = match std::fs::read_to_string(file) {
        Ok(text) => match toml::from_str::<CatalogFile>(&text) {
            Ok(parsed) => parsed.models,
            Err(e) => return (entries, Some(format!("{}: {e}", file.display()))),
        },
        Err(_) => Vec::new(),
    };
    for entry in extra {
        entries.retain(|e| e.id != entry.id);
        entries.push(entry);
    }
    (entries, None)
}

impl CatalogEntry {
    pub fn dir(&self, folder: &Path) -> PathBuf {
        folder.join(&self.id)
    }

    /// The path jevons-api loads: the model file, or the directory.
    pub fn model_path(&self, folder: &Path) -> PathBuf {
        match &self.model_file {
            Some(file) => self.dir(folder).join(file),
            None => self.dir(folder),
        }
    }

    pub fn mmproj_path(&self, folder: &Path) -> Option<PathBuf> {
        self.mmproj_file.as_ref().map(|f| self.dir(folder).join(f))
    }

    /// Whether a download finished: it writes a marker after verifying every file.
    pub fn is_ready(&self, folder: &Path) -> bool {
        self.dir(folder)
            .join(crate::download::COMPLETE_MARKER)
            .is_file()
    }

    pub fn serves(&self, service: Service) -> bool {
        self.services.contains(&service)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_entries_extend_and_override_the_builtin_catalog() {
        let file = std::env::temp_dir().join(format!("jevons-models-{}.toml", std::process::id()));
        std::fs::write(
            &file,
            r#"[[models]]
id = "parakeet-tdt-0.6b-v3"
name = "Pinned Parakeet"
services = ["speech"]
repo = "nvidia/parakeet-tdt-0.6b-v3"
revision = "abc123"
files = ["*"]

[[models]]
id = "gemma"
name = "DiffusionGemma"
services = ["generative", "decision"]
repo = "someone/diffusiongemma-gguf"
files = ["*Q4_K_M.gguf", "mmproj-*.gguf"]
model_file = "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
"#,
        )
        .unwrap();
        let (entries, error) = load(&file);
        std::fs::remove_file(&file).unwrap();
        assert_eq!(error, None);
        assert_eq!(entries.len(), builtin().len() + 1);
        let parakeet = entries
            .iter()
            .find(|e| e.id == "parakeet-tdt-0.6b-v3")
            .unwrap();
        assert_eq!(parakeet.revision, "abc123");
        let gemma = entries.iter().find(|e| e.id == "gemma").unwrap();
        assert_eq!(
            gemma.model_path(Path::new("/m")),
            Path::new("/m/gemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
        );
    }
}
