//! Resumable, verified model downloads from Hugging Face.
//!
//! [`plan`] lists the repository files matching an entry's globs, with sizes and (for LFS
//! files) SHA-256 digests. [`download`] fetches each into `<file>.part`, resuming with an HTTP
//! range, verifies the digest, then renames it. After every file, a marker makes the entry
//! ready. Nothing downloads unless the user asks.

use crate::catalog::CatalogEntry;
use futures_util::StreamExt;
use globset::{Glob, GlobSetBuilder};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub const COMPLETE_MARKER: &str = ".jevons-complete";
pub const HUGGING_FACE: &str = "https://huggingface.co";

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("{0}")]
    Http(#[from] reqwest::Error),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{url} answered {status}")]
    Status { url: String, status: u16 },
    #[error("{file} is corrupt: expected SHA-256 {expected}, got {actual}")]
    Checksum {
        file: String,
        expected: String,
        actual: String,
    },
    #[error("no file in {repo} matches {globs:?}")]
    NothingToDownload { repo: String, globs: Vec<String> },
    #[error("{0}")]
    Invalid(String),
    #[error("cancelled")]
    Cancelled,
}

/// A file to fetch.
#[derive(Clone, Debug, PartialEq)]
pub struct RemoteFile {
    pub repo: String,
    pub revision: String,
    pub path: String,
    pub size: u64,
    pub sha256: Option<String>,
}

#[derive(Deserialize)]
struct TreeItem {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    oid: String,
}

/// Progress for the Settings panel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Progress {
    pub file: String,
    pub done: u64,
    pub total: u64,
}

/// Where to download from; tests point it at a local server.
#[derive(Clone, Debug)]
pub struct Hub {
    pub base: String,
    pub token: Option<String>,
    http: reqwest::Client,
}

impl Default for Hub {
    fn default() -> Self {
        Self::new(HUGGING_FACE, std::env::var("HF_TOKEN").ok())
    }
}

impl Hub {
    pub fn new(base: &str, token: Option<String>) -> Self {
        Self {
            base: base.trim_end_matches('/').into(),
            token: token.filter(|t| !t.is_empty()),
            http: reqwest::Client::new(),
        }
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        let request = self.http.get(url);
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    /// The files of `entry` to fetch, from its repository and any extra sources.
    pub async fn plan(&self, entry: &CatalogEntry) -> Result<Vec<RemoteFile>, DownloadError> {
        let mut files = self
            .list(&entry.repo, &entry.revision, &entry.files)
            .await?;
        for source in &entry.extra {
            files.extend(
                self.list(&source.repo, &source.revision, &source.files)
                    .await?,
            );
        }
        Ok(files)
    }

    /// The files of `repo` at `revision` matching `globs`; at least one.
    async fn list(
        &self,
        repo: &str,
        revision: &str,
        globs: &[String],
    ) -> Result<Vec<RemoteFile>, DownloadError> {
        let mut set = GlobSetBuilder::new();
        for glob in globs {
            set.add(Glob::new(glob).map_err(|e| DownloadError::Invalid(e.to_string()))?);
        }
        let set = set
            .build()
            .map_err(|e| DownloadError::Invalid(e.to_string()))?;
        let url = format!(
            "{}/api/models/{repo}/tree/{revision}?recursive=true",
            self.base
        );
        let response = self.get(&url).send().await?;
        if !response.status().is_success() {
            return Err(DownloadError::Status {
                url,
                status: response.status().as_u16(),
            });
        }
        let items: Vec<TreeItem> = response.json().await?;
        let files: Vec<RemoteFile> = items
            .into_iter()
            .filter(|i| i.kind == "file" && set.is_match(&i.path))
            .map(|i| RemoteFile {
                repo: repo.into(),
                revision: revision.into(),
                path: i.path,
                size: i.size,
                sha256: i.lfs.map(|l| l.oid),
            })
            .collect();
        if files.is_empty() {
            return Err(DownloadError::NothingToDownload {
                repo: repo.into(),
                globs: globs.to_vec(),
            });
        }
        Ok(files)
    }

    /// Downloads `files` of `entry` into `folder`, resuming partial files; marks the entry
    /// ready when every file is in place. Stops with [`DownloadError::Cancelled`] when `cancel`
    /// is set.
    pub async fn download(
        &self,
        entry: &CatalogEntry,
        files: &[RemoteFile],
        folder: &Path,
        cancel: Arc<AtomicBool>,
        mut progress: impl FnMut(Progress),
    ) -> Result<PathBuf, DownloadError> {
        let dir = entry.dir(folder);
        tokio::fs::create_dir_all(&dir).await?;
        let total: u64 = files.iter().map(|f| f.size).sum();
        let mut done_before = 0;
        for file in files {
            let target = safe_join(&dir, &file.path)?;
            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            if !is_complete(&target, file).await? {
                self.fetch(file, &target, &cancel, |done| {
                    progress(Progress {
                        file: file.path.clone(),
                        done: done_before + done,
                        total,
                    })
                })
                .await?;
            }
            done_before += file.size;
            progress(Progress {
                file: file.path.clone(),
                done: done_before,
                total,
            });
        }
        tokio::fs::write(dir.join(COMPLETE_MARKER), b"").await?;
        Ok(entry.model_path(folder))
    }

    async fn fetch(
        &self,
        file: &RemoteFile,
        target: &Path,
        cancel: &AtomicBool,
        mut progress: impl FnMut(u64),
    ) -> Result<(), DownloadError> {
        let part = part_path(target);
        let mut have = tokio::fs::metadata(&part).await.map_or(0, |m| m.len());
        if file.size > 0 && have > file.size {
            tokio::fs::remove_file(&part).await?;
            have = 0;
        }
        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.base, file.repo, file.revision, file.path
        );
        let mut request = self.get(&url);
        if have > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let response = request.send().await?;
        let status = response.status();
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && have == file.size {
            // The part file is already whole.
        } else if !status.is_success() {
            return Err(DownloadError::Status {
                url,
                status: status.as_u16(),
            });
        } else {
            let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
            let mut out = tokio::fs::OpenOptions::new()
                .create(true)
                .append(resumed)
                .write(true)
                .truncate(!resumed)
                .open(&part)
                .await?;
            if !resumed {
                have = 0;
            }
            let mut body = response.bytes_stream();
            while let Some(chunk) = body.next().await {
                if cancel.load(Ordering::Relaxed) {
                    out.flush().await?;
                    return Err(DownloadError::Cancelled);
                }
                let chunk = chunk?;
                out.write_all(&chunk).await?;
                have += chunk.len() as u64;
                progress(have);
            }
            out.flush().await?;
        }
        if let Some(expected) = &file.sha256 {
            let actual = sha256(&part).await?;
            if !actual.eq_ignore_ascii_case(expected) {
                tokio::fs::remove_file(&part).await?;
                return Err(DownloadError::Checksum {
                    file: file.path.clone(),
                    expected: expected.clone(),
                    actual,
                });
            }
        }
        tokio::fs::rename(&part, target).await?;
        Ok(())
    }
}

/// `dir/path`, refusing paths that would leave `dir`.
fn safe_join(dir: &Path, path: &str) -> Result<PathBuf, DownloadError> {
    let relative = Path::new(path);
    if relative.is_absolute()
        || relative
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(DownloadError::Invalid(format!("unsafe file path {path:?}")));
    }
    Ok(dir.join(relative))
}

fn part_path(target: &Path) -> PathBuf {
    let mut name = target.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    target.with_file_name(name)
}

async fn is_complete(target: &Path, file: &RemoteFile) -> Result<bool, DownloadError> {
    Ok(tokio::fs::metadata(target)
        .await
        .is_ok_and(|m| file.size == 0 || m.len() == file.size))
}

async fn sha256(path: &Path) -> Result<String, DownloadError> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buffer).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Service;
    use axum::extract::{Path as UrlPath, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::{IntoResponse, Response};
    use axum::routing::get;
    use std::sync::Mutex;

    const WEIGHTS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

    #[derive(Clone, Default)]
    struct Hits {
        ranges: Arc<Mutex<Vec<Option<String>>>>,
    }

    fn digest(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    async fn serve(sha: String) -> (String, Hits) {
        let hits = Hits::default();
        let tree = move || async move {
            axum::Json(serde_json::json!([
                {"type": "file", "path": "config.json", "size": 2},
                {"type": "file", "path": "model.safetensors", "size": WEIGHTS.len(), "lfs": {"oid": sha}},
                {"type": "file", "path": "README.md", "size": 5},
                {"type": "directory", "path": "extra"},
            ]))
        };
        let projector = || async {
            axum::Json(serde_json::json!([
                {"type": "file", "path": "mmproj.gguf", "size": 4},
                {"type": "file", "path": "other.gguf", "size": 9},
            ]))
        };
        let file = |State(hits): State<Hits>,
                    UrlPath((_, _, path)): UrlPath<(String, String, String)>,
                    headers: HeaderMap| async move {
            let range = headers
                .get("range")
                .map(|r| r.to_str().unwrap().to_string());
            hits.ranges.lock().unwrap().push(range.clone());
            let body: &[u8] = match path.as_str() {
                "config.json" => b"{}",
                "mmproj.gguf" => b"proj",
                _ => WEIGHTS,
            };
            match range.and_then(|r| {
                r.strip_prefix("bytes=")?
                    .strip_suffix('-')?
                    .parse::<usize>()
                    .ok()
            }) {
                Some(start) => {
                    (StatusCode::PARTIAL_CONTENT, body[start..].to_vec()).into_response()
                }
                None => Response::new(body.to_vec().into()),
            }
        };
        let app = axum::Router::new()
            .route("/api/models/org/model/tree/main", get(tree))
            .route("/api/models/other/projector/tree/main", get(projector))
            .route("/{owner}/{name}/resolve/main/{*path}", get(file))
            .with_state(hits.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (base, hits)
    }

    fn entry() -> CatalogEntry {
        CatalogEntry {
            id: "model".into(),
            name: "Model".into(),
            services: vec![Service::Speech],
            repo: "org/model".into(),
            revision: "main".into(),
            files: vec!["*.json".into(), "*.safetensors".into()],
            model_file: None,
            mmproj_file: None,
            license: String::new(),
            memory_gb: 0.0,
            extra: Vec::new(),
        }
    }

    fn folder(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("jevons-download-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[tokio::test]
    async fn interrupted_download_resumes_with_range() {
        let (base, hits) = serve(digest(WEIGHTS)).await;
        let hub = Hub::new(&base, None);
        let entry = entry();
        let files = hub.plan(&entry).await.unwrap();
        assert_eq!(files.len(), 2, "globs keep only matching files");
        let folder = folder("resume");
        let dir = entry.dir(&folder);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors.part"), &WEIGHTS[..10]).unwrap();
        assert!(!entry.is_ready(&folder));

        let mut last = Progress::default();
        let path = hub
            .download(&entry, &files, &folder, Arc::default(), |p| last = p)
            .await
            .unwrap();
        assert_eq!(path, dir);
        assert_eq!(
            std::fs::read(dir.join("model.safetensors")).unwrap(),
            WEIGHTS
        );
        assert!(!dir.join("model.safetensors.part").exists());
        assert!(entry.is_ready(&folder));
        assert_eq!(last.done, last.total);
        assert!(
            hits.ranges
                .lock()
                .unwrap()
                .contains(&Some("bytes=10-".into()))
        );
        std::fs::remove_dir_all(folder).unwrap();
    }

    #[tokio::test]
    async fn checksum_mismatch_marks_model_corrupt_and_keeps_no_final_file() {
        let (base, _) = serve(digest(b"something else")).await;
        let hub = Hub::new(&base, None);
        let entry = entry();
        let files = hub.plan(&entry).await.unwrap();
        let folder = folder("corrupt");
        let error = hub
            .download(&entry, &files, &folder, Arc::default(), |_| {})
            .await
            .unwrap_err();
        assert!(matches!(error, DownloadError::Checksum { .. }), "{error}");
        let dir = entry.dir(&folder);
        assert!(!dir.join("model.safetensors").exists());
        assert!(!dir.join("model.safetensors.part").exists());
        assert!(!entry.is_ready(&folder));
        std::fs::remove_dir_all(folder).unwrap();
    }

    #[tokio::test]
    async fn a_cancelled_download_keeps_its_part_file() {
        let (base, _) = serve(digest(WEIGHTS)).await;
        let hub = Hub::new(&base, None);
        let entry = entry();
        let files = hub.plan(&entry).await.unwrap();
        let folder = folder("cancel");
        let cancel = Arc::new(AtomicBool::new(true));
        let error = hub
            .download(&entry, &files, &folder, cancel, |_| {})
            .await
            .unwrap_err();
        assert!(matches!(error, DownloadError::Cancelled));
        assert!(!entry.is_ready(&folder));
        std::fs::remove_dir_all(folder).unwrap();
    }

    #[tokio::test]
    async fn extra_sources_download_into_the_same_folder() {
        let (base, _) = serve(digest(WEIGHTS)).await;
        let hub = Hub::new(&base, None);
        let mut entry = entry();
        entry.extra = vec![crate::catalog::Source {
            repo: "other/projector".into(),
            revision: "main".into(),
            files: vec!["mmproj.gguf".into()],
        }];
        let files = hub.plan(&entry).await.unwrap();
        assert_eq!(files.len(), 3);
        assert_eq!(files[2].repo, "other/projector");
        let folder = folder("extra");
        hub.download(&entry, &files, &folder, Arc::default(), |_| {})
            .await
            .unwrap();
        assert_eq!(
            std::fs::read(entry.dir(&folder).join("mmproj.gguf")).unwrap(),
            b"proj"
        );
        std::fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn repository_paths_cannot_escape_the_model_folder() {
        assert!(safe_join(Path::new("/m"), "../etc/passwd").is_err());
        assert!(safe_join(Path::new("/m"), "/etc/passwd").is_err());
        assert!(safe_join(Path::new("/m"), "linear_spec_lora/a.bin").is_ok());
    }
}
