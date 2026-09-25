//! Validated safetensors reader for single-file and sharded Hugging Face checkpoints.
//!
//! Headers are parsed up front; tensor bytes are read on demand with positional reads, so a
//! loader can stream one tensor at a time to the device without mapping whole shards.
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

/// Largest accepted JSON header, as in the reference implementation.
const MAX_HEADER: u64 = 100 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum SafetensorsError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid safetensors data in {path}: {message}")]
    Format { path: PathBuf, message: String },
    #[error("tensor {0} is not in the checkpoint")]
    Missing(String),
}

pub type Result<T> = std::result::Result<T, SafetensorsError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I64,
    I32,
    U8,
    Bool,
}

impl Dtype {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I64" => Self::I64,
            "I32" => Self::I32,
            "U8" => Self::U8,
            "BOOL" => Self::Bool,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
            Self::U8 | Self::Bool => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte offset of the tensor data within its file.
    offset: u64,
    len: u64,
    shard: usize,
}

impl TensorInfo {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_len(&self) -> usize {
        self.len as usize
    }
}

#[derive(Deserialize)]
struct RawInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

struct Shard {
    path: PathBuf,
    file: File,
}

/// One or more safetensors files presented as a single tensor namespace.
pub struct Checkpoint {
    shards: Vec<Shard>,
    tensors: BTreeMap<String, TensorInfo>,
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> SafetensorsError + '_ {
    move |source| SafetensorsError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn format(path: &Path, message: impl Into<String>) -> SafetensorsError {
    SafetensorsError::Format {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

impl Checkpoint {
    /// Opens `model.safetensors.index.json` in `dir` if present, otherwise `model.safetensors`.
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        if index.exists() {
            return Self::open_index(&index);
        }
        Self::open_files(&[dir.join("model.safetensors")])
    }

    /// Opens every shard named by a sharded-checkpoint index; each tensor must be in the shard
    /// the index names.
    pub fn open_index(index: &Path) -> Result<Self> {
        #[derive(Deserialize)]
        struct Index {
            weight_map: HashMap<String, String>,
        }
        let text = std::fs::read_to_string(index).map_err(io(index))?;
        let parsed: Index =
            serde_json::from_str(&text).map_err(|e| format(index, format!("index: {e}")))?;
        let dir = index.parent().unwrap_or(Path::new("."));
        let mut files: Vec<&String> = parsed.weight_map.values().collect();
        files.sort();
        files.dedup();
        if files
            .iter()
            .any(|f| f.contains('/') || f.contains('\\') || f.starts_with('.'))
        {
            return Err(format(index, "shard names must be plain file names"));
        }
        let paths: Vec<PathBuf> = files.iter().map(|f| dir.join(f)).collect();
        let checkpoint = Self::open_files(&paths)?;
        for (name, file) in &parsed.weight_map {
            let info = checkpoint
                .tensors
                .get(name)
                .ok_or_else(|| SafetensorsError::Missing(name.clone()))?;
            if checkpoint.shards[info.shard].path != dir.join(file) {
                return Err(format(index, format!("{name} is not in {file}")));
            }
        }
        Ok(checkpoint)
    }

    /// Opens safetensors files as one namespace; tensor names must be unique across files.
    pub fn open_files(paths: &[PathBuf]) -> Result<Self> {
        let mut shards = Vec::with_capacity(paths.len());
        let mut tensors = BTreeMap::new();
        for (shard, path) in paths.iter().enumerate() {
            let mut file = File::open(path).map_err(io(path))?;
            let file_len = file.metadata().map_err(io(path))?.len();
            let mut len = [0u8; 8];
            file.read_exact(&mut len).map_err(io(path))?;
            let header_len = u64::from_le_bytes(len);
            if header_len > MAX_HEADER || 8 + header_len > file_len {
                return Err(format(path, "header length out of range"));
            }
            let mut header = vec![0u8; header_len as usize];
            file.read_exact(&mut header).map_err(io(path))?;
            let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)
                .map_err(|e| format(path, format!("header: {e}")))?;
            let data_start = 8 + header_len;
            let data_len = file_len - data_start;
            for (name, value) in raw {
                if name == "__metadata__" {
                    continue;
                }
                let info: RawInfo = serde_json::from_value(value)
                    .map_err(|e| format(path, format!("{name}: {e}")))?;
                let dtype = Dtype::parse(&info.dtype)
                    .ok_or_else(|| format(path, format!("{name}: dtype {}", info.dtype)))?;
                let [begin, end] = info.data_offsets;
                let elements = info
                    .shape
                    .iter()
                    .try_fold(1u64, |n, &d| n.checked_mul(d as u64));
                let expected = elements.and_then(|n| n.checked_mul(dtype.size() as u64));
                if begin > end || end > data_len || expected != Some(end - begin) {
                    return Err(format(
                        path,
                        format!("{name}: data offsets do not match shape"),
                    ));
                }
                let info = TensorInfo {
                    dtype,
                    shape: info.shape,
                    offset: data_start + begin,
                    len: end - begin,
                    shard,
                };
                if tensors.insert(name.clone(), info).is_some() {
                    return Err(format(path, format!("duplicate tensor {name}")));
                }
            }
            shards.push(Shard {
                path: path.clone(),
                file,
            });
        }
        Ok(Self { shards, tensors })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::Missing(name.into()))
    }

    /// Reads a tensor's raw little-endian bytes.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let info = self.tensor(name)?;
        let mut bytes = vec![0u8; info.byte_len()];
        self.read_into(info, &mut bytes)?;
        Ok(bytes)
    }

    /// Reads a tensor's raw bytes into `out`, which must be exactly its size.
    pub fn read_into(&self, info: &TensorInfo, out: &mut [u8]) -> Result<()> {
        let shard = &self.shards[info.shard];
        if out.len() != info.byte_len() {
            return Err(format(&shard.path, "output buffer size mismatch"));
        }
        crate::io::read_exact_at(&shard.file, out, info.offset).map_err(io(&shard.path))
    }

    /// Reads a floating-point tensor widened to f32.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.tensor(name)?;
        let bytes = self.read(name)?;
        let path = &self.shards[info.shard].path;
        Ok(match info.dtype {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect(),
            Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes(*c).to_f32())
                .collect(),
            other => return Err(format(path, format!("{name}: {other:?} is not a float"))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tensors as a safetensors file.
    fn write(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let mut data: Vec<u8> = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let begin = data.len();
            data.extend(bytes);
            header.insert(
                (*name).into(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [begin, data.len()]}),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(data);
        std::fs::write(path, file).unwrap();
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("jevons-st-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn sharded_checkpoints_read_tensors_from_the_indexed_shard() {
        let dir = temp_dir("sharded");
        let bf16: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        write(
            &dir.join("a.safetensors"),
            &[("x", "BF16", vec![2], bf16.clone())],
        );
        write(
            &dir.join("b.safetensors"),
            &[("y", "F32", vec![1, 1], 3.25f32.to_le_bytes().to_vec())],
        );
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"metadata":{},"weight_map":{"x":"a.safetensors","y":"b.safetensors"}}"#,
        )
        .unwrap();
        let checkpoint = Checkpoint::open_dir(&dir).unwrap();
        assert_eq!(checkpoint.names().collect::<Vec<_>>(), ["x", "y"]);
        assert_eq!(checkpoint.tensor("x").unwrap().dtype, Dtype::BF16);
        assert_eq!(checkpoint.read("x").unwrap(), bf16);
        assert_eq!(checkpoint.read_f32("x").unwrap(), [1.5, -2.0]);
        assert_eq!(checkpoint.read_f32("y").unwrap(), [3.25]);
        assert!(matches!(
            checkpoint.read("z"),
            Err(SafetensorsError::Missing(_))
        ));
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"b.safetensors","y":"b.safetensors"}}"#,
        )
        .unwrap();
        assert!(Checkpoint::open_dir(&dir).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn offsets_that_disagree_with_shape_or_file_size_are_rejected() {
        let dir = temp_dir("invalid");
        let path = dir.join("model.safetensors");
        write(&path, &[("x", "F32", vec![3], vec![0; 8])]);
        assert!(matches!(
            Checkpoint::open_dir(&dir),
            Err(SafetensorsError::Format { .. })
        ));
        write(&path, &[("x", "Q4", vec![1], vec![0; 1])]);
        assert!(Checkpoint::open_dir(&dir).is_err());
        std::fs::write(&path, u64::MAX.to_le_bytes()).unwrap();
        assert!(Checkpoint::open_dir(&dir).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
