//! Minimal validated GGUF v2/v3 reader: metadata, tensor directory, and bounded tensor reads.
//!
//! Only the value and tensor types used by DiffusionGemma checkpoints are interpreted; every
//! other tensor type is rejected when its data is requested rather than guessed.
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    os::unix::fs::FileExt,
    path::Path,
};

const MAGIC: &[u8; 4] = b"GGUF";
const MAX_STRING: u64 = 1 << 24;
const MAX_ARRAY: u64 = 1 << 26;
const MAX_DIMS: u32 = 4;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("I/O error reading the model: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GGUF file: {0}")]
    Format(String),
    #[error("Missing GGUF metadata key {0}")]
    MissingKey(String),
    #[error("Missing tensor {0}")]
    MissingTensor(String),
}

pub type Result<T> = std::result::Result<T, GgufError>;

fn format_err<T>(message: impl Into<String>) -> Result<T> {
    Err(GgufError::Format(message.into()))
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Value::I8(v) => Some(v.into()),
            Value::I16(v) => Some(v.into()),
            Value::I32(v) => Some(v.into()),
            Value::I64(v) => Some(v),
            _ => self.as_u64().and_then(|v| i64::try_from(v).ok()),
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v.into()),
            Value::F64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// GGML storage types this runtime can decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorType {
    F32,
    F16,
    Q4K,
    Q5_0,
    Q6K,
    Q8_0,
    /// Any other GGML type id; rejected before decoding.
    Other(u32),
}

impl TensorType {
    fn from_id(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            6 => Self::Q5_0,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            14 => Self::Q6K,
            other => Self::Other(other),
        }
    }

    /// (elements per block, bytes per block).
    pub fn block(self) -> Option<(u64, u64)> {
        match self {
            Self::F32 => Some((1, 4)),
            Self::F16 => Some((1, 2)),
            Self::Q4K => Some((256, 144)),
            Self::Q5_0 => Some((32, 22)),
            Self::Q6K => Some((256, 210)),
            Self::Q8_0 => Some((32, 34)),
            Self::Other(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: `dims[0]` is the contiguous row length.
    pub dims: Vec<u64>,
    pub kind: TensorType,
    /// Absolute byte offset in the file.
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    file: File,
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
}

struct Reader<R> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut buf = [0; N];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING {
            return format_err("string is too long");
        }
        let mut buf = vec![0; len as usize];
        self.inner.read_exact(&mut buf)?;
        String::from_utf8(buf).or_else(|_| format_err("string is not UTF-8"))
    }
    fn value(&mut self, kind: u32, depth: u32) -> Result<Value> {
        Ok(match kind {
            0 => Value::U8(self.bytes::<1>()?[0]),
            1 => Value::I8(self.bytes::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.bytes()?)),
            3 => Value::I16(i16::from_le_bytes(self.bytes()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.bytes()?)),
            6 => Value::F32(f32::from_le_bytes(self.bytes()?)),
            7 => match self.bytes::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return format_err("invalid boolean"),
            },
            8 => Value::String(self.string()?),
            9 => {
                if depth > 0 {
                    return format_err("nested arrays are not supported");
                }
                let inner = self.u32()?;
                let len = self.u64()?;
                if len > MAX_ARRAY {
                    return format_err("array is too long");
                }
                let mut values = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    values.push(self.value(inner, depth + 1)?);
                }
                Value::Array(values)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            other => return format_err(format!("unknown metadata type {other}")),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = Reader {
            inner: BufReader::with_capacity(1 << 20, file.try_clone()?),
        };
        if &reader.bytes::<4>()? != MAGIC {
            return format_err("bad magic");
        }
        let version = reader.u32()?;
        if !(2..=3).contains(&version) {
            return format_err(format!("unsupported version {version}"));
        }
        let tensor_count = reader.u64()?;
        let kv_count = reader.u64()?;
        if tensor_count > 1 << 20 || kv_count > 1 << 20 {
            return format_err("implausible header counts");
        }
        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = reader.string()?;
            let kind = reader.u32()?;
            let value = reader.value(kind, 0)?;
            if metadata.insert(key, value).is_some() {
                return format_err("duplicate metadata key");
            }
        }
        let alignment = match metadata.get("general.alignment") {
            Some(v) => v
                .as_u64()
                .filter(|a| a.is_power_of_two())
                .ok_or_else(|| GgufError::Format("invalid alignment".into()))?,
            None => 32,
        };
        let mut infos = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = reader.string()?;
            let n_dims = reader.u32()?;
            if n_dims == 0 || n_dims > MAX_DIMS {
                return format_err(format!("tensor {name} has {n_dims} dimensions"));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(reader.u64()?);
            }
            let kind = TensorType::from_id(reader.u32()?);
            let offset = reader.u64()?;
            infos.push((name, dims, kind, offset));
        }
        let header_end = reader.inner.stream_position()?;
        let data_start = header_end.next_multiple_of(alignment);
        // Rewind the buffered reader's file handle; later reads use positional I/O.
        reader.inner.seek(SeekFrom::Start(0))?;
        let mut tensors = HashMap::new();
        for (name, dims, kind, offset) in infos {
            let elements = dims
                .iter()
                .try_fold(1u64, |a, &d| a.checked_mul(d))
                .ok_or_else(|| GgufError::Format(format!("tensor {name} is too large")))?;
            let bytes = match kind.block() {
                Some((block, size)) => {
                    if dims[0] % block != 0 {
                        return format_err(format!("tensor {name} rows are not block aligned"));
                    }
                    elements / block * size
                }
                None => 0,
            };
            let start = data_start
                .checked_add(offset)
                .filter(|s| {
                    offset % alignment == 0 && s.checked_add(bytes).is_some_and(|e| e <= file_len)
                })
                .ok_or_else(|| GgufError::Format(format!("tensor {name} is out of bounds")))?;
            let info = TensorInfo {
                name: name.clone(),
                dims,
                kind,
                offset: start,
                bytes,
            };
            if tensors.insert(name, info).is_some() {
                return format_err("duplicate tensor name");
            }
        }
        Ok(Self {
            file,
            metadata,
            tensors,
        })
    }

    pub fn get(&self, key: &str) -> Result<&Value> {
        self.metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingKey(key.into()))
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        self.get(key)?
            .as_u64()
            .ok_or_else(|| GgufError::Format(format!("{key} is not an unsigned integer")))
    }

    pub fn f32(&self, key: &str) -> Result<f32> {
        self.get(key)?
            .as_f64()
            .map(|v| v as f32)
            .ok_or_else(|| GgufError::Format(format!("{key} is not a float")))
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingTensor(name.into()))
    }

    /// Reads raw tensor bytes; rejects types this runtime cannot decode.
    pub fn read(&self, info: &TensorInfo) -> Result<Vec<u8>> {
        if info.kind.block().is_none() {
            return format_err(format!(
                "tensor {} has unsupported type {:?}",
                info.name, info.kind
            ));
        }
        let mut buf = vec![0; info.bytes as usize];
        self.file.read_exact_at(&mut buf, info.offset)?;
        Ok(buf)
    }

    /// Reads a contiguous byte range inside one tensor (e.g. a single expert).
    pub fn read_range(&self, info: &TensorInfo, start: u64, len: u64) -> Result<Vec<u8>> {
        if start.checked_add(len).is_none_or(|end| end > info.bytes) {
            return format_err(format!("range is outside tensor {}", info.name));
        }
        let mut buf = vec![0; len as usize];
        self.file.read_exact_at(&mut buf, info.offset + start)?;
        Ok(buf)
    }

    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.tensor(name)?;
        if info.kind != TensorType::F32 {
            return format_err(format!("tensor {name} is not F32"));
        }
        Ok(self
            .read(info)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    fn sample() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(MAGIC);
        out.extend(3u32.to_le_bytes());
        out.extend(1u64.to_le_bytes());
        out.extend(2u64.to_le_bytes());
        string(&mut out, "a.count");
        out.extend(4u32.to_le_bytes());
        out.extend(7u32.to_le_bytes());
        string(&mut out, "a.names");
        out.extend(9u32.to_le_bytes());
        out.extend(8u32.to_le_bytes());
        out.extend(2u64.to_le_bytes());
        string(&mut out, "x");
        string(&mut out, "yz");
        string(&mut out, "w");
        out.extend(2u32.to_le_bytes());
        out.extend(2u64.to_le_bytes());
        out.extend(1u64.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend(1.5f32.to_le_bytes());
        out.extend((-2.0f32).to_le_bytes());
        out
    }

    fn write(bytes: &[u8], name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("gguf-{}-{name}", std::process::id()));
        File::create(&path).unwrap().write_all(bytes).unwrap();
        path
    }

    #[test]
    fn well_formed_files_expose_metadata_and_tensors() {
        let path = write(&sample(), "ok");
        let gguf = Gguf::open(&path).unwrap();
        assert_eq!(gguf.u64("a.count").unwrap(), 7);
        let names = gguf.get("a.names").unwrap().as_array().unwrap();
        assert_eq!(names[1].as_str(), Some("yz"));
        assert_eq!(gguf.read_f32("w").unwrap(), vec![1.5, -2.0]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn truncated_or_corrupt_files_are_rejected() {
        let good = sample();
        for (name, bytes) in [
            ("magic", [b"GGUX".as_slice(), &good[4..]].concat()),
            ("short", good[..good.len() - 4].to_vec()),
            ("header", good[..40].to_vec()),
        ] {
            let path = write(&bytes, name);
            assert!(Gguf::open(&path).is_err(), "{name}");
            std::fs::remove_file(path).unwrap();
        }
    }
}
