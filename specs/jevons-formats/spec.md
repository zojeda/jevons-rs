# jevons-formats

## Purpose

jevons-formats reads model files on the host. It parses GGUF files (metadata, the tensor directory
and tensor bytes), decodes and repacks GGML block quantization, and reads Hugging Face safetensors
checkpoints in one file or many shards. It validates every header before a caller touches tensor
data, reads tensors on demand, and needs no GPU.

## Scope

jevons-formats owns `gguf` (`Gguf`, `Value`, `TensorType`, `TensorInfo`, `GgufError`), `quant`
(`dequantize`, `pack`, `Packed`, `packed_words`, `packed_supported`, `q4k_scale_min`) and
`safetensors` (`Checkpoint`, `Dtype`, `TensorInfo`, `SafetensorsError`).

It leaves to other crates:

- Choosing an architecture from a file: `jevons-models`.
- Tokenizer metadata: `jevons-tokenizer`, which reads it through `Gguf`.
- Tensor names, shapes and model configuration: the model crates.
- Uploading weights to the GPU: `jevons-kernels` and `jevons-burn`.

## Requirements

### R1 GGUF files expose metadata and tensors

`Gguf::open` reads a file that starts with the magic `GGUF` and has version 2 or 3. It exposes the
metadata as key-value pairs and the tensor directory by name, each tensor with its dimensions
(`dims[0]` is the row length), its type, its absolute byte offset and its byte length.

Tests: `well_formed_files_expose_metadata_and_tensors`

### R2 Malformed GGUF headers are rejected

`Gguf::open` fails with `GgufError::Format` on a bad magic, a version other than 2 or 3, more than
2^20 tensors or metadata keys, a string longer than 2^24 bytes or not UTF-8, an array longer than
2^26 values, a nested array, a boolean byte other than 0 or 1, an unknown value type, a duplicate
key, a duplicate tensor name, or a tensor with no dimensions or more than four. A file that ends
inside the header fails with `GgufError::Io`.

Tests: `truncated_or_corrupt_files_are_rejected`

### R3 GGUF tensors must lie inside the file

Tensor data starts at the end of the header rounded up to `general.alignment`, which must be a
power of two and defaults to 32. `Gguf::open` fails with `GgufError::Format` when a tensor's offset
is not a multiple of the alignment, when its bytes run past the end of the file, when its element
count overflows, or when the row length of a known type is not a multiple of its block size.

Tests: `truncated_or_corrupt_files_are_rejected`

### R4 Metadata values convert by kind

A metadata value is one of 13 kinds: the signed and unsigned integers of 8, 16, 32 and 64 bits,
`f32`, `f64`, a boolean, a string or an array. `as_u64` accepts any integer that is not negative,
`as_i64` any integer that fits, and `as_f64` either float. `Gguf::get`, `u64` and `f32` fail with
`MissingKey` for an absent key and with `Format` for a value that does not convert.

Tests: `well_formed_files_expose_metadata_and_tensors`

### R5 Six GGML types have known blocks

`TensorType::block` gives elements and bytes per block: F32 (1, 4), F16 (1, 2), Q4_K (256, 144),
Q5_0 (32, 22), Q6_K (256, 210) and Q8_0 (32, 34). These are GGML type IDs 0, 1, 12, 6, 14 and 8.
Any other type ID opens as `TensorType::Other` with no block and a byte length of zero.

Tests: none yet

### R6 GGUF tensor reads are bounded

`Gguf::read` returns a tensor's bytes and fails with `Format` for a type with no block.
`read_range(info, start, len)` returns bytes inside one tensor and fails with `Format` when the
range passes the tensor's end. `read_f32(name)` decodes an F32 tensor, fails with `MissingTensor`
for an unknown name and with `Format` for another type.

Tests: `well_formed_files_expose_metadata_and_tensors`

### R7 Positioned reads report a short file

Tensor reads fill the whole buffer from the requested file offset without moving a shared cursor,
on Unix and Windows. A read past the end of the file fails with an I/O error of kind
`UnexpectedEof`.

Tests: `positioned_reads_fill_the_buffer_and_report_a_short_file`

### R8 Dequantization matches the GGML reference

`dequantize(kind, bytes, out)` decodes whole blocks of F32, F16, Q4_K, Q6_K, Q5_0 and Q8_0 to
`f32` with the values of the pinned `ggml-quants.c`. It fails for any other type, when `bytes` is
not a whole number of blocks, and when `out` does not hold one value per decoded element.
`q4k_scale_min(j, scales)` returns the 6-bit scale and minimum of Q4_K sub-block `j`.

Tests: `q4k_decodes_scale_high_bits_for_upper_groups`, `q8_0_and_q5_0_dequantize_signed_values`

### R9 Packing keeps values on word boundaries

`pack(kind, bytes)` splits Q4_K, Q6_K, Q5_0 and Q8_0 blocks into four regions: quants `q`, high
bits `h` and scales `s` as 32-bit words, and `d` as `f32` scales, so every block starts on a
word. `packed_words` gives the words per block in each region: Q4_K (36, 0, 0, 0), Q6_K
(32, 16, 4, 1), Q5_0 (4, 1, 0, 1) and Q8_0 (8, 0, 0, 1). Packed blocks decode to the same values
as the source blocks. `pack` fails for other types and for input that is not a whole number of
blocks, and `packed_supported` is true for the four packed types. `Packed::extend` appends another
tensor's regions.

Tests: `packing_splits_quants_and_scales_without_changing_values`

### R10 A checkpoint directory prefers its index

`Checkpoint::open_dir(dir)` opens `model.safetensors.index.json` when the directory has one, and
`model.safetensors` otherwise.

Tests: `sharded_checkpoints_read_tensors_from_the_indexed_shard`

### R11 Safetensors headers must match their data

`Checkpoint::open_files` fails with `SafetensorsError::Format`, naming the file, when the header
length exceeds 100 MiB or the file, when the header is not valid JSON, when a tensor's dtype is
not F32, F16, BF16, I64, I32, U8 or BOOL, or when its data offsets fall outside the data or do not
equal its shape times its dtype size. I/O errors name the file too. The `__metadata__` entry is
skipped. A tensor name must be unique across all files.

Tests: `offsets_that_disagree_with_shape_or_file_size_are_rejected`

### R12 Sharded tensors live in their indexed shard

`Checkpoint::open_index` reads the index's `weight_map` and opens each named shard once. A shard
name with `/` or `\`, or one that starts with `.`, fails with `Format`. A mapped tensor that no
shard holds fails with `Missing`, and one held by a shard other than the mapped one fails with
`Format`.

Tests: `sharded_checkpoints_read_tensors_from_the_indexed_shard`

### R13 Safetensors reads return raw or widened data

`names` lists tensor names in sorted order. `tensor` and `read` fail with `Missing` for an unknown
name. `read` returns a tensor's raw little-endian bytes, and `read_into` fails with `Format` unless
the buffer has the tensor's byte length. `read_f32` widens F32, F16 and BF16 tensors to `f32` and
fails with `Format` for other dtypes.

Tests: `sharded_checkpoints_read_tensors_from_the_indexed_shard`
