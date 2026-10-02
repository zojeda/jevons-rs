# jevons-tokenizer

## Purpose

jevons-tokenizer turns text into token IDs and back for the supported model families. `gemma4`
reads DiffusionGemma's vocabulary from GGUF metadata and tokenizes as the pinned llama.cpp does,
token for token. `hf` loads a Hugging Face `tokenizer.json` (byte-level BPE such as Tekken, and
the Parakeet vocabulary) and implements `TextTokenizer`, keeping user text from encoding to a
chat marker.

## Scope

jevons-tokenizer owns `gemma4::Tokenizer` with `TokenizerError`, and `hf::HfTokenizer`.

It leaves to other crates:

- Reading GGUF files: `jevons-formats`.
- The `TextTokenizer` contract: `jevons-core`. The Gemma 4 tokenizer implements it through an
  adapter in `jevons-models`.
- Chat markers and framing: the model crates, through `ChatFormat`.

The Hugging Face tokenizer runs on the `tokenizers` crate without its C and C++ features, so the
crate needs no C/C++ build.

## Requirements

### R1 A Gemma 4 vocabulary loads from GGUF

`Tokenizer::from_gguf` reads `tokenizer.ggml.model`, `tokenizer.ggml.tokens`,
`tokenizer.ggml.merges`, the optional `tokenizer.ggml.token_type`, the special-token IDs and the
optional `tokenizer.ggml.add_eos_token`. An empty token text becomes `[EMPTY_<id>]`, and a special
ID outside the vocabulary is ignored. `n_vocab`, `bos`, `eos` and `mask` report the vocabulary
size and those IDs, and `mask` is -1 when the vocabulary has no mask token.

Tests: `tokenizer_matches_llama_reference`

### R2 Invalid Gemma 4 vocabularies are rejected

`from_gguf` fails with `UnsupportedModel` when the model is not `gemma4`, with the GGUF error when
the model, tokens or merges key is missing, and with `Invalid` when a value has the wrong kind, the
vocabulary is empty or larger than `i32::MAX`, `token_type` is shorter than the vocabulary, two
tokens share a text, there is no BOS token, `add_eos_token` is set without an EOS token, or more
than one token matches a special token that llama.cpp finds by its text.

Tests: none yet

### R3 Gemma 4 output matches llama.cpp

`tokenize(text, add_special, parse_special)` returns the IDs of
`llama_tokenize(vocab, text, add_special, parse_special)`. `token_to_piece` returns the bytes of
`llama_token_to_piece` with `lstrip = 0` and `special = false`, and `is_control` agrees with
llama.cpp for every token. The check covers all 262,144 token pieces and 269,290 strings of a
llama.cpp dump.

Tests: `tokenizer_matches_llama_reference`

### R4 Gemma 4 adds BOS and EOS by its own rule

With `add_special`, `tokenize` prepends BOS whatever `tokenizer.ggml.add_bos_token` says, and
appends EOS when `add_eos_token` is true. Empty text with `add_special` gives `[BOS]`, and
without it gives no tokens.

Tests: `gemma4_always_adds_bos_but_not_eos`

### R5 Special tokens split the text, longest first

`tokenize` splits text around special tokens, trying longer texts first. Control and unknown tokens
match when `parse_special` is set. User-defined tokens match in every call. Without
`parse_special`, text that spells a control token encodes to tokens that are not control tokens.

Tests: `control_tokens_are_parsed_only_when_parse_special_is_set`, `user_defined_tokens_are_always_partitioned_and_spaces_stay_raw`

### R6 Token attributes follow llama.cpp's fixes

A token whose text is on llama.cpp's end-of-generation list, such as `<eos>`, `<turn|>` or
`<|tool_response>`, becomes a control token. A control token whose text contains `unused` is
marked unused. `<|channel|>`, `<|message|>`, `<|start|>` and `<|constrain|>` become user-defined.
When `<|tool_response>` ends generation, `</s>` becomes a normal token.

Tests: `user_defined_tokens_are_always_partitioned_and_spaces_stay_raw`, `pieces_hide_control_tokens_and_decode_bytes`

### R7 Merges apply by rank, then from the left

Spaces in the raw text between special tokens become `▁` before merging. Merges apply lowest rank
first, and among equal ranks the leftmost pair first. A merge line that repeats keeps its first
rank.

Tests: `lower_rank_merges_win_over_leftmost_pairs`, `spaces_are_escaped_and_merged_as_markers`, `user_defined_tokens_are_always_partitioned_and_spaces_stay_raw`

### R8 Newline runs stay whole when they are tokens

Raw text splits into runs of newlines and runs of other characters. A run of newlines that is a
token in the vocabulary becomes that one token. Other runs merge character by character.

Tests: `newline_runs_use_whole_tokens_or_merge_per_character`

### R9 Unknown pieces fall back to bytes

A merged piece with no token becomes one `<0xXX>` token per UTF-8 byte. A byte with no byte token
in the vocabulary is dropped.

Tests: `unknown_characters_fall_back_to_byte_tokens`

### R10 Pieces hide control tokens

`token_to_piece` returns no bytes for control, unknown and undefined tokens and for IDs outside the
vocabulary. It returns a user-defined token's text as stored, a normal token's text with `▁` as a
space, and a byte token's byte. `is_control` is false for IDs outside the vocabulary.

Tests: `pieces_hide_control_tokens_and_decode_bytes`, `spaces_are_escaped_and_merged_as_markers`, `unknown_characters_fall_back_to_byte_tokens`

### R11 A Hugging Face tokenizer loads from tokenizer.json

`HfTokenizer::from_file(path, bos)` fails with `ModelLoad` when the file cannot be read, and with
`UnsupportedModel` naming the path when it is not a valid tokenizer. `n_vocab` counts added tokens.

Tests: `markers_parse_only_in_framing_and_user_text_never_yields_them`, `nemotron_chat_markers_are_single_tokens_and_literal_text_stays_plain`

### R12 Chat markers parse in framing

With `special = true`, `tokenize` encodes added tokens, special or not (Nemotron's `</think>`), as
their IDs. `single_token(text)` returns the ID when `text` encodes to one token with markers
parsed, and `None` otherwise.

Tests: `markers_parse_only_in_framing_and_user_text_never_yields_them`, `nemotron_chat_markers_are_single_tokens_and_literal_text_stays_plain`

### R13 User text yields no added token

With `special = false`, `tokenize` encodes with the vocabulary and none of the file's added tokens,
so text that spells a marker stays plain text. A result that still holds an added token ID fails
with `InvalidInput` ("Text encodes to a control token"). `is_added` is true for the IDs in the
file's `added_tokens`.

Tests: `markers_parse_only_in_framing_and_user_text_never_yields_them`, `nemotron_chat_markers_are_single_tokens_and_literal_text_stays_plain`

### R14 BOS comes from the caller

With `bos = true`, `tokenize` prepends the BOS ID given to `from_file`. Without one it prepends
nothing.

Tests: `markers_parse_only_in_framing_and_user_text_never_yields_them`

### R15 Code pieces are short alphanumeric words

`code_piece` decodes a vocabulary token, drops one leading space, and returns the text when it is
1 to 16 ASCII letters and digits. IDs outside the vocabulary and any other text give `None`, and so
do Hugging Face added tokens and Gemma 4 control tokens. Both tokenizers follow the same rule, so a
Gemma 4 `▁word` token gives `word`.

Tests: `code_pieces_exclude_added_tokens_and_non_alphanumeric_text`, `nemotron_chat_markers_are_single_tokens_and_literal_text_stays_plain`, `code_pieces_ignore_one_leading_space_and_skip_control_tokens`

### R16 Decoding checks token IDs

`decode` returns the text of token IDs and leaves out added tokens flagged special. A negative ID
or one outside the vocabulary fails with `InvalidInput`, and a decoder failure with `Backend`.
`piece(id)` returns the vocabulary entry as stored, such as `▁word`.

Tests: none yet
