# Voice output: the models

[Proposal](proposal.md) · [Design](design.md) · [Tasks](tasks.md)

Read on 2026-10-11 from model cards, licence files, configs, inference code, papers and
leaderboards. Nobody downloaded weights or listened to anything. A number is **independent**
when a third party ran the model, and **vendor** when the model's authors or a competitor did.

## Filters

1. Spanish and English, both usable. Latin American Spanish counts at least as much as Castilian.
2. A licence that allows personal and commercial use, with weights you can download.
3. No ONNX Runtime, libtorch or C library at run time. A phoneme model needs a pure-Rust
   phonemizer for both languages, and espeak-ng is GPL-3.0 C.
4. Every network in the pipeline ports to Burn 0.22 on HIP: the codec decoder or vocoder too.
5. It fits and keeps up beside a 17.7 GB language model on an APU whose GPU memory is host memory.

## Shortlist

Spanish and English word error rates are independent: the Hugging Face Open TTS Leaderboard,
snapshot 2026-10-09, CV3-Eval `es` and the mean of Seed-TTS-eval and CV3-Eval `en`. "Port" is
effort against the Parakeet port as 1.

| Model | Licence | Size | Spanish WER | English WER | First audio | Port |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-TTS-12Hz 0.6B | Apache-2.0 | 915M BF16 + 171M codec | 2.94 preset, 2.81 cloned (the 1.7B; the 0.6B was not run). Vendor: 1.49 | 1.71 (1.7B) | 413 ms on an RTX 4060 at an 8-frame chunk | 1.5 |
| Pocket TTS | CC-BY-4.0 | 100M per language | 8.87 preset, 15.32 cloned. Vendor-affiliated: 1.15 | 1.62 | 41 ms on an A100, 126 ms on a CPU | 0.8 |
| Magpie Multilingual 357M | NVIDIA Open Model | 364M + 39M codec | 4.15. Vendor CER: 0.60 | 2.04 | not measured | 1.5 to 2 |
| Kokoro-82M | Apache-2.0 | 82M | 2.18 | 1.46 | 103 ms for a sentence, no streaming | 2 to 2.5 |
| Chatterbox Multilingual V3 | MIT | 0.8B in five networks | 8.99 preset, 4.28 cloned | 1.70 | no streaming | 3 |

### Qwen3-TTS-12Hz 0.6B (Alibaba, 2026-01-22)

- **Shape.** A 28-layer talker (d=1024, 16 heads, 8 KV heads, RMSNorm, SiLU, RoPE) predicts the
  first codebook. A 5-layer code predictor fills 15 more. The codec decoder is an 8-layer
  transformer and a causal convolution stack, with no diffusion or flow steps. 24 kHz.
- **Text.** Qwen BPE over raw text in a ChatML wrapper. The language is one token in the prompt.
  The package normalizes nothing.
- **Voices.** The CustomVoice checkpoint bakes in nine speakers, none a native Spanish speaker.
  The Base checkpoint takes a speaker vector or a prompt of reference codes, both computable in
  Python once, so neither encoder needs a port.
- **Speed.** Stock PyTorch runs the 0.6B at 0.23 times real time on an RTX 4060, and 2.26 times
  once each decode step runs as one captured graph: the cost is launch overhead. Stock PyTorch
  on ROCm runs the 1.7B three times slower than real time on a Radeon 8060S.
- **Elsewhere.** A C++ GGML build runs it on HIP in BF16 and Q4_K_M. Two candle ports and an MLX
  port exist to read beside the reference.
- **Known problems.** An English accent on Spanish words with non-native voices. Skips and
  invented words on long text, so sentences go in one at a time. Scattered reports of noise or a
  wrong timbre in the first second and on one-word replies. No blind listening test in any
  language.

### Pocket TTS (Kyutai, Spanish since 2026-04, retrained 2026-09-24)

- **Shape.** A 6-layer causal transformer, an MLP that produces one continuous latent per frame
  in one step, and a Mimi decoder. A 24-layer variant exists.
- **Text.** A 4,000-piece Unigram tokenizer over raw text. No number or date rules.
- **Voices.** 27 named voices per language, shipped as the transformer's KV state after it read
  the voice prompt. A port loads them only if its attention matches their key rotation and
  layout: check that first.
- **Why second.** One checkpoint per language rules out a sentence that mixes both. Its Spanish
  is contested: 8.87% independent (the distilled model) against 1.15% from a lab spun out of
  Kyutai.

### Magpie Multilingual 357M (NVIDIA, v2607 of 2026-07-21)

- An encoder, a 12-layer decoder with cross-attention, a local transformer over codebooks and
  the NanoCodec decoder. Classifier-free guidance doubles the batch.
- Text goes through a pronunciation dictionary to IPA, portable as a hash map, after text
  normalization the card calls required. Upstream runs C++ grammars for it.
- Five fixed voices, recorded by English speakers, with cloning removed. 22.05 kHz. Utterances
  cap at 20 seconds.

### Kokoro-82M and Chatterbox Multilingual V3

- **Kokoro** has the lowest error rates in both languages and renders a sentence in about
  100 ms. Spanish goes through espeak-ng. Its author warns that non-English support "may be absent
  or thin", and its three Spanish voices carry no grade. The port is many small operators: BiLSTMs, AdaIN
  stacks, an inverse STFT.
- **Chatterbox** ships the only checkpoint aimed at Latin American Spanish
  (`Chatterbox-Multilingual-es-mx-latam`), unmeasured by anyone else. It runs a 30-layer
  language model, a flow-matching decoder with ten steps and guidance, and a vocoder.

## Out

- **Non-commercial weights:** OmniVoice, Voxtral TTS, Fish S2 Pro, OpenAudio S1-mini, Higgs
  TTS 3, F5-TTS and its Spanish finetune, XTTS-v2, Llasa. S1-mini tops the open models on the
  one blind Spanish board (Artificial Analysis, launched 2026-09-22).
- **No Spanish:** Breeze TTS 2 (first among open weights in English blind tests), Sesame CSM-1B,
  Dia and Dia2, VibeVoice, Kyutai TTS 1.6B, Chatterbox Turbo and Nano, Spark-TTS, MegaTTS 3,
  IndexTTS-2, Marvis, NeuTTS Air.
- **Too large for this APU:** Fish S2 Pro reached 7.6 of the 21 tokens a second it needs on a
  Radeon 8060S. Voxtral, Higgs TTS 3, ZONOS2 and MOSS-TTS 8B sit in the same class.
- **No source or no future:** Supertonic 3 ships ONNX graphs only, and Supertone archived it.
- **Second language model needed:** Orpheus needs one 3.3B model per language.
- **Too slow or too many stages:** VoxCPM2 (2.3B, 1.04 times real time on an A100), CosyVoice 3
  (a language model, a 22-layer DiT and a vocoder, and English digits inside Spanish text).

## What nobody has measured

- Latin American Spanish by ear. The one blind Spanish board omits Qwen3-TTS, Pocket TTS and
  Kokoro, and does not state its reference voices' accent.
- Spanish and English in one sentence ("abrí GitHub Copilot"), for any open model.
- One-word and two-word Spanish replies.
- Any of these models on a Radeon 8060S through a runtime tuned for it.
- Spanish numbers and dates. Only Magpie claims to read them, so plan rules for them.

## Operators the workspace lacks

Each of these serves several candidates: transposed 1-D convolution with streaming state (all),
Snake and SnakeBeta (Qwen3-TTS, Magpie, Kokoro, Chatterbox), residual vector quantizer lookup
(Qwen3-TTS), a Unigram tokenizer (Pocket TTS), Spanish and English number, date and currency
rules (all). Burn 0.22 has `conv_transpose1d`; nothing here has run it on HIP.

## Sources

- [Open TTS Leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard) and its
  [announcement](https://huggingface.co/blog/open-tts-leaderboard) (2026-09-30).
- Artificial Analysis: [open weights](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights),
  [controlled voice](https://artificialanalysis.ai/text-to-speech/leaderboard/controlled-voice).
- Qwen3-TTS: [repository](https://github.com/QwenLM/Qwen3-TTS),
  [0.6B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base),
  [codec](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz),
  [report](https://arxiv.org/abs/2601.15621),
  [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (consumer GPU timings,
  HIP validation), [Crane](https://github.com/lucasjinreal/Crane) (candle port),
  [accent discussion](https://github.com/QwenLM/Qwen3-TTS/discussions/230),
  [short replies](https://github.com/QwenLM/Qwen3-TTS/issues/343).
- Pocket TTS: [repository](https://github.com/kyutai-labs/pocket-tts),
  [weights](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning),
  [ptts](https://github.com/gradium-ai/xn-ptts) (a Rust runtime with Spanish and English text
  rules).
- Magpie: [card](https://huggingface.co/nvidia/magpie_tts_multilingual_357m),
  [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp).
- Kokoro: [voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
  Chatterbox: [repository](https://github.com/resemble-ai/chatterbox),
  [Latin American pack](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-es-mx-latam).
- The Radeon 8060S: [Qwen3-TTS on three machines](https://tinycomputers.io/posts/the-real-cost-of-running-qwen-tts-locally-three-machines-compared.html)
  (2026-03-12), [Fish S2 Pro on Strix Halo](https://sleepingrobots.com/dreams/fish-audio-strix-halo/)
  (2026-03-15).
- Long text: [Taming Long-form Text-to-Speech](https://arxiv.org/abs/2609.16989) (2026-09-15).
