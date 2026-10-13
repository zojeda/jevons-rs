# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "qwen-tts==0.1.1",
#   "torch>=2.8,<2.10",
#   "torchaudio>=2.8,<2.10",
#   "safetensors",
#   "soundfile",
#   "numpy",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchaudio = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Dump Qwen3-TTS reference tensors from the `qwen-tts` package.

Runs on CPU in float32. For each case (a voice file from `qwen3_tts_voice.py`, a language and a
text) it dumps the prompt's token ids, the talker's input for the prompt, its hidden states
after layers 0 and 13 and after the final norm, the logits at the prompt's last position, the
greedy codes of the first frames, the codes of one sampled run, and what the codec decoder makes
of the sampled codes: the summed codebook vectors, the transformer's output and the audio.
Output: raw little-endian f32/i32 arrays plus manifest.json, for parity tests of the Rust port,
and `samples/<case>.wav` to listen to. Keep the output out of git.

    uv run scripts/reference/qwen3_tts_dump.py MODEL_DIR OUT_DIR \
        NAME=VOICE.safetensors:LANGUAGE:TEXT [NAME=VOICE.safetensors:LANGUAGE:TEXT ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile
import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
from safetensors.numpy import load_file

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("out_dir")
parser.add_argument("cases", nargs="+", help="NAME=VOICE.safetensors:LANGUAGE:TEXT")
parser.add_argument("--greedy-frames", type=int, default=40)
parser.add_argument("--max-frames", type=int, default=600)
parser.add_argument("--seed", type=int, default=7)
args = parser.parse_args()

out = Path(args.out_dir)
(out / "samples").mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)

tts = Qwen3TTSModel.from_pretrained(args.model_dir, device_map="cpu", dtype=torch.float32)
model = tts.model
talker = model.talker
decoder = model.speech_tokenizer.model.decoder


def save(name, tensor, dtype):
    array = np.ascontiguousarray(tensor.detach().cpu().numpy().astype(dtype))
    array.tofile(out / name)
    return list(array.shape)


def generate(ids, prompt, language, **kwargs):
    codes, _ = model.generate(
        input_ids=[ids],
        voice_clone_prompt=prompt,
        languages=[language],
        non_streaming_mode=False,
        **tts._merge_generate_kwargs(**kwargs),
    )
    return codes[0]


manifest = {"model": str(args.model_dir), "seed": args.seed, "cases": {}}
for case in args.cases:
    name, rest = case.split("=", 1)
    voice_file, language, text = rest.split(":", 2)
    speaker = torch.from_numpy(load_file(voice_file)["speaker"])
    prompt = tts._prompt_items_to_voice_clone_prompt(
        [
            VoiceClonePromptItem(
                ref_code=None,
                ref_spk_embedding=speaker,
                x_vector_only_mode=True,
                icl_mode=False,
                ref_text=None,
            )
        ]
    )
    ids = tts._tokenize_texts([tts._build_assistant_text(text)])[0]
    entry = {"text": text, "language": language, "voice": Path(voice_file).stem}
    entry["ids"] = save(f"{name}.ids.i32", ids[0], np.int32)
    save(f"{name}.speaker.f32", speaker, np.float32)

    # The prompt: the first call of the talker's transformer, and of its head.
    seen = {}

    def on_talker(_module, _args, kwargs, output):
        if "embeds" not in seen:
            seen["embeds"] = kwargs["inputs_embeds"][0]
            seen["hidden"] = [h[0] for h in output.hidden_states]

    def on_head(_module, _args, output):
        seen.setdefault("logits", output[0, -1])

    hooks = [
        talker.model.register_forward_hook(on_talker, with_kwargs=True),
        talker.codec_head.register_forward_hook(on_head),
    ]
    greedy = generate(
        ids,
        prompt,
        language,
        do_sample=False,
        subtalker_dosample=False,
        max_new_tokens=args.greedy_frames + 1,
    )
    for hook in hooks:
        hook.remove()
    entry["prefill"] = save(f"{name}.prefill.f32", seen["embeds"], np.float32)
    layers = len(seen["hidden"]) - 1
    entry["hidden_first"] = save(f"{name}.hidden_first.f32", seen["hidden"][1], np.float32)
    entry["hidden_middle"] = save(f"{name}.hidden_middle.f32", seen["hidden"][layers // 2], np.float32)
    entry["hidden_final"] = save(f"{name}.hidden_final.f32", seen["hidden"][layers], np.float32)
    entry["middle_layer"] = layers // 2 - 1
    entry["logits"] = save(f"{name}.logits.f32", seen["logits"], np.float32)
    entry["greedy"] = save(f"{name}.greedy.i32", greedy, np.int32)

    # One sampled run with the package's defaults, and what the decoder makes of it.
    torch.manual_seed(args.seed)
    codes = generate(ids, prompt, language, max_new_tokens=args.max_frames)
    entry["codes"] = save(f"{name}.codes.i32", codes, np.int32)
    by_codebook = codes.unsqueeze(0).transpose(1, 2)
    quantized = decoder.quantizer.decode(by_codebook)
    entry["quantized"] = save(f"{name}.quantized.f32", quantized[0], np.float32)
    latent = decoder.pre_transformer(inputs_embeds=decoder.pre_conv(quantized).transpose(1, 2))
    entry["latent"] = save(f"{name}.latent.f32", latent.last_hidden_state[0], np.float32)
    wav = decoder(by_codebook)[0, 0]
    entry["wav"] = save(f"{name}.wav.f32", wav, np.float32)
    soundfile.write(out / "samples" / f"{name}.wav", wav.numpy(), 24000, subtype="PCM_16")
    manifest["cases"][name] = entry
    print(f"{name}: {ids.shape[1]} ids, {greedy.shape[0]} greedy frames, {codes.shape[0]} frames sampled")

(out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
