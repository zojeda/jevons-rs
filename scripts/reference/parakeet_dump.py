# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.8,<2.10",
#   "transformers>=5.17,<6",  # ParakeetForTDT
#   "librosa",
#   "soundfile",
#   "numpy",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Dump Parakeet TDT reference tensors from the official HF implementation.

Runs on CPU in float32. For each clip (16 kHz mono audio files) it dumps the waveform, the
normalized log-mel features, the subsampling output, encoder layers 0, n/2 and n-1, the encoder
output, its joint projection, the joint logits of the first greedy steps, and the greedy TDT
tokens and durations with the decoded text. Output: raw little-endian f32/i32 arrays plus
manifest.json, for parity tests of the Rust port. Keep the output out of git.

    uv run scripts/reference/parakeet_dump.py MODEL_DIR OUT_DIR NAME=AUDIO [NAME=AUDIO ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile
import torch
from transformers import AutoProcessor, ParakeetForTDT

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("out_dir")
parser.add_argument("clips", nargs="+", help="NAME=AUDIO, a 16 kHz mono file")
parser.add_argument("--steps", type=int, default=24, help="joint logit steps to dump")
args = parser.parse_args()

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
manifest = {"tensors": {}, "clips": {}}


def save(name, value):
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    array = np.ascontiguousarray(array)
    if array.dtype.kind in "iub":
        array, suffix = array.astype("<i4"), "i32"
    else:
        array, suffix = array.astype("<f4"), "f32"
    array.tofile(out / f"{name}.{suffix}")
    manifest["tensors"][name] = {"dtype": suffix, "shape": list(array.shape)}


torch.manual_seed(0)
processor = AutoProcessor.from_pretrained(args.model_dir)
model = ParakeetForTDT.from_pretrained(args.model_dir, dtype=torch.float32).eval()
config = model.config
vocab = config.vocab_size
blank = config.blank_token_id
layers = model.encoder.layers
traced = sorted({0, len(layers) // 2, len(layers) - 1})

for spec in args.clips:
    name, path = spec.split("=", 1)
    audio, rate = soundfile.read(path, dtype="float32")
    assert rate == 16000 and audio.ndim == 1, f"{path}: need 16 kHz mono, got {rate} Hz {audio.shape}"
    save(f"{name}_audio", audio)

    inputs = processor(audio, sampling_rate=rate, return_tensors="pt")
    features = inputs["input_features"]
    save(f"{name}_mel", features[0])

    captured = {}
    hooks = [model.encoder.subsampling.register_forward_hook(lambda m, i, o: captured.__setitem__("sub", o))]
    for index in traced:
        hooks.append(
            layers[index].register_forward_hook(lambda m, i, o, index=index: captured.__setitem__(index, o))
        )
    with torch.no_grad():
        encoded = model.get_audio_features(input_features=features, attention_mask=inputs["attention_mask"])
    for hook in hooks:
        hook.remove()
    save(f"{name}_subsampling", captured["sub"][0])
    for index in traced:
        save(f"{name}_layer{index}", captured[index][0])
    save(f"{name}_encoder", encoded.last_hidden_state[0])
    projected = encoded.pooler_output[0]
    save(f"{name}_projected", projected)

    # Greedy TDT decoding written out, so the per-step joint logits can be dumped. It must match
    # model.generate exactly, which is checked below.
    frames = int(encoded.attention_mask[0].sum())
    state = [None]

    def predict(token):
        embedded = model.decoder.embedding(torch.tensor([[token]]))
        output, state[0] = model.decoder.lstm(embedded, state[0])
        return model.decoder.decoder_projector(output)[0, 0]

    with torch.no_grad():
        prediction = predict(blank)
        t, tokens, durations, starts, step_logits = 0, [], [], [], []
        while t < frames:
            logits = model.joint.head(torch.relu(projected[t] + prediction))
            if len(step_logits) < args.steps:
                step_logits.append(logits)
            token = int(logits[:vocab].argmax())
            duration = config.durations[int(logits[vocab:].argmax())]
            if token == blank and duration == 0:
                duration = 1
            if token != blank:
                tokens.append(token)
                starts.append(t)
                durations.append(duration)
                prediction = predict(token)
            t += duration

        generated = model.generate(**inputs, return_dict_in_generate=True)
    reference = [int(x) for x in generated.sequences[0].tolist()[1:] if int(x) != blank]
    assert reference == tokens, f"{name}: manual greedy loop diverged from generate()"

    save(f"{name}_step_logits", torch.stack(step_logits))
    save(f"{name}_tokens", np.array(tokens))
    save(f"{name}_token_frames", np.array(starts))
    save(f"{name}_token_durations", np.array(durations))
    text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
    manifest["clips"][name] = {"path": str(path), "frames": frames, "text": text}
    print(f"{name}: {len(audio) / rate:.2f} s, {frames} frames, {len(tokens)} tokens: {text}")

manifest["traced_layers"] = traced
manifest["vocab_size"] = vocab
manifest["blank"] = blank
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
