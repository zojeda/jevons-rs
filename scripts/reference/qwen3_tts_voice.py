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
"""Make a Qwen3-TTS voice from a recording.

Runs the Base checkpoint's speaker encoder over a clip, on CPU in float32, and writes the
speaker vector as `speaker` (1,024 f32) in a safetensors file. Put the file in the checkpoint's
`voices/` folder and its name is a voice. The runtime ports neither the speaker encoder nor the
codec encoder: a voice is made here, once. Use a recording you have the rights to.

    uv run scripts/reference/qwen3_tts_voice.py MODEL_DIR CLIP OUT.safetensors [--seconds 12]
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile
import torch
from qwen_tts import Qwen3TTSModel
from safetensors.numpy import save_file

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("clip", help="an audio file soundfile reads (WAV, FLAC, OGG)")
parser.add_argument("out", help="the voice file to write")
parser.add_argument("--start", type=float, default=0.0, help="seconds to skip at the start")
parser.add_argument("--seconds", type=float, default=12.0, help="seconds of the clip to use")
parser.add_argument("--source", default="", help="where the clip comes from, kept in the file")
args = parser.parse_args()

audio, rate = soundfile.read(args.clip, dtype="float32", always_2d=True)
audio = audio.mean(axis=1)
audio = audio[int(args.start * rate) : int((args.start + args.seconds) * rate)]

tts = Qwen3TTSModel.from_pretrained(args.model_dir, device_map="cpu", dtype=torch.float32)
item = tts.create_voice_clone_prompt(ref_audio=(audio, rate), x_vector_only_mode=True)[0]
speaker = item.ref_spk_embedding.detach().float().cpu().numpy().reshape(-1)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
save_file(
    {"speaker": np.ascontiguousarray(speaker, dtype=np.float32)},
    str(out),
    metadata={"source": args.source or Path(args.clip).name, "seconds": f"{len(audio) / rate:.2f}"},
)
print(f"{out}: {speaker.shape[0]} values from {len(audio) / rate:.1f} s of {args.clip}")
