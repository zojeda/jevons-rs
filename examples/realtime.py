# /// script
# requires-python = ">=3.10"
# dependencies = ["websockets>=13", "soundfile", "numpy"]
# ///
"""Stream an audio file to the Realtime transcription API at real-time pace.

Opens a transcription session on /v1/realtime, sends the file as 24 kHz PCM16 in 100 ms
chunks with server-side turn detection, and prints live deltas and final transcripts.

    uv run examples/realtime.py examples/speech-es.flac [--url ws://127.0.0.1:8080/v1/realtime]
        [--model parakeet-tdt-0.6b-v3] [--language es] [--fast]

Set TYPESAFE_API_KEY when the server requires a key.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time

import numpy as np
import soundfile
import websockets

RATE = 24000
CHUNK = RATE // 10

parser = argparse.ArgumentParser()
parser.add_argument("audio")
parser.add_argument("--url", default="ws://127.0.0.1:8080/v1/realtime")
parser.add_argument("--model", default="parakeet-tdt-0.6b-v3")
parser.add_argument("--language", default="")
parser.add_argument("--fast", action="store_true", help="send as fast as possible")
args = parser.parse_args()

audio, rate = soundfile.read(args.audio, dtype="float32", always_2d=True)
audio = audio.mean(axis=1)
if rate != RATE:
    times = np.arange(int(len(audio) * RATE / rate)) / RATE
    audio = np.interp(times, np.arange(len(audio)) / rate, audio).astype(np.float32)
# Trailing silence lets turn detection end the last turn.
audio = np.concatenate([audio, np.zeros(RATE, dtype=np.float32)])
pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


async def send_audio(socket):
    started = time.monotonic()
    for i, offset in enumerate(range(0, len(pcm), CHUNK * 2)):
        chunk = base64.b64encode(pcm[offset : offset + CHUNK * 2]).decode()
        await socket.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        if not args.fast:
            await asyncio.sleep(max(0.0, started + (i + 1) * 0.1 - time.monotonic()))


async def main():
    protocols = ["realtime"]
    if key := os.environ.get("TYPESAFE_API_KEY"):
        protocols.append(f"openai-insecure-api-key.{key}")
    async with websockets.connect(
        f"{args.url}?intent=transcription", subprotocols=protocols
    ) as socket:
        await socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": RATE},
                                "transcription": {"model": args.model, "language": args.language},
                                "turn_detection": {"type": "server_vad", "silence_duration_ms": 600},
                            }
                        },
                    },
                }
            )
        )
        sender = asyncio.create_task(send_audio(socket))
        started = time.monotonic()
        pending = 0
        while True:
            try:
                message = await asyncio.wait_for(socket.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                if sender.done() and pending == 0:
                    break
                continue
            event = json.loads(message)
            kind = event["type"]
            at = f"{time.monotonic() - started:6.2f}s"
            if kind == "input_audio_buffer.committed":
                pending += 1
            elif kind == "conversation.item.input_audio_transcription.delta":
                print(event["delta"], end="", flush=True)
            elif kind == "conversation.item.input_audio_transcription.completed":
                print(f"\n{at} ✔ {event['transcript']}")
                pending -= 1
            elif kind == "input_audio_buffer.speech_started":
                print(f"{at} ▶ ", end="", flush=True)
            elif kind == "error":
                print(f"\n{at} error: {event['error']['message']}", file=sys.stderr)
        await sender


asyncio.run(main())
