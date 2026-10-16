# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.100"]
# ///
"""Every OpenAI-compatible jevons-rs API through the official OpenAI Python SDK.

    uv run examples/openai-sdk.py [--url http://127.0.0.1:5380/v1]

Needs a server with the generative, speech and voice services enabled (see
jevons.example.toml); sections for a missing service are skipped. Set TYPESAFE_API_KEY when the
server requires a key.
"""

import argparse
import os
import tempfile
from pathlib import Path

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:5380/v1")
args = parser.parse_args()
client = OpenAI(base_url=args.url, api_key=os.environ.get("TYPESAFE_API_KEY", "unused"))
examples = Path(__file__).resolve().parent
models = {model.id for model in client.models.list()}

if "jev-latest" in models:
    print("== Chat Completions")
    reply = client.chat.completions.create(
        model="jev-latest",
        messages=[{"role": "user", "content": "What is 15% of 240? Answer with the number."}],
        max_tokens=128,
    )
    print(reply.choices[0].message.content, f"({reply.usage.completion_tokens} tokens)")

    print("== Chat Completions, streamed")
    stream = client.chat.completions.create(
        model="jev-latest",
        messages=[{"role": "user", "content": "Name three rivers in South America."}],
        max_tokens=64,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    print("== Responses")
    response = client.responses.create(
        model="jev-latest",
        instructions="Answer in one short sentence.",
        input="Why is the sky blue?",
        max_output_tokens=64,
    )
    print(response.output_text)

    print("== Completions")
    completion = client.completions.create(
        model="jev-latest", prompt="The three primary colors of light are", max_tokens=16
    )
    print(completion.choices[0].text.strip())

if "parakeet-latest" in models:
    print("== Transcription (Spanish, word timestamps)")
    with open(examples / "speech-es.flac", "rb") as audio:
        transcript = client.audio.transcriptions.create(
            model="parakeet-latest",
            file=audio,
            language="es",
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
    print(transcript.text)
    print("first words:", [(w.word, w.start) for w in transcript.words[:4]])

    print("== Transcription of a browser recording (WebM/Opus), streamed")
    with open(examples / "speech-es-browser.webm", "rb") as audio:
        stream = client.audio.transcriptions.create(model="parakeet-latest", file=audio, stream=True)
        for event in stream:
            if event.type == "transcript.text.delta":
                print(event.delta, end="", flush=True)
    print()

if "qwen3-tts-latest" in models:
    print("== Speech (WAV, saved as it arrives)")
    spoken = Path(tempfile.gettempdir()) / "jevons-speech.wav"
    with client.audio.speech.with_streaming_response.create(
        model="qwen3-tts-latest",
        voice="lector",
        input="Hola. El total es $1.500, y la reunión empieza a las 15:30.",
        response_format="wav",
    ) as response:
        response.stream_to_file(spoken)
    print(f"{spoken}: {spoken.stat().st_size} bytes")

    print("== Speech (24 kHz PCM16, an English voice)")
    pcm = client.audio.speech.create(
        model="qwen3-tts-latest",
        voice="reader",
        input="The build finished in 42 seconds.",
        response_format="pcm",
        extra_body={"language": "en"},
    ).content
    print(f"{len(pcm) / 48000:.1f} s of audio")
