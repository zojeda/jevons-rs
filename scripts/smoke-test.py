#!/usr/bin/env python3
"""Exercise the running service using only the Python standard library."""

import argparse
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def transcribe(url, headers):
    """Transcribes the Spanish fixture through the multipart transcription route."""
    audio = (Path(__file__).resolve().parents[1] / "examples/speech-es.flac").read_bytes()
    boundary = "jevons-smoke-test"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nparakeet-latest\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\nverbose_json\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech-es.flac"\r\n'
        "Content-Type: audio/flac\r\n\r\n".encode()
        + audio
        + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    request_headers = {k: v for k, v in headers.items() if k != "Content-Type"}
    request_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    request = Request(url + "/v1/audio/transcriptions", data=b"".join(parts), headers=request_headers)
    with urlopen(request, timeout=180) as response:
        assert response.headers.get("x-typesafe-request-id"), "Missing request ID"
        body = json.load(response)
    assert "grabación" in body["text"].lower(), body["text"]
    assert body["segments"] and 15 < body["duration"] < 17, body


def speak(url, headers, health):
    """Speaks the request fixture through the speech route and checks the WAV stream."""
    voice = health["voice"]
    assert voice["default_voice"] in voice["voices"], voice
    example = Path(__file__).resolve().parents[1] / "examples/audio-speech.json"
    body = json.loads(example.read_text())
    body["model"] = health["services"]["voice"]
    body["voice"] = voice["default_voice"]

    def post(body):
        request = Request(url + "/v1/audio/speech", data=json.dumps(body).encode(), headers=headers)
        try:
            response = urlopen(request, timeout=180)
        except HTTPError as error:
            response = error
        with response:
            assert response.headers.get("x-typesafe-request-id"), "Missing request ID"
            return response.status, response.headers.get("Content-Type"), response.read()

    status, content_type, audio = post(body)
    assert status == 200 and content_type == "audio/wav", (status, content_type)
    assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE", audio[:12]
    # A streamed header: the length was unknown when it left.
    assert audio[4:8] == audio[40:44] == b"\xff\xff\xff\xff", audio[:44]
    rate = int.from_bytes(audio[24:28], "little")
    assert rate == voice["sample_rate"], (rate, voice)
    seconds = (len(audio) - 44) / 2 / rate
    assert len(audio) % 2 == 0 and 2 < seconds < 60, seconds
    status, content_type, pcm = post(dict(body, input="Hola.", response_format="pcm"))
    assert status == 200 and content_type == "audio/pcm", (status, content_type)
    assert pcm and len(pcm) % 2 == 0, len(pcm)
    status, _, error = post(dict(body, voice="nobody"))
    assert status == 400 and json.loads(error)["error"]["param"] == "voice", error
    status, _, error = post(dict(body, instructions="Speak cheerfully."))
    assert status == 400 and json.loads(error)["error"]["code"] == "unsupported_parameter", error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("TYPESAFE_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["TYPESAFE_API_KEY"]

    def call(path, body=None):
        request = Request(
            args.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
        )
        try:
            response = urlopen(request, timeout=180)
        except HTTPError as error:
            response = error
        with response:
            assert response.headers.get("x-typesafe-request-id"), "Missing request ID"
            return response.status, json.load(response)

    status, models = call("/v1/models")
    assert status == 200, models
    names = {model["name"] for model in models["models"]}
    if "parakeet-latest" in names:
        transcribe(args.url, headers)
    status, health = call("/health")
    assert status == 200, health
    if health["services"]["voice"]:
        assert health["services"]["voice"] in names, names
        speak(args.url, headers, health)
    if "jev-latest" not in names:
        print("Speech and voice routes OK; no language model is loaded")
        return
    example = Path(__file__).resolve().parents[1] / "examples/system-one.json"
    request = json.loads(example.read_text())
    # The generic alias is served by every architecture.
    request["model"] = "jev-latest"
    status, response = call("/v1/systemone", request)
    assert status == 200, response
    assert set(response["answers"]) == set(request["questions"])
    for name, answer in response["answers"].items():
        assert answer["type"] == request["questions"][name]["type"]
        if answer["type"] == "noul":
            assert 0 <= answer["noul"] <= 1
            continue
        probabilities = answer["probabilities"]
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
        assert abs(sum(probabilities.values()) - 1) < 1e-9
        entropy = -sum(p * math.log(p) for p in probabilities.values() if p)
        assert abs(answer["confidence"] - (1 - entropy / math.log(len(probabilities)))) < 1e-9
        if answer["type"] == "choice":
            assert set(probabilities) == set(request["questions"][name]["criteria"])
            assert probabilities[answer["choice"]] == max(probabilities.values())
        else:
            expected = sum(int(level) * p for level, p in probabilities.items())
            assert abs(answer["score"] - expected) < 1e-9
            assert answer["legend"] == {
                str(i): level for i, level in enumerate(request["questions"][name]["criteria"])
            }
    assert response["usage"]["input_tokens"] > 0
    assert response["usage"]["output_tokens"] == 0
    status, error = call("/v1/systemone", dict(request, steps=9))
    assert status == 422 and isinstance(error["detail"], list), error
    status, error = call("/v1/systemone", dict(request, model="unknown-model"))
    assert status == 404 and error["detail"]["error_type"] == "not_found_error", error
    print(json.dumps(response, indent=2))
    print("HTTP smoke test passed")


if __name__ == "__main__":
    main()
