#!/usr/bin/env python3
"""Exercise the running service using only the Python standard library."""

import argparse
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


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
    assert any(model["name"] == "jev-latest" for model in models["models"])
    example = Path(__file__).resolve().parents[1] / "examples/system-one.json"
    request = json.loads(example.read_text())
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
