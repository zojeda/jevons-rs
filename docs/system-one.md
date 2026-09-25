# System One API

[Back to README](../README.md)

`POST /v1/systemone` implements [TypeSafe AI's System One API](https://docs.typesafe.ai/introduction): you send a `state` and named `questions`, and get typed, probabilistic answers instead of free text: the probability of yes (`noul`), a distribution over named options (`choice`), or an expected rubric level (`score`). It runs on the diffusion language models (DiffusionGemma, Nemotron-Labs-Diffusion), which read every answer at once from fixed slots on a masked canvas. See the [HTTP API reference](api.md#questions-and-answers) for question types, extensions and limits, and [benchmarks](../benchmarks/README.md) for accuracy and latency.

## The masked canvas

A canvas is a block of token positions. For this API, we fix the question labels and leave one answer position per question:

```text
Prompt: state + questions + answer codes (A = yes, B = no, ...)

Canvas: Question 1       Question 2
        Answer: [ ? ]   Answer: [ ? ]
                 ^               ^
             answer slot     answer slot
```

The brackets mark unknown answers. We fill those positions with seeded random vocabulary tokens, excluding the special mask token. After caching the prompt, we evaluate the whole canvas once. Bidirectional attention lets each slot use the surrounding canvas and prompt context. We read the logits at each slot and normalize them over its allowed answer codes.

DiffusionGemma's full text generator refines a noisy canvas over multiple denoising steps. By default, this service includes an empty, closed thought channel in the prompt, takes one read with fixed surrounding text, and returns the answer distributions. It generates no thought tokens. Optional extensions add denoising steps, noise samples, a bounded thought, sequential question chunks, and images. See [diffusion and canvas inference](inference.md) for the model explanation, a worked example, and the limits of these probabilities.

## Text example

With the service running, in another terminal:

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  --data-binary @examples/system-one.json
```

The example asks three questions about a construction material. Set `TYPESAFE_API_KEY` on the server to enable bearer authentication, then add `-H "Authorization: Bearer $TYPESAFE_API_KEY"` to client calls. The default listener is `127.0.0.1:8080`.

## Image example

For images, start the service with a compatible projector (see [image setup](build.md#image-input)):

```bash
export DIFFUSION_MMPROJ="$HOME/models/diffusiongemma/mmproj-diffusiongemma-26b-a4b-f16.gguf"
cargo run --release --locked -p jevons-rs -- -m "$DIFFUSION_MODEL" --mmproj "$DIFFUSION_MMPROJ"
```

Ask what’s in a photo and get structured answers:

<table>
  <tr>
    <td width="40%" align="center" valign="middle">
      <img src="../examples/hotdog.jpg" width="330" alt="A hot dog in a bun topped with mustard">
      <br><sub><strong>INPUT</strong> · Answer about the photo.</sub>
    </td>
    <td width="60%" align="center" valign="middle">
      <img src="assets/hotdog-response.svg" width="520" alt="Example response: hot dog, 96.3% probability of yes. Condiment probabilities: mustard 91.3%, ketchup 7.1%, none 1.5%. 204 input tokens, 0 output tokens.">
    </td>
  </tr>
</table>

Probabilities are rounded from the example response below; model answers can vary.

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  --data-binary @examples/hotdog.json
```

<details>
<summary>View the request</summary>

The image data is abbreviated here; [hotdog.json](../examples/hotdog.json) contains the complete request.

```json
{
  "model": "gemmadiffusion-latest",
  "state": "Answer about the photo.",
  "images": [
    "data:image/jpeg;base64,/9j/4gJASU..."
  ],
  "questions": {
    "hotdog": {
      "type": "noul",
      "instructions": "The photo shows a hot dog."
    },
    "condiment": {
      "type": "choice",
      "instructions": "Which condiment is on it?",
      "criteria": {
        "mustard": null,
        "ketchup": null,
        "none": null
      }
    }
  }
}
```

</details>

<details>
<summary>View the full JSON response</summary>

```json
{
  "model": "gemmadiffusion-0.1",
  "answers": {
    "condiment": {
      "type": "choice",
      "choice": "mustard",
      "probabilities": {
        "ketchup": 0.07114288211805858,
        "mustard": 0.9134373998609046,
        "none": 0.01541971802103676
      },
      "confidence": 0.6950052214787418
    },
    "hotdog": {
      "type": "noul",
      "noul": 0.9629650092899195
    }
  },
  "usage": {
    "input_tokens": 204,
    "output_tokens": 0
  }
}
```

</details>


Try the [hot dog photo example](api.md#hot-dog-photo), including the [bundled JPEG](../examples/hotdog.jpg), [ready-to-send request](../examples/hotdog.json), and startup instructions for the vision projector.

Use [JavaScript SDK examples](../examples/javascript/README.md) for application code. The [API reference](api.md) covers request types, model aliases, and errors. Use the [extensions](api.md#extensions) for `steps`, `samples`, `think`, `sequential`, and `images`. Text defaults are `steps=1`, `samples=1`, and `think=0`, with an 8,192-token context. Override the context with `--context-size`; larger contexts allocate more cache memory. Image requests require `--mmproj` or `DIFFUSION_MMPROJ`.

