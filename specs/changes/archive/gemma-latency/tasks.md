# DiffusionGemma latency: tasks

[Proposal](proposal.md) · [Design](design.md)

- [x] `latency_bench`: mean latency of Decision reads over the System One corpus and of
      Generative chat answers, with answers scored and repeated reads compared bit for bit.
- [x] Baseline on Windows, and where the time goes.
- [x] `DiffusionModel::predictions` with the f64 default; `uniform::refine` asks the model.
- [x] `predict_rows` and `Model::predictions`; `Conditioning::Previous` without logits.
- [x] Prefill chunks token rows across prompt parts.
- [x] `Model::forward` with prompt rows before canvas rows; `prefill_with_canvas`.
- [x] `DiffusionModel::prefill_and_forward_canvas`; one-step reads use it.
- [x] GPU kernel tests, the three model tests, `prefix_check`.
- [x] Before and after, back to back, under `benchmarks/gemma-latency-2026-10-10/`.
- [x] The user's choice on Burn (2026-10-11): the forward through Burn.
- [x] One Burn operation per kernel: the same bits, over a thousand queued operations a forward.
      Dropped for two operations per layer.
- [x] `model::burn`: the embedding, an attention block and a feed-forward block per layer, and
      the head, as Burn backend extensions.
- [x] `Model::forward`, self-conditioning, logits, picks and predictions on Burn tensors; the
      direct-launch forward removed.
- [x] The same bits as direct launches on the corpus and the generated texts.
- [x] Before, direct launches and through Burn, interleaved, in the benchmark report.
- [x] The GPU kernel tests, the three model tests and `prefix_check` on the final tree.
- [x] The folder moves to `specs/changes/archive/` and its row leaves the index.
