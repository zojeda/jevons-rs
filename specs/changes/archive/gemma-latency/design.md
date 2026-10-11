# DiffusionGemma latency: design

[Proposal](proposal.md) · [Tasks](tasks.md)

## Row predictions on the device

`predict_rows` runs one 256-thread workgroup per chosen row of the logits. A thread owns
`vocabulary / 256` consecutive tokens. With `e = exp((logit - max) * inverse temperature)`:

1. Each thread finds its largest logit and that token. The workgroup takes the first thread with
   the largest value, which is the first such token.
2. Each thread sums `e` and `e * ln e` over its tokens. Every thread then adds the 256 partial
   sums in thread order, so it knows `z` and the sum of the threads before it.
3. The target is `draw * z`. A thread whose partial sum crosses what is left of the target scans
   its tokens again for the first one that does. Threads after it report their first token, and
   the workgroup takes the smallest.

The kernel writes the best token, the sampled token, `z` and `sum(e * ln e)`; the host computes
the entropy as `ln z - sum(e * ln e) / z`. The draws come from the host's seeded generator in
the order the f64 sampler used, so a seed gives the same noise and, on the twelve benchmark
answers, the same text. Sums are f32 in a fixed order: a row gives the same values wherever it
is launched.

`Conditioning::Previous` lost its logits slice. The model kept the device copy already; the
slice only let a caller pass logits the model no longer held, which no caller did.

## One forward for the end of a prompt and a canvas

`Model::forward` takes `canvas_from`, the first canvas row, and `exact`. Prompt rows come first
and end the prompt.

- **Embedding:** the kernel already took the first canvas row.
- **Products:** `exact` selects the row-invariant plans (matrix tiles, no split-K) and the tiled
  expert products for every row.
- **Attention:** two launches, one per kind of row, each the launch those rows would get in a
  forward of their own. The kernel takes the row at which its queries start. A tile that held
  both kinds would put causal rows in matrix instructions with keys they cannot see, which
  changes their rounding.
- **End of layer:** `post_ffn` takes both layer scales and the first canvas row.

A prompt row's values equal a prefill's because every kernel it passes is row invariant. A
canvas row's values equal those of a canvas forwarded alone for the same reason, which is why
`Model::canvas` uses `exact` for every canvas without conditioning that is read for candidate
logits. Without that, a read would depend on what the cache held, and the second sample of a
read would differ from the same read done alone.

`prefill_with_canvas` puts the canvas in the last token chunk when both fit in the batch.
Otherwise the canvas gets its own forward, still with `exact`.

## The forward as Burn operations

`model::burn` declares `GemmaOps`, a Burn backend extension. A forward is one operation for the
embedding, two per layer, and a few for what follows the layers:

- **`g4_attention_block`** takes the residual rows and the layer's input rows as tensors and runs
  the Q, K and V products, their preparation (which writes the caches), attention in its one or
  two launches, the output product and the norms after it.
- **`g4_feed_forward_block`** runs the shared expert, routing, grouping, the two routed products
  and the norms that end the layer. It returns the residual and the next layer's input rows.
- **The head:** the quantized product, the soft cap, the softmax and scaled norm of
  self-conditioning, candidate picks and row predictions, each an operation.

An operation takes an argument struct with everything that is not an activation: the layer's
weights and the scratch its kernels write between them (both shared, not copied), the launch
plans the tuner gives for the forward's rows, rotary tables, positions.

- **Same launches.** A block calls the launchers the direct path called, in its order, with its
  plans, on the same scratch. The bits are equal on all 84 corpus slots and the generated texts.
- **Order.** Burn queues operations on the model's thread and runs them in the order they were
  made, which keeps an attention after the cache write of its layer, and one block's use of the
  scratch after the last one's. A direct launch or read on the same device runs ahead of what is
  still queued, so prefill ends with a sync of the Burn device, and the cache and trace
  diagnostics sync before they read.
- **What stays direct.** Loading and uploads, the tuner's measurements, the one-time transposed
  embedding table, and the vision encoder.

### One operation per kernel, built and dropped

The spike of issue #5 wrapped each of a layer's sixteen kernels and measured 110 µs more per
layer on six layers. The whole model that way gave the same bits as direct launches, and queues
more than a thousand operations and dropped tensors per forward. Its reads with a resident prompt
took 90, 103 and 172 ms against 72 to 80 ms for direct launches. No load log covered the first
run and the last two ran while another project's builds loaded the machine, so those runs do not
isolate what it costs. Two
operations per layer queue about sixty operations a forward and measure within 1% of direct
launches on a quiet machine (a fresh read in 265 against 264 ms), so that is the version kept.

## What moved

Figures are from the interleaved runs of the
[measurements](../../../../benchmarks/gemma-latency-2026-10-10/README.md).

- **Reads with a resident prompt got slower: 56 to 70 ms on 2026-10-11, 70 to 77 ms the evening
  before.** Their canvas runs alone, and to match the bits of a canvas forwarded with prompt
  rows it now uses the tiled kernels. At ten rows the small-row expert kernel was faster. Every
  extra sample of a read pays the same.
- **Decision canvases under 32 rows changed precision.** The small-row expert kernel multiplies
  f32 weights; the tiled one rounds weights to FP16, as prefill does. Probabilities moved: the
  median over the corpus's 84 slots is 1e-4, three slots moved more than 0.01, and the near-tie
  of `relational-join_missing` moved by 0.21. No slot changed its best candidate, and the corpus
  scores 75 of 84 before and after. JevBench was not rerun.
- **`forward_ms`** of a one-step read no longer holds its first canvas, which now counts as
  prefill.
- **Load time** grows by the one new kernel's compilation.

## The fallback

Chunking token rows across prompt parts, without the canvas in the prompt's forward, changes no
bit of a decision read and slows nothing. Measured on 2026-10-10: fresh reads 479 to 335 ms,
extended prompts 276 to 160 ms, resident prompts unchanged. The canvas in the prompt's forward
took fresh reads to 291 ms and extended prompts to 124 ms that evening. Going back means `together = false` in `read_with_options`
and `exact = false` in `Model::canvas`; the rest of that path then never runs and can go.

## Not done

- **Canvas rows on the small-row kernels inside the shared forward.** The dense products would
  still be shared, the expert products launched once more for the canvas rows. That keeps the
  resident-prompt read at its old time and most of the gain, and needs row offsets in routing,
  grouping and the grouped products.
- Generation does not forward its first canvas step with the block's prefill. Its canvases keep
  the tuned plans, and the gain would be one forward in up to 48.
- A canvas that does not fit in the last chunk is not split across chunks.
