import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { once } from "node:events";
import test from "node:test";
import { createClient } from "./client.mjs";
import { evaluate, percentile, shuffled, summarize, validateCorpus } from "./benchmark-lib.mjs";
import { measure } from "./benchmark.mjs";

const corpus = JSON.parse(readFileSync(new URL("../../benchmarks/system-one/cases.json", import.meta.url)));
const cases = validateCorpus(corpus);

function perfectResponse(entry) {
  return { model: "fixture", answers: Object.fromEntries(Object.entries(entry.request.questions).map(([id, q]) => {
    const expected = entry.expected[id];
    if (q.type === "noul") return [id, { type: "noul", noul: Number(expected) }];
    const labels = q.type === "score" ? q.criteria.map((_, i) => String(i)) : Object.keys(q.criteria);
    const probabilities = Object.fromEntries(labels.map((label) => [label, Number(label === String(expected))]));
    return [id, { type: q.type, probabilities, confidence: 1,
      ...(q.type === "choice" ? { choice: expected } : { score: expected,
        legend: Object.fromEntries(q.criteria.map((value, i) => [String(i), value])) }) }];
  })), usage: { input_tokens: 123, output_tokens: 0 } };
}

test("corpus has 72 cases across nine categories and 84 individually keyed questions", () => {
  assert.equal(cases.length, 72);
  const categories = new Set(cases.map((entry) => entry.category));
  assert.equal(categories.size, 9);
  for (const category of categories) assert.equal(cases.filter((entry) => entry.category === category).length, 8);
  assert.equal(cases.reduce((n, entry) => n + Object.keys(entry.expected).length, 0), 84);
  for (const entry of cases) assert.ok(evaluate(entry, perfectResponse(entry)).every((answer) => answer.correct));
});

test("invalid answer keys and duplicate IDs fail before network access", () => {
  const bad = structuredClone(corpus);
  bad.cases[0].expected.q1 = "not_an_option";
  assert.throws(() => validateCorpus(bad), /Invalid question or target/);
  bad.cases[0] = bad.cases[1];
  assert.throws(() => validateCorpus(bad), /unique IDs/);
});

test("noul uses probability threshold, and scores preserve fractional expected levels", () => {
  const entry = cases.find((entry) => entry.id === "temporal-offset_before");
  const result = evaluate(entry, { answers: { q1: { type: "noul", noul: 0.8 } } })[0];
  assert.equal(result.correct, true);
  assert.ok(Math.abs(result.brier - 0.04) < 1e-12);
  const rubric = cases.find((entry) => entry.id === "rubric-blocked");
  const response = perfectResponse(rubric);
  Object.assign(response.answers.q1, { score: 1.6, probabilities: { 0: 0, 1: 0.4, 2: 0.6, 3: 0 }, confidence: 0.3 });
  const score = evaluate(rubric, response)[0];
  assert.equal(score.correct, true);
  assert.ok(Math.abs(score.absolute_error - 0.4) < 1e-12);
  assert.equal(evaluate(rubric, response, 0.25)[0].correct, false);
  response.answers.q1.score = 2.8;
  assert.equal(evaluate(rubric, response)[0].valid, false);
});

test("missing, nonfinite, out-of-range and inconsistent probabilities cannot pass", () => {
  const entry = cases[0];
  for (const corrupt of [
    (response) => { delete response.answers.q1; },
    (response) => { response.answers.q1.probabilities.eligible = NaN; },
    (response) => { response.answers.q1.probabilities.eligible = 1.2; },
    (response) => { response.answers.q1.probabilities.extra = 0; },
    (response) => { response.answers.q1.choice = "ineligible"; },
  ]) {
    const response = perfectResponse(entry);
    corrupt(response);
    assert.equal(evaluate(entry, response)[0].valid, false);
  }
});

test("failures count against accuracy and never make successful latency look faster", () => {
  const entry = cases[0];
  const good = { endpoint: "local", case_id: entry.id, category: entry.category, repeat: 0,
    warmup: false, http_ok: true, valid_response: true, elapsed_ms: 100,
    evaluations: evaluate(entry, perfectResponse(entry)) };
  const failed = { ...good, case_id: "failed", http_ok: false, valid_response: false,
    elapsed_ms: 1, evaluations: evaluate(entry, null) };
  const summary = summarize([good, failed, { ...failed, warmup: true },
    { ...good, endpoint: "remote", elapsed_ms: 200 }]);
  assert.equal(summary.endpoints.local.questions.accuracy, 0.5);
  assert.equal(summary.endpoints.local.questions.accuracy_valid_only, 1);
  assert.equal(summary.endpoints.local.latency_valid_responses.p50_ms, 100);
  assert.equal(summary.endpoints.local.latency_failed_attempts.p50_ms, 1);
  assert.equal(summary.paired.valid_pairs, 1);
  assert.equal(summary.paired.median_remote_over_local_latency, 2);
  assert.equal(summary.paired.local_minus_remote_ms.mean_ms, -100);
  assert.equal(percentile([], 0.5), null);
  assert.equal(percentile([10, 20], 0.5), 15);
});

test("seeded scheduling is reproducible, shuffled and leaves corpus unchanged", () => {
  const original = cases.map((entry) => entry.id);
  assert.deepEqual(shuffled(cases, 42), shuffled(cases, 42));
  assert.notDeepEqual(shuffled(cases, 42), shuffled(cases, 43));
  assert.deepEqual(cases.map((entry) => entry.id), original);
});

test("real SDK serializes all corpus cases without answer keys and uses model overrides", async (t) => {
  const received = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks));
    received.push({ body, path: request.url, auth: request.headers.authorization });
    response.writeHead(200, { "Content-Type": "application/json", "x-typesafe-request-id": "test-request" });
    response.end(JSON.stringify(perfectResponse(cases[received.length - 1])));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => { server.closeAllConnections(); server.close(); });
  const client = createClient({ baseURL: `http://127.0.0.1:${server.address().port}`,
    apiKey: "fixture-key", logLevel: "off", retry: { maxRetries: 0 } });
  for (const entry of cases) {
    const row = await measure(client, "local", entry, "fixture-model");
    assert.equal(row.valid_response, true);
    assert.ok(row.evaluations.every((answer) => answer.correct));
    assert.equal(row.request_id, "test-request");
    const wire = received.at(-1);
    assert.equal(wire.path, "/v1/systemone");
    assert.equal(wire.auth, "Bearer fixture-key");
    assert.deepEqual(wire.body, { ...entry.request, model: "fixture-model" });
  }
  assert.equal(received.length, 72);
});

test("SDK errors are attempted once and response bodies are not logged", async () => {
  let attempts = 0;
  const client = createClient({ apiKey: "fixture-key", logLevel: "off", retry: { maxRetries: 0 },
    fetch: async () => { attempts++; return new Response(JSON.stringify({ detail: "sensitive echo" }), { status: 529 }); } });
  const row = await measure(client, "remote", cases[0], "fixture-model");
  assert.equal(attempts, 1);
  assert.equal(row.status, 529);
  assert.equal(row.valid_response, false);
  assert.equal(row.evaluations[0].correct, false);
  assert.ok(!JSON.stringify(row).includes("sensitive echo"));
});
