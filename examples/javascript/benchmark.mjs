#!/usr/bin/env node
import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { VERSION } from "@typesafe-ai/sdk";
import { createClient } from "./client.mjs";
import { evaluate, shuffled, summarize, validateCorpus } from "./benchmark-lib.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

export async function measure(client, endpoint, entry, model, tolerance = 0.5) {
  const start = performance.now();
  let data;
  let metadata;
  let elapsed;
  try {
    const result = await client.systemOne({ ...entry.request, model }).withResponse();
    elapsed = performance.now() - start;
    data = result.data;
    metadata = { http_ok: true, status: result.response.status, request_id: result.requestId ?? null };
  } catch (error) {
    elapsed = performance.now() - start;
    // SDK error messages/bodies can echo payloads. Store only class and status.
    metadata = { http_ok: false, status: error.status ?? null, error: error.constructor.name };
  }
  const evaluations = evaluate(entry, data, tolerance);
  return { endpoint, case_id: entry.id, category: entry.category, elapsed_ms: elapsed,
    ...metadata, valid_response: metadata.http_ok && evaluations.every((answer) => answer.valid),
    // Store only response fields used in the benchmark, never headers or credentials.
    response: data ? { model: data.model, answers: data.answers, usage: data.usage } : null,
    evaluations };
}

function numberOption(values, key, fallback, min, max, integer = true) {
  const value = values[key] === undefined ? fallback : Number(values[key]);
  if (!Number.isFinite(value) || (integer && !Number.isInteger(value)) || value < min || value > max) {
    throw new Error(`--${key} must be ${integer ? "an integer" : "a number"} between ${min} and ${max}`);
  }
  return value;
}

function baseURL(value) {
  const url = new URL(value);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error("Base URLs must be HTTP(S) API roots without credentials, query strings, or fragments");
  }
  return url.href.replace(/\/$/, "");
}

function display(summary) {
  const percent = (value) => value === null ? "n/a" : `${(100 * value).toFixed(1)}%`;
  const ms = (value) => value === null ? "n/a" : value.toFixed(1);
  console.table(Object.entries(summary.endpoints).map(([endpoint, value]) => ({
    endpoint, requests: value.requests, valid: value.valid_responses,
    "question accuracy": percent(value.questions.accuracy),
    "all answers correct": percent(value.request_exact_match),
    "p50 ms (valid)": ms(value.latency_valid_responses.p50_ms),
    "p95 ms (valid)": ms(value.latency_valid_responses.p95_ms),
  })));
}

export async function main(args = process.argv.slice(2)) {
  const { values } = parseArgs({ args, options: {
    help: { type: "boolean" }, "dry-run": { type: "boolean" },
    endpoint: { type: "string", default: "both" }, corpus: { type: "string" },
    category: { type: "string" }, limit: { type: "string" },
    "local-url": { type: "string" }, "remote-url": { type: "string" },
    "local-model": { type: "string" }, "remote-model": { type: "string" },
    seed: { type: "string" }, repeat: { type: "string" }, warmup: { type: "string" },
    "timeout-ms": { type: "string" }, "score-tolerance": { type: "string" },
    output: { type: "string" }, "env-file": { type: "string" },
  } });
  if (values.help) {
    console.log(`Compare local and hosted System One using the existing TypeSafe JavaScript client.
  --dry-run                 Validate cases and print the plan; no HTTP calls
  --endpoint both|local|remote  Default: both
  --category NAME           Filter one category (8 cases per category)
  --limit N                 Select N cases after seeded shuffle (default: all)
  --seed N                  Shuffle seed, default 42 (does not set model RNG)
  --repeat N                Measured rounds, default 1
  --warmup N                Unmeasured calls per endpoint, default 0
  --local-url URL           Default http://127.0.0.1:8080
  --remote-url URL          Default https://api.typesafe.ai
  --local-model ID          Default gemmadiffusion-latest
  --remote-model ID         Default jev-latest
  --timeout-ms N            Default 180000; no retries
  --score-tolerance N       Score accuracy tolerance in levels, default 0.5
  --env-file PATH           Default repository .env, existing environment wins
  --corpus PATH             Default benchmarks/system-one/cases.json
  --output DIRECTORY        New results directory; refuses overwrite
Remote key: TYPESAFE_API_KEY. Local key: LOCAL_TYPESAFE_API_KEY (optional).
Default: 72 cases × 2 endpoints = 144 calls, sequential and paired.`);
    return;
  }
  const envFile = resolve(values["env-file"] ?? resolve(root, ".env"));
  if (values["env-file"] || existsSync(envFile)) {
    if (!process.loadEnvFile) throw new Error("Loading .env requires Node 20.12+; upgrade Node or export the variables first");
    process.loadEnvFile(envFile);
  }
  const corpusPath = resolve(values.corpus ?? resolve(root, "benchmarks/system-one/cases.json"));
  const corpusBytes = readFileSync(corpusPath);
  const corpus = JSON.parse(corpusBytes);
  const allCases = validateCorpus(corpus);
  const seed = numberOption(values, "seed", 42, 0, 0xffffffff);
  const repeats = numberOption(values, "repeat", 1, 1, 100);
  const warmups = numberOption(values, "warmup", 0, 0, 100);
  const timeout = numberOption(values, "timeout-ms", 180000, 1, 3600000);
  const tolerance = numberOption(values, "score-tolerance", 0.5, 0, 1, false);
  const filtered = values.category ? allCases.filter((entry) => entry.category === values.category) : allCases;
  if (!filtered.length) throw new Error("No cases match --category");
  const count = numberOption(values, "limit", filtered.length, 1, filtered.length);
  const cases = shuffled(filtered, seed).slice(0, count);
  const endpoints = values.endpoint === "both" ? ["local", "remote"] : [values.endpoint];
  if (endpoints.some((name) => !["local", "remote"].includes(name))) throw new Error("Invalid --endpoint");
  const config = {
    local: { baseURL: baseURL(values["local-url"] ?? "http://127.0.0.1:8080"),
      defaultModel: values["local-model"] ?? "gemmadiffusion-latest" },
    remote: { baseURL: baseURL(values["remote-url"] ?? "https://api.typesafe.ai"),
      defaultModel: values["remote-model"] ?? "jev-latest" },
  };
  const totalCalls = endpoints.length * (warmups + cases.length * repeats);
  console.log(`${cases.length} cases, ${cases.reduce((n, entry) => n + Object.keys(entry.expected).length, 0)} questions/round, ${totalCalls} HTTP calls (${warmups} warmups/endpoint).`);
  if (values["dry-run"]) {
    console.table([...new Set(cases.map((entry) => entry.category))].map((category) => {
      const entries = cases.filter((entry) => entry.category === category);
      return { category, cases: entries.length,
        questions: entries.reduce((n, entry) => n + Object.keys(entry.expected).length, 0) };
    }));
    return;
  }
  if (endpoints.includes("remote") && !process.env.TYPESAFE_API_KEY) throw new Error("Missing TYPESAFE_API_KEY in environment or .env");
  const clients = Object.fromEntries(endpoints.map((name) => [name, createClient({
    ...config[name], apiKey: name === "remote" ? process.env.TYPESAFE_API_KEY : process.env.LOCAL_TYPESAFE_API_KEY || "local-no-auth",
    timeout, retry: { maxRetries: 0 }, logLevel: "off",
  })]));
  const timestamp = new Date().toISOString();
  const output = resolve(values.output ?? resolve(root, "benchmarks/results", timestamp.replace(/[:.]/g, "-")));
  mkdirSync(dirname(output), { recursive: true });
  mkdirSync(output); // An existing run must never be overwritten.
  const metadata = { started_at: timestamp, corpus: corpus.name, corpus_version: corpus.version,
    corpus_sha256: createHash("sha256").update(corpusBytes).digest("hex"),
    sdk_version: VERSION, node_version: process.version, seed, repeats, warmups, timeout_ms: timeout,
    score_tolerance: tolerance, retry_count: 0, concurrency: 1, total_calls: totalCalls,
    endpoints: Object.fromEntries(endpoints.map((name) => [name, config[name]])),
    case_ids: cases.map((entry) => entry.id), note: "End-to-end SDK latency; includes network and response parsing. No model seed override." };
  writeFileSync(resolve(output, "run.json"), JSON.stringify(metadata, null, 2) + "\n");
  const rows = [];
  let completed = 0;
  async function run(entry, name, repeat, warmup) {
    const row = { ...await measure(clients[name], name, entry, config[name].defaultModel, tolerance), repeat, warmup };
    rows.push(row);
    appendFileSync(resolve(output, "results.jsonl"), JSON.stringify(row) + "\n");
    console.log(`[${++completed}/${totalCalls}] ${name} ${entry.id}: ${row.http_ok ? row.status : row.error} ${row.elapsed_ms.toFixed(1)}ms; ${row.evaluations.filter((a) => a.correct).length}/${row.evaluations.length}${warmup ? " (warmup)" : ""}`);
  }
  for (let i = 0; i < warmups; i++) {
    for (const name of endpoints) await run(cases[i % cases.length], name, -1, true);
  }
  for (let repeat = 0; repeat < repeats; repeat++) {
    const order = shuffled(cases, seed + repeat);
    for (let i = 0; i < order.length; i++) {
      const endpointOrder = (i + repeat) % 2 ? [...endpoints].reverse() : endpoints;
      for (const name of endpointOrder) await run(order[i], name, repeat, false);
    }
  }
  const summary = summarize(rows);
  writeFileSync(resolve(output, "summary.json"), JSON.stringify(summary, null, 2) + "\n");
  display(summary);
  console.log(`Results: ${output}`);
  if (rows.some((row) => !row.valid_response)) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => { console.error(`Benchmark failed: ${error.message}`); process.exitCode = 1; });
}
