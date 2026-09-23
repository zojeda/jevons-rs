// Pure scoring and scheduling helpers; no network or credential access.
export function validateCorpus(corpus) {
  const cases = corpus?.cases;
  if (corpus?.version !== 1 || !Array.isArray(cases) || cases.length < 1 || cases.length > 100) {
    throw new Error("Corpus must have version 1 and 1–100 cases");
  }
  const ids = new Set();
  for (const entry of cases) {
    if (!entry.id || ids.has(entry.id) || !entry.category || !entry.rationale) {
      throw new Error("Cases need unique IDs, categories, and rationales");
    }
    ids.add(entry.id);
    const { request, expected } = entry;
    if (!request || !["string", "object"].includes(typeof request.state) || request.state === null ||
        typeof request.model !== "string" || !request.questions || !expected ||
        Object.keys(request).some((key) => !["state", "model", "questions"].includes(key))) {
      throw new Error(`Invalid request in ${entry.id}`);
    }
    const questions = Object.entries(request.questions);
    if (!questions.length || JSON.stringify(Object.keys(expected).sort()) !==
        JSON.stringify(questions.map(([id]) => id).sort())) {
      throw new Error(`Answer keys must match questions in ${entry.id}`);
    }
    for (const [id, question] of questions) {
      const { type, criteria, instructions } = question;
      if (typeof instructions !== "string" || !instructions) throw new Error(`Missing instructions: ${entry.id}/${id}`);
      const target = expected[id];
      const valid = type === "noul" ? typeof target === "boolean"
        : type === "choice" ? criteria && !Array.isArray(criteria) &&
          Object.keys(criteria).length >= 1 && Object.keys(criteria).length <= 128 &&
          typeof target === "string" && Object.hasOwn(criteria, target)
        : type === "score" ? Array.isArray(criteria) && criteria.length >= 2 && criteria.length <= 10 &&
          criteria.every((level) => typeof level === "string") &&
          Number.isInteger(target) && target >= 0 && target < criteria.length
        : false;
      if (!valid) throw new Error(`Invalid question or target: ${entry.id}/${id}`);
    }
  }
  return cases;
}

const finiteNumber = (value) => typeof value === "number" && Number.isFinite(value);
const probability = (value) => finiteNumber(value) && value >= 0 && value <= 1;
const mean = (values) => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;

function validDistribution(values, labels) {
  return values && typeof values === "object" && !Array.isArray(values) &&
    Object.keys(values).length === labels.length && labels.every((key) => probability(values[key])) &&
    Math.abs(Object.values(values).reduce((a, b) => a + b, 0) - 1) <= 0.001;
}

export function evaluate(entry, response, tolerance = 0.5) {
  return Object.entries(entry.request.questions).map(([id, question]) => {
    const expected = entry.expected[id];
    const answer = response?.answers?.[id];
    const result = { id, type: question.type, expected, valid: false, correct: false };
    if (!answer || answer.type !== question.type) return { ...result, error: "missing_or_wrong_type" };
    if (question.type === "noul") {
      if (!probability(answer.noul)) return { ...result, error: "invalid_probability" };
      const target = Number(expected);
      return { ...result, valid: true, predicted: answer.noul >= 0.5,
        correct: (answer.noul >= 0.5) === expected, probability: answer.noul,
        brier: (answer.noul - target) ** 2,
        log_loss: -Math.log(Math.max(1e-15, expected ? answer.noul : 1 - answer.noul)) };
    }
    const labels = question.type === "choice" ? Object.keys(question.criteria)
      : question.criteria.map((_, i) => String(i));
    if (!validDistribution(answer.probabilities, labels) || !probability(answer.confidence)) {
      return { ...result, error: "invalid_distribution_or_confidence" };
    }
    if (question.type === "choice") {
      if (!labels.includes(answer.choice) ||
          answer.probabilities[answer.choice] + 0.001 < Math.max(...Object.values(answer.probabilities))) {
        return { ...result, error: "invalid_choice" };
      }
      return { ...result, valid: true, predicted: answer.choice, correct: answer.choice === expected,
        log_loss: -Math.log(Math.max(1e-15, answer.probabilities[expected])) };
    }
    if (!finiteNumber(answer.score) || answer.score < 0 || answer.score > labels.length - 1 ||
        !answer.legend || labels.some((label) => answer.legend[label] !== question.criteria[Number(label)])) {
      return { ...result, error: "invalid_score_or_legend" };
    }
    const weighted = labels.reduce((total, label) => total + Number(label) * answer.probabilities[label], 0);
    if (Math.abs(weighted - answer.score) > 0.01) return { ...result, error: "score_distribution_mismatch" };
    const error = Math.abs(answer.score - expected);
    return { ...result, valid: true, predicted: answer.score, correct: error <= tolerance,
      absolute_error: error, normalized_absolute_error: error / (labels.length - 1) };
  });
}

export function percentile(values, quantile) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * quantile;
  const low = Math.floor(index);
  const high = Math.ceil(index);
  return sorted[low] + (sorted[high] - sorted[low]) * (index - low);
}

function latency(values) {
  return { n: values.length, mean_ms: mean(values), p50_ms: percentile(values, 0.5),
    p95_ms: percentile(values, 0.95), min_ms: values.length ? Math.min(...values) : null,
    max_ms: values.length ? Math.max(...values) : null };
}

function accuracy(answers) {
  return { n: answers.length, valid: answers.filter((a) => a.valid).length,
    correct: answers.filter((a) => a.correct).length,
    accuracy: mean(answers.map((a) => Number(a.correct))),
    accuracy_valid_only: mean(answers.filter((a) => a.valid).map((a) => Number(a.correct))),
    noul_brier: mean(answers.filter((a) => a.valid && a.type === "noul").map((a) => a.brier)),
    classification_log_loss: mean(answers.filter((a) => a.valid && a.log_loss !== undefined).map((a) => a.log_loss)),
    score_mae: mean(answers.filter((a) => a.valid && a.type === "score").map((a) => a.absolute_error)),
    score_normalized_mae: mean(answers.filter((a) => a.valid && a.type === "score").map((a) => a.normalized_absolute_error)) };
}

function aggregate(rows) {
  return { requests: rows.length, http_successes: rows.filter((row) => row.http_ok).length,
    valid_responses: rows.filter((row) => row.valid_response).length,
    request_exact_match: mean(rows.map((row) => Number(row.evaluations.every((a) => a.correct)))),
    questions: accuracy(rows.flatMap((row) => row.evaluations)),
    latency_valid_responses: latency(rows.filter((row) => row.valid_response).map((row) => row.elapsed_ms)),
    latency_all_attempts: latency(rows.map((row) => row.elapsed_ms)),
    latency_failed_attempts: latency(rows.filter((row) => !row.valid_response).map((row) => row.elapsed_ms)) };
}

export function summarize(rows) {
  const measured = rows.filter((row) => !row.warmup);
  const result = { endpoints: {} };
  for (const endpoint of [...new Set(measured.map((row) => row.endpoint))]) {
    const selected = measured.filter((row) => row.endpoint === endpoint);
    result.endpoints[endpoint] = { ...aggregate(selected),
      by_category: Object.fromEntries([...new Set(selected.map((row) => row.category))].map((category) =>
        [category, aggregate(selected.filter((row) => row.category === category))])),
      by_question_type: Object.fromEntries(["noul", "choice", "score"].map((type) =>
        [type, accuracy(selected.flatMap((row) => row.evaluations.filter((a) => a.type === type)))])),
      by_question_count: Object.fromEntries([...new Set(selected.map((row) => row.evaluations.length))].map((count) =>
        [count, aggregate(selected.filter((row) => row.evaluations.length === count))])) };
    result.endpoints[endpoint].category_macro_accuracy = mean(Object.values(result.endpoints[endpoint].by_category)
      .map((value) => value.questions.accuracy));
  }
  const pairs = new Map();
  for (const row of measured) {
    const key = `${row.repeat}/${row.case_id}`;
    if (!pairs.has(key)) pairs.set(key, {});
    pairs.get(key)[row.endpoint] = row;
  }
  const paired = [...pairs.values()].filter((pair) => pair.local?.valid_response && pair.remote?.valid_response);
  result.paired = { valid_pairs: paired.length,
    local_minus_remote_ms: latency(paired.map((pair) => pair.local.elapsed_ms - pair.remote.elapsed_ms)),
    median_remote_over_local_latency: percentile(paired.map((pair) => pair.remote.elapsed_ms / pair.local.elapsed_ms), 0.5),
    local_accuracy_wins: 0, remote_accuracy_wins: 0, accuracy_ties: 0 };
  for (const pair of paired) {
    const local = pair.local.evaluations.filter((a) => a.correct).length;
    const remote = pair.remote.evaluations.filter((a) => a.correct).length;
    result.paired[local > remote ? "local_accuracy_wins" : remote > local ? "remote_accuracy_wins" : "accuracy_ties"]++;
  }
  return result;
}

export function shuffled(cases, seed) {
  let state = seed >>> 0;
  const result = [...cases];
  const random = () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
