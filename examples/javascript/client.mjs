import { TypeSafeClient } from "@typesafe-ai/sdk";

export function createClient(overrides = {}) {
  return new TypeSafeClient({
    baseURL: process.env.TYPESAFE_BASE_URL || "http://127.0.0.1:8080",
    defaultModel: process.env.TYPESAFE_DEFAULT_MODEL || "gemmadiffusion-latest",
    // The SDK requires a key. This placeholder works when server auth is disabled.
    apiKey: process.env.TYPESAFE_API_KEY || "local-no-auth",
    // Local inference can take longer than the SDK's default timeout.
    timeout: 180_000,
    // Keep each example to one attempt, including when the local queue is full.
    retry: { maxRetries: 0 },
    ...overrides,
  });
}

export const client = createClient();
