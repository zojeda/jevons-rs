import { UnprocessableEntityError, noul } from "@typesafe-ai/sdk";
import { client } from "./client.mjs";

try {
  // The SDK forwards extensions; steps above eight fail validation.
  await client.systemOne({
    state: "Ground granulated blast furnace slag is used in concrete.",
    questions: { is_scm: noul("Is the material an SCM?") },
    steps: 9,
  });
  throw new Error("Expected this server to reject steps=9 with HTTP 422");
} catch (error) {
  if (!(error instanceof UnprocessableEntityError)) {
    throw error;
  }
  console.log(JSON.stringify({
    name: error.name,
    status: error.status,
    requestId: error.requestId,
    details: error.body,
  }, null, 2));
}
