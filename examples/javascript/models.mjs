import { client } from "./client.mjs";

// The SDK unwraps the server's { models: [...] } response into an array.
const { data: models, requestId } = await client.models.list().withResponse();

console.log(JSON.stringify({ requestId, models }, null, 2));
