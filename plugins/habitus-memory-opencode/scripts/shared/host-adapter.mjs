// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
const REQUIRED_FUNCTIONS = [
  "contextOutput",
  "conversationId",
  "nativeSessionId",
  "prompt",
  "readTranscriptDelta",
  "successOutput",
];

export function requireHostAdapter(adapter) {
  if (!adapter || typeof adapter !== "object") throw new TypeError("host adapter must be an object");
  for (const name of REQUIRED_FUNCTIONS) {
    if (typeof adapter[name] !== "function") throw new TypeError(`host adapter must implement ${name}`);
  }
  if (typeof adapter.host !== "string" || !/^[a-z0-9](?:[a-z0-9-]{0,62})$/.test(adapter.host)) {
    throw new TypeError("host adapter host must be a safe lowercase identifier");
  }
  if (typeof adapter.protocol !== "string" || !/^[a-z][a-z0-9_]{0,127}$/.test(adapter.protocol)) {
    throw new TypeError("host adapter protocol must be a safe protocol identifier");
  }
  return adapter;
}

export function safeIdentifier(value) {
  const normalized = String(value || "").trim().replace(/[^A-Za-z0-9._-]/g, "-");
  if (!normalized) throw new Error("host session identity is missing");
  return normalized.slice(0, 220);
}
