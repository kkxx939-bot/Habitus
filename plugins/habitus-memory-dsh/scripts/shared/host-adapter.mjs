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

// 适配器的 selectCompleted 收尾：boundaries 是这一段里每一轮结束的位置（读到第几条、收下了几条），按先后排；
// 一轮都没结束给 null。最后一轮的结束处就是这一次能确认到的位置
export function completedTurns(accepted, boundaries) {
  const last = boundaries.at(-1);
  if (last == null) return null;
  return { consumedCount: last.consumedCount, records: accepted.slice(0, last.recordCount), afterTurn: true, boundaries };
}
