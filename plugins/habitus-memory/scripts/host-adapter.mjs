import { hasMeaningfulMessageContent, stripInjectionReceipts } from "./shared/injection-receipt.mjs";
import { readJsonlDelta } from "./shared/jsonl.mjs";
import { safeIdentifier } from "./shared/host-adapter.mjs";

function selectCompleted(records, injectionReceipts) {
  const accepted = [];
  const pending = new Set();
  let safe = null;
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (record == null) continue;
    const item = record?.type === "response_item" ? record.payload : null;
    const itemType = item?.type;
    if (["function_call", "custom_tool_call", "tool_search_call"].includes(itemType)) {
      const callId = item.call_id || item.id;
      if (typeof callId === "string" && callId) pending.add(callId);
    } else if (["function_call_output", "custom_tool_call_output", "tool_search_output"].includes(itemType)) {
      const callId = item.call_id || item.id;
      if (typeof callId === "string") pending.delete(callId);
    }
    const stripped = stripInjectionReceipts(record, injectionReceipts);
    const cleanedItem = stripped.value?.type === "response_item" ? stripped.value.payload : null;
    const injectionOnly = stripped.removed
      && cleanedItem?.type === "message"
      && !hasMeaningfulMessageContent(cleanedItem.content);
    if (!injectionOnly) accepted.push(stripped.value);
    if (itemType === "message" && item?.role === "assistant" && pending.size === 0) {
      safe = { consumedCount: index + 1, recordCount: accepted.length };
    }
  }
  if (safe == null) return null;
  return { consumedCount: safe.consumedCount, records: accepted.slice(0, safe.recordCount), afterTurn: true };
}

export const hostAdapter = Object.freeze({
  host: "codex",
  protocol: "codex_rollout",
  // agent 向人提问、等人作答的工具（方案 32 第 4.2 条）：CLI 是 request_user_input，Desktop 用 request_user_input_async
  questionTool: "request_user_input（Desktop 里是 request_user_input_async）",
  nativeSessionId(input) { return safeIdentifier(input?.session_id || input?.conversation_id); },
  conversationId(input) { return `codex-${this.nativeSessionId(input)}`; },
  prompt(input) { return typeof input?.prompt === "string" ? input.prompt.trim() : ""; },
  successOutput() { return {}; },
  contextOutput(context) {
    return { hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context } };
  },
  transcriptPath(input) {
    return typeof input?.transcript_path === "string" && input.transcript_path ? input.transcript_path : null;
  },
  readTranscriptDelta(input, cursor, config, injectionReceipts) {
    return readJsonlDelta(this.transcriptPath(input), cursor, {
      maxBytes: config.maxTranscriptBytes,
      selectCompleted: (records) => selectCompleted(records, injectionReceipts),
    });
  },
});
