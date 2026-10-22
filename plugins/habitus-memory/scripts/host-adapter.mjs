import { hasMeaningfulMessageContent, stripInjectionReceipts } from "./shared/injection-receipt.mjs";
import { readJsonlDelta } from "./shared/jsonl.mjs";
import { safeIdentifier } from "./shared/host-adapter.mjs";

// 一批只切在一轮结束处：Codex 给每一轮的最终答复标 phase = "final_answer"（本机 0.147–0.162 的 74 个做完的轮，每轮恰好
// 一条、都是最后一条答复）。中途的进度答复不切——压缩前（PreCompact）也只送已经结束的轮，做到一半的那一轮留在转写文件里，
// 等它做完连同压缩后的那一半一起送；一轮不再被拆成两批（裁定 69：一轮 = 宿主的一轮）。
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
    if (itemType === "message" && item?.role === "assistant" && item?.phase === "final_answer" && pending.size === 0) {
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
