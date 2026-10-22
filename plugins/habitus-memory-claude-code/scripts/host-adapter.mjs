import { hasMeaningfulMessageContent, stripInjectionReceipts } from "./shared/injection-receipt.mjs";
import { readJsonlDelta } from "./shared/jsonl.mjs";
import { safeIdentifier } from "./shared/host-adapter.mjs";
import { zcodeAdapter } from "./zcode-adapter.mjs";

function accepted(record) {
  if (!["user", "assistant"].includes(record?.type) || record?.isSidechain === true || record?.isMeta === true) return false;
  return hasMeaningfulMessageContent(record?.message?.content);
}

function blocks(record) {
  const content = record?.message?.content;
  return Array.isArray(content) ? content : [];
}

function selectCompleted(records, injectionReceipts) {
  const selected = [];
  const pending = new Set();
  let safe = null;
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (record == null) continue;
    for (const block of blocks(record)) {
      if (block?.type === "tool_use" && typeof block.id === "string") pending.add(block.id);
      if (block?.type === "tool_result" && typeof block.tool_use_id === "string") pending.delete(block.tool_use_id);
    }
    const stripped = stripInjectionReceipts(record, injectionReceipts);
    if (accepted(stripped.value)) selected.push(stripped.value);
    const stopReason = record?.message?.stop_reason;
    if (
      record?.type === "assistant"
      && record?.message?.role === "assistant"
      && stopReason !== "tool_use"
      && pending.size === 0
    ) safe = { consumedCount: index + 1, recordCount: selected.length };
  }
  if (safe == null) return null;
  return { consumedCount: safe.consumedCount, records: selected.slice(0, safe.recordCount), afterTurn: true };
}

export const claudeCodeAdapter = Object.freeze({
  host: "claude-code",
  protocol: "claude_code",
  // agent 向人提问、等人作答的工具（方案 32 第 4.2 条）
  questionTool: "AskUserQuestion",
  nativeSessionId(input) {
    const parent = safeIdentifier(input?.session_id);
    return input?.agent_id ? safeIdentifier(`${parent}.subagent.${input.agent_id}`) : parent;
  },
  conversationId(input) { return `claude-${this.nativeSessionId(input)}`; },
  prompt(input) { return typeof input?.prompt === "string" ? input.prompt.trim() : ""; },
  successOutput() { return {}; },
  contextOutput(context) {
    return { hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context } };
  },
  transcriptPath(input) {
    const path = input?.agent_transcript_path || input?.transcript_path;
    return typeof path === "string" && path ? path : null;
  },
  readTranscriptDelta(input, cursor, config, injectionReceipts) {
    return readJsonlDelta(this.transcriptPath(input), cursor, {
      maxBytes: config.maxTranscriptBytes,
      selectCompleted: (records) => selectCompleted(records, injectionReceipts),
    });
  },
});

// 同一份插件也装在 ZCode 里（它认 Claude Code 的插件格式）：ZCode 给 hook 进程设 ZCODE_SESSION_ID，据此换成 ZCode 的适配器
export function selectHostAdapter(env = process.env) {
  return env.ZCODE_SESSION_ID ? zcodeAdapter : claudeCodeAdapter;
}

export const hostAdapter = selectHostAdapter();
