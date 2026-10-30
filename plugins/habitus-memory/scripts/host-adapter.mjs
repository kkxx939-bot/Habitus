import { hasMeaningfulMessageContent, stripInjectionReceipts } from "./shared/injection-receipt.mjs";
import { readFirstRecord, readJsonlDelta } from "./shared/jsonl.mjs";
import { completedTurns, safeIdentifier } from "./shared/host-adapter.mjs";

// 一批只切在一轮结束处：Codex 给每一轮的最终答复标 phase = "final_answer"（本机 0.147–0.162 的 74 个做完的轮，每轮恰好
// 一条、都是最后一条答复）。中途的进度答复不切——压缩前（PreCompact）也只送已经结束的轮，做到一半的那一轮留在转写文件里，
// 等它做完连同压缩后的那一半一起送；一轮不再被拆成两批（裁定 69：一轮 = 宿主的一轮）。
// skipped 里的是原样超过读取额度、没读的行（jsonl.mjs），看不到内容：它可能就是挂着的那个工具调用的输出，越过它时挂着的全部放下，
// 不然这一轮再也等不到输出、这个会话往后一轮都切不出来；一轮照样在之后的 final_answer 处结束
function selectCompleted(records, injectionReceipts, { skipped = new Set() } = {}) {
  const accepted = [];
  const pending = new Set();
  const boundaries = [];
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (skipped.has(index)) pending.clear();
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
      boundaries.push({ consumedCount: index + 1, recordCount: accepted.length });
    }
  }
  return completedTurns(accepted, boundaries);
}

export const hostAdapter = Object.freeze({
  host: "codex",
  protocol: "codex_rollout",
  // agent 向人提问、等人作答的工具（方案 32 第 4.2 条）：CLI 是 request_user_input，Desktop 用 request_user_input_async
  questionTool: "request_user_input（Desktop 里是 request_user_input_async）",
  nativeSessionId(input) { return safeIdentifier(input?.session_id || input?.conversation_id); },
  // 子代理线程不进记忆（提示是主 agent 写的任务，结果已回到主线程）。认法：hook 输入带 agent_id，或转写开头的 session_meta
  // 记着它是 spawn 出来的子线程（source.subagent.thread_spawn / thread_source = "subagent"；子线程 rollout 的 session_id 是
  // 父线程的，不能拿它认）
  async prepareInput(input) {
    const meta = await readFirstRecord(this.transcriptPath(input));
    const payload = meta?.type === "session_meta" ? meta.payload : null;
    const spawned = payload?.thread_source === "subagent" || (payload?.source != null && typeof payload.source === "object" && "subagent" in payload.source);
    // `codex exec` 的会话（脚本调用，不是他在说话）：session_meta.source 是 "exec"（Codex 的 SessionSource：cli / vscode /
    // exec / mcp / subagent{…}，与上面认子线程是同一个字段）
    const exec = payload?.source === "exec";
    if (!spawned && !exec) return input;
    return { ...input, ...(spawned ? { subagent_thread: true } : {}), ...(exec ? { exec_session: true } : {}) };
  },
  isSubagent(input) { return Boolean(input?.agent_id) || input?.subagent_thread === true; },
  isNonInteractive(input) { return input?.exec_session === true; },
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
      selectCompleted: (records, segment) => selectCompleted(records, injectionReceipts, segment),
    });
  },
});
