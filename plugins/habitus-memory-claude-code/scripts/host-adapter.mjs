import { hasMeaningfulMessageContent, stripInjectionReceipts } from "./shared/injection-receipt.mjs";
import { readJsonlDelta } from "./shared/jsonl.mjs";
import { completedTurns, safeIdentifier } from "./shared/host-adapter.mjs";
import { zcodeAdapter } from "./zcode-adapter.mjs";

function conversational(record) {
  return ["user", "assistant"].includes(record?.type);
}

function accepted(record) {
  if (!conversational(record) || record?.isSidechain === true || record?.isMeta === true) return false;
  return hasMeaningfulMessageContent(record?.message?.content);
}

// 工具调用只在 user / assistant 记录里配对：api-request-blob 等记录会重复已配对过的 tool_use，算进去就再也清不空
function blocks(record) {
  const content = conversational(record) ? record?.message?.content : null;
  return Array.isArray(content) ? content : [];
}

// Claude Code 把一条答复的每个内容块（thinking、text、tool_use）各写成一条记录，每条都带这条答复的 stop_reason：
// 后面紧跟着同一 message.id 的记录时，这条答复还没写完（裁定 69：一轮不拆开）。段尾那条看不到后面：这一段停在读取额度上
// （more，后面还有没读到的记录）时当它还没写完，不在它后面切；读到了文件末尾才按它自己算
function continuesMessage(records, index, more) {
  const id = records[index]?.message?.id;
  if (typeof id !== "string" || !id) return false;
  for (let next = index + 1; next < records.length; next += 1) {
    if (records[next] == null) continue;
    return records[next].type === "assistant" && records[next].message?.id === id;
  }
  return more === true;
}

// 切点：没有挂着的工具调用、stop_reason 不是 tool_use、同一条答复后面没有别的块的答复之后。这也可能落在一轮中间（流式写下的
// 答复块 stop_reason 为空），压缩前送的那一批会把一轮拆开；但子代理一轮的最后一条答复 stop_reason 同样为空，不能只认 end_turn
// （2026-10-22 对本机 60 份转写试过：57 份会一条都送不出）。Claude Code 怎么认"一轮结束"待它的真实转写核清（待办），在那之前保持原样。
// skipped 里的是原样超过读取额度、没读的行（jsonl.mjs），看不到内容：它可能就是挂着的那个工具调用的结果（截图、读大文件），
// 越过它时挂着的全部放下，不然这一轮再也等不到结果、这个会话往后一轮都切不出来；一轮照样在之后第一条说完了的答复处结束
function selectCompleted(records, injectionReceipts, { more = false, skipped = new Set() } = {}) {
  const selected = [];
  const pending = new Set();
  const boundaries = [];
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (skipped.has(index)) pending.clear();
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
      && !continuesMessage(records, index, more)
    ) boundaries.push({ consumedCount: index + 1, recordCount: selected.length });
  }
  return completedTurns(selected, boundaries);
}

export const claudeCodeAdapter = Object.freeze({
  host: "claude-code",
  protocol: "claude_code",
  // agent 向人提问、等人作答的工具（方案 32 第 4.2 条）
  questionTool: "AskUserQuestion",
  nativeSessionId(input) { return safeIdentifier(input?.session_id); },
  // 子代理里触发的 hook 带 agent_id：子代理的会话不进记忆（它的提示是主 agent 写的任务，结果已回到主会话的转写里）
  isSubagent(input) { return Boolean(input?.agent_id); },
  // 非交互会话（`claude -p`、Agent SDK）不进记忆：Claude Code 启动时把入口写进 CLAUDE_CODE_ENTRYPOINT（非交互为 sdk-cli，
  // Agent SDK 自己设 sdk-ts / sdk-py），hook 进程继承它；转写里每条记录的 entrypoint 就是这个值（本机 17,921 份 sdk-cli 会话）
  isNonInteractive(_input, env = process.env) { return String(env.CLAUDE_CODE_ENTRYPOINT || "").startsWith("sdk-"); },
  conversationId(input) { return `claude-${this.nativeSessionId(input)}`; },
  prompt(input) { return typeof input?.prompt === "string" ? input.prompt.trim() : ""; },
  successOutput() { return {}; },
  contextOutput(context) {
    return { hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context } };
  },
  transcriptPath(input) {
    return typeof input?.transcript_path === "string" && input.transcript_path ? input.transcript_path : null;
  },
  readTranscriptDelta(input, cursor, config, injectionReceipts) {
    // 只有 user / assistant 记录与切点、送出有关：api-request 之类的日志记录（本机一份转写里一轮夹着 140 MiB）不留、不占读取额度
    return readJsonlDelta(this.transcriptPath(input), cursor, {
      maxBytes: config.maxTranscriptBytes,
      relevant: conversational,
      selectCompleted: (records, segment) => selectCompleted(records, injectionReceipts, segment),
    });
  },
});

// 同一份插件也装在 ZCode 里（它认 Claude Code 的插件格式）：ZCode 给 hook 进程设 ZCODE_SESSION_ID，据此换成 ZCode 的适配器
export function selectHostAdapter(env = process.env) {
  return env.ZCODE_SESSION_ID ? zcodeAdapter : claudeCodeAdapter;
}

export const hostAdapter = selectHostAdapter();
