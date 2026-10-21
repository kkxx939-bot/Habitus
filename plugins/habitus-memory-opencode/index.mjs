// Habitus 的 OpenCode 插件（方案 36 第四节）：人说话时注入召回的记忆与"先问他"的提醒指令块（合成段，不算他说的话）；
// 会话空闲 / 压缩 / 删除时把已完成的消息镜像成本地转写、经共享 outbox 送回本地 Habitus 服务（协议 opencode）。
// 失败一律静默放行，不拦 OpenCode。

import { loadPluginConfig } from "./scripts/shared/config.mjs";
import { createContextInjection, renderReminderInstruction } from "./scripts/shared/hook-runner.mjs";
import { INJECTION_TAGS } from "./scripts/shared/injection-receipt.mjs";
import { writeOperationLog } from "./scripts/shared/operation-log.mjs";
import { PluginCore } from "./scripts/shared/plugin-core.mjs";
import { hostAdapter } from "./scripts/host-adapter.mjs";
import { TranscriptMirror, transcriptPath } from "./scripts/mirror.mjs";

const [, REMINDER_TAG] = INJECTION_TAGS;

export function userText(parts) {
  return (Array.isArray(parts) ? parts : [])
    .filter((part) => part?.type === "text" && typeof part.text === "string" && !part.synthetic && !part.ignored)
    .map((part) => part.text)
    .join(" ")
    .trim();
}

export async function HabitusMemoryPlugin({ client } = {}, { config = loadPluginConfig(), core, mirror } = {}) {
  if (!config.enabled) return {};
  const pluginCore = core || new PluginCore(config);
  const transcripts = mirror || new TranscriptMirror(config.stateRoot);
  const parents = new Map();

  const input = (sessionId, extra = {}) => ({
    session_id: sessionId,
    parent_id: parents.get(sessionId) || undefined,
    transcript_path: transcriptPath(config.stateRoot, sessionId),
    ...extra,
  });

  async function guarded(stage, sessionId, work) {
    try { return await work(); }
    catch (error) {
      await writeOperationLog(config, hostAdapter, input(sessionId), { hook: stage, stage, status: "error", retryable: true, error })
        .catch(() => {});
      return undefined;
    }
  }

  async function capture(sessionId, { flush = false } = {}) {
    const response = await client.session.messages({ path: { id: sessionId } });
    const messages = Array.isArray(response?.data) ? response.data : Array.isArray(response) ? response : [];
    await transcripts.mirror(sessionId, messages);
    await pluginCore.enqueueCapture(hostAdapter, input(sessionId));
    if (flush) await pluginCore.flush(hostAdapter, input(sessionId));
    else await pluginCore.drain(hostAdapter, input(sessionId));
  }

  return {
    "chat.message": async (hookInput, output) => {
      const sessionId = hookInput?.sessionID ?? output?.message?.sessionID;
      const messageId = hookInput?.messageID ?? output?.message?.id;
      const prompt = userText(output?.parts);
      if (!sessionId || !messageId || !prompt) return;
      await guarded("chat.message", sessionId, async () => {
        const blocks = [];
        const context = await pluginCore.recall(hostAdapter, input(sessionId, { prompt }));
        if (context) {
          const prepared = createContextInjection(context);
          await pluginCore.recordInjection(hostAdapter, input(sessionId), prepared.receipt);
          blocks.push(prepared.injection);
        }
        const reminders = await pluginCore.claimReminders(hostAdapter, input(sessionId, { prompt }));
        if (reminders.length > 0) {
          const block = createContextInjection(renderReminderInstruction(reminders, hostAdapter.questionTool), undefined, REMINDER_TAG);
          await pluginCore.recordInjection(hostAdapter, input(sessionId), block.receipt);
          blocks.push(block.injection);
        }
        if (blocks.length === 0) return;
        output.parts.unshift({
          id: `prt-habitus-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          type: "text",
          text: blocks.join("\n"),
          synthetic: true,
          sessionID: sessionId,
          messageID: messageId,
        });
      });
    },

    event: async ({ event } = {}) => {
      const type = event?.type;
      const properties = event?.properties ?? {};
      if (type === "session.created" || type === "session.updated") {
        const info = properties.info;
        if (info?.id && info?.parentID) parents.set(info.id, info.parentID);
        return;
      }
      if (type === "question.asked") { transcripts.asked(properties); return; }
      if (type === "question.replied") { transcripts.replied(properties); return; }
      const sessionId = properties.sessionID ?? properties.info?.id;
      if (!sessionId) return;
      if (type === "session.idle") await guarded("session.idle", sessionId, () => capture(sessionId));
      else if (type === "session.deleted") await guarded("session.deleted", sessionId, () => capture(sessionId, { flush: true }));
    },

    "experimental.session.compacting": async (hookInput) => {
      const sessionId = hookInput?.sessionID;
      if (sessionId) await guarded("session.compacting", sessionId, () => capture(sessionId, { flush: true }));
    },
  };
}

export default HabitusMemoryPlugin;
