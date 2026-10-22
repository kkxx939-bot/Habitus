// Habitus 的 OpenCode 插件（方案 36 第四节）：人说话时注入召回的记忆与"先问他"的提醒指令块（合成段，不算他说的话）；
// 会话空闲 / 压缩 / 删除时把已完成的消息镜像成本地转写、经共享 outbox 送回本地 Habitus 服务（协议 opencode）。
// 失败一律静默放行，不拦 OpenCode。

import { loadPluginConfig } from "./scripts/shared/config.mjs";
import { promptInjection } from "./scripts/shared/hook-runner.mjs";
import { writeOperationLog } from "./scripts/shared/operation-log.mjs";
import { PluginCore } from "./scripts/shared/plugin-core.mjs";
import { hostAdapter } from "./scripts/host-adapter.mjs";
import { TranscriptMirror, transcriptPath } from "./scripts/shared/parts-transcript.mjs";

export function userText(parts) {
  return (Array.isArray(parts) ? parts : [])
    .filter((part) => part?.type === "text" && typeof part.text === "string" && !part.synthetic && !part.ignored)
    .map((part) => part.text)
    .join(" ")
    .trim();
}

export async function HabitusMemoryPlugin({ client } = {}, { config: given, core, mirror } = {}) {
  let config;
  try { config = given ?? loadPluginConfig(); }
  catch (error) {
    // 配置读不了（连接文件权限、HABITUS_URL 不是本机）不能把 OpenCode 的插件加载拖垮：这次不接，说一声
    process.stderr.write(`habitus-memory: plugin disabled: ${error?.message || error}\n`);
    return {};
  }
  if (!config.enabled) return {};
  const pluginCore = core || new PluginCore(config);
  const transcripts = mirror || new TranscriptMirror(config.stateRoot, hostAdapter.host);
  const parents = new Map();

  const input = (sessionId, extra = {}) => ({
    session_id: sessionId,
    parent_id: parents.get(sessionId) || undefined,
    transcript_path: transcriptPath(config.stateRoot, sessionId, hostAdapter.host),
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

  // 别的会话留在 outbox 里没送出去的（服务当时没起、会话之后再没动静），本进程第一次送的时候一并补送
  let recovered = false;

  // 子会话（task 工具起的）认父会话：先看事件里记下的，没有（插件进程重启过）就问 OpenCode 一次并记住。
  // 消息上的 parentID 是它回应的那条消息，不是父会话，不能拿来用
  async function resolveParent(sessionId) {
    if (parents.has(sessionId)) return parents.get(sessionId);
    let parent = null;
    try {
      const response = await client.session.get({ path: { id: sessionId } });
      const value = (response?.data ?? response)?.parentID;
      parent = typeof value === "string" && value ? value : null;
    } catch {
      return null; // 问不到这次先按主会话算，下次再问
    }
    parents.set(sessionId, parent);
    return parent;
  }

  async function capture(sessionId, { flush = false, fetch = true } = {}) {
    await resolveParent(sessionId);
    if (fetch) {
      const response = await client.session.messages({ path: { id: sessionId } });
      const messages = Array.isArray(response?.data) ? response.data : Array.isArray(response) ? response : [];
      await transcripts.mirror(sessionId, messages);
    }
    await pluginCore.enqueueCapture(hostAdapter, input(sessionId));
    if (flush) return pluginCore.flush(hostAdapter, input(sessionId));
    if (recovered) return pluginCore.drain(hostAdapter, input(sessionId));
    const result = await pluginCore.recoverPending(hostAdapter, input(sessionId));
    recovered = !result?.retryable;
    return result;
  }

  // 会话删了：最后一批送完、服务确认收尾后，镜像文件和过了宽限期的会话状态都清掉
  async function close(sessionId) {
    // session.deleted 发出来时 OpenCode 已经删了消息（拉不到），只送已经镜像下来的
    const flushed = await capture(sessionId, { flush: true, fetch: false });
    if (!flushed?.ok) return;
    await transcripts.forget(sessionId);
    parents.delete(sessionId);
    await pluginCore.retireIdleSessions(hostAdapter);
  }

  return {
    "chat.message": async (hookInput, output) => {
      const sessionId = hookInput?.sessionID ?? output?.message?.sessionID;
      const messageId = hookInput?.messageID ?? output?.message?.id;
      const prompt = userText(output?.parts);
      if (!sessionId || !messageId || !prompt) return;
      await guarded("chat.message", sessionId, async () => {
        // 子会话的 prompt 是主 agent 写的任务，不是他说的话：不召回、不取提醒（子代理也没有提问工具）
        if (await resolveParent(sessionId)) return;
        const injection = await promptInjection(pluginCore, hostAdapter, input(sessionId, { prompt }), { config, hook: "chat.message" });
        if (!injection) return;
        output.parts.unshift({
          id: `prt-habitus-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          type: "text",
          text: injection,
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
        if (info?.id) parents.set(info.id, info.parentID || null);
        return;
      }
      const sessionId = properties.sessionID ?? properties.info?.id;
      if (!sessionId) return;
      if (type === "session.idle") await guarded("session.idle", sessionId, () => capture(sessionId));
      else if (type === "session.deleted") await guarded("session.deleted", sessionId, () => close(sessionId));
    },

    "experimental.session.compacting": async (hookInput) => {
      const sessionId = hookInput?.sessionID;
      if (sessionId) await guarded("session.compacting", sessionId, () => capture(sessionId, { flush: true }));
    },
  };
}

export default HabitusMemoryPlugin;
