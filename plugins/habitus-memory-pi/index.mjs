// Habitus 的 pi 扩展（方案 36 第四节）：人说话时召回记忆、取提醒，作为一条不显示的 custom 消息排在他这句话后面
// （转写里是 role:"custom"，不算他说的话）；一轮跑完、压缩前、会话结束时把 pi 自己的会话 JSONL 经共享 outbox 送回本地
// Habitus 服务（协议 pi）。另注册提问工具 ask_user（pi 没有内置的）。
// 没有会话文件的进程（-p --no-session，比如子代理示例另起的 pi）不是他在说话，什么都不做。失败一律静默放行，不拦 pi。

import { loadPluginConfig } from "./scripts/shared/config.mjs";
import { promptInjection } from "./scripts/shared/hook-runner.mjs";
import { writeOperationLog } from "./scripts/shared/operation-log.mjs";
import { PluginCore } from "./scripts/shared/plugin-core.mjs";
import { hostAdapter } from "./scripts/host-adapter.mjs";
import { questionTool } from "./scripts/question-tool.mjs";

const CUSTOM_TYPE = "habitus-memory";

export default function HabitusMemoryExtension(pi, { config: given, core } = {}) {
  let config;
  try { config = given ?? loadPluginConfig(); }
  catch (error) {
    // 配置读不了（连接文件权限、HABITUS_URL 不是本机）不能把 pi 的扩展加载拖垮：这次不接，说一声
    process.stderr.write(`habitus-memory: extension disabled: ${error?.message || error}\n`);
    return;
  }
  if (!config.enabled) return;
  const pluginCore = core || new PluginCore(config);
  pi.registerTool(questionTool());

  // 这一刻的会话：没有会话文件就是不落盘的进程，给 null
  function input(ctx, extra = {}) {
    const transcript = ctx?.sessionManager?.getSessionFile?.();
    const sessionId = ctx?.sessionManager?.getSessionId?.();
    if (!transcript || !sessionId) return null;
    return { session_id: sessionId, transcript_path: transcript, ...extra };
  }

  async function guarded(stage, value, work) {
    try { return await work(); }
    catch (error) {
      await writeOperationLog(config, hostAdapter, value, { hook: stage, stage, status: "error", retryable: true, error })
        .catch(() => {});
      return undefined;
    }
  }

  // 别的会话留在 outbox 里没送出去的（服务当时没起），本进程第一次送的时候一并补送
  let recovered = false;

  async function capture(value, { flush = false } = {}) {
    await pluginCore.enqueueCapture(hostAdapter, value);
    if (flush) return pluginCore.flush(hostAdapter, value);
    if (recovered) return pluginCore.drain(hostAdapter, value);
    const result = await pluginCore.recoverPending(hostAdapter, value);
    recovered = !result?.retryable;
    return result;
  }

  pi.on("before_agent_start", async (event, ctx) => {
    if (event?.messageType !== "user") return undefined;
    const prompt = typeof event.prompt === "string" ? event.prompt.trim() : "";
    const value = input(ctx, { prompt });
    if (!value || !prompt) return undefined;
    const injection = await guarded("before_agent_start", value, () => (
      promptInjection(pluginCore, hostAdapter, value, { config, hook: "before_agent_start" })
    ));
    return injection ? { message: { customType: CUSTOM_TYPE, content: injection, display: false } } : undefined;
  });

  pi.on("agent_settled", async (_event, ctx) => {
    const value = input(ctx);
    if (value) await guarded("agent_settled", value, () => capture(value));
  });

  pi.on("session_before_compact", async (_event, ctx) => {
    const value = input(ctx);
    if (value) await guarded("session_before_compact", value, () => capture(value, { flush: true }));
    return undefined; // 不拦压缩
  });

  pi.on("session_shutdown", async (_event, ctx) => {
    const value = input(ctx);
    if (!value) return;
    await guarded("session_shutdown", value, async () => {
      const flushed = await capture(value, { flush: true });
      if (flushed?.ok) await pluginCore.retireIdleSessions(hostAdapter);
    });
  });
}
