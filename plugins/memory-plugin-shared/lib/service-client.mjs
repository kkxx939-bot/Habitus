import { rememberPartBody } from "./remember-parts.mjs";

// 服务的错误体是 {code, message, retryable}：记日志、写 outbox 时转成一行字（直接 String() 只得到 "[object Object]"）
export function errorText(error) {
  if (error == null) return "";
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  if (typeof error === "object" && typeof error.message === "string") {
    return error.code ? `${error.code}: ${error.message}` : error.message;
  }
  try { return JSON.stringify(error); } catch { return String(error); }
}

// remember 的请求体：送的与按服务上限量大小的是同一份字节
export function rememberBody(item) {
  return JSON.stringify({
    delivery_id: item.deliveryId,
    conversation_id: item.conversationId,
    started_on: item.startedOn,
    protocol: item.protocol,
    payload: item.payload,
    start_sequence: item.startSequence,
    occurred_at: item.occurredAt,
    after_turn: item.afterTurn,
  });
}

function retryableStatus(status) {
  return !status || status === 408 || status === 429 || status >= 500;
}

export class HabitusServiceClient {
  constructor(config, fetchImpl = globalThis.fetch) {
    if (typeof fetchImpl !== "function") throw new TypeError("fetch implementation is required");
    this.config = config;
    this.fetchImpl = fetchImpl;
  }

  // reportedStatuses：契约里仍回 SuccessResponse 的非 2xx 状态码（/ready 未就绪的 503），按成功读出 result
  async request(path, init = {}, timeoutMs = this.config.timeoutMs, { reportedStatuses = [] } = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      // 不跟随重定向：只认 baseUrl 这个本机地址（config.mjs），回 3xx 当请求失败，转写不会跟着被送到别处
      const response = await this.fetchImpl(`${this.config.baseUrl}${path}`, {
        ...init,
        headers: { "Content-Type": "application/json", ...(init.headers || {}) },
        redirect: "error",
        signal: controller.signal,
      });
      const body = await response.json().catch(() => null);
      if (!body || typeof body !== "object") {
        return { ok: false, status: response.status, retryable: true, error: "invalid JSON response" };
      }
      const reported = reportedStatuses.includes(response.status) && body.status === "ok";
      if ((!response.ok && !reported) || body.status === "error") {
        const retryable = body?.error?.retryable;
        return {
          ok: false,
          status: response.status,
          retryable: typeof retryable === "boolean" ? retryable : retryableStatus(response.status),
          error: body.error || body,
        };
      }
      return { ok: true, status: response.status, result: body.result };
    } catch (error) {
      return { ok: false, status: 0, retryable: true, error: error?.message || String(error) };
    } finally {
      clearTimeout(timeout);
    }
  }

  health() { return this.request("/health"); }
  // 未就绪时回 503 + HealthResult（ready: false 与各项检查），调用方按 result.ready 判
  ready(timeoutMs) { return this.request("/ready", {}, timeoutMs, { reportedStatuses: [503] }); }
  capabilities(timeoutMs) { return this.request("/api/v1/capabilities", {}, timeoutMs); }

  cursor(conversationId, startedOn) {
    const query = new URLSearchParams({ conversation_id: conversationId, started_on: startedOn });
    return this.request(`/api/v1/memory/conversations/cursor?${query}`);
  }

  remember(item) {
    return this.request("/api/v1/memory/remember", { method: "POST", body: rememberBody(item) });
  }

  // 单独一轮超过请求体上限时分块上传的一块（remember-parts.mjs）
  rememberPart(part) {
    return this.request("/api/v1/memory/remember/parts", { method: "POST", body: rememberPartBody(part) });
  }

  recall(query, state, timeoutMs) {
    return this.request("/api/v1/memory/recall", {
      method: "POST",
      body: JSON.stringify({
        query,
        conversation_id: state.conversationId,
        started_on: state.startedOn,
      }),
    }, timeoutMs);
  }

  flush(state) {
    return this.request("/api/v1/memory/flush", {
      method: "POST",
      body: JSON.stringify({ conversation_id: state.conversationId, started_on: state.startedOn }),
    });
  }

  // 提醒出口的取口（方案 32 第 4.1 条）：session 就是送转写时的 conversation_id；单独的短超时。
  claimReminders({ host, session, at, limit }, timeoutMs = this.config.reminderTimeoutMs || 2_000) {
    return this.request(
      "/api/v1/reminders/claim",
      { method: "POST", body: JSON.stringify({ host, session, at, limit }) },
      timeoutMs,
    );
  }
}
