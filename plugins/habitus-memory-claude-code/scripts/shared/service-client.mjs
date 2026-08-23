// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
function retryableStatus(status) {
  return !status || status === 408 || status === 429 || status >= 500;
}

export class HabitusServiceClient {
  constructor(config, fetchImpl = globalThis.fetch) {
    if (typeof fetchImpl !== "function") throw new TypeError("fetch implementation is required");
    this.config = config;
    this.fetchImpl = fetchImpl;
  }

  async request(path, init = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.config.timeoutMs);
    try {
      const response = await this.fetchImpl(`${this.config.baseUrl}${path}`, {
        ...init,
        headers: { "Content-Type": "application/json", ...(init.headers || {}) },
        signal: controller.signal,
      });
      const body = await response.json().catch(() => null);
      if (!body || typeof body !== "object") {
        return { ok: false, status: response.status, retryable: true, error: "invalid JSON response" };
      }
      if (!response.ok || body.status === "error") {
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
  ready() { return this.request("/ready"); }
  capabilities() { return this.request("/api/v1/capabilities"); }

  cursor(conversationId, startedOn) {
    const query = new URLSearchParams({ conversation_id: conversationId, started_on: startedOn });
    return this.request(`/api/v1/memory/conversations/cursor?${query}`);
  }

  remember(item) {
    return this.request("/api/v1/memory/remember", {
      method: "POST",
      body: JSON.stringify({
        delivery_id: item.deliveryId,
        conversation_id: item.conversationId,
        started_on: item.startedOn,
        protocol: item.protocol,
        payload: item.payload,
        start_sequence: item.startSequence,
        occurred_at: item.occurredAt,
        after_turn: item.afterTurn,
      }),
    });
  }

  recall(query, state) {
    return this.request("/api/v1/memory/recall", {
      method: "POST",
      body: JSON.stringify({
        query,
        conversation_id: state.conversationId,
        started_on: state.startedOn,
      }),
    });
  }

  flush(state) {
    return this.request("/api/v1/memory/flush", {
      method: "POST",
      body: JSON.stringify({ conversation_id: state.conversationId, started_on: state.startedOn }),
    });
  }
}
