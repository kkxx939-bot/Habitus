// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
import { randomUUID } from "node:crypto";
import { requireHostAdapter } from "./host-adapter.mjs";
import {
  acknowledgeOutboxItem,
  bindOutboxItem,
  blockOutboxItem,
  claimIsActive,
  claimOutboxItem,
  enqueueDelta,
  listOutbox,
  listPendingSessions,
  releaseOutboxItem,
} from "./outbox.mjs";
import { M2BOSServiceClient } from "./service-client.mjs";
import { requireInjectionReceipt } from "./injection-receipt.mjs";
import { readState, withSessionLock, writeState } from "./state-store.mjs";

const REQUIRED_FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];

function utcDate() { return new Date().toISOString().slice(0, 10); }

function occurredAt(input) {
  const candidate = input?.timestamp || input?.occurred_at;
  const parsed = candidate == null ? new Date() : new Date(candidate);
  return Number.isNaN(parsed.getTime()) ? new Date().toISOString() : parsed.toISOString();
}

export class PluginCore {
  constructor(config, { service } = {}) {
    this.config = config;
    this.service = service || new M2BOSServiceClient(config);
  }

  async sessionStart(adapterValue, input) {
    const adapter = requireHostAdapter(adapterValue);
    const compatibility = await this.#compatible(adapter);
    if (!compatibility.ok) throw new Error(`m2bOS service unavailable: ${compatibility.error || "capabilities failed"}`);
    return this.#locked(adapter, input, (state) => state);
  }

  async recoverPending(adapterValue, input) {
    const adapter = requireHostAdapter(adapterValue);
    const compatibility = await this.#compatible(adapter);
    if (!compatibility.ok) return { retryable: true, reason: "service_incompatible", error: compatibility.error };
    const current = adapter.nativeSessionId(input);
    const pending = await listPendingSessions(this.config.stateRoot, adapter.host);
    const sessions = [current, ...pending.filter((value) => value !== current)];
    const results = [];
    for (const nativeSessionId of sessions) results.push(await this.#drainSession(adapter, nativeSessionId));
    return { current: results[0], recoveredSessions: sessions.length };
  }

  async enqueueCapture(adapterValue, input) {
    const adapter = requireHostAdapter(adapterValue);
    return this.#locked(adapter, input, async (state) => {
      const queued = await listOutbox(this.config.stateRoot, adapter.host, state.nativeSessionId);
      const captureCursor = queued.length === 0
        ? state.acknowledgedTranscriptCursor
        : queued.at(-1).transcriptEndCursor;
      const delta = await adapter.readTranscriptDelta(input, captureCursor, this.config, state.injectionReceipts);
      if (delta == null) return { state, enqueued: false };
      if (delta.records.length === 0) {
        const advanced = { ...state, acknowledgedTranscriptCursor: delta.nextCursor };
        await writeState(this.config.stateRoot, adapter.host, state.nativeSessionId, advanced);
        return { state: advanced, enqueued: false };
      }
      await enqueueDelta(this.config.stateRoot, {
        host: adapter.host,
        nativeSessionId: state.nativeSessionId,
        conversationId: state.conversationId,
        startedOn: state.startedOn,
        protocol: adapter.protocol,
        payload: { records: delta.records },
        occurredAt: occurredAt(input),
        afterTurn: delta.afterTurn,
        transcriptStartCursor: delta.startCursor,
        transcriptEndCursor: delta.nextCursor,
        createdAt: new Date().toISOString(),
      });
      return { state, enqueued: true };
    });
  }

  async drain(adapterValue, input) {
    const adapter = requireHostAdapter(adapterValue);
    const compatibility = await this.#compatible(adapter);
    if (!compatibility.ok) return { retryable: true, reason: "service_incompatible", error: compatibility.error };
    return this.#drainSession(adapter, adapter.nativeSessionId(input));
  }

  async recall(adapterValue, input) {
    const adapter = requireHostAdapter(adapterValue);
    const prompt = adapter.prompt(input);
    if (!prompt) return "";
    const compatibility = await this.#compatible(adapter);
    if (!compatibility.ok) return "";
    const ready = await this.service.ready();
    if (!ready.ok || ready.result?.ready !== true) return "";
    const state = await this.#locked(adapter, input, (value) => value);
    const response = await this.service.recall(prompt, state);
    return response.ok && typeof response.result?.context === "string" ? response.result.context : "";
  }

  async recordInjection(adapterValue, input, receiptValue) {
    const adapter = requireHostAdapter(adapterValue);
    const receipt = requireInjectionReceipt(receiptValue);
    return this.#locked(adapter, input, async (state) => {
      const unique = new Map(state.injectionReceipts.map((item) => [item.nonce || item.digest, item]));
      unique.set(receipt.nonce, receipt);
      const injectionReceipts = [...unique.values()].slice(-32);
      const updated = { ...state, injectionReceipts };
      await writeState(this.config.stateRoot, adapter.host, state.nativeSessionId, updated);
      return updated;
    });
  }

  async flush(adapterValue, input) {
    const adapter = requireHostAdapter(adapterValue);
    const drained = await this.drain(adapter, input);
    const state = await this.#locked(adapter, input, (value) => value);
    const remaining = await listOutbox(this.config.stateRoot, adapter.host, state.nativeSessionId);
    if (remaining.length > 0) return { ok: false, retryable: true, reason: "outbox_not_empty", drained };
    return this.service.flush(state);
  }

  async #compatible(adapter) {
    const capabilities = await this.service.capabilities();
    if (!capabilities.ok) return capabilities;
    try { requireCompatibleService(capabilities.result, adapter.protocol); return { ok: true }; }
    catch (error) { return { ok: false, retryable: true, error: error?.message || String(error) }; }
  }

  async #locked(adapter, input, callback) {
    const nativeSessionId = adapter.nativeSessionId(input);
    return this.#lockedSession(adapter, nativeSessionId, () => adapter.conversationId(input), callback);
  }

  async #lockedSession(adapter, nativeSessionId, conversationId, callback) {
    return withSessionLock(this.config.stateRoot, adapter.host, nativeSessionId, async () => {
      let state = await readState(this.config.stateRoot, adapter.host, nativeSessionId);
      if (state == null) {
        state = {
          nativeSessionId,
          conversationId: conversationId(),
          startedOn: utcDate(),
          protocol: adapter.protocol,
          nextSequence: 0,
          acknowledgedTranscriptCursor: null,
          injectionReceipts: [],
        };
        await writeState(this.config.stateRoot, adapter.host, nativeSessionId, state);
      }
      return callback(state);
    });
  }

  async #drainSession(adapter, nativeSessionId) {
    let drained = 0;
    while (true) {
      const decision = await this.#lockedSession(
        adapter,
        nativeSessionId,
        () => { throw new Error("cannot reconstruct a missing state for an orphaned outbox"); },
        async (state) => {
          const items = await listOutbox(this.config.stateRoot, adapter.host, nativeSessionId);
          const item = items[0];
          if (item == null) return { kind: "done", state };
          if (item.status === "blocked") return { kind: "blocked", state, item };
          if (item.status === "queued") return { kind: "cursor", state, item };
          if (item.status === "inflight" && claimIsActive(item)) return { kind: "busy", state, item };
          const claimToken = randomUUID();
          const leaseMs = Math.max(30_000, (this.config.timeoutMs || 15_000) * 3);
          const claimed = await claimOutboxItem(this.config.stateRoot, item, claimToken, leaseMs);
          return { kind: "deliver", state, item: claimed, claimToken };
        },
      );
      if (decision.kind === "done") return { state: decision.state, drained };
      if (decision.kind === "blocked") return { state: decision.state, blocked: decision.item.id, drained };
      if (decision.kind === "busy") return { state: decision.state, pending: decision.item.id, retryable: true, drained };
      if (decision.kind === "cursor") {
        const cursor = await this.service.cursor(decision.state.conversationId, decision.state.startedOn);
        if (!cursor.ok) return { state: decision.state, pending: decision.item.id, retryable: cursor.retryable, drained };
        const nextSequence = cursor.result?.next_sequence;
        if (!Number.isSafeInteger(nextSequence) || nextSequence < 0) {
          await this.#blockCurrent(adapter, nativeSessionId, decision.item.id, "service returned an invalid cursor");
          return { state: decision.state, blocked: decision.item.id, drained };
        }
        if (nextSequence < decision.state.nextSequence) {
          await this.#blockCurrent(adapter, nativeSessionId, decision.item.id, "service cursor moved behind acknowledged plugin state");
          return { state: decision.state, blocked: decision.item.id, drained };
        }
        await this.#lockedSession(adapter, nativeSessionId, () => "", async (state) => {
          const current = (await listOutbox(this.config.stateRoot, adapter.host, nativeSessionId))[0];
          if (current?.id !== decision.item.id || current.status !== "queued") return;
          await bindOutboxItem(this.config.stateRoot, current, nextSequence);
          if (nextSequence !== state.nextSequence) {
            await writeState(this.config.stateRoot, adapter.host, nativeSessionId, { ...state, nextSequence });
          }
        });
        continue;
      }
      const response = await this.service.remember(decision.item);
      const result = await this.#lockedSession(adapter, nativeSessionId, () => "", async (state) => {
        const current = (await listOutbox(this.config.stateRoot, adapter.host, nativeSessionId))[0];
        if (current?.id !== decision.item.id || current.claimToken !== decision.claimToken) {
          return { kind: "superseded", state };
        }
        if (!response.ok) {
          if (!response.retryable) {
            await blockOutboxItem(this.config.stateRoot, current, response.error, decision.claimToken);
            return { kind: "blocked", state };
          }
          await releaseOutboxItem(this.config.stateRoot, current, decision.claimToken);
          return { kind: "retry", state };
        }
        const nextSequence = response.result?.next_sequence;
        if (!Number.isSafeInteger(nextSequence) || nextSequence <= current.startSequence) {
          await blockOutboxItem(
            this.config.stateRoot,
            current,
            "service returned an invalid remember acknowledgement",
            decision.claimToken,
          );
          return { kind: "blocked", state };
        }
        const updated = {
          ...state,
          nextSequence,
          acknowledgedTranscriptCursor: current.transcriptEndCursor,
        };
        await writeState(this.config.stateRoot, adapter.host, nativeSessionId, updated);
        await acknowledgeOutboxItem(this.config.stateRoot, current, decision.claimToken);
        return { kind: "ack", state: updated };
      });
      if (result.kind === "ack") { drained += 1; continue; }
      if (result.kind === "blocked") return { state: result.state, blocked: decision.item.id, drained };
      return { state: result.state, pending: decision.item.id, retryable: true, drained };
    }
  }

  async #blockCurrent(adapter, nativeSessionId, itemId, reason) {
    await this.#lockedSession(adapter, nativeSessionId, () => "", async () => {
      const current = (await listOutbox(this.config.stateRoot, adapter.host, nativeSessionId))[0];
      if (current?.id === itemId) await blockOutboxItem(this.config.stateRoot, current, reason);
    });
  }
}

function requireCompatibleService(capabilities, protocol) {
  if (capabilities?.api_version !== "1.0") throw new Error(`unsupported m2bOS API version: ${capabilities?.api_version}`);
  if (!Array.isArray(capabilities.protocols) || !capabilities.protocols.includes(protocol)) {
    throw new Error(`m2bOS service does not support protocol: ${protocol}`);
  }
  const features = new Set(Array.isArray(capabilities.features) ? capabilities.features : []);
  const missing = REQUIRED_FEATURES.filter((feature) => !features.has(feature));
  if (missing.length > 0) throw new Error(`m2bOS service is missing features: ${missing.join(", ")}`);
}
