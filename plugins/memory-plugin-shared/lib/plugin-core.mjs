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
import {
  DEFAULT_MAX_REQUEST_BYTES,
  REMEMBER_PARTS_FEATURE,
  REMINDERS_FEATURE,
  hasFeature,
  maxRequestBytes,
  requireCompatibleService,
} from "./compat.mjs";
import { HabitusServiceClient, errorText, rememberBody } from "./service-client.mjs";
import { requireInjectionReceipt } from "./injection-receipt.mjs";
import { stripMedia } from "./media.mjs";
import { sendRememberParts, splitRememberBody } from "./remember-parts.mjs";
import { replaceJsonDurably } from "./atomic-file.mjs";
import { writeOperationLog } from "./operation-log.mjs";
import { readFile, rm, stat } from "node:fs/promises";
import { makePrivateDirectory } from "./private-dir.mjs";
import { join, relative, isAbsolute } from "node:path";
import {
  listStateSessions,
  readState,
  removeState,
  stateModifiedAt,
  withSessionLock,
  writeState,
} from "./state-store.mjs";

const CLAIMED_REMINDERS_KEPT = 64;
// 注入回执平时留最近这么多条；读转写一直失败（captureFailure）时注入的还没被读过、剥过，一条不丢，读成之后再收回到这么多条
const INJECTION_RECEIPTS_KEPT = 32;
// 原样超过读取额度、跳过没读的转写行（jsonl.mjs）：会话状态里留最近这么多条给 doctor 列
const SKIPPED_LINES_KEPT = 16;
// 服务上次给出的请求体上限：入队时（Stop 的前台，不问服务）按它把一段增量在轮的边界拆成几批
const SERVICE_LIMITS_FILE = "service-limits.json";
const MiB = 1024 * 1024;

function utcDate() { return new Date().toISOString().slice(0, 10); }

async function pathExists(path) {
  try { await stat(path); return true; }
  catch (error) { if (error?.code === "ENOENT") return false; throw error; }
}

async function knownRequestLimit(stateRoot) {
  try {
    const value = JSON.parse(await readFile(join(stateRoot, SERVICE_LIMITS_FILE), "utf8"));
    return Number.isSafeInteger(value?.maxRequestBytes) && value.maxRequestBytes > 0 ? value.maxRequestBytes : DEFAULT_MAX_REQUEST_BYTES;
  } catch {
    return DEFAULT_MAX_REQUEST_BYTES;
  }
}

async function recordRequestLimit(stateRoot, limit) {
  if (await knownRequestLimit(stateRoot) === limit) return;
  await makePrivateDirectory(stateRoot);
  await replaceJsonDurably(join(stateRoot, SERVICE_LIMITS_FILE), { schemaVersion: 1, maxRequestBytes: limit });
}

function megabytes(bytes) {
  return `${(bytes / MiB).toFixed(1)} MiB`;
}

// 把这一段增量的各轮按请求体上限贪心拼成几批：一轮绝不拆开（裁定 69），每批都以一轮结束收尾；
// 单独一轮就超限的自成一批，送的时候把这一批的请求体分块上传（remember-parts.mjs）
function groupTurns(turns, envelopeBytes, limit) {
  const groups = [];
  let current = null;
  for (const turn of turns) {
    const sizes = turn.records.map((record) => Buffer.byteLength(JSON.stringify(record)));
    const turnBytes = sizes.reduce((sum, size) => sum + size, 0) + Math.max(0, sizes.length - 1);
    const separator = current != null && current.records.length > 0 && sizes.length > 0 ? 1 : 0;
    if (current != null && current.records.length > 0 && sizes.length > 0 && current.bytes + separator + turnBytes > limit) {
      groups.push(current);
      current = null;
    }
    if (current == null) current = { records: [], bytes: envelopeBytes, nextCursor: null };
    if (sizes.length > 0) current.bytes += (current.records.length > 0 ? 1 : 0) + turnBytes;
    current.records.push(...turn.records);
    current.nextCursor = turn.nextCursor;
  }
  if (current != null) groups.push(current);
  return groups;
}

// 送一批时要的本地服务口径：请求体上限，以及单独一轮超限时能不能分块上传
function deliveryLimits(capabilities) {
  return { limit: maxRequestBytes(capabilities), parts: hasFeature(capabilities, REMEMBER_PARTS_FEATURE) };
}

function occurredAt(input) {
  const candidate = input?.timestamp || input?.occurred_at;
  const parsed = candidate == null ? new Date() : new Date(candidate);
  return Number.isNaN(parsed.getTime()) ? new Date().toISOString() : parsed.toISOString();
}

// 召回 / 取提醒没成（不抛异常的那些）：原因记进这一轮的 failures，返回"什么都没有"（召回给空串、取提醒给空清单）。
// retryable 是服务回的那个（请求被拒、不可重试的照记成不可重试），没有服务回话的那几种（连不上、不兼容、预算用完）算可重试
function failed(turn, stage, reason, detail, nothing, retryable = true) {
  turn?.failures?.push({ stage, reason, detail: errorText(detail), retryable });
  return nothing;
}

// 一个请求没成的原因：连不上、超时（status 0）与回了非 2xx 分开记，一个是服务没起或不回话，一个是起了但出错
function requestFailure(response) {
  return response.status ? "request_failed" : "service_unavailable";
}

function failureDetail(response) {
  return response.status ? `HTTP ${response.status}: ${errorText(response.error)}` : errorText(response.error);
}

export class PluginCore {
  constructor(config, { service } = {}) {
    this.config = config;
    this.service = service || new HabitusServiceClient(config);
  }

  async sessionStart(adapterValue, input) {
    const adapter = requireHostAdapter(adapterValue);
    const compatibility = await this.#compatible(adapter);
    if (!compatibility.ok) throw new Error(`Habitus service unavailable: ${compatibility.error || "capabilities failed"}`);
    return this.#locked(adapter, input, (state) => state);
  }

  async recoverPending(adapterValue, input) {
    const adapter = requireHostAdapter(adapterValue);
    const compatibility = await this.#compatible(adapter);
    if (!compatibility.ok) return { retryable: true, reason: "service_incompatible", error: compatibility.error };
    const limits = deliveryLimits(compatibility.capabilities);
    const current = adapter.nativeSessionId(input);
    const pending = await listPendingSessions(this.config.stateRoot, adapter.host);
    const sessions = [current, ...pending.filter((value) => value !== current)];
    const results = [];
    for (const nativeSessionId of sessions) results.push(await this.#drainSession(adapter, nativeSessionId, limits));
    return { current: results[0], recoveredSessions: sessions.length };
  }

  async enqueueCapture(adapterValue, input) {
    const adapter = requireHostAdapter(adapterValue);
    const limit = await knownRequestLimit(this.config.stateRoot);
    return this.#locked(adapter, input, async (state) => {
      const queued = await listOutbox(this.config.stateRoot, adapter.host, state.nativeSessionId);
      let captureCursor = queued.length === 0 ? state.acknowledgedTranscriptCursor : queued.at(-1).transcriptEndCursor;
      let current = state;
      let enqueued = false;
      // 一次只读一段（maxTranscriptBytes），读到文件末尾为止；每段读完就落进 outbox，中途被杀下次从最后一批的结束处接着读
      while (true) {
        let delta;
        try {
          delta = await adapter.readTranscriptDelta(input, captureCursor, this.config, current.injectionReceipts);
        } catch (error) {
          // 读不出来（比如剥完媒体一轮仍超读取额度）每次 Stop 都会照样失败：记在会话状态里，doctor 列出来（错误照常往上抛、进操作日志）。
          // since 是这一串失败的第一次（从那以后没读成过），at 是最近一次
          const at = new Date().toISOString();
          const since = current.captureFailure?.since ?? current.captureFailure?.at ?? at;
          const captureFailure = { reason: String(error?.message || error).slice(0, 512), since, at };
          await writeState(this.config.stateRoot, adapter.host, current.nativeSessionId, { ...current, captureFailure }).catch(() => {});
          throw error;
        }
        if (delta == null) break;
        // 媒体载荷在入队前换成占位（media.mjs）：outbox 里与送出去的都不带它（readJsonlDelta 读的时候已剥过，这里对哪种适配器都兜住）
        const turns = (delta.turns ?? [{ records: delta.records, nextCursor: delta.nextCursor }])
          .map((turn) => ({ records: turn.records.map(stripMedia), nextCursor: turn.nextCursor }));
        if (turns.every((turn) => turn.records.length === 0)) {
          current = { ...current, acknowledgedTranscriptCursor: delta.nextCursor };
          await writeState(this.config.stateRoot, adapter.host, current.nativeSessionId, current);
        } else {
          await this.#enqueueTurns(adapter, current, input, delta, turns, limit);
          enqueued = true;
        }
        // 这一段已经落下（游标越过了这几行）之后再记：落下之前失败的话，下一次重读还会再碰到它们
        if (delta.skippedLines?.length > 0) current = await this.#recordSkippedLines(adapter, input, current, delta.skippedLines);
        captureCursor = delta.nextCursor;
        if (delta.more !== true) break;
      }
      if (current.captureFailure != null) {
        const { captureFailure: _resolved, ...rest } = current;
        current = { ...rest, injectionReceipts: current.injectionReceipts.slice(-INJECTION_RECEIPTS_KEPT) };
        await writeState(this.config.stateRoot, adapter.host, current.nativeSessionId, current);
      }
      return { state: current, enqueued };
    });
  }

  // 游标越过的、原样超过读取额度没读的转写行：每行记一条操作日志，会话状态里留最近几条（doctor 列出来），采集照常往下走。
  // 游标没越过的不在这里（jsonl.mjs 只报越过了的），同一行不会记两次
  async #recordSkippedLines(adapter, input, state, lines) {
    const at = new Date().toISOString();
    for (const line of lines) {
      await writeOperationLog(this.config, adapter, input, {
        hook: "capture",
        stage: "read_transcript",
        status: "skipped",
        retryable: false,
        reason: "line_exceeds_read_limit",
        error: new Error(`transcript line ${line.lineCount} (${line.bytes} bytes) exceeds the ${this.config.maxTranscriptBytes}-byte read limit and was skipped`),
      });
    }
    const skipped = [...(state.skippedTranscriptLines ?? []), ...lines.map((line) => ({ ...line, at }))].slice(-SKIPPED_LINES_KEPT);
    const next = { ...state, skippedTranscriptLines: skipped };
    await writeState(this.config.stateRoot, adapter.host, next.nativeSessionId, next);
    return next;
  }

  async #enqueueTurns(adapter, state, input, delta, turns, limit) {
    const at = occurredAt(input);
    const draft = {
      host: adapter.host,
      nativeSessionId: state.nativeSessionId,
      conversationId: state.conversationId,
      startedOn: state.startedOn,
      protocol: adapter.protocol,
      occurredAt: at,
    };
    // 请求体除去 records 的部分按最长的写法量（序号取最大、after_turn 取 false），拼出来的每批都不会少算
    const envelopeBytes = Buffer.byteLength(rememberBody({
      ...draft, deliveryId: "0".repeat(64), payload: { records: [] }, startSequence: Number.MAX_SAFE_INTEGER, afterTurn: false,
    }));
    const groups = groupTurns(turns, envelopeBytes, limit);
    let startCursor = delta.startCursor;
    for (const [index, group] of groups.entries()) {
      await enqueueDelta(this.config.stateRoot, {
        ...draft,
        payload: { records: group.records },
        afterTurn: index === groups.length - 1 ? delta.afterTurn : true,
        transcriptStartCursor: startCursor,
        transcriptEndCursor: group.nextCursor,
        createdAt: new Date().toISOString(),
      });
      startCursor = group.nextCursor;
    }
  }

  async drain(adapterValue, input) {
    const adapter = requireHostAdapter(adapterValue);
    const compatibility = await this.#compatible(adapter);
    if (!compatibility.ok) return { retryable: true, reason: "service_incompatible", error: compatibility.error };
    return this.#drainSession(adapter, adapter.nativeSessionId(input), deliveryLimits(compatibility.capabilities));
  }

  // 一次 UserPromptSubmit（召回 + 取提醒）共用的这一轮：capabilities 只问一次，所有请求落在同一个预算里（config.promptBudgetMs）。
  // 取提醒占预算的最后一段（reminderTimeoutMs，最多一半），召回这边（capabilities、ready、recall）用前面的部分。
  // failures 收这一轮召回 / 取提醒没成的原因（不是异常的那些：连不上、不兼容、未就绪、预算用完、非 2xx），调用方据此记操作日志
  beginPrompt() {
    const budgetMs = this.config.promptBudgetMs || 10_000;
    const reminderMs = Math.min(this.config.reminderTimeoutMs || 2_000, Math.floor(budgetMs / 2));
    const end = Date.now() + budgetMs;
    return { end, recallEnd: end - reminderMs, compatibility: null, failures: [] };
  }

  async recall(adapterValue, input, turn = null) {
    const adapter = requireHostAdapter(adapterValue);
    const prompt = adapter.prompt(input);
    if (!prompt) return "";
    const compatibility = await this.#compatible(adapter, turn);
    if (!compatibility.ok) return failed(turn, "recall", compatibility.reason, compatibility.error, "");
    const readyTimeout = this.#remaining(turn?.recallEnd, this.config.timeoutMs);
    if (readyTimeout === 0) return failed(turn, "recall", "prompt_budget_exhausted", "no time left for ready", "");
    const ready = await this.service.ready(readyTimeout);
    if (!ready.ok) return failed(turn, "recall", requestFailure(ready), ready.error, "", ready.retryable !== false);
    // 只看召回所需的就绪（recall_ready）：记忆写入侧停住（Job 失败待处置、在等模型恢复）时已有的记忆照样召回
    if (ready.result?.recall_ready !== true) return failed(turn, "recall", "service_not_ready", `status ${ready.result?.status}`, "");
    const state = await this.#locked(adapter, input, (value) => value);
    const recallTimeout = this.#remaining(turn?.recallEnd, this.config.timeoutMs);
    if (recallTimeout === 0) return failed(turn, "recall", "prompt_budget_exhausted", "no time left for recall", "");
    const response = await this.service.recall(prompt, state, recallTimeout);
    if (!response.ok) return failed(turn, "recall", requestFailure(response), failureDetail(response), "", response.retryable !== false);
    return typeof response.result?.context === "string" ? response.result.context : "";
  }

  async claimReminders(adapterValue, input, turn = null) {
    // 只在 UserPromptSubmit 取（裁定 64 第 9 条）；失败、超时、服务没这个特性一律当这一轮没有提醒。
    // 同一会话在租约内重取会拿回自己手里那几条（claimed_at 不变）：注入过的不再注入，免得 agent 每轮都问一遍。
    const adapter = requireHostAdapter(adapterValue);
    if (!adapter.prompt(input)) return [];
    const compatibility = await this.#compatible(adapter, turn);
    if (!compatibility.ok) return failed(turn, "claim_reminders", compatibility.reason, compatibility.error, []);
    if (!hasFeature(compatibility.capabilities, REMINDERS_FEATURE)) return [];
    const timeoutMs = this.#remaining(turn?.end, this.config.reminderTimeoutMs);
    if (timeoutMs === 0) return failed(turn, "claim_reminders", "prompt_budget_exhausted", "no time left for claiming reminders", []);
    const state = await this.#locked(adapter, input, (value) => value);
    const response = await this.service.claimReminders({
      host: adapter.host,
      session: state.conversationId,
      at: new Date().toISOString(),
      limit: this.config.reminderClaimLimit || 3,
    }, timeoutMs);
    if (!response.ok) {
      return failed(turn, "claim_reminders", requestFailure(response), failureDetail(response), [], response.retryable !== false);
    }
    const reminders = Array.isArray(response.result?.reminders) ? response.result.reminders : [];
    const usable = reminders.filter((item) => (
      item && typeof item.request_id === "string" && typeof item.text === "string" && typeof item.claimed_at === "string"
    ));
    if (usable.length === 0) return [];
    return this.#locked(adapter, input, async (current) => {
      const seen = new Set((current.remindersClaimed || []).map((item) => `${item.request_id}@${item.claimed_at}`));
      const fresh = usable.filter((item) => !seen.has(`${item.request_id}@${item.claimed_at}`));
      if (fresh.length === 0) return [];
      const remindersClaimed = [
        ...(current.remindersClaimed || []),
        ...fresh.map((item) => ({ request_id: item.request_id, claimed_at: item.claimed_at })),
      ].slice(-CLAIMED_REMINDERS_KEPT);
      await writeState(this.config.stateRoot, adapter.host, current.nativeSessionId, { ...current, remindersClaimed });
      return fresh;
    });
  }

  async recordInjection(adapterValue, input, receiptValue) {
    const adapter = requireHostAdapter(adapterValue);
    const receipt = requireInjectionReceipt(receiptValue);
    return this.#locked(adapter, input, async (state) => {
      const unique = new Map(state.injectionReceipts.map((item) => [item.nonce || item.digest, item]));
      unique.set(receipt.nonce, receipt);
      const receipts = [...unique.values()];
      const injectionReceipts = state.captureFailure == null ? receipts.slice(-INJECTION_RECEIPTS_KEPT) : receipts;
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

  // turn 是 beginPrompt 给的那一轮：同一轮里只问一次 capabilities，超时取预算剩下的；没有 turn（后台 worker）照常用 timeoutMs
  async #compatible(adapter, turn = null) {
    if (turn == null) return this.#negotiate(adapter);
    turn.compatibility ??= this.#negotiate(adapter, this.#remaining(turn.recallEnd, this.config.timeoutMs));
    return turn.compatibility;
  }

  async #negotiate(adapter, timeoutMs) {
    if (timeoutMs === 0) return { ok: false, retryable: true, reason: "prompt_budget_exhausted", error: "prompt budget exhausted" };
    const capabilities = await this.service.capabilities(timeoutMs);
    if (!capabilities.ok) return { ...capabilities, reason: requestFailure(capabilities) };
    try { requireCompatibleService(capabilities.result, adapter.protocol); }
    catch (error) { return { ok: false, retryable: true, reason: "service_incompatible", error: error?.message || String(error) }; }
    // 记下服务的请求体上限给入队用；记不下（盘满等）入队按默认上限拆，送的时候仍按服务这次给的核
    await recordRequestLimit(this.config.stateRoot, maxRequestBytes(capabilities.result)).catch(() => {});
    return { ok: true, capabilities: capabilities.result };
  }

  // 这一轮还剩多少毫秒给下一个请求（不超过它自己的超时）；没有截止时刻就用它自己的超时（undefined 走客户端默认）；用完了是 0
  #remaining(end, ceiling) {
    if (end == null) return ceiling;
    const left = end - Date.now();
    if (left <= 0) return 0;
    return ceiling ? Math.min(ceiling, left) : left;
  }

  async retireIdleSessions(adapterValue) {
    // 会话状态的退场（26 号清单 7.4）：宿主那边这个会话已经不在、该会话 outbox 为空、状态超过宽限期没动过，才删 state；
    // 锁目录在锁放掉时随之删掉，插件自己的转写镜像（状态目录 transcripts/ 下的）一并删掉。
    // "不在了"看宿主的原始数据：转写就是宿主文件的（Codex、Claude Code、pi）看那个文件；转写是插件镜像的（ZCode、DeepSeek Harness）
    // 由适配器的 sessionExists 去宿主的库 / 会话日志里查。老的状态没存 transcript 路径（或宿主会话 id）的不删（不知道它还在不在）。
    const adapter = requireHostAdapter(adapterValue);
    const retired = [];
    for (const nativeSessionId of await listStateSessions(this.config.stateRoot, adapter.host)) {
      try {
        const done = await withSessionLock(this.config.stateRoot, adapter.host, nativeSessionId, async () => {
          const state = await readState(this.config.stateRoot, adapter.host, nativeSessionId);
          if (!state?.transcriptPath) return false;
          if ((await listOutbox(this.config.stateRoot, adapter.host, nativeSessionId)).length > 0) return false;
          const modified = await stateModifiedAt(this.config.stateRoot, adapter.host, nativeSessionId);
          if (modified == null || Date.now() - modified < this.config.sessionRetireGraceMs) return false;
          if (!(await this.#hostSessionGone(adapter, state))) return false;
          await removeState(this.config.stateRoot, adapter.host, nativeSessionId);
          if (this.#isMirror(state.transcriptPath)) await rm(state.transcriptPath, { force: true });
          return true;
        });
        if (done) retired.push(nativeSessionId);
      } catch {
        // 某个会话的锁一时拿不到（正在用）或状态、宿主数据读不了：这次跳过，下次退场时再看
      }
    }
    return { retired };
  }

  async #hostSessionGone(adapter, state) {
    if (typeof adapter.sessionExists === "function") return !(await adapter.sessionExists(state));
    return !(await pathExists(state.transcriptPath));
  }

  // 插件自己写的转写镜像（parts-transcript.mjs、dsh 的 session-log.mjs）都在状态目录的 transcripts/ 下；宿主自己的文件一概不碰
  #isMirror(path) {
    const inside = relative(join(this.config.stateRoot, "transcripts"), path);
    return Boolean(inside) && !inside.startsWith("..") && !isAbsolute(inside);
  }

  async #locked(adapter, input, callback) {
    const nativeSessionId = adapter.nativeSessionId(input);
    const transcriptPath = typeof adapter.transcriptPath === "function" ? adapter.transcriptPath(input) : null;
    const hostSessionId = typeof adapter.hostSessionId === "function" ? adapter.hostSessionId(input) : null;
    return this.#lockedSession(adapter, nativeSessionId, () => adapter.conversationId(input), async (state) => {
      // 记下 transcript 路径与宿主原样的会话 id（由宿主适配器给）：会话状态退场时要看宿主那边它还在不在（26 号清单 7.4）
      const recorded = {};
      if (typeof transcriptPath === "string" && transcriptPath && state.transcriptPath !== transcriptPath) recorded.transcriptPath = transcriptPath;
      if (typeof hostSessionId === "string" && hostSessionId && state.hostSessionId !== hostSessionId) recorded.hostSessionId = hostSessionId;
      if (Object.keys(recorded).length > 0) {
        state = { ...state, ...recorded };
        await writeState(this.config.stateRoot, adapter.host, nativeSessionId, state);
      }
      return callback(state);
    });
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

  async #drainSession(adapter, nativeSessionId, { limit, parts: partsSupported }) {
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
          // 单独一轮仍超过本地服务的请求体上限：本地服务有 remember_parts_v1 就把这一批的请求体分块上传；
          // 没有就不送（送了也是不可重试的 413），标 blocked 并写明这一批多大、超了多少
          const body = rememberBody(item);
          const bytes = Buffer.byteLength(body);
          let parts = null;
          if (bytes > limit) {
            if (!partsSupported) {
              await blockOutboxItem(
                this.config.stateRoot,
                item,
                `request_too_large: this batch is ${bytes} bytes (${megabytes(bytes)}), over the local service limit of ${limit} bytes (${megabytes(limit)}) by ${bytes - limit} bytes; a single turn is never split, and this local service has no ${REMEMBER_PARTS_FEATURE} to upload it in parts`,
              );
              return { kind: "blocked", state, item };
            }
            parts = splitRememberBody(item.deliveryId, body, limit);
          }
          const claimToken = randomUUID();
          // 分块上传要送几次请求：租约按次数放长，免得送到一半被另一个进程认领走
          const timeoutMs = this.config.timeoutMs || 15_000;
          const leaseMs = Math.max(30_000, timeoutMs * 3) + timeoutMs * Math.max(0, (parts?.length ?? 1) - 1);
          const claimed = await claimOutboxItem(this.config.stateRoot, item, claimToken, leaseMs);
          return { kind: "deliver", state, item: claimed, claimToken, parts };
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
      const response = decision.parts == null
        ? await this.service.remember(decision.item)
        : await sendRememberParts(this.service, decision.parts);
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
