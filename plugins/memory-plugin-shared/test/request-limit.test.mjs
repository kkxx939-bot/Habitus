// 一批送给本地服务之前：媒体载荷换成占位（media.mjs）、超过请求体上限的在轮的边界拆成几个请求；单独一轮仍超限的，
// 本地服务有 remember_parts_v1 就把这一批的请求体分块上传、拼回后按原样处理（裁定 72 第 2 条），没有就标 blocked
// 并让 doctor 说清是哪个会话、多大、超了多少。
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { hostAdapter as claude } from "../../habitus-memory-claude-code/scripts/host-adapter.mjs";
import { hostAdapter as codex } from "../../habitus-memory/scripts/host-adapter.mjs";
import { runPluginDoctor } from "../doctor.mjs";
import { stripMedia } from "../lib/media.mjs";
import { listOutbox } from "../lib/outbox.mjs";
import { PluginCore } from "../lib/plugin-core.mjs";
import { rememberPartBody, sendRememberParts } from "../lib/remember-parts.mjs";
import { rememberBody } from "../lib/service-client.mjs";
import { readState } from "../lib/state-store.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];
const PARTS_FEATURES = [...FEATURES, "remember_parts_v1"];
const MiB = 1024 * 1024;

async function temporaryRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "habitus-request-limit-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

// 像真本地服务一样：请求体超过上限回 413（不可重试）；同一投递号重投不重复记（remember_idempotency_v1）；
// 有 remember_parts_v1 时按投递号在内存里暂存各块，第 0 块重新开始，接不上的回 ABORTED（可重试，从第 0 块重送），
// 收齐、核对字节数与摘要后拼回请求体按 remember 处理。restart() 像本地服务重启：暂存丢掉，已记下的还在
function service({ maxRequestBytes, parts = false } = {}) {
  const limit = maxRequestBytes ?? 16 * MiB;
  const tooLarge = (body) => (Buffer.byteLength(body) > limit
    ? { ok: false, status: 413, retryable: false, error: { code: "REQUEST_TOO_LARGE", message: `Request body exceeds the ${limit}-byte limit` } }
    : null);
  const accept = (body) => {
    const request = JSON.parse(body);
    if (!remote.delivered.has(request.delivery_id)) {
      remote.delivered.add(request.delivery_id);
      remote.bodies.push(body);
      remote.sequence = request.start_sequence + request.payload.records.length;
    }
    return { ok: true, result: { next_sequence: remote.sequence } };
  };
  const remote = {
    bodies: [],
    parts: [],
    delivered: new Set(),
    staging: new Map(),
    sequence: 0,
    restart() { remote.staging.clear(); },
    capabilities: async () => ({
      ok: true,
      result: {
        api_version: "1.0",
        protocols: ["codex_rollout"],
        features: parts ? PARTS_FEATURES : FEATURES,
        ...(maxRequestBytes ? { max_request_bytes: maxRequestBytes } : {}),
      },
    }),
    cursor: async () => ({ ok: true, result: { next_sequence: remote.sequence } }),
    remember: async (item) => {
      const body = rememberBody(item);
      return tooLarge(body) ?? accept(body);
    },
    rememberPart: async (part) => {
      const body = rememberPartBody(part);
      remote.parts.push({ index: part.part_index, count: part.part_count, bytes: Buffer.byteLength(body) });
      const refused = tooLarge(body);
      if (refused) return refused;
      const chunk = Buffer.from(part.data, "base64");
      let upload = part.part_index === 0 ? { chunks: [], digest: part.body_sha256 } : remote.staging.get(part.delivery_id);
      if (upload == null || upload.digest !== part.body_sha256 || upload.chunks.length !== part.part_index) {
        remote.staging.delete(part.delivery_id);
        return { ok: false, status: 409, retryable: true, error: { code: "ABORTED", message: "upload again from part 0" } };
      }
      upload.chunks.push(chunk);
      remote.staging.set(part.delivery_id, upload);
      if (upload.chunks.length < part.part_count) {
        return { ok: true, result: { received_parts: upload.chunks.length, part_count: part.part_count, remember: null } };
      }
      remote.staging.delete(part.delivery_id);
      const whole = Buffer.concat(upload.chunks);
      assert.equal(whole.length, part.body_bytes);
      assert.equal(createHash("sha256").update(whole).digest("hex"), part.body_sha256);
      const remembered = accept(whole.toString("utf8"));
      return { ok: true, result: { received_parts: part.part_count, part_count: part.part_count, remember: remembered.result } };
    },
    flush: async () => ({ ok: true, result: {} }),
  };
  return remote;
}

const line = (value) => `${JSON.stringify(value)}\n`;

function codexTurn(index, answer) {
  return [
    { type: "event_msg", payload: { type: "task_started" } },
    { type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: `第 ${index} 句` }] } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: [{ type: "output_text", text: answer }] } },
  ];
}

test("the media placeholders in the shared fixture are exactly what the plugin sends in place of the payloads", async () => {
  // 服务端那一半（剥前剥后交给各协议适配器得到同样的消息）在 tests/unit/integrations/test_service_contract.py
  const fixture = JSON.parse(await readFile(join(HERE, "fixtures", "media-placeholder.json"), "utf8"));
  assert.deepEqual(Object.keys(fixture.turns).sort(), ["claude_code", "codex_rollout", "dsh", "opencode", "pi", "zcode"]);
  for (const [protocol, { raw, stripped }] of Object.entries(fixture.turns)) {
    assert.deepEqual(raw.map(stripMedia), stripped, protocol);
  }
});

function longStrings(value, found = []) {
  if (typeof value === "string") { if (value.length >= 1024) found.push(value.slice(0, 40)); }
  else if (value && typeof value === "object") for (const item of Object.values(value)) longStrings(item, found);
  return found;
}

// T1：Claude Code 读 PDF 的记录上 toolUseResult 是 {type:"pdf", file:{base64}}（本机真实转写里的形状）：经适配器读、入队之后，
// outbox 里一段载荷都不剩，toolUseResult 换成占位
test("a Claude Code PDF read leaves no payload in the outbox: its toolUseResult becomes a placeholder", async (t) => {
  const { raw, stripped } = JSON.parse(await readFile(join(HERE, "fixtures", "media-placeholder.json"), "utf8")).turns.claude_code;
  assert.ok(raw.some((record) => record.toolUseResult?.type === "pdf"), "the fixture has a PDF read");
  const root = await temporaryRoot(t);
  const transcript = join(root, "transcript.jsonl");
  await writeFile(transcript, raw.map((record) => JSON.stringify(record)).join("\n") + "\n");
  const core = new PluginCore({ stateRoot: join(root, "state"), timeoutMs: 1000, maxTranscriptBytes: MiB }, { service: {} });
  await core.enqueueCapture(claude, { session_id: "s", transcript_path: transcript });
  const records = (await listOutbox(join(root, "state"), "claude-code", "s")).flatMap((item) => item.payload.records);
  assert.deepEqual(records, stripped);
  assert.deepEqual(longStrings(records), []);
  const pdf = records.find((record) => record.toolUseResult?.type === "pdf").toolUseResult;
  assert.deepEqual(Object.keys(pdf).sort(), ["omitted_bytes", "omitted_sha256", "type"]);
});

// R18：只有媒体的工具结果跨插件 / 本地服务两边：插件剥成占位，本地服务认出"只有媒体"（Python 那一半在 test_service_contract.py）
test("a tool result holding only media becomes exactly the placeholders in the shared fixture", async () => {
  const { media_only_tool_results: cases } = JSON.parse(await readFile(join(HERE, "fixtures", "media-placeholder.json"), "utf8"));
  assert.deepEqual(
    [...new Set(cases.flatMap((item) => item.stripped.map((part) => part.type)))].sort(),
    ["document", "file", "image", "input_image"],
  );
  for (const { name, raw, stripped, omitted } of cases) {
    assert.deepEqual(stripMedia(raw), stripped, name);
    for (const part of stripped) {
      assert.equal(typeof part.omitted_bytes, "number", name);
      assert.match(part.omitted_sha256, /^[0-9a-f]{64}$/, name);
      assert.equal("text" in part, false, name);
    }
    // 本地服务据占位记下的 original_size_bytes / original_sha256（Python 那一半断言经不经插件逐字段等于 omitted）：
    // 字节数相加；一段时就是这段载荷文本的 sha256，多段时是各段 sha256 按顺序以换行连接后的 sha256
    assert.equal(omitted.original_size_bytes, stripped.reduce((total, part) => total + part.omitted_bytes, 0), name);
    const digests = stripped.map((part) => part.omitted_sha256);
    const combined = digests.length === 1 ? digests[0] : createHash("sha256").update(digests.join("\n"), "ascii").digest("hex");
    assert.equal(omitted.original_sha256, combined, name);
  }
  assert.ok(cases.some(({ stripped }) => stripped.length > 1), "the fixture has a tool result with several media parts");
});

// T21：共享 fixture 的 not_media 组（没有 type 的 data 对象、没有 type 也没有 mime 字段的 data URL）插件不当媒体、原样不动；
// 本地服务那一半（不当媒体、内容原样存）在 test_service_contract.py
test("the shared not-media cases are left exactly as they are", async () => {
  const { not_media: cases } = JSON.parse(await readFile(join(HERE, "fixtures", "media-placeholder.json"), "utf8"));
  assert.ok(cases.length >= 2);
  for (const { name, content } of cases) {
    assert.equal(stripMedia(content), content, name);
    const record = { type: "user", message: { role: "user", content: [{ type: "tool_result", tool_use_id: "t", content }] } };
    assert.equal(stripMedia(record), record, name);
  }
});

test("a Codex turn with a 1 MB screenshot goes out without the image payload and with every word of text", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const image = `data:image/png;base64,${"A".repeat(1024 * 1024)}`;
  const records = [
    { type: "event_msg", payload: { type: "task_started" } },
    { type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: "看看截图" }, { type: "input_image", image_url: image }] } },
    { type: "response_item", payload: { type: "function_call", name: "view_image", call_id: "c1", arguments: "{\"path\":\"a.png\"}" } },
    { type: "response_item", payload: { type: "function_call_output", call_id: "c1", output: [{ type: "input_text", text: "Image a.png" }, { type: "input_image", image_url: image, detail: "high" }] } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: [{ type: "output_text", text: "是登录页" }] } },
  ];
  const transcript = join(stateRoot, "rollout.jsonl");
  await writeFile(transcript, records.map(line).join(""));
  const remote = service();
  const core = new PluginCore({ stateRoot, timeoutMs: 1000, maxTranscriptBytes: 64 * MiB }, { service: remote });
  const input = { session_id: "image-turn", transcript_path: transcript };
  await core.enqueueCapture(codex, input);
  const queued = JSON.stringify(await listOutbox(stateRoot, "codex", "image-turn"));
  assert.ok(!queued.includes("base64,"), "the outbox keeps the image payload");
  await core.drain(codex, input);
  assert.equal(remote.bodies.length, 1);
  const [body] = remote.bodies;
  assert.ok(!body.includes("base64,") && !body.includes("A".repeat(1024)), "the request carries the image payload");
  assert.ok(Buffer.byteLength(body) < 4096, `request is ${Buffer.byteLength(body)} bytes`);
  const sent = JSON.parse(body).payload.records;
  assert.deepEqual(sent, records.map(stripMedia));
  assert.deepEqual(sent[1].payload.content, [
    { type: "input_text", text: "看看截图" },
    { type: "input_image", media_type: "image/png", omitted_bytes: image.length, omitted_sha256: createHash("sha256").update(image).digest("hex") },
  ]);
  assert.deepEqual(sent[3].payload.output[0], { type: "input_text", text: "Image a.png" });
});

test("a backlog over the service's request limit goes out as several requests, each ending a turn, that add up to the whole", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const turns = [0, 1, 2].map((index) => codexTurn(index, `${index}`.repeat(7 * MiB)));
  const transcript = join(stateRoot, "rollout.jsonl");
  await writeFile(transcript, turns.flat().map(line).join(""));
  const whole = await codex.readTranscriptDelta({ transcript_path: transcript }, null, { maxTranscriptBytes: 64 * MiB }, []);
  const remote = service();
  const core = new PluginCore({ stateRoot, timeoutMs: 1000, maxTranscriptBytes: 64 * MiB }, { service: remote });
  const input = { session_id: "backlog", transcript_path: transcript };
  await core.enqueueCapture(codex, input);
  const drained = await core.drain(codex, input);
  assert.equal(drained.blocked, undefined);
  assert.ok(remote.bodies.length >= 2, `${remote.bodies.length} request(s)`);
  const sent = remote.bodies.map((body) => JSON.parse(body));
  for (const [index, body] of sent.entries()) {
    assert.ok(Buffer.byteLength(remote.bodies[index]) <= 16 * MiB, `request ${index} is ${Buffer.byteLength(remote.bodies[index])} bytes`);
    assert.equal(body.payload.records.at(-1).payload.phase, "final_answer", `request ${index} does not end a turn`);
    assert.equal(body.after_turn, true);
  }
  assert.deepEqual(sent.flatMap((body) => body.payload.records), [...whole.records]);
  assert.deepEqual(await listOutbox(stateRoot, "codex", "backlog"), []);
  assert.equal((await readState(stateRoot, "codex", "backlog")).acknowledgedTranscriptCursor.byteOffset, whole.nextCursor.byteOffset);
});

test("the request limit the service declares in its capabilities is the one batches are cut to", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const turns = [0, 1, 2].map((index) => codexTurn(index, `${index}`.repeat(3 * MiB)));
  const transcript = join(stateRoot, "rollout.jsonl");
  await writeFile(transcript, turns.flat().map(line).join(""));
  const remote = service({ maxRequestBytes: 4 * MiB });
  const core = new PluginCore({ stateRoot, timeoutMs: 1000, maxTranscriptBytes: 64 * MiB }, { service: remote });
  const input = { session_id: "declared", transcript_path: transcript };
  await core.sessionStart(codex, input);
  await core.enqueueCapture(codex, input);
  await core.drain(codex, input);
  assert.equal(remote.bodies.length, 3);
  assert.ok(remote.bodies.every((body) => Buffer.byteLength(body) <= 4 * MiB));
});

test("against a local service without remember_parts_v1 a single turn over the limit is blocked, not sent, and the doctor names the session, its size and the excess", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const transcript = join(stateRoot, "rollout.jsonl");
  await writeFile(transcript, codexTurn(0, "长".repeat(40_000)).map(line).join(""));
  const remote = service({ maxRequestBytes: 64 * 1024 });
  let attempts = 0;
  const remember = remote.remember;
  remote.remember = async (item) => { attempts += 1; return remember(item); };
  const core = new PluginCore({ stateRoot, timeoutMs: 1000, maxTranscriptBytes: 64 * MiB }, { service: remote });
  const input = { session_id: "one-big-turn", transcript_path: transcript };
  await core.sessionStart(codex, input);
  await core.enqueueCapture(codex, input);
  const drained = await core.drain(codex, input);
  assert.ok(drained.blocked);
  assert.equal(attempts, 0, "an oversize batch was sent anyway");
  const [item] = await listOutbox(stateRoot, "codex", "one-big-turn");
  assert.equal(item.status, "blocked");
  assert.equal(item.payload.records.length, 3, "the turn was cut");
  const bytes = Buffer.byteLength(rememberBody(item));
  const report = await runPluginDoctor({ stateRoot }, {
    service: { health: async () => ({ ok: false, error: "down" }) },
    inspectHarnesses: async () => [], inspectFeatures: async () => [], inspectMarketplace: async () => ({ name: "marketplace", status: "pass", detail: "" }),
    inspectHookReviews: async () => [],
  });
  const outbox = report.checks.find((check) => check.name === "outbox");
  assert.equal(outbox.status, "fail");
  assert.ok(outbox.detail.includes(`blocked codex session one-big-turn: request_too_large: this batch is ${bytes} bytes`), outbox.detail);
  assert.ok(outbox.detail.includes(`over the local service limit of 65536 bytes (0.1 MiB) by ${bytes - 65536} bytes`), outbox.detail);
  assert.ok(outbox.detail.includes("has no remember_parts_v1"), outbox.detail);
  assert.equal(remote.parts.length, 0, "parts were sent to a local service that does not declare remember_parts_v1");
});

// 一轮 40 MiB 的纯文字：几个大工具结果（Claude Read 大文件、命令输出那样的），没有一个字节是媒体
function hugeToolTurn() {
  const records = [
    { type: "event_msg", payload: { type: "task_started" } },
    { type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: "把这几个日志都读一遍" }] } },
  ];
  for (let index = 0; index < 5; index += 1) {
    const callId = `call-${index}`;
    records.push({ type: "response_item", payload: { type: "function_call", name: "shell", call_id: callId, arguments: JSON.stringify({ command: ["cat", `log-${index}.txt`] }) } });
    const text = `第 ${index} 份日志 "引号" 与 \\ 反斜杠\n`.repeat(Math.ceil((10 * MiB) / 40)).slice(0, 4 * MiB);
    records.push({ type: "response_item", payload: { type: "function_call_output", call_id: callId, output: text } });
  }
  records.push({ type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: [{ type: "output_text", text: "读完了" }] } });
  return records;
}

async function partsSetup(t, sessionId, remoteOptions) {
  const stateRoot = await temporaryRoot(t);
  const transcript = join(stateRoot, "rollout.jsonl");
  await writeFile(transcript, hugeToolTurn().map(line).join(""));
  const remote = service(remoteOptions);
  const core = new PluginCore({ stateRoot, timeoutMs: 1000, maxTranscriptBytes: 128 * MiB }, { service: remote });
  const input = { session_id: sessionId, transcript_path: transcript, timestamp: "2026-10-23T08:00:00.000Z" };
  await core.sessionStart(codex, input);
  await core.enqueueCapture(codex, input);
  return { stateRoot, remote, core, input };
}

test("a single 40 MiB turn goes out in at least three parts under the limit and is assembled into exactly the request a one-shot send makes", async (t) => {
  const oneShot = await partsSetup(t, "huge-turn", { maxRequestBytes: 64 * MiB });
  const bytes = Buffer.byteLength(rememberBody((await listOutbox(oneShot.stateRoot, "codex", "huge-turn"))[0]));
  assert.ok(bytes > 40 * MiB, `the turn is only ${bytes} bytes`);
  await oneShot.core.drain(codex, oneShot.input);
  assert.equal(oneShot.remote.bodies.length, 1);

  const chunked = await partsSetup(t, "huge-turn", { parts: true });
  const drained = await chunked.core.drain(codex, chunked.input);
  assert.equal(drained.blocked, undefined);
  assert.equal(drained.drained, 1);
  assert.ok(chunked.remote.parts.length >= 3, `${chunked.remote.parts.length} part(s)`);
  assert.ok(chunked.remote.parts.every((part) => part.bytes <= 16 * MiB), JSON.stringify(chunked.remote.parts));
  // 投递号取决于各自转写文件的游标，除它之外拼回的请求体与一次送的逐字节相同
  const sameBut = (body) => ({ ...JSON.parse(body), delivery_id: null });
  assert.equal(chunked.remote.bodies.length, 1);
  assert.deepEqual(sameBut(chunked.remote.bodies[0]), sameBut(oneShot.remote.bodies[0]), "the assembled request differs from the one-shot request");
  assert.equal(
    chunked.remote.bodies[0],
    oneShot.remote.bodies[0].replace(JSON.parse(oneShot.remote.bodies[0]).delivery_id, JSON.parse(chunked.remote.bodies[0]).delivery_id),
  );
  assert.deepEqual(await listOutbox(chunked.stateRoot, "codex", "huge-turn"), []);
  assert.equal((await readState(chunked.stateRoot, "codex", "huge-turn")).nextSequence, oneShot.remote.sequence);
});

test("a part that fails midway keeps the batch in the outbox, and the resend from part 0 records it exactly once", async (t) => {
  const { stateRoot, remote, core, input } = await partsSetup(t, "part-fails", { parts: true });
  const rememberPart = remote.rememberPart;
  let failed = false;
  const attempts = [];
  remote.rememberPart = async (part) => {
    attempts.push(part.part_index);
    if (part.part_index === 1 && !failed) {
      failed = true;
      return { ok: false, status: 0, retryable: true, error: "connection reset" };
    }
    return rememberPart(part);
  };
  const first = await core.drain(codex, input);
  assert.equal(first.retryable, true);
  assert.equal(remote.bodies.length, 0);
  const [kept] = await listOutbox(stateRoot, "codex", "part-fails");
  assert.equal(kept.status, "bound", "the batch left the outbox before its last part was acknowledged");
  const second = await core.drain(codex, input);
  assert.equal(second.drained, 1);
  assert.equal(remote.bodies.length, 1);
  assert.deepEqual(attempts.slice(0, 3), [0, 1, 0], "the resend did not start from part 0");
  assert.deepEqual(await listOutbox(stateRoot, "codex", "part-fails"), []);
});

test("when the local service restarts mid-upload the plugin resends from part 0 and the turn is recorded once", async (t) => {
  const { stateRoot, remote, core, input } = await partsSetup(t, "service-restart", { parts: true });
  const rememberPart = remote.rememberPart;
  let restarted = false;
  remote.rememberPart = async (part) => {
    if (part.part_index === 2 && !restarted) { restarted = true; remote.restart(); }
    return rememberPart(part);
  };
  const first = await core.drain(codex, input);
  assert.equal(first.retryable, true);
  assert.equal(remote.bodies.length, 0);
  await core.drain(codex, input);
  assert.equal(remote.bodies.length, 1);
  assert.deepEqual(await listOutbox(stateRoot, "codex", "service-restart"), []);

  // 最后一块的确认丢了：插件整批重送，本地服务按投递号不重复记
  const again = remote.bodies[0];
  const [part] = remote.parts;
  assert.equal(part.index, 0);
  const { splitRememberBody } = await import("../lib/remember-parts.mjs");
  const resent = splitRememberBody(JSON.parse(again).delivery_id, again, 16 * MiB);
  for (const value of resent) await remote.rememberPart(value);
  assert.equal(remote.bodies.length, 1, "a resent upload was recorded twice");
});

// 杀变异 ME03 / ME04 / ME05：嵌套的 image_url.url 也剥；不是媒体的对象（带 data: 的普通链接、带字符串 data 的普通结果）原样不动
test("media payloads nested under image_url.url are stripped, and objects that are not media are left exactly as they are", () => {
  const url = `data:image/png;base64,${"A".repeat(64)}`;
  assert.deepEqual(stripMedia({ type: "image_url", image_url: { url } }), {
    type: "image_url",
    media_type: "image/png",
    omitted_bytes: url.length,
    omitted_sha256: createHash("sha256").update(url, "utf8").digest("hex"),
  });
  const notMedia = [
    { type: "link", url: "data:text/plain,hello" },
    { type: "tool_output", data: "exit 0" },
    { kind: "event", data: "AAAA" },
    [{ type: "text", text: "data:image/png;base64,AAAA" }],
  ];
  for (const value of notMedia) assert.equal(stripMedia(value), value, JSON.stringify(value));
});

// 杀变异 PC02 / PC03 / PC04（拼批时少算了轮之间的逗号、轮里各条之间的逗号、信封里序号与 after_turn 的最长写法）：
// 上限正好放得下两轮时拼成一批；少一个字节就拆成两批，每批按最长写法量都不超过上限
test("batches are cut on exact request sizes: two turns that fit to the byte share a batch, one byte less splits them", async (t) => {
  const records = [codexTurn(0, "第一轮的答复"), codexTurn(1, "第二轮的答复")].flat();
  const capture = async (limit) => {
    const stateRoot = await temporaryRoot(t);
    await writeFile(join(stateRoot, "service-limits.json"), JSON.stringify({ schemaVersion: 1, maxRequestBytes: limit }));
    const transcript = join(stateRoot, "rollout.jsonl");
    await writeFile(transcript, records.map(line).join(""));
    await new PluginCore({ stateRoot, timeoutMs: 1000, maxTranscriptBytes: MiB }, { service: {} })
      .enqueueCapture(codex, { session_id: "exact", transcript_path: transcript });
    return listOutbox(stateRoot, "codex", "exact");
  };
  const longest = (item) => Buffer.byteLength(rememberBody({ ...item, startSequence: Number.MAX_SAFE_INTEGER, afterTurn: false }));
  const [whole] = await capture(64 * MiB);
  const exact = longest(whole);
  assert.equal((await capture(exact)).length, 1);
  const split = await capture(exact - 1);
  assert.equal(split.length, 2);
  for (const item of split) assert.ok(longest(item) <= exact - 1, `a batch is ${longest(item)} bytes, over ${exact - 1}`);
});

// 杀变异 RP06（最后一块的回应里没有拼回的结果也当送成功）：没拼回就是没送成，可重试、整批重送
test("a last part whose response carries no assembled remember result is a retryable failure", async () => {
  const parts = [0, 1].map((index) => ({ delivery_id: "d", part_index: index, part_count: 2, body_bytes: 2, body_sha256: "0".repeat(64), data: "" }));
  for (const result of [{}, { remember: null }, { remember: "ok" }]) {
    const sent = await sendRememberParts({ rememberPart: async () => ({ ok: true, status: 200, result }) }, parts);
    assert.deepEqual([sent.ok, sent.retryable], [false, true], JSON.stringify(result));
  }
  const done = await sendRememberParts({ rememberPart: async () => ({ ok: true, status: 200, result: { remember: { next_sequence: 3 } } }) }, parts);
  assert.deepEqual(done, { ok: true, status: 200, result: { next_sequence: 3 } });
});

// 杀变异 PC13（分块上传的租约不按块数放长）：块越多送得越久，租约要够送完，免得送到一半被另一个进程认领走
test("the claim on a batch uploaded in parts lasts one request timeout longer for every part after the first", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const transcript = join(stateRoot, "rollout.jsonl");
  await writeFile(transcript, codexTurn(0, "长".repeat(8 * 1024)).map(line).join(""));
  const remote = service({ maxRequestBytes: 4096, parts: true });
  const timeoutMs = 20_000;
  const leases = [];
  const rememberPart = remote.rememberPart;
  remote.rememberPart = async (part) => {
    const [item] = await listOutbox(stateRoot, "codex", "lease");
    leases.push({ count: part.part_count, left: Date.parse(item.claimExpiresAt) - Date.now() });
    return rememberPart(part);
  };
  const core = new PluginCore({ stateRoot, timeoutMs, maxTranscriptBytes: MiB }, { service: remote });
  const input = { session_id: "lease", transcript_path: transcript };
  await core.enqueueCapture(codex, input);
  const drained = await core.drain(codex, input);
  assert.equal(drained.drained, 1);
  const [{ count, left }] = leases;
  assert.ok(count >= 3, `${count} parts`);
  const expected = timeoutMs * 3 + timeoutMs * (count - 1);
  assert.ok(left > expected - 5_000 && left <= expected, `lease ${left} ms, expected about ${expected} ms`);
});
