import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { appendFile, mkdtemp, rm, stat, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { openNoFollow, readFirstRecord, readJsonlDelta } from "../lib/jsonl.mjs";

function completed(records) {
  const index = records.findLastIndex((record) => record?.done === true);
  return index < 0 ? null : { consumedCount: index + 1, records: records.slice(0, index + 1), afterTurn: true };
}

async function root(t) {
  const value = await mkdtemp(join(tmpdir(), "habitus-jsonl-test-"));
  t.after(() => rm(value, { recursive: true, force: true }));
  return value;
}

test("partial final JSONL line is not acknowledged", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  await writeFile(path, '{"done":true}\n{"done":true}');
  const delta = await readJsonlDelta(path, null, { maxBytes: 1024, selectCompleted: completed });
  assert.equal(delta.records.length, 1);
  assert.equal(delta.nextCursor.lineCount, 1);
});

test("transcript replacement starts a new cursor generation", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  await writeFile(path, '{"value":"old","done":true}\n');
  const first = await readJsonlDelta(path, null, { maxBytes: 1024, selectCompleted: completed });
  await rm(path);
  await writeFile(path, '{"value":"new","done":true}\n');
  const second = await readJsonlDelta(path, first.nextCursor, { maxBytes: 1024, selectCompleted: completed });
  assert.equal(second.nextCursor.generation, 1);
  assert.equal(second.records[0].value, "new");
});

test("same-inode transcript rewrite is detected by the prefix anchors", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  await writeFile(path, '{"value":"one","done":true}\n');
  const first = await readJsonlDelta(path, null, { maxBytes: 1024, selectCompleted: completed });
  await writeFile(path, '{"value":"two","done":true}\n');
  const second = await readJsonlDelta(path, first.nextCursor, { maxBytes: 1024, selectCompleted: completed });
  assert.equal(second.nextCursor.generation, 1);
  assert.equal(second.records[0].value, "two");
});

test("early same-inode rewrite is detected when the bytes just before the cursor are unchanged", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  const tail = "x".repeat(256);
  await writeFile(path, `${JSON.stringify({ value: "one", tail, done: true })}\n`);
  const first = await readJsonlDelta(path, null, { maxBytes: 4096, selectCompleted: completed });
  await writeFile(path, `${JSON.stringify({ value: "two", tail, done: true })}\n`);
  const second = await readJsonlDelta(path, first.nextCursor, { maxBytes: 4096, selectCompleted: completed });
  assert.equal(second.nextCursor.generation, 1);
  assert.equal(second.records[0].value, "two");
});

// B3：转写比读取上限大时不再整份读、也不再每次抛错：从游标往后一段一段读，每段最多留 maxBytes，读到末尾为止
test("a transcript larger than the read limit is read in increments from the cursor", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  const lines = Array.from({ length: 20 }, (_, index) => JSON.stringify({ index, pad: "x".repeat(200), done: true }));
  await writeFile(path, `${lines.join("\n")}\n`);
  assert.ok((await stat(path)).size > 4 * 1024);
  const seen = [];
  let cursor = null;
  let reads = 0;
  while (true) {
    const delta = await readJsonlDelta(path, cursor, { maxBytes: 1024, selectCompleted: completed });
    if (delta == null) break;
    reads += 1;
    seen.push(...delta.records.map((record) => record.index));
    assert.ok(delta.turns.every((turn) => turn.nextCursor.byteOffset <= delta.nextCursor.byteOffset));
    cursor = delta.nextCursor;
    if (!delta.more) break;
  }
  assert.ok(reads > 1, `${reads} read(s)`);
  assert.deepEqual(seen, lines.map((_, index) => index));
  assert.equal(cursor.byteOffset, (await stat(path)).size);
  // 追加之后从游标接着读，前缀没变就不换代
  await appendFile(path, `${JSON.stringify({ index: 20, done: true })}\n`);
  const next = await readJsonlDelta(path, cursor, { maxBytes: 1024, selectCompleted: completed });
  assert.equal(next.nextCursor.generation, 0);
  assert.deepEqual(next.records.map((record) => record.index), [20]);
});

test("records the reader is told are irrelevant do not count against the read limit", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  await writeFile(path, [
    JSON.stringify({ kind: "said", done: false }),
    JSON.stringify({ kind: "log", pad: "x".repeat(800) }),
    JSON.stringify({ kind: "log", pad: "x".repeat(800) }),
    JSON.stringify({ kind: "said", done: true }),
  ].map((value) => `${value}\n`).join(""));
  const options = { maxBytes: 1024, selectCompleted: completed };
  await assert.rejects(readJsonlDelta(path, null, options), /a transcript turn exceeds the 1024-byte read limit/);
  const delta = await readJsonlDelta(path, null, { ...options, relevant: (record) => record.kind === "said" });
  assert.deepEqual(delta.records.map((record) => record?.kind ?? null), ["said", null, null, "said"]);
  assert.equal(delta.nextCursor.lineCount, 4);
  assert.deepEqual(delta.skippedLines, []);
});

test("a cursor written before the anchor schema (whole-prefix digest) still continues an unchanged transcript", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  const first = `${JSON.stringify({ value: "one", done: true })}\n`;
  await writeFile(path, `${first}${JSON.stringify({ value: "two", done: true })}\n`);
  const metadata = await stat(path);
  const legacy = {
    schemaVersion: 2, generation: 0, fileIdentity: `${metadata.dev}:${metadata.ino}`,
    byteOffset: Buffer.byteLength(first), lineCount: 1, prefixDigest: createHash("sha256").update(first).digest("hex"),
  };
  const delta = await readJsonlDelta(path, legacy, { maxBytes: 1024, selectCompleted: completed });
  assert.equal(delta.nextCursor.generation, 0);
  assert.equal(delta.nextCursor.schemaVersion, 3);
  assert.deepEqual(delta.records.map((record) => record.value), ["two"]);
  const rewritten = await readJsonlDelta(path, { ...legacy, prefixDigest: "0".repeat(64) }, { maxBytes: 1024, selectCompleted: completed });
  assert.equal(rewritten.nextCursor.generation, 1);
  assert.deepEqual(rewritten.records.map((record) => record.value), ["one", "two"]);
});

// 游标前的锚点只看最后 64 KiB：更早处的同长度改写由开头那一段的锚点认出来
test("a same-length rewrite at the head of a long transcript starts a new generation", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  const pad = "x".repeat(70 * 1024);
  const tail = `${JSON.stringify({ value: "tail", pad, done: true })}\n`;
  await writeFile(path, `${JSON.stringify({ value: "one", done: true })}\n${tail}`);
  const first = await readJsonlDelta(path, null, { maxBytes: 1024 * 1024, selectCompleted: completed });
  await writeFile(path, `${JSON.stringify({ value: "two", done: true })}\n${tail}`);
  const second = await readJsonlDelta(path, first.nextCursor, { maxBytes: 1024 * 1024, selectCompleted: completed });
  assert.equal(second.nextCursor.generation, 1);
  assert.deepEqual(second.records.map((record) => record.value), ["two", "tail"]);
});

// R2：读取额度按剥掉媒体载荷之后的大小算：一轮原样远超额度（几张截图）、剥完只剩几百字节，照样读得出、游标前进；剥完的才是给出的记录
test("the read limit counts records after their media payloads are stripped", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  const shot = (index) => ({ shot: index, content: [{ type: "image", source: { type: "base64", media_type: "image/png", data: "A".repeat(600) } }] });
  const records = [shot(0), shot(1), shot(2), { value: "看完了", done: true }];
  await writeFile(path, records.map((value) => `${JSON.stringify(value)}\n`).join(""));
  const delta = await readJsonlDelta(path, null, { maxBytes: 1024, selectCompleted: completed });
  assert.equal(delta.more, false);
  assert.equal(delta.nextCursor.lineCount, 4);
  assert.deepEqual(delta.records[0].content, [
    { type: "image", media_type: "image/png", omitted_bytes: 600, omitted_sha256: createHash("sha256").update("A".repeat(600)).digest("hex") },
  ]);
});

// 剥完只剩文字仍超过额度的一轮读不出来
test("a turn whose text alone is over the read limit still cannot be read", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  await writeFile(path, [{ value: "x".repeat(700) }, { value: "y".repeat(700) }, { value: "z", done: true }].map((value) => `${JSON.stringify(value)}\n`).join(""));
  await assert.rejects(readJsonlDelta(path, null, { maxBytes: 1024, selectCompleted: completed }), /a transcript turn exceeds the 1024-byte read limit/);
});

// T3（裁定 75 第 1 条）：一行原样就超过额度的一律跳过、不报错，不看它在文件里的什么位置（开头、整行落在一个 4 MiB 读块里、
// 跨读块边界、边界前只有一小段）、也不看它与对话相不相干；游标越过它，后面的照常读，skippedLines 报出第几行、多少字节。
// 杀变异 J07（单行阈值放宽到几倍）：额度 1 MiB、一行 2 MiB，放宽了的话块内那几处会把它读进来
test("a raw line over the read limit is skipped wherever it sits in the file, and the read goes on past it", async (t) => {
  const directory = await root(t);
  const MiB = 1024 * 1024;
  const big = (kind) => `${JSON.stringify({ kind, content: [{ type: "image", source: { type: "base64", media_type: "image/png", data: "A".repeat(2 * MiB) } }] })}\n`;
  const line = (value) => `${JSON.stringify(value)}\n`;
  const pad = (bytes) => line({ kind: "pad", pad: "p".repeat(bytes) });
  // 垫的是不相干的记录：不占额度，一次读就从开头读到那一行，读块边界落在文件的 4 MiB 处
  const relevant = (record) => record.kind === "said";
  const after = line({ kind: "said", value: "after", done: true });
  // 起点：开头、1.31 MiB（整行在第一个读块里）、2.62 MiB（跨 4 MiB 边界）、3.9 MiB（跨边界、边界前只有一小段）、4.2 MiB（整行在第二块里）
  const offsets = [0, 1.31, 2.62, 3.9, 4.2].map((value) => Math.floor(value * MiB));
  for (const kind of ["said", "progress"]) {
    for (const offset of offsets) {
      const path = join(directory, `${kind}-${offset}.jsonl`);
      // 垫在前面的拆成小于额度的几行（每行 ≤ 512 KiB）
      const head = [];
      for (let left = offset; left > 0;) {
        const size = Math.min(left, 512 * 1024);
        head.push(pad(Math.max(0, size - 40)));
        left -= Buffer.byteLength(head.at(-1));
      }
      const before = head.join("");
      const raw = big(kind);
      await writeFile(path, before + raw + after);
      const lines = head.length + 2;
      let cursor = null;
      const records = [];
      const skippedLines = [];
      for (let round = 0; round < 20; round += 1) {
        const delta = await readJsonlDelta(path, cursor, { maxBytes: MiB, selectCompleted: completed, relevant });
        if (delta == null) break;
        records.push(...delta.records);
        skippedLines.push(...delta.skippedLines);
        cursor = delta.nextCursor;
        if (!delta.more) break;
      }
      const where = `${kind} at ${offset}`;
      assert.equal(cursor.lineCount, lines, where);
      assert.equal(cursor.byteOffset, Buffer.byteLength(before + raw + after), where);
      assert.deepEqual(skippedLines, [{ lineCount: head.length + 1, bytes: Buffer.byteLength(raw) - 1 }], where);
      assert.equal(records.at(-1).value, "after", where);
      assert.equal(records.filter((record) => record != null && record.value !== "after").length, 0, where);
    }
  }
});

// 跳过的那一行只在游标越过它时报：它后面的一轮还没做完，这一次不报、游标停在它前面，下一次一轮做完了才报一次
test("a skipped line is reported once, when the cursor moves past it", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  const first = `${JSON.stringify({ value: "one", done: true })}\n`;
  await writeFile(path, `${first}${JSON.stringify({ pad: "x".repeat(2048) })}\n${JSON.stringify({ value: "two" })}\n`);
  const options = { maxBytes: 1024, selectCompleted: completed };
  const delta = await readJsonlDelta(path, null, options);
  assert.equal(delta.nextCursor.lineCount, 1);
  assert.deepEqual(delta.skippedLines, []);
  assert.equal(await readJsonlDelta(path, delta.nextCursor, options), null);
  await appendFile(path, `${JSON.stringify({ value: "three", done: true })}\n`);
  const next = await readJsonlDelta(path, delta.nextCursor, options);
  assert.equal(next.nextCursor.lineCount, 4);
  assert.deepEqual(next.records.map((record) => record?.value ?? null), [null, "two", "three"]);
  assert.equal(next.skippedLines.length, 1);
  assert.equal(next.skippedLines[0].lineCount, 2);
  assert.equal(next.skippedLines[0].bytes, Buffer.byteLength(JSON.stringify({ pad: "x".repeat(2048) })));
});

// 杀变异 JL04（跨块那一行的起点按这一块的开头算）：一行跨过 4 MiB 读块时，后面各行的结束位置与游标都要按文件里的真实字节位置
test("a line that spans two read chunks still leaves the cursor at its true byte offset", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  const first = `${JSON.stringify({ value: "one", pad: "x".repeat(4 * 1024 * 1024 + 100) })}\n`;
  const second = `${JSON.stringify({ value: "two", done: true })}\n`;
  await writeFile(path, first + second);
  const delta = await readJsonlDelta(path, null, { maxBytes: 64 * 1024 * 1024, selectCompleted: completed });
  assert.equal(delta.nextCursor.byteOffset, Buffer.byteLength(first + second));
  assert.equal(delta.nextCursor.lineCount, 2);
  await appendFile(path, `${JSON.stringify({ value: "three", done: true })}\n`);
  const next = await readJsonlDelta(path, delta.nextCursor, { maxBytes: 1024, selectCompleted: completed });
  assert.equal(next.nextCursor.generation, 0);
  assert.deepEqual(next.records.map((record) => record.value), ["three"]);
});

// 杀变异 JL01 / JL12（不核游标前 64 KiB 的尾锚点 / 锚点缩成几个字节）：开头 64 KiB 不变、游标前一段被同长度改写，也要换代重读
test("a same-length rewrite past the first 64 KiB but before the cursor starts a new generation", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  const head = `${JSON.stringify({ value: "head", pad: "x".repeat(70 * 1024), done: true })}\n`;
  const body = (value) => `${JSON.stringify({ value, pad: "y".repeat(200), done: true })}\n`;
  await writeFile(path, head + body("one"));
  const first = await readJsonlDelta(path, null, { maxBytes: 1024 * 1024, selectCompleted: completed });
  await writeFile(path, head + body("two"));
  const second = await readJsonlDelta(path, first.nextCursor, { maxBytes: 1024 * 1024, selectCompleted: completed });
  assert.equal(second.nextCursor.generation, 1);
  assert.deepEqual(second.records.map((record) => record.value), ["head", "two"]);
});

// 杀变异 JL05（打开转写时跟随符号链接）：转写路径是符号链接就不读（宿主给的路径被换成链接时，不顺着它读到别处的文件）
test("a transcript path that is a symlink is not followed", async (t) => {
  const directory = await root(t);
  const target = join(directory, "elsewhere.jsonl");
  await writeFile(target, '{"done":true}\n');
  const path = join(directory, "transcript.jsonl");
  if (!await symlinkOrSkip(t, target, path)) return;
  await assert.rejects(readJsonlDelta(path, null, { maxBytes: 1024, selectCompleted: completed }), { code: "ELOOP" });
  assert.equal(await readFirstRecord(path), null);
});

// Windows 上普通用户建文件符号链接要 SeCreateSymbolicLinkPrivilege 或开发者模式（CI runner 是管理员，建得了）。
// 建不了就跳过：那台机器上普通进程也就造不出这种链接，没有可防的
async function symlinkOrSkip(t, target, path) {
  try {
    await symlink(target, path);
    return true;
  } catch (error) {
    if (process.platform !== "win32" || error?.code !== "EPERM") throw error;
    t.skip("Windows 上当前用户没有建符号链接的特权（未开开发者模式）");
    return false;
  }
}

// Windows 那一支（没有 O_NOFOLLOW：先 lstat 拒链接、打开后核对文件身份）在 POSIX 上也照样走一遍：
// 链接被拒、普通文件照常打开读到内容
test("without O_NOFOLLOW a transcript symlink is still refused and a regular file still opens", async (t) => {
  const directory = await root(t);
  const target = join(directory, "elsewhere.jsonl");
  await writeFile(target, '{"done":true}\n');
  const path = join(directory, "transcript.jsonl");
  if (!await symlinkOrSkip(t, target, path)) return;
  await assert.rejects(openNoFollow(path, 0), { code: "ELOOP" });
  const handle = await openNoFollow(target, 0);
  try { assert.equal((await handle.readFile("utf8")), '{"done":true}\n'); }
  finally { await handle.close(); }
  await assert.rejects(openNoFollow(join(directory, "missing.jsonl"), 0), { code: "ENOENT" });
});

// 杀变异 JL08（坏行当空行跳过）：一行完整却不是 JSON 时报错、游标不往前走（plugin-core 记成 captureFailure、doctor 列出来），
// 不悄悄丢掉那一行接着读
test("a complete line that is not JSON stops the read instead of being skipped", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  await writeFile(path, '{"value":"a"}\nnot json\n{"value":"b","done":true}\n');
  await assert.rejects(
    readJsonlDelta(path, null, { maxBytes: 1024, selectCompleted: completed }),
    /transcript line 2 is invalid JSON/,
  );
});

// 杀变异 J08（more 不看文件到没到头）：一轮还没做完、文件正好在额度处结束：这是读到了文件末尾、不是留满了额度，
// 返回 null 等这一轮做完，不报"一轮超过额度"
test("an unfinished turn that ends exactly at the read limit waits for the turn instead of failing", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  await writeFile(path, `${JSON.stringify({ value: "x".repeat(200) })}\n${JSON.stringify({ value: "y" })}\n`);
  const { size } = await stat(path);
  assert.equal(await readJsonlDelta(path, null, { maxBytes: size, selectCompleted: completed }), null);
});
