import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { readJsonlDelta } from "../lib/jsonl.mjs";

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

test("same-inode transcript rewrite is detected by the full acknowledged-prefix digest", async (t) => {
  const directory = await root(t);
  const path = join(directory, "transcript.jsonl");
  await writeFile(path, '{"value":"one","done":true}\n');
  const first = await readJsonlDelta(path, null, { maxBytes: 1024, selectCompleted: completed });
  await writeFile(path, '{"value":"two","done":true}\n');
  const second = await readJsonlDelta(path, first.nextCursor, { maxBytes: 1024, selectCompleted: completed });
  assert.equal(second.nextCursor.generation, 1);
  assert.equal(second.records[0].value, "two");
});

test("early same-inode rewrite is detected when the final 128 bytes are unchanged", async (t) => {
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
