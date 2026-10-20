import assert from "node:assert/strict";
import { mkdir, mkdtemp, readdir, readFile, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { loadPluginConfig } from "../lib/config.mjs";
import { writeOperationLog } from "../lib/operation-log.mjs";

const ADAPTER = { host: "test-host", nativeSessionId: () => "session-1" };

function config(stateRoot, overrides = {}) {
  return { ...loadPluginConfig({ HABITUS_PLUGIN_STATE_DIR: stateRoot }), ...overrides };
}

test("the operation log rotates by size under unique names and keeps a bounded number of copies", async () => {
  const stateRoot = await mkdtemp(join(tmpdir(), "habitus-oplog-"));
  const settings = config(stateRoot, { operationLogMaxBytes: 4096, operationLogKeep: 2 });
  const logs = join(stateRoot, "logs");
  for (let index = 0; index < 120; index += 1) {
    await writeOperationLog(settings, ADAPTER, {}, { hook: "stop", stage: "flush", status: "ok", delivery: `d${index}` });
  }
  const names = await readdir(logs);
  const rotated = names.filter((name) => name.startsWith("operations-"));
  assert.ok(names.includes("operations.jsonl"));
  assert.ok(rotated.length >= 1 && rotated.length <= 2, `kept ${rotated.length} rotated copies`);
  assert.equal(new Set(rotated).size, rotated.length);
  assert.ok((await stat(join(logs, "operations.jsonl"))).size < 4096 + 1024);
  const last = (await readFile(join(logs, "operations.jsonl"), "utf8")).trim().split("\n").at(-1);
  assert.equal(JSON.parse(last).delivery, "d119");
});

test("concurrent writers rotating at once never overwrite each other and the hook never fails", async () => {
  const stateRoot = await mkdtemp(join(tmpdir(), "habitus-oplog-race-"));
  const settings = config(stateRoot, { operationLogMaxBytes: 4096, operationLogKeep: 50 });
  await writeOperationLog(settings, ADAPTER, {}, { hook: "stop", stage: "seed", status: "ok" });
  await writeFile(join(stateRoot, "logs", "operations.jsonl"), `${"x".repeat(5000)}\n`);
  await Promise.all(
    Array.from({ length: 20 }, (_, index) =>
      writeOperationLog(settings, ADAPTER, {}, { hook: "stop", stage: "race", status: "ok", delivery: `r${index}` }),
    ),
  );
  const names = await readdir(join(stateRoot, "logs"));
  const contents = await Promise.all(names.map((name) => readFile(join(stateRoot, "logs", name), "utf8")));
  const deliveries = contents.join("").split("\n").filter(Boolean).filter((line) => line.startsWith("{")).map((line) => JSON.parse(line).delivery);
  for (let index = 0; index < 20; index += 1) assert.ok(deliveries.includes(`r${index}`), `r${index} survived`);
});

test("the rotation settings come from the environment with bounded defaults", () => {
  const defaults = loadPluginConfig({ HABITUS_PLUGIN_STATE_DIR: tmpdir() });
  assert.equal(defaults.operationLogMaxBytes, 5 * 1024 * 1024);
  assert.equal(defaults.operationLogKeep, 3);
  const set = loadPluginConfig({ HABITUS_PLUGIN_STATE_DIR: tmpdir(), HABITUS_PLUGIN_LOG_MAX_BYTES: "8192", HABITUS_PLUGIN_LOG_KEEP: "5" });
  assert.equal(set.operationLogMaxBytes, 8192);
  assert.equal(set.operationLogKeep, 5);
});

test("an old copy that cannot be deleted is skipped and pruning carries on", async () => {
  // L10.12：原先删不掉的那一份让清理整段停下，份数无限增长，而且这一条记录也丢了
  const stateRoot = await mkdtemp(join(tmpdir(), "habitus-oplog-stuck-"));
  const settings = config(stateRoot, { operationLogMaxBytes: 4096, operationLogKeep: 2 });
  const logs = join(stateRoot, "logs");
  await mkdir(join(logs, "operations-0000-01-01T00-00-00.000Z-1-aaaaaaaa.jsonl"), { recursive: true });
  for (let index = 0; index < 120; index += 1) {
    await writeOperationLog(settings, ADAPTER, {}, { hook: "stop", stage: "flush", status: "ok", delivery: `s${index}` });
  }
  const rotated = (await readdir(logs, { withFileTypes: true })).filter(
    (entry) => entry.isFile() && entry.name.startsWith("operations-"),
  );
  assert.ok(rotated.length <= 2, `kept ${rotated.length} rotated copies`);
  const last = (await readFile(join(logs, "operations.jsonl"), "utf8")).trim().split("\n").at(-1);
  assert.equal(JSON.parse(last).delivery, "s119");
});

test("blank or non-positive settings fall back to their defaults instead of the minimum", () => {
  // L10.12：Number("") 是 0，原先被收成下限——宽限期从 24 小时静默降到 1 小时
  const root = { HABITUS_PLUGIN_STATE_DIR: "/tmp/habitus-config" };
  for (const value of ["", "   ", "0", "-3"]) {
    const settings = loadPluginConfig({ ...root, HABITUS_PLUGIN_SESSION_GRACE_HOURS: value, HABITUS_PLUGIN_LOG_KEEP: value });
    assert.equal(settings.sessionRetireGraceMs, 24 * 3_600_000, `grace for ${JSON.stringify(value)}`);
    assert.equal(settings.operationLogKeep, 3, `keep for ${JSON.stringify(value)}`);
  }
  assert.equal(loadPluginConfig({ ...root, HABITUS_PLUGIN_SESSION_GRACE_HOURS: "2" }).sessionRetireGraceMs, 2 * 3_600_000);
  assert.equal(loadPluginConfig({ ...root, HABITUS_PLUGIN_TIMEOUT_MS: "100" }).timeoutMs, 250);
});
