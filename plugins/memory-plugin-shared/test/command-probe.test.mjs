// 宿主命令探测（CommandRunner.available）：启动慢的宿主不能被误判成"没装"，挂死的宿主也不能把探测拖住（W19）。
// 假宿主都放在临时 bin 里，PATH 只有它和 node。
import assert from "node:assert/strict";
import { chmod, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { CommandRunner, PROBE_TIMEOUT_MS } from "../../registrations.mjs";

async function hostBin(t, scripts) {
  const root = await mkdtemp(join(tmpdir(), "habitus-probe-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await symlink(process.execPath, join(root, "node"));
  for (const [name, body] of Object.entries(scripts)) {
    const path = join(root, name);
    await writeFile(path, `#!/bin/sh\n${body}\n`);
    await chmod(path, 0o755);
  }
  return { PATH: root, HOME: root };
}

// 用 node 计时，不依赖 PATH 上有 sleep
const wait = (ms, code) => `exec node -e "setTimeout(() => process.exit(${code}), ${ms})"`;

test("a host whose --version takes 4 s is still detected with the default probe timeout", async (t) => {
  const env = await hostBin(t, { slowhost: wait(4000, 0) });
  assert.ok(PROBE_TIMEOUT_MS >= 8000, `default probe timeout ${PROBE_TIMEOUT_MS} ms is too short for cold-starting CLIs`);
  assert.equal(new CommandRunner(env).available("slowhost"), true);
});

// node 启动时会把信号处置恢复成默认，shell 里 trap '' TERM 管不到 exec 出来的 node；要在 node 里装处理函数才是真忽略 TERM。
// 超时留 1.5 s，让 node 来得及装上处理函数；假宿主 20 s 后自己退出，撤掉 SIGKILL 时这条会拖到 20 s、超过 5 s 而变红（不会卡死整套测试）
test("a host that never answers and ignores SIGTERM is killed at the probe timeout and reported as missing", async (t) => {
  const env = await hostBin(t, { hunghost: `exec node -e "process.on('SIGTERM', () => {}); setTimeout(() => {}, 20000)"` });
  const started = Date.now();
  assert.equal(new CommandRunner(env, { probeTimeoutMs: 1500 }).available("hunghost"), false);
  assert.ok(Date.now() - started < 5000, `probe took ${Date.now() - started} ms`);
});

test("a missing command, a failing --version and a working one", async (t) => {
  const env = await hostBin(t, { okhost: "exit 0", badhost: "exit 3" });
  const runner = new CommandRunner(env);
  assert.equal(runner.available("okhost"), true);
  assert.equal(runner.available("badhost"), false);
  assert.equal(runner.available("nohost"), false);
});

test("the probe timeout must be a positive integer", () => {
  for (const value of [0, -1, 1.5, Number.NaN, "3000"]) {
    assert.throws(() => new CommandRunner({}, { probeTimeoutMs: value }), /probeTimeoutMs/);
  }
});
