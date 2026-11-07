import assert from "node:assert/strict";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";

import { HARNESS_REGISTRY } from "../../harnesses.mjs";
import { fileEntryPath, patchFilePath } from "../../registrations.mjs";

// 阶段 3：各宿主的配置目录在 Windows 上同样落在用户主目录下（OpenCode 的 xdg-basedir 在 Windows 上也是 ~/.config，
// pi 是 ~/.pi，DeepSeek Harness 是 ~/.dsh）；用户名含中文与空格时路径原样拼出，不被转义或截断。
const HOME = resolve(tmpdir(), "Users", "张三 李四");

test("host config entries resolve under a home directory with Chinese characters and spaces", () => {
  const env = { HOME };
  const opencode = fileEntryPath(HARNESS_REGISTRY.resolve("opencode"), env);
  assert.ok(opencode.startsWith(join(HOME, ".config", "opencode")), opencode);
  const pi = fileEntryPath(HARNESS_REGISTRY.resolve("pi"), env);
  assert.ok(pi.startsWith(join(HOME, ".pi")), pi);
  assert.equal(patchFilePath(HARNESS_REGISTRY.resolve("dsh"), env), join(HOME, ".dsh", "cordis.patch.yml"));
});

test("without HOME the host config entries fall back to the OS home directory", () => {
  const opencode = fileEntryPath(HARNESS_REGISTRY.resolve("opencode"), {});
  assert.ok(opencode.startsWith(join(homedir(), ".config")), opencode);
});
