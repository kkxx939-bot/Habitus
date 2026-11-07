import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { createRequire, syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { basename, join, win32 } from "node:path";
import test from "node:test";

import { formatError, main } from "../../bin/habitus.mjs";
import { HARNESS_REGISTRY, createHarnessRegistry } from "../../harnesses.mjs";
import { atStep, dedicatedRoot, executePluginLifecycle, sourceDigest } from "../../install-memory-plugin.mjs";
import { fileEntryPath, patchBlock, patchFilePath } from "../../registrations.mjs";
import { DENIAL_SKIP_REASON, allowWritesSync, denyWritesSync, writeDenialTakesEffect } from "./support/read-only.mjs";

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function sink() {
  const chunks = [];
  return { write: (value) => { chunks.push(String(value)); }, text: () => chunks.join("") };
}

async function workspace(t) {
  const directory = await mkdtemp(join(tmpdir(), "habitus-installer-test-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

// 只有 Codex 与 OpenCode 的小插件源：安装器只关心目录怎么换，不关心插件内容
async function fakeSource(directory) {
  const source = join(directory, "source");
  await mkdir(join(source, "habitus-memory"), { recursive: true });
  await mkdir(join(source, "habitus-memory-opencode"), { recursive: true });
  await mkdir(join(source, "memory-plugin-shared"), { recursive: true });
  await writeFile(join(source, "habitus-memory", "plugin.txt"), "codex\n");
  await writeFile(join(source, "habitus-memory-opencode", "index.mjs"), "export default {};\n");
  await writeFile(join(source, "memory-plugin-shared", "lib.txt"), "shared\n");
  await writeFile(join(source, "harnesses.mjs"), "// fake\n");
  return source;
}

const REGISTRY = createHarnessRegistry([HARNESS_REGISTRY.resolve("opencode"), HARNESS_REGISTRY.resolve("codex")]);

// 有 codex 命令的假宿主；failOn 里的操作抛错（模拟宿主命令失败）
class CodexRunner {
  constructor(failOn = null) { this.failOn = failOn; this.marketplaceSource = null; this.installed = false; }
  available(command) { return command === "codex"; }
  json(_command, args) {
    if (args.includes("marketplace")) return { marketplaces: this.marketplaceSource ? [{ name: "habitus-local", root: this.marketplaceSource }] : [] };
    return { installed: this.installed ? [{ pluginId: "habitus-memory@habitus-local", installed: true, enabled: true }] : [] };
  }
  run(_command, args) {
    const operation = args.join(" ");
    if (this.failOn && operation.startsWith(this.failOn)) throw new Error(`simulated failure: ${operation}`);
    if (operation.startsWith("plugin marketplace add ")) this.marketplaceSource = args.at(-1);
    else if (operation === "plugin marketplace remove habitus-local") this.marketplaceSource = null;
    else if (operation.startsWith("plugin add ")) this.installed = true;
    else if (operation.startsWith("plugin remove ")) this.installed = false;
  }
}

const NO_COMMANDS = Object.freeze({
  available: () => false,
  json: () => { throw new Error("no harness command"); },
  run: (command) => { throw Object.assign(new Error(`spawnSync ${command} ENOENT`), { code: "ENOENT" }); },
});

async function setup(t) {
  const directory = await workspace(t);
  const source = await fakeSource(directory);
  const parent = join(directory, "roots");
  await mkdir(parent);
  const env = { HOME: join(directory, "home"), XDG_CONFIG_HOME: join(directory, "config") };
  await mkdir(join(env.XDG_CONFIG_HOME, "opencode"), { recursive: true });
  const lifecycle = (options, runner = NO_COMMANDS) => executePluginLifecycle(options, { runner, sourceRoot: source, registry: REGISTRY, env });
  return { directory, source, parent, env, lifecycle };
}

// ── --root 的安全闸（B1，测试报告 C5）─────────────────────────────────────────────────────────────

test("install and update refuse a non-empty --root without a Habitus marker and leave it untouched", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const root = join(parent, "my-notes");
  await mkdir(root);
  await writeFile(join(root, "thesis.txt"), "important\n");
  for (const action of ["install", "update"]) {
    await assert.rejects(lifecycle({ action, harnesses: [], root }), /not empty and has no Habitus install marker/);
    await assert.rejects(lifecycle({ action, harnesses: [], root, prepareOnly: true }), /not empty and has no Habitus install marker/);
  }
  // T12：用户只用默认根，拒绝时不叫他去换 --root（那不是用户选项）
  await assert.rejects(lifecycle({ action: "install", harnesses: [], root }), (error) => /Move it aside/.test(error.message) && !/--root/.test(error.message));
  assert.deepEqual(await readdir(root), ["thesis.txt"]);
  assert.equal(await readFile(join(root, "thesis.txt"), "utf8"), "important\n");
  assert.deepEqual(await readdir(parent), ["my-notes"]);  // 没有留下暂存或备份
  const file = join(parent, "a-file");
  await writeFile(file, "x");
  await assert.rejects(lifecycle({ action: "install", harnesses: [], root: file }), /exists and is not a directory/);
  assert.equal(await readFile(file, "utf8"), "x");
});

test("an empty --root or one carrying a valid marker is replaced", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const root = join(parent, "marketplace");
  await mkdir(root);
  await lifecycle({ action: "install", harnesses: [], root });
  assert.equal(JSON.parse(await readFile(join(root, ".habitus-install.json"), "utf8")).marketplace, "habitus-local");
  await lifecycle({ action: "update", harnesses: [], root });
  assert.ok((await stat(join(root, "plugins", "habitus-memory", "plugin.txt"))).isFile());
  assert.deepEqual(await readdir(parent), ["marketplace"]);
});

test("a marker that is not ours does not count: install refuses and remove keeps the directory", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const root = join(parent, "forged");
  await mkdir(root);
  await writeFile(join(root, ".habitus-install.json"), JSON.stringify({ schemaVersion: 1, marketplace: "someone-else" }));
  await writeFile(join(root, "keep.txt"), "x");
  await assert.rejects(lifecycle({ action: "install", harnesses: [], root }), /no Habitus install marker/);
  const removed = await lifecycle({ action: "remove", harnesses: [], root });
  assert.equal(removed.removed, false);
  assert.deepEqual((await readdir(root)).sort(), [".habitus-install.json", "keep.txt"]);
});

test("remove never deletes a --root without the marker", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const root = join(parent, "other");
  await mkdir(root);
  await writeFile(join(root, "keep.txt"), "x");
  const removed = await lifecycle({ action: "remove", harnesses: [], root });
  assert.equal(removed.removed, false);
  assert.deepEqual(await readdir(root), ["keep.txt"]);
});

test("the home directory is never a marketplace root, even when it is empty", async (t) => {
  const { directory, lifecycle } = await setup(t);
  const home = join(directory, "empty-home");
  await mkdir(home);
  // os.homedir() 读的是它：POSIX 是 HOME，Windows 是 USERPROFILE
  const variable = process.platform === "win32" ? "USERPROFILE" : "HOME";
  const previous = process.env[variable];
  process.env[variable] = home;
  try {
    await assert.rejects(lifecycle({ action: "install", harnesses: [], root: home }), /dedicated child directory/);
  } finally {
    process.env[variable] = previous;
  }
  assert.deepEqual(await readdir(home), []);
});

test("a marketplace root inside the plugin source is refused", async (t) => {
  const { source, lifecycle } = await setup(t);
  const root = join(source, "inside");
  await assert.rejects(lifecycle({ action: "install", harnesses: [], root }), /outside the source plugin directory/);
  await assert.rejects(stat(root), /ENOENT/);
});

async function preparedWithSentinel(parent, lifecycle) {
  const root = join(parent, "marketplace");
  await lifecycle({ action: "install", harnesses: [], root, prepareOnly: true });
  await writeFile(join(root, "old.txt"), "previous install\n");
  return root;
}

test("a failed registration rolls the marketplace root back to the previous install", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const root = await preparedWithSentinel(parent, lifecycle);
  await assert.rejects(
    lifecycle({ action: "install", harnesses: ["codex"], root }, new CodexRunner("plugin add")),
    /simulated failure: plugin add/,
  );
  assert.equal(await readFile(join(root, "old.txt"), "utf8"), "previous install\n");
  assert.deepEqual(await readdir(parent), ["marketplace"]);
});

test("a swap that fails after moving the old root aside puts the old root back", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const root = await preparedWithSentinel(parent, lifecycle);
  const fsPromises = createRequire(import.meta.url)("node:fs/promises");
  const renameOriginal = fsPromises.rename;
  fsPromises.rename = async (from, to) => {
    if (basename(String(from)).includes(".staging-")) throw Object.assign(new Error("simulated rename failure"), { code: "EIO" });
    return renameOriginal(from, to);
  };
  syncBuiltinESMExports();
  try {
    await assert.rejects(lifecycle({ action: "install", harnesses: [], root, prepareOnly: true }), /simulated rename failure/);
  } finally {
    fsPromises.rename = renameOriginal;
    syncBuiltinESMExports();
  }
  assert.equal(await readFile(join(root, "old.txt"), "utf8"), "previous install\n");
  assert.deepEqual(await readdir(parent), ["marketplace"]);
});

test("a failed removal restores the registrations it already took away", async (t) => {
  const { parent, env, lifecycle } = await setup(t);
  const root = join(parent, "marketplace");
  await lifecycle({ action: "install", harnesses: [], root }, new CodexRunner());
  const entry = fileEntryPath(REGISTRY.resolve("opencode"), env);
  const runner = new CodexRunner("plugin remove");
  runner.marketplaceSource = root;
  runner.installed = true;
  await assert.rejects(lifecycle({ action: "remove", harnesses: [], root }, runner), /simulated failure: plugin remove/);
  assert.match(await readFile(entry, "utf8"), /habitus-memory: managed by `habitus plugin`/);
  assert.ok((await stat(join(root, ".habitus-install.json"))).isFile());
});

// ── 本机有没有这个宿主（B8）──────────────────────────────────────────────────────────────────────

async function hostsWithoutCommands(t) {
  const directory = await workspace(t);
  const env = { HOME: join(directory, "home") };
  for (const path of [[".dsh"], [".config", "opencode"], [".pi"]]) await mkdir(join(env.HOME, ...path), { recursive: true });
  return { env, root: join(directory, "marketplace") };
}

test("a present harness whose command is not on PATH is skipped with the reason; the others still install", async (t) => {
  const { env, root } = await hostsWithoutCommands(t);
  const result = await executePluginLifecycle({ action: "install", harnesses: [], root }, { runner: NO_COMMANDS, env });
  const byId = Object.fromEntries(result.harnesses.map((item) => [item.harness, item]));
  assert.equal(byId.opencode.installed, true);
  assert.equal(byId.pi.installed, true);
  assert.equal(byId.dsh.installed, false);
  // 路径按本平台分隔符显示（Windows 上是 ~\\.dsh）
  assert.equal(byId.dsh.skipped, `本机有 ${join("~", ".dsh")} 但 PATH 上没有 dsh 命令`);
  await assert.rejects(readFile(patchFilePath(HARNESS_REGISTRY.resolve("dsh"), env), "utf8"), /ENOENT/);
  // 点名要装 dsh 仍然报错
  await assert.rejects(
    executePluginLifecycle({ action: "install", harnesses: ["dsh"], root }, { runner: NO_COMMANDS, env }),
    new RegExp(`dsh CLI is not installed: 本机有 ${escapeRegExp(join("~", ".dsh"))} 但 PATH 上没有 dsh 命令`),
  );
  const out = sink();
  const lifecycle = (options, context) => executePluginLifecycle(options, { ...context, runner: NO_COMMANDS });
  assert.equal(await main(["plugin", "install", "--root", root], { stdout: out, stderr: sink(), env, lifecycle }), 0);
  assert.match(out.text(), new RegExp(`dsh +跳过：本机有 ${escapeRegExp(join("~", ".dsh"))} 但 PATH 上没有 dsh 命令`));
});

test("habitus harnesses, plugin status and the installer agree on which harnesses are present", async (t) => {
  const { env, root } = await hostsWithoutCommands(t);
  const listed = await executePluginLifecycle({ action: "harnesses" }, { runner: NO_COMMANDS, env });
  const status = await executePluginLifecycle({ action: "status", harnesses: [], root }, { runner: NO_COMMANDS, env });
  const present = Object.fromEntries(status.harnesses.map((item) => [item.harness, item.available]));
  assert.deepEqual(Object.fromEntries(listed.harnesses.map((item) => [item.id, item.available])), present);
  assert.deepEqual(present, { codex: false, "claude-code": false, zcode: false, opencode: true, pi: true, dsh: true });
});

// R5：update 不带 --harness 只重新注册已经注册过的宿主；install 不带 --harness 仍是本机有的全部
test("update without --harness re-registers only the harnesses already registered", async (t) => {
  const { parent, env, lifecycle } = await setup(t);
  const root = join(parent, "marketplace");
  const codex = new CodexRunner();
  const calls = [];
  const runner = new Proxy(codex, {
    get: (target, name) => (name === "run" ? (command, args) => { calls.push(args.join(" ")); target.run(command, args); } : target[name]),
  });
  await lifecycle({ action: "install", harnesses: ["codex"], root }, runner);
  await lifecycle({ action: "update", harnesses: [], root }, runner);
  assert.equal(codex.installed, true);
  await assert.rejects(readFile(fileEntryPath(REGISTRY.resolve("opencode"), env), "utf8"), { code: "ENOENT" });
  await lifecycle({ action: "remove", harnesses: ["codex"], root }, runner);
  calls.length = 0;
  await lifecycle({ action: "update", harnesses: [], root }, runner);
  assert.deepEqual(calls, []);
  assert.equal(codex.installed, false);
  await lifecycle({ action: "install", harnesses: [], root }, runner);
  assert.equal(codex.installed, true);
  assert.ok((await readFile(fileEntryPath(REGISTRY.resolve("opencode"), env), "utf8")).length > 0);
});

// 杀变异 IN17（不点名时本机没有的宿主也装）：OpenCode 的配置目录不在、命令也不在时不替它写入口；没有的 Codex 也不报"跳过"
test("install without --harness leaves harnesses that are not on this machine alone", async (t) => {
  const { parent, env, lifecycle } = await setup(t);
  await rm(join(env.XDG_CONFIG_HOME, "opencode"), { recursive: true });
  const result = await lifecycle({ action: "install", harnesses: [], root: join(parent, "marketplace") });
  assert.deepEqual(
    result.harnesses.map(({ harness, available, installed, skipped }) => ({ harness, available, installed, skipped })),
    [
      { harness: "opencode", available: false, installed: false, skipped: undefined },
      { harness: "codex", available: false, installed: false, skipped: undefined },
    ],
  );
  await assert.rejects(readFile(fileEntryPath(REGISTRY.resolve("opencode"), env), "utf8"), { code: "ENOENT" });
});

// 杀变异 IN16（源摘要不算共享库）：只改了 memory-plugin-shared（还没同步进各插件）时源摘要也要变，doctor 才认得出装着的副本旧了
test("the source digest changes when only the shared library changes", async (t) => {
  const { source } = await setup(t);
  const before = await sourceDigest(source, REGISTRY);
  assert.equal(await sourceDigest(source, REGISTRY), before);
  await writeFile(join(source, "memory-plugin-shared", "lib.txt"), "shared, edited\n");
  assert.notEqual(await sourceDigest(source, REGISTRY), before);
});

// 记下宿主命令的假 Codex：插件可以装着但被停用
class RecordingCodexRunner extends CodexRunner {
  constructor(failOnce = null) { super(); this.calls = []; this.enabled = true; this.failOnce = failOnce; }
  json(command, args) {
    if (args.includes("marketplace")) return super.json(command, args);
    return { installed: this.installed ? [{ pluginId: "habitus-memory@habitus-local", installed: true, enabled: this.enabled }] : [] };
  }
  run(command, args) {
    const operation = args.join(" ");
    this.calls.push(operation);
    if (this.failOnce && operation.startsWith(this.failOnce)) {
      this.failOnce = null;
      throw new Error(`simulated failure: ${operation}`);
    }
    super.run(command, args);
    if (operation.startsWith("plugin add ")) this.enabled = true;
  }
}

// 杀变异 RG02（装着却停用的插件不重装）：Codex 里插件装着但被停用时，install 先卸掉再重新注册，装完是启用的
test("install re-registers a Codex plugin that is installed but disabled", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const root = join(parent, "marketplace");
  const runner = new RecordingCodexRunner();
  await lifecycle({ action: "install", harnesses: ["codex"], root }, runner);
  runner.enabled = false;
  runner.calls.length = 0;
  await lifecycle({ action: "install", harnesses: ["codex"], root }, runner);
  assert.deepEqual(runner.calls, [
    "plugin remove habitus-memory@habitus-local --json",
    "plugin marketplace remove habitus-local",
    `plugin marketplace add ${root}`,
    "plugin add habitus-memory@habitus-local",
  ]);
  assert.equal(runner.enabled, true);
});

// 杀变异 RG12（回滚时不管原来装没装都把插件加回去）：原来只登记了 marketplace、没装插件的 Codex，安装失败回滚后仍然没装插件
test("rolling back a failed install does not add a Codex plugin that was not installed before", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const root = join(parent, "marketplace");
  const runner = new RecordingCodexRunner("plugin add");
  runner.marketplaceSource = root;
  await assert.rejects(lifecycle({ action: "install", harnesses: ["codex"], root }, runner), /simulated failure: plugin add/);
  assert.equal(runner.installed, false);
  assert.equal(runner.marketplaceSource, root);
  assert.equal(runner.calls.filter((call) => call.startsWith("plugin add ")).length, 1);
});

// 杀变异 RG14（插件清单里 installed: false 的条目也算装着）：宿主列出来但标着没装的，不算已注册
test("a Codex plugin listed with installed: false is not registered", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const runner = new CodexRunner();
  runner.marketplaceSource = join(parent, "marketplace");
  runner.json = (_command, args) => (args.includes("marketplace")
    ? { marketplaces: [{ name: "habitus-local", root: runner.marketplaceSource }] }
    : { installed: [{ pluginId: "habitus-memory@habitus-local", installed: false, enabled: true }] });
  const status = await lifecycle({ action: "status", harnesses: ["codex"], root: runner.marketplaceSource }, runner);
  assert.equal(status.harnesses[0].installed, false);
});

// ── dsh 的 home 级 patch ─────────────────────────────────────────────────────────────────────────

async function dshHome(t) {
  const directory = await workspace(t);
  const env = { HOME: join(directory, "home"), DSH_HOME: join(directory, "dsh") };
  await mkdir(env.DSH_HOME, { recursive: true });
  const definition = HARNESS_REGISTRY.resolve("dsh");
  const runner = { available: (command) => command === "dsh", json: () => [], run: () => {} };
  return { directory, env, definition, path: patchFilePath(definition, env), runner };
}

// 杀变异 RG07（卸完只剩注释也写回去）：去掉受管那段后 patch 里只剩注释，就整份删掉（只剩注释的 patch 会让 dsh 起不来）
test("removing the dsh entry deletes a patch file that would be left with only comments", async (t) => {
  const { directory, env, definition, path, runner } = await dshHome(t);
  const root = join(directory, "marketplace");
  await writeFile(path, `# 我自己的说明\n\n${patchBlock(definition, root)}\n`);
  await executePluginLifecycle({ action: "remove", harnesses: ["dsh"], root }, { runner, env });
  await assert.rejects(readFile(path, "utf8"), { code: "ENOENT" });
});

// 杀变异 RG13（patch 里有受管段就算装着）：受管段指向的插件目录没了（根目录删了、换过 --root）算悬空，不算已注册
test("a dsh patch entry whose plugin directory is gone is dangling, not registered", async (t) => {
  const { directory, env, definition, path, runner } = await dshHome(t);
  await writeFile(path, `${patchBlock(definition, join(directory, "gone"))}\n`);
  const status = await executePluginLifecycle({ action: "status", harnesses: ["dsh"], root: join(directory, "gone") }, { runner, env });
  assert.equal(status.harnesses[0].installed, false);
  assert.equal(status.harnesses[0].dangling, true);
});

// 杀变异 RG15（有开始标记没结束标记不报错）：受管段被人改坏、只剩开始标记时不猜着改，报错让人手动修，文件不动
test("a dsh patch with a begin marker but no end marker is refused and left as it is", async (t) => {
  const { directory, env, definition, path, runner } = await dshHome(t);
  const broken = `${patchBlock(definition, join(directory, "old")).split("\n").slice(0, -1).join("\n")}\n- name: mine\n`;
  await writeFile(path, broken);
  await assert.rejects(
    executePluginLifecycle({ action: "install", harnesses: ["dsh"], root: join(directory, "marketplace") }, { runner, env }),
    /begin marker without its end marker/,
  );
  assert.equal(await readFile(path, "utf8"), broken);
});

// 杀变异 J33（update 不点名时不认"入口还在、指向的副本没了"）：入口指向的是以前 --root 装的、已经删掉的副本，
// 不点名的 update 照样把它重新注册到这次的根
test("update without --harness re-registers an entry left dangling by a removed earlier --root", async (t) => {
  const { parent, env, lifecycle } = await setup(t);
  const old = join(parent, "old-root");
  const root = join(parent, "marketplace");
  await lifecycle({ action: "install", harnesses: ["opencode"], root: old });
  await rm(old, { recursive: true, force: true });
  const before = await lifecycle({ action: "status", harnesses: ["opencode"], root });
  assert.equal(before.harnesses[0].dangling, true);
  await lifecycle({ action: "update", harnesses: [], root });
  const after = await lifecycle({ action: "status", harnesses: ["opencode"], root });
  assert.equal(after.harnesses[0].installed, true);
  assert.match(await readFile(fileEntryPath(REGISTRY.resolve("opencode"), env), "utf8"), new RegExp(basename(root)));
});

// 杀变异 J34（update 点名了宿主也只碰已注册的）：点名的宿主没注册过，update --harness 照样把它装上
test("update --harness registers the named harness even when it was not registered before", async (t) => {
  const { parent, env, lifecycle } = await setup(t);
  const root = join(parent, "marketplace");
  const codex = new CodexRunner();
  await lifecycle({ action: "install", harnesses: ["codex"], root }, codex);
  await assert.rejects(readFile(fileEntryPath(REGISTRY.resolve("opencode"), env), "utf8"), { code: "ENOENT" });
  await lifecycle({ action: "update", harnesses: ["opencode"], root }, codex);
  assert.equal((await lifecycle({ action: "status", harnesses: ["opencode"], root }, codex)).harnesses[0].installed, true);
});

// 杀变异 J38（没装插件也强制刷新 marketplace）：marketplace 已经登记在这个根、插件没装：install 只补装插件，不先删再加 marketplace
test("install adds only the plugin when the Codex marketplace is already registered at this root", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const root = join(parent, "marketplace");
  const runner = new RecordingCodexRunner();
  runner.marketplaceSource = root;
  await lifecycle({ action: "install", harnesses: ["codex"], root }, runner);
  assert.deepEqual(runner.calls, ["plugin add habitus-memory@habitus-local"]);
  assert.equal(runner.installed, true);
});

// 杀变异 J39（插件条目缺 installed 字段判成没装）：Codex 的插件清单里条目没写 installed 时按装着算（只有写明 false 才算没装）
test("a Codex plugin entry without an installed field counts as installed", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const runner = new CodexRunner();
  runner.marketplaceSource = join(parent, "marketplace");
  runner.json = (_command, args) => (args.includes("marketplace")
    ? { marketplaces: [{ name: "habitus-local", root: runner.marketplaceSource }] }
    : { installed: [{ pluginId: "habitus-memory@habitus-local", enabled: true }] });
  const status = await lifecycle({ action: "status", harnesses: ["codex"], root: runner.marketplaceSource }, runner);
  assert.equal(status.harnesses[0].installed, true);
});

// ── 报错写清卡在哪（W5）────────────────────────────────────────────────────────────────────────────

test("a failed registration names the harness and the step without changing the original message", async (t) => {
  const { parent, lifecycle } = await setup(t);
  const root = join(parent, "marketplace");
  const error = await lifecycle({ action: "install", harnesses: ["codex"], root }, new CodexRunner("plugin add")).then(
    () => assert.fail("install should fail"),
    (value) => value,
  );
  assert.equal(error.message, "simulated failure: plugin add habitus-memory@habitus-local");
  assert.deepEqual(error.habitusStep, { harness: "codex", action: "注册（marketplace）" });
  const err = sink();
  const failing = () => Promise.reject(error);
  assert.equal(await main(["plugin", "install", "--root", root], { stdout: sink(), stderr: err, env: {}, lifecycle: failing }), 1);
  assert.equal(
    err.text(),
    "habitus plugin: [codex · 注册（marketplace）] simulated failure: plugin add habitus-memory@habitus-local\n  （加 --verbose 看调用栈）\n",
  );
});

test("a registration whose rollback also fails reports the harness left behind", async (t) => {
  if (!writeDenialTakesEffect()) { t.skip(DENIAL_SKIP_REASON); return; }
  const { directory, parent, env } = await setup(t);
  const source = await fakeSource(join(directory, "again"));
  const registry = createHarnessRegistry([HARNESS_REGISTRY.resolve("codex"), HARNESS_REGISTRY.resolve("opencode")]);
  // codex 原本装着、指向旧目录，这次先改注册成功；OpenCode 的配置目录只读，写入口失败；
  // 回滚 codex 时要把它指回旧目录，这一步也失败
  const opencodeConfig = join(env.XDG_CONFIG_HOME, "opencode");
  denyWritesSync(opencodeConfig);
  const previous = join(directory, "previous-marketplace");
  const runner = new CodexRunner(`plugin marketplace add ${previous}`);
  runner.marketplaceSource = previous;
  runner.installed = true;
  const error = await executePluginLifecycle(
    { action: "install", harnesses: [], root: join(parent, "marketplace") },
    { runner, sourceRoot: source, registry, env },
  ).then(() => assert.fail("install should fail"), (value) => value)
    .finally(() => allowWritesSync(opencodeConfig));
  assert.equal(error.habitusStep.harness, "opencode");
  assert.match(error.habitusStep.action, /^注册（file）$/);
  assert.equal(error.restoreFailures.length, 1);
  assert.equal(error.restoreFailures[0].harness, "codex");
  assert.match(error.restoreFailures[0].message, /simulated failure: plugin marketplace add/);
  const text = formatError("plugin", error);
  assert.match(text, /^habitus plugin: \[opencode · 注册（file）\] /);
  assert.match(text, /\n  回滚时 codex 的注册没能恢复：simulated failure: plugin marketplace add/);
});

test("--verbose adds the stack and the cause chain, and is stripped before the subcommand sees it", async () => {
  const inner = new Error("EACCES: permission denied, open '/x/y'");
  const error = Object.assign(new Error("could not write the plugin entry", { cause: inner }), { habitusStep: { harness: "pi", action: "注册（file）" } });
  const plain = formatError("plugin", error);
  assert.match(plain, /^habitus plugin: \[pi · 注册（file）\] could not write the plugin entry\n  原因：EACCES: permission denied, open '\/x\/y'\n  （加 --verbose 看调用栈）\n$/);
  const verbose = formatError("plugin", error, true);
  assert.ok(verbose.includes(error.stack) && verbose.includes(inner.stack), verbose);
  assert.doesNotMatch(verbose, /加 --verbose/);
  // 位置不限，摘掉后子命令照常解析
  const out = sink();
  assert.equal(await main(["--verbose", "harnesses", "--json"], { stdout: out, stderr: sink(), env: { PATH: "", HOME: "/nonexistent" } }), 0);
  assert.ok(Array.isArray(JSON.parse(out.text()).harnesses));
  const err = sink();
  const failing = () => Promise.reject(error);
  assert.equal(await main(["plugin", "install", "--verbose", "--root", "/tmp/x"], { stdout: sink(), stderr: err, env: {}, lifecycle: failing }), 1);
  assert.ok(err.text().includes(error.stack));
});

test("a plain usage error stays one line without the --verbose hint", async () => {
  assert.equal(formatError("recall", new Error("usage: habitus recall <问题>")), "habitus recall: usage: habitus recall <问题>\n");
  assert.equal(formatError("recall", "boom"), "habitus recall: boom\n");
});

test("a plugin copy that cannot be rolled back keeps the original error, still restores registrations, and is reported", async (t) => {
  if (!writeDenialTakesEffect()) { t.skip(DENIAL_SKIP_REASON); return; }
  const { parent, lifecycle } = await setup(t);
  const root = join(parent, "marketplace");
  const previous = join(parent, "previous-marketplace");
  // codex 原本指向旧目录；这次注册失败的同时把根目录的父目录改成只读，插件副本的回滚（删暂存目录）随之失败
  const runner = new CodexRunner();
  runner.marketplaceSource = previous;
  runner.installed = true;
  const run = runner.run.bind(runner);
  runner.run = (command, args) => {
    if (args.join(" ").startsWith("plugin add ")) {
      denyWritesSync(parent);
      throw new Error("simulated failure: plugin add");
    }
    return run(command, args);
  };
  const error = await lifecycle({ action: "install", harnesses: ["codex"], root }, runner).then(
    () => assert.fail("install should fail"),
    (value) => value,
  ).finally(() => allowWritesSync(parent));
  assert.equal(error.message, "simulated failure: plugin add");
  assert.deepEqual(error.habitusStep, { harness: "codex", action: "注册（marketplace）" });
  assert.equal(runner.marketplaceSource, previous, "codex registration must still be restored");
  assert.equal(error.marketplaceRollbackFailure.root, root);
  assert.match(error.marketplaceRollbackFailure.message, /EACCES|EPERM/);
  const text = formatError("plugin", error);
  assert.match(text, /^habitus plugin: \[codex · 注册（marketplace）\] simulated failure: plugin add\n/);
  assert.match(text, new RegExp(`\\n  回滚时插件副本 ${root.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")} 没能恢复：.*(EACCES|EPERM)`));
  assert.match(text, /（加 --verbose 看调用栈）\n$/);
});

test("a remove that cannot delete the plugin copy names the step", async (t) => {
  if (!writeDenialTakesEffect()) { t.skip(DENIAL_SKIP_REASON); return; }
  const { parent, lifecycle } = await setup(t);
  const root = join(parent, "marketplace");
  await lifecycle({ action: "install", harnesses: [], root });
  denyWritesSync(parent);
  const error = await lifecycle({ action: "remove", harnesses: [], root }).then(
    () => assert.fail("remove should fail"),
    (value) => value,
  ).finally(() => allowWritesSync(parent));
  assert.deepEqual(error.habitusStep, { action: `删除插件副本 ${root}` });
  assert.match(formatError("plugin", error), /^habitus plugin: \[删除插件副本 .*\] .*(EACCES|EPERM)/);
});

test("a frozen error passes through atStep unchanged", async () => {
  const frozen = Object.freeze(new Error("frozen original"));
  await assert.rejects(atStep({ action: "x" }, () => { throw frozen; }), (error) => error === frozen);
});

// 撤回即失败（Windows 真机 bug）：原来判"在源目录外"用的是 path.parse(relation).isAbsolute（parse 没有这个字段，恒 undefined），
// 只剩 startsWith("..")。Windows 上临时目录在 C:、源码在 D:（CI 的 D:\a\Habitus），relative() 给的是绝对路径，
// 于是每次 install 都被误拒成"在源目录里面"。盘符语义用 path.win32 在任何平台上验
test("a marketplace root on another drive than the plugin source is outside it, one inside still is not", () => {
  const source = "D:\\a\\Habitus\\plugins";
  assert.equal(dedicatedRoot("C:\\Users\\张三 李四\\.habitus\\plugin-marketplace", source, win32), "C:\\Users\\张三 李四\\.habitus\\plugin-marketplace");
  assert.equal(dedicatedRoot("D:\\a\\marketplace", source, win32), "D:\\a\\marketplace");
  assert.throws(() => dedicatedRoot("D:\\a\\Habitus\\plugins\\nested", source, win32), /outside the source plugin directory/);
  assert.throws(() => dedicatedRoot(source, source, win32), /outside the source plugin directory/);
  assert.throws(() => dedicatedRoot("C:\\", source, win32), /dedicated child directory/);
});
