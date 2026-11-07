import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { cp, mkdir, mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test, { after } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { HARNESS_REGISTRY } from "../../harnesses.mjs";
import { prepareMarketplace } from "../../install-memory-plugin.mjs";
import { HEADER, sharedSources, sharedTargets } from "../sync.mjs";
import { isolatedEnv } from "./support/isolated-env.mjs";

const PLUGINS = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const LIB = join(PLUGINS, "memory-plugin-shared", "lib");
const ISOLATED = await isolatedEnv({ after });

test("every plugin's shared copy matches the library exactly, with no leftovers", async () => {
  const sources = await sharedSources();
  for (const target of sharedTargets()) {
    assert.deepEqual((await readdir(target)).sort(), sources, `${target} has stale or missing files; run npm run sync`);
    for (const filename of sources) {
      assert.equal(
        await readFile(join(target, filename), "utf8"),
        HEADER + await readFile(join(LIB, filename), "utf8"),
        `${target}/${filename} is stale; run npm run sync`,
      );
    }
  }
});

test("the shared library only imports node builtins and its own siblings (it is copied into each plugin)", async () => {
  for (const filename of await sharedSources()) {
    const source = await readFile(join(LIB, filename), "utf8");
    for (const [, specifier] of source.matchAll(/(?:from|import)\s*\(?\s*["']([^"']+)["']/g)) {
      assert.ok(
        specifier.startsWith("node:") || /^\.\/[^/]+\.mjs$/.test(specifier),
        `${filename} imports ${specifier}`,
      );
    }
  }
});

test("the npm package ships every registered plugin directory", async () => {
  const manifest = JSON.parse(await readFile(join(PLUGINS, "package.json"), "utf8"));
  const shipped = manifest.files.map((entry) => entry.replace(/\/$/, ""));
  for (const definition of HARNESS_REGISTRY.list()) {
    assert.ok(shipped.includes(definition.pluginDirectory), `package.json files misses ${definition.pluginDirectory}`);
  }
});

function runEntry(script, stateRoot) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [script], {
      env: { ...ISOLATED.env, HABITUS_MEMORY_ENABLED: "0", HABITUS_PLUGIN_STATE_DIR: stateRoot, HABITUS_HOOK_WORKER: "" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk) => { stderr += chunk; });
    child.once("error", reject);
    child.once("close", (code) => resolve({ code, stdout, stderr }));
    child.stdin.end("{}");
  });
}

// 宿主只加载安装器暂存出来的插件目录（<根>/plugins/<名>），仓库里的兄弟目录不在那里：每个入口在暂存副本里都要加载得起来。
// hook 脚本按宿主的方式起一个进程跑（记忆关着，只验证模块能解析、能跑完）；进程内插件的入口直接 import
test("every plugin entry loads from the installer's staged copy, without the repository around it", async (t) => {
  const parent = await mkdtemp(join(tmpdir(), "habitus-packaging-test-"));
  t.after(() => rm(parent, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 }));
  const root = await prepareMarketplace(join(parent, "marketplace"));
  const stateRoot = join(parent, "state");
  for (const definition of HARNESS_REGISTRY.list()) {
    const staged = join(root, "plugins", definition.pluginDirectory);
    const entries = [];
    if (definition.hooksManifest) {
      const manifest = JSON.parse(await readFile(join(staged, definition.hooksManifest), "utf8"));
      for (const groups of Object.values(manifest.hooks)) {
        for (const hook of groups.flatMap((group) => group.hooks)) {
          const [, script] = hook.command.match(/^node "\$\{[A-Z_]+\}\/(\S+\.mjs)"$/) || [];
          assert.ok(script, `${definition.id}: unexpected hook command ${hook.command}`);
          entries.push(script);
        }
      }
    }
    for (const script of [...new Set(entries)]) {
      const result = await runEntry(join(staged, script), stateRoot);
      assert.equal(result.code, 0, `${definition.id}: ${script} fails in the staged copy\n${result.stderr}`);
      assert.deepEqual(JSON.parse(result.stdout), {}, `${definition.id}: ${script}`);
    }
    if (definition.fileEntry) {
      const module = await import(pathToFileURL(join(staged, definition.fileEntry.module)).href);
      assert.equal(typeof module.default, "function", `${definition.id}: ${definition.fileEntry.module} has no default export`);
    }
  }
});

// 撤回即失败（hook 命令不加引号）：marketplace 根默认在用户主目录下，Windows 用户名常带空格（C:\\Users\\张三 李四）。
// 宿主把 ${CLAUDE_PLUGIN_ROOT} / ${PLUGIN_ROOT} 换成插件目录再交给 shell 跑；路径不带引号时 node 只收到空格前那一截。
// 这里照宿主的做法：把插件暂存到带中文和空格的目录，替换变量后经 shell 执行（POSIX 是 /bin/sh，Windows 是 cmd.exe）
test("every hook command still runs when the plugin directory has Chinese characters and spaces", async (t) => {
  const parent = await mkdtemp(join(tmpdir(), "habitus 张三 李四-"));
  t.after(() => rm(parent, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 }));
  const root = await prepareMarketplace(join(parent, "plugin marketplace"));
  const stateRoot = join(parent, "state");
  let ran = 0;
  for (const definition of HARNESS_REGISTRY.list().filter((item) => item.hooksManifest)) {
    const staged = join(root, "plugins", definition.pluginDirectory);
    const manifest = JSON.parse(await readFile(join(staged, definition.hooksManifest), "utf8"));
    const commands = new Set(Object.values(manifest.hooks).flat().flatMap((group) => group.hooks).map((hook) => hook.command));
    for (const command of commands) {
      const expanded = command.replace(/\$\{[A-Z_]+\}/g, staged);
      const result = await runShell(expanded, stateRoot);
      assert.equal(result.code, 0, `${definition.id}: ${expanded}\n${result.stderr}`);
      assert.deepEqual(JSON.parse(result.stdout), {}, `${definition.id}: ${expanded}`);
      ran += 1;
    }
  }
  assert.ok(ran >= 8, `only ${ran} hook commands ran`);
});

function runShell(command, stateRoot) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, {
      shell: true,
      env: { ...ISOLATED.env, HABITUS_MEMORY_ENABLED: "0", HABITUS_PLUGIN_STATE_DIR: stateRoot, HABITUS_HOOK_WORKER: "" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk) => { stderr += chunk; });
    child.once("error", reject);
    child.once("close", (code) => resolve({ code, stdout, stderr }));
    child.stdin.end("{}");
  });
}

// 仓库放在带中文和空格的路径下时，靠 import.meta.url 找文件的测试照样找得到（URL 的 pathname 保留百分号编码，要用 fileURLToPath）。
// 把 plugins/ 与根目录的 marketplace 清单复制到这样的目录里，跑按文件位置找夹具与清单的那两条
test("tests that locate files from their own URL still pass when the repository path has non-ASCII characters", async (t) => {
  const parent = await mkdtemp(join(tmpdir(), "habitus 中文路径-"));
  t.after(() => rm(parent, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 }));
  await cp(PLUGINS, join(parent, "plugins"), { recursive: true });
  for (const definition of HARNESS_REGISTRY.list().filter((item) => item.marketplaceManifest)) {
    const destination = join(parent, definition.marketplaceManifest.path);
    await mkdir(dirname(destination), { recursive: true });
    await cp(join(PLUGINS, "..", definition.marketplaceManifest.path), destination);
  }
  const env = { ...ISOLATED.env };
  const result = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [
      "--test",
      "--test-reporter=tap",
      "--test-name-pattern=^checked-in host transcript fixtures remain consumable$",
      "--test-name-pattern=^the repository root carries the same marketplace manifests the installer stages$",
      "memory-plugin-shared/test/host-adapters.test.mjs",
      "memory-plugin-shared/test/cli.test.mjs",
    ], { cwd: join(parent, "plugins"), env, stdio: ["ignore", "pipe", "pipe"] });
    let output = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => { output += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk) => { output += chunk; });
    child.once("error", reject);
    child.once("close", (code) => resolve({ code, output }));
  });
  assert.equal(result.code, 0, result.output);
  assert.match(result.output, /^# pass 2$/m, result.output);
});
