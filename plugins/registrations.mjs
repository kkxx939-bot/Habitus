// 宿主插件的注册方式（方案 36）：每种方式一组 present / inspect / install / remove / restore，安装器与 doctor 都按定义里的
// ``registration`` 选一组来用，不在各处按方式写分支。加一种方式（比如 pi 的 npm 扩展）就是在这里加一组、注册表里声明。
//
// - marketplace：调宿主原生命令（`plugin marketplace add` / `plugin add` …），宿主自己记它装了什么（Codex、Claude Code）。
// - file：在宿主的插件目录放一个带受管标记的入口文件，指向 marketplace 根里的插件（OpenCode、pi）。
// - cordis-patch：装宿主的 hook 桥，在它的 home 级 patch 文件里写一段受管配置（DeepSeek Harness）。

import { mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { MARKETPLACE_NAME, PLUGIN_ID, PLUGIN_NAME } from "./harnesses.mjs";
import { killHostProcess, runHostCommand, spawnHostCommand } from "./host-command.mjs";

// 探测"本机有没有这个宿主"（`<命令> --version`）的超时。宿主 CLI 多是 Node 程序，冷启动在慢机器、Windows 杀毒扫描
// 或高负载下可以到好几秒；超时被判成"没装"会让安装器和 doctor 直接跳过一个装好的宿主，比多等几秒糟得多。
// 没装的命令是立刻 ENOENT，不吃这个超时；只有装了但卡住的宿主会等满（W19）。
export const PROBE_TIMEOUT_MS = 10_000;

// 宿主命令一律同步跑、带超时：到点直接 SIGKILL（spawnSync 的超时信号只发一次、发完还要同步等它退出，
// 宿主若接管了 SIGTERM，默认的 SIGTERM 等于没有超时）。env 是要查的那个环境（PATH、HOME 都取它）
export class CommandRunner {
  constructor(env = process.env, { probeTimeoutMs = PROBE_TIMEOUT_MS } = {}) {
    if (!Number.isInteger(probeTimeoutMs) || probeTimeoutMs <= 0) {
      throw new Error(`probeTimeoutMs must be a positive integer, got ${probeTimeoutMs}`);
    }
    this.env = env;
    this.probeTimeoutMs = probeTimeoutMs;
  }

  available(command) {
    const result = runHostCommand(command, ["--version"], {
      encoding: "utf8", timeout: this.probeTimeoutMs, killSignal: "SIGKILL", env: this.env,
    });
    return result.status === 0 && !result.error;
  }

  json(command, args) {
    const result = runHostCommand(command, args, {
      encoding: "utf8", timeout: 10_000, killSignal: "SIGKILL", maxBuffer: 16 * 1024 * 1024, env: this.env,
    });
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed with status ${result.status}`);
    try { return JSON.parse(result.stdout); }
    catch (error) { throw new Error(`${command} returned invalid JSON`, { cause: error }); }
  }

  run(command, args) {
    const result = runHostCommand(command, args, { stdio: "inherit", timeout: 30_000, killSignal: "SIGKILL", env: this.env });
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed with status ${result.status}`);
  }
}

function commandArguments(values, root) {
  return values.map((value) => value === "{root}" ? root : value);
}

function marketplaceEntries(value) {
  if (Array.isArray(value?.marketplaces)) return value.marketplaces;
  if (Array.isArray(value)) return value;
  return [];
}

function pluginEntries(value) {
  if (Array.isArray(value?.installed)) return value.installed;
  if (Array.isArray(value?.plugins)) return value.plugins;
  if (Array.isArray(value)) return value;
  return [];
}

function marketplaceSource(entry) {
  return entry?.root || entry?.path || entry?.source?.path || entry?.marketplaceSource?.source || null;
}

function pluginIsInstalled(entry) {
  return entry?.installed !== false;
}

// ── 文件注册（OpenCode）：在宿主的插件目录放一个入口文件，指向 marketplace 根里的插件 ────────────────────────
const ENTRY_MARKER = "// habitus-memory: managed by `habitus plugin`. Do not edit.";

// 入口文件放在哪棵目录下：config 是 XDG 配置目录（OpenCode），home 是用户主目录（pi 的 ~/.pi）
function configBase(definition, env) {
  if (definition.fileEntry.base === "home") return resolve(env.HOME || homedir());
  return env.XDG_CONFIG_HOME ? resolve(env.XDG_CONFIG_HOME) : join(env.HOME || homedir(), ".config");
}

export function fileEntryPath(definition, env = process.env) {
  return join(configBase(definition, env), ...definition.fileEntry.configPath, definition.fileEntry.name);
}

async function exists(path) {
  try { await stat(path); return true; }
  catch (error) { if (error?.code === "ENOENT") return false; throw error; }
}

export function fileEntrySource(definition, root) {
  const target = pathToFileURL(join(root, "plugins", definition.pluginDirectory, definition.fileEntry.module)).href;
  return `${ENTRY_MARKER}\nexport { default } from ${JSON.stringify(target)};\n`;
}

// 文件注册不需要宿主 CLI：宿主的配置目录在（桌面版不一定把 CLI 放进 PATH）、或入口已经在，就算本机有这个宿主。
async function fileEntryPresent(definition, runner, env = process.env) {
  return await exists(fileEntryPath(definition, env))
    || await exists(join(configBase(definition, env), definition.fileEntry.configPath[0]))
    || runner.available(definition.command);
}

// 入口指向的模块不在了（根目录被删、换过 --root）算悬空，不算已注册。
async function inspectFileEntry(definition, runner, env = process.env) {
  let text = null;
  try { text = await readFile(fileEntryPath(definition, env), "utf8"); }
  catch (error) { if (error?.code !== "ENOENT") throw error; }
  const available = await fileEntryPresent(definition, runner, env);
  const ours = typeof text === "string" && text.startsWith(ENTRY_MARKER);
  const targetUrl = ours ? text.match(/from "(file:[^"]+)"/)?.[1] : null;
  const target = targetUrl ? fileURLToPath(targetUrl) : null;
  const suffix = join("plugins", definition.pluginDirectory, definition.fileEntry.module);
  const root = target?.endsWith(suffix) ? target.slice(0, -suffix.length - 1) : null;
  const live = Boolean(root) && await exists(target);
  return {
    harness: definition.id,
    available,
    marketplaceSource: root,
    installed: live,
    enabled: live,
    dangling: Boolean(ours && !live),
    foreignEntry: typeof text === "string" && !ours,
  };
}

async function writeFileEntry(definition, root, env) {
  const path = fileEntryPath(definition, env);
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  let existing = null;
  try { existing = await readFile(path, "utf8"); }
  catch (error) { if (error?.code !== "ENOENT") throw error; }
  if (existing != null && !existing.startsWith(ENTRY_MARKER)) {
    throw new Error(
      `${path} exists and is not managed by Habitus; move it away first`
      + ` (or install the other harnesses only, e.g. --harness codex --harness claude-code)`,
    );
  }
  await writeFile(path, fileEntrySource(definition, root), { mode: 0o600 });
}

async function removeFileEntry(definition, env) {
  const path = fileEntryPath(definition, env);
  let existing = null;
  try { existing = await readFile(path, "utf8"); }
  catch (error) { if (error?.code === "ENOENT") return false; throw error; }
  if (!existing.startsWith(ENTRY_MARKER)) return false;
  await rm(path, { force: true });
  return true;
}

const MARKETPLACE = Object.freeze({
  // 宿主自己记装了什么，本机有没有它就看它的 CLI
  async present(definition, runner) { return runner.available(definition.command); },

  async inspect(definition, runner) {
    if (!runner.available(definition.command)) {
      return { harness: definition.id, available: false, marketplaceSource: null, installed: false, enabled: false };
    }
    const marketplaces = marketplaceEntries(runner.json(definition.command, definition.marketplaceInventory));
    const plugins = pluginEntries(runner.json(definition.command, definition.pluginInventory));
    const marketplace = marketplaces.find((entry) => entry?.name === MARKETPLACE_NAME);
    const plugin = plugins.find((entry) => (
      entry?.pluginId === PLUGIN_ID
      || (entry?.name === PLUGIN_NAME
        && (entry?.marketplaceName === MARKETPLACE_NAME || entry?.marketplace === MARKETPLACE_NAME))
    ));
    return {
      harness: definition.id,
      available: true,
      marketplaceSource: marketplaceSource(marketplace),
      installed: Boolean(plugin && pluginIsInstalled(plugin)),
      enabled: Boolean(plugin && plugin?.enabled !== false),
    };
  },

  async remove(snapshot, definition, runner) {
    if (snapshot.installed) runner.run(definition.command, definition.pluginRemove);
    if (snapshot.marketplaceSource) runner.run(definition.command, definition.marketplaceRemove);
  },

  async install(snapshot, definition, root, runner, { refresh }) {
    const sourceChanged = snapshot.marketplaceSource != null
      && resolve(snapshot.marketplaceSource) !== resolve(root);
    const mustRefresh = refresh || sourceChanged || (snapshot.installed && !snapshot.enabled);
    if (mustRefresh) await MARKETPLACE.remove(snapshot, definition, runner);
    const marketplacePresent = snapshot.marketplaceSource && !mustRefresh;
    const pluginPresent = snapshot.installed && snapshot.enabled && !mustRefresh;
    if (!marketplacePresent) {
      runner.run(definition.command, commandArguments(definition.marketplaceAdd, root));
    }
    if (!pluginPresent) runner.run(definition.command, definition.pluginAdd);
    if (definition.pluginEnable) runner.run(definition.command, definition.pluginEnable);
  },

  async restore(snapshot, definition, runner) {
    if (!snapshot.available) return;
    const current = await MARKETPLACE.inspect(definition, runner);
    try { await MARKETPLACE.remove(current, definition, runner); } catch {}
    if (snapshot.marketplaceSource) {
      runner.run(
        definition.command,
        commandArguments(definition.marketplaceAdd, snapshot.marketplaceSource),
      );
      if (snapshot.installed) {
        runner.run(definition.command, definition.pluginAdd);
        if (definition.pluginEnable) runner.run(definition.command, definition.pluginEnable);
      }
    }
  },
});

const FILE = Object.freeze({
  present: fileEntryPresent,
  inspect: (definition, runner, env) => inspectFileEntry(definition, runner, env),
  // 文件入口不在宿主 CLI 上登记，删除不需要它：在就删（只删我们自己的）
  async remove(_snapshot, definition, _runner, env) { await removeFileEntry(definition, env); },
  async install(_snapshot, definition, root, _runner, { env }) { await writeFileEntry(definition, root, env); },
  async restore(snapshot, definition, _runner, env) {
    if (snapshot.installed && snapshot.marketplaceSource) await writeFileEntry(definition, snapshot.marketplaceSource, env);
    else await removeFileEntry(definition, env);
  },
  // 宿主 CLI 不在 PATH 上（桌面版）也能装、能卸
  needsCommand: false,
});

// ── cordis patch（DeepSeek Harness）：用 dsh 自己的插件命令把它的 Claude Code hook 桥装进 profile，再在 home 级
// `$DSH_HOME/cordis.patch.yml` 里插一条桥的配置，指向 marketplace 根里的插件（hooks.json 与 ${CLAUDE_PLUGIN_ROOT}）。
// 这段只写在受管标记之间，别人写的内容不动；那份文件是流式写法（`[...]`）时不改、报错说怎么手动加。
const PATCH_BEGIN = "# habitus-memory: begin (managed by `habitus plugin`; do not edit between these lines)";
const PATCH_END = "# habitus-memory: end";

function dshHome(env) {
  return env.DSH_HOME ? resolve(env.DSH_HOME) : join(env.HOME || homedir(), ".dsh");
}

export function patchFilePath(_definition, env = process.env) {
  return join(dshHome(env), "cordis.patch.yml");
}

export function patchBlock(definition, root) {
  const plugin = join(root, "plugins", definition.pluginDirectory);
  return [
    PATCH_BEGIN,
    `- name: ${JSON.stringify(definition.patchEntry.package)}`,
    "  config:",
    `    configPath: ${JSON.stringify(join(plugin, definition.hooksManifest))}`,
    `    pluginRoot: ${JSON.stringify(plugin)}`,
    PATCH_END,
  ].join("\n");
}

function splitPatch(text) {
  const begin = text.indexOf(PATCH_BEGIN);
  if (begin < 0) return null;
  const end = text.indexOf(PATCH_END, begin);
  if (end < 0) throw new Error("cordis.patch.yml has a Habitus begin marker without its end marker; fix it by hand");
  return { before: text.slice(0, begin), block: text.slice(begin, end + PATCH_END.length), after: text.slice(end + PATCH_END.length) };
}

async function readPatch(path) {
  try { return await readFile(path, "utf8"); }
  catch (error) { if (error?.code === "ENOENT") return null; throw error; }
}

function onlyComments(text) {
  return text.split("\n").every((line) => !line.trim() || line.trim().startsWith("#"));
}

const CORDIS_PATCH = Object.freeze({
  // 装要跑 dsh 自己的插件命令；卸与恢复只改 patch 文件，不要命令
  needsCommand: true,
  // 与文件注册同口径：dsh 的 home 在、或命令在，就算本机有
  async present(definition, runner, env) {
    return runner.available(definition.command) || await exists(dshHome(env));
  },
  missingCommandReason(definition, env) {
    return `本机有 ${shownPath(dshHome(env), env)} 但 PATH 上没有 ${definition.command} 命令`;
  },
  async inspect(definition, runner, env) {
    const text = await readPatch(patchFilePath(definition, env));
    const available = await CORDIS_PATCH.present(definition, runner, env);
    const parts = text == null ? null : splitPatch(text);
    const pluginRoot = parts?.block.match(/pluginRoot: ("[^"\n]*")/)?.[1];
    const plugin = pluginRoot ? JSON.parse(pluginRoot) : null;
    const suffix = join("plugins", definition.pluginDirectory);
    const root = plugin?.endsWith(suffix) ? plugin.slice(0, -suffix.length - 1) : null;
    const live = Boolean(root) && await exists(join(plugin, definition.hooksManifest));
    return {
      harness: definition.id,
      available,
      marketplaceSource: root,
      installed: live,
      enabled: live,
      dangling: Boolean(parts && !live),
    };
  },
  async install(_snapshot, definition, root, runner, { env }) {
    const { package: name, profile } = definition.patchEntry;
    runner.run(definition.command, ["plugin", "--profile", profile, "add", name]);
    await writePatchBlock(definition, patchBlock(definition, root), env);
  },
  async remove(_snapshot, definition, _runner, env) {
    const path = patchFilePath(definition, env);
    const text = await readPatch(path);
    const parts = text == null ? null : splitPatch(text);
    if (!parts) return;
    const rest = `${parts.before.trimEnd()}\n${parts.after.replace(/^\n+/, "")}`.trim();
    // 只剩注释的空 patch 会让 dsh 起不来：没别的内容就整份删掉（等于没有 home 级这一层）
    if (!rest || onlyComments(rest)) await rm(path, { force: true });
    else await writeFile(path, `${rest}\n`, { mode: 0o600 });
  },
  async restore(snapshot, definition, runner, env) {
    if (snapshot.installed && snapshot.marketplaceSource) {
      await writePatchBlock(definition, patchBlock(definition, snapshot.marketplaceSource), env);
    } else {
      await CORDIS_PATCH.remove(snapshot, definition, runner, env);
    }
  },
});

async function writePatchBlock(definition, block, env) {
  const path = patchFilePath(definition, env);
  const text = await readPatch(path);
  let next;
  if (text == null || !text.trim() || text.trim() === "[]") next = `${block}\n`;
  else {
    const parts = splitPatch(text);
    if (parts) next = `${parts.before}${block}${parts.after}`;
    else if (text.trimStart().startsWith("[")) {
      throw new Error(`${path} is a flow-style YAML list; add this entry to it by hand:\n${block}`);
    } else next = `${text.trimEnd()}\n\n${block}\n`;
  }
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  await writeFile(path, next, { mode: 0o600 });
}

function shownPath(path, env) {
  const home = resolve(env.HOME || homedir());
  return path === home || path.startsWith(`${home}${sep}`) ? `~${path.slice(home.length)}` : path;
}

const REGISTRATIONS = Object.freeze({
  marketplace: {
    ...MARKETPLACE,
    needsCommand: true,
    missingCommandReason: (definition) => `PATH 上没有 ${definition.command} 命令`,
  },
  file: FILE,
  "cordis-patch": CORDIS_PATCH,
});

export function registrationOf(definition) {
  const registration = REGISTRATIONS[definition.registration];
  if (!registration) throw new Error(`unknown harness registration: ${definition.registration}`);
  return registration;
}

// "本机有没有这个宿主"只有这一个口径：habitus harnesses、plugin install / status、doctor 都经它（inspect 也是）
export function harnessPresent(definition, runner, env = process.env) {
  return registrationOf(definition).present(definition, runner, env);
}

// 本机有这个宿主、但它的注册方式要跑宿主命令而命令不在 PATH 上：返回装不了的原因，装得了返回 null
export function installBlocker(definition, runner, env = process.env) {
  const registration = registrationOf(definition);
  if (!registration.needsCommand || runner.available(definition.command)) return null;
  return registration.missingCommandReason(definition, env);
}

// 宿主要求用户逐条批准插件 hook 时（定义里的 hookReview），问宿主我们的 hook 批没批：
// {available, hooks: [{event, trustStatus}], pending: [没批准的 event]}；宿主没装或没答给 available: false。
// 怎么问是宿主的协议，按 hookReview.kind 选一组（与注册方式按 registration 选一组同理），协调层不写宿主分支
export function inspectHookReview(definition, options = {}) {
  if (!definition.hookReview) return Promise.resolve(null);
  const review = HOOK_REVIEWS[definition.hookReview.kind];
  if (!review) throw new Error(`${definition.id} has an unknown hookReview kind: ${definition.hookReview.kind}`);
  return review(definition, options);
}

// 问完（答了、没答、出错）都把 app-server 收干净再交结果：关 stdin、断开两条管道、SIGTERM，宽限内没退就 SIGKILL，
// 等到它退出。宿主可能接管 SIGTERM、它的子进程可能占着 stdout：只发一次 SIGTERM 不等它退，doctor / plugin status
// 印完结果也会被子进程的句柄吊着不退出。最长用时是 timeoutMs + 2 × HOOK_REVIEW_KILL_GRACE_MS
export const HOOK_REVIEW_KILL_GRACE_MS = 1000;

// Codex：`codex app-server` 的 JSON-RPC（initialize → hooks/list），trustStatus 由 Codex 自己判（trusted / managed 才算批了）
function inspectCodexAppServerHooks(definition, { spawnImpl = spawnHostCommand, timeoutMs = 15_000, env = process.env, cwd = env.HOME || homedir() } = {}) {
  return new Promise((resolvePromise) => {
    let child;
    let settled = false;
    let exited = false;
    let buffer = "";
    const signal = (name) => {
      try { if (child) killHostProcess(child, name); } catch { /* 已经退出 */ }
    };
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { child?.stdin?.end(); } catch { /* 管道已断 */ }
      child?.stdin?.destroy?.();
      child?.stdout?.destroy?.();
      if (exited) { resolvePromise(value); return; }
      signal("SIGTERM");
      // 没起来的（没有 pid）没有进程可等
      if (!child?.pid) { resolvePromise(value); return; }
      const kill = setTimeout(() => {
        signal("SIGKILL");
        // SIGKILL 之后仍等不到 exit 就不再等，结果照样交回去
        setTimeout(() => resolvePromise(value), HOOK_REVIEW_KILL_GRACE_MS);
      }, HOOK_REVIEW_KILL_GRACE_MS);
      child.once("exit", () => {
        clearTimeout(kill);
        resolvePromise(value);
      });
    };
    const timer = setTimeout(() => finish({ available: false, error: "harness did not answer" }), timeoutMs);
    const send = (message) => child.stdin.write(`${JSON.stringify(message)}\n`);
    try {
      child = spawnImpl(definition.command, definition.hookReview.args, { cwd, env, stdio: ["pipe", "pipe", "ignore"] });
    } catch (error) {
      finish({ available: false, error: error?.message || String(error) });
      return;
    }
    child.on("error", (error) => finish({ available: false, error: error?.code === "ENOENT" ? "harness is not installed" : error.message }));
    child.on("exit", () => {
      exited = true;
      finish({ available: false, error: "harness exited before answering" });
    });
    child.stdin.on("error", () => {});
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      buffer += chunk;
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        let message;
        try { message = JSON.parse(line); } catch { continue; }
        if (message.id === 1) {
          send({ method: "initialized" });
          send({ id: 2, method: "hooks/list", params: { cwds: [cwd] } });
        } else if (message.id === 2) {
          if (message.error) {
            finish({ available: false, error: message.error.message || "hooks/list failed" });
            return;
          }
          const hooks = (message.result?.data || [])
            .flatMap((entry) => entry.hooks || [])
            .filter((hook) => hook.pluginId === PLUGIN_ID)
            .map((hook) => ({ event: hook.eventName, trustStatus: hook.trustStatus }));
          const pending = hooks.filter((hook) => !["trusted", "managed"].includes(hook.trustStatus)).map((hook) => hook.event);
          finish({ available: true, hooks, pending });
        }
      }
    });
    send({
      id: 1,
      method: "initialize",
      params: { clientInfo: { name: "habitus", title: null, version: "1" }, capabilities: { experimentalApi: true, requestAttestation: false } },
    });
  });
}

const HOOK_REVIEWS = { "codex-app-server": inspectCodexAppServerHooks };
