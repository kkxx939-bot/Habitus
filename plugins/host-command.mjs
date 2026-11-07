// 跑宿主命令（codex、claude、opencode、pi …）的唯一入口：安装器、doctor、habitus plugin status 都经这里起宿主进程。
//
// POSIX 原样交给 spawn / spawnSync，行为与以前逐字一致。
//
// Windows 上 Node 按 PATH 只找 .exe / .com，而 npm 装的宿主是 .cmd 批处理（codex.cmd、opencode.cmd、pi.cmd），
// 直接 spawn 找不到；Node 也不许不经 shell 起 .cmd。所以 Windows 上按 PATH × PATHEXT 自己找：.exe / .com 直接起，
// .cmd / .bat 经 cmd.exe /d /s /c 起、逐个转义参数（与 cross-spawn 同一套转义）。批处理起的宿主是 cmd.exe 的子进程
// （npm 的 codex 还会再起原生 codex.exe），Windows 杀进程不连带子进程，所以要结束时整棵树一起杀（taskkill /T /F）。
// 同步调用要带超时又要杀整棵树，spawnSync 自己做不到（它到点只杀直接子进程），于是 Windows 上同步调用经本文件的
// 运行器起：运行器异步起宿主、到点杀树后退出，spawnSync 只等运行器。
import { ChildProcess, spawn, spawnSync } from "node:child_process";
import { statSync } from "node:fs";
import { win32 } from "node:path";
import { fileURLToPath } from "node:url";

const RUNNER = fileURLToPath(import.meta.url);
const DEFAULT_PATHEXT = ".COM;.EXE;.BAT;.CMD";
const LAUNCHABLE = new Set([".COM", ".EXE", ".BAT", ".CMD"]);
const CMD_META = /([()\][%!^"`<>&|;, *?])/g;
// 运行器到点杀树后的退出码（与 GNU timeout 相同）；调用方只看非 0
export const HOST_TIMEOUT_STATUS = 124;
// 运行器自己卡住时 spawnSync 的后备超时比宿主超时多出的余量
const RUNNER_GRACE_MS = 5000;

// 普通对象当 env 时键是大小写敏感的（Windows 进程 env 里常是 Path 而不是 PATH），按 Windows 的口径不分大小写取
function envValue(env, name) {
  const key = Object.keys(env).find((candidate) => candidate.toUpperCase() === name);
  return key === undefined ? undefined : env[key];
}

function isFile(path) {
  try { return statSync(path).isFile(); } catch { return false; }
}

/**
 * 按 Windows 的找法定位宿主命令：带路径分隔符的只看那一处，否则逐个 PATH 目录、每个目录里按 PATHEXT 顺序试扩展名。
 * 只认 CreateProcess 或 cmd.exe 起得了的 .com / .exe / .bat / .cmd（PATHEXT 里的 .JS / .VBS 等交给脚本宿主，不起）。
 * 找不到返回 null。纯函数（文件存在与否由 exists 判），POSIX 上也能测。
 */
export function resolveWindowsCommand(command, env, exists = isFile) {
  const extensions = (envValue(env, "PATHEXT") || DEFAULT_PATHEXT)
    .split(";").map((item) => item.trim().toUpperCase()).filter((item) => LAUNCHABLE.has(item));
  const own = win32.extname(command).toUpperCase();
  const names = LAUNCHABLE.has(own) ? [command] : extensions.map((extension) => command + extension.toLowerCase());
  const directories = /[\\/]/.test(command)
    ? [""]
    : (envValue(env, "PATH") || "").split(";").map((item) => item.trim().replace(/^"(.*)"$/, "$1")).filter(Boolean);
  for (const directory of directories) {
    for (const name of names) {
      const file = directory ? win32.join(directory, name) : name;
      if (exists(file)) return { file, batch: [".BAT", ".CMD"].includes(win32.extname(file).toUpperCase()) };
    }
  }
  return null;
}

function quoteForCmd(value) {
  // 先按 MSVCRT 规则给参数加引号（引号前的反斜杠加倍、引号转义、结尾反斜杠加倍），再把 cmd 的元字符逐个 ^ 转义
  const quoted = `"${String(value).replace(/(\\*)"/g, '$1$1\\"').replace(/(\\*)$/, "$1$1")}"`;
  return quoted.replace(CMD_META, "^$1");
}

/** 起一个已定位的宿主命令要交给 spawn 的 [文件, 参数, 是否原样传参]。纯函数。 */
export function windowsLaunch(resolved, args, comspec) {
  if (!resolved.batch) return { file: resolved.file, args: [...args], verbatim: false };
  const line = [resolved.file.replace(CMD_META, "^$1"), ...args.map(quoteForCmd)].join(" ");
  return { file: comspec, args: ["/d", "/s", "/c", `"${line}"`], verbatim: true };
}

function systemRoot() {
  return process.env.SystemRoot || process.env.SYSTEMROOT || "C:\\Windows";
}

function comspec() {
  return process.env.ComSpec || win32.join(systemRoot(), "System32", "cmd.exe");
}

function taskkill() {
  return win32.join(systemRoot(), "System32", "taskkill.exe");
}

function killTree(pid, executable = taskkill()) {
  spawnSync(executable, ["/T", "/F", "/PID", String(pid)], { stdio: "ignore", windowsHide: true });
}

/** 同 spawnSync(command, args, options)。Windows 上见文件头：经运行器起，到点杀整棵进程树，超时的退出码是 HOST_TIMEOUT_STATUS。 */
export function runHostCommand(command, args, options = {}) {
  if (process.platform !== "win32") return spawnSync(command, args, options);
  const resolved = resolveWindowsCommand(command, options.env || process.env);
  // 找不到就照原样交给 Node：它报的 ENOENT 与 POSIX 上是同一种错误，调用方按它判"没装"
  if (!resolved) return spawnSync(command, args, options);
  const { timeout, killSignal: _ignored, ...rest } = options;
  const spec = { ...windowsLaunch(resolved, args, comspec()), timeoutMs: timeout || 0, taskkill: taskkill() };
  return spawnSync(process.execPath, [RUNNER, JSON.stringify(spec)], {
    ...rest,
    windowsHide: true,
    killSignal: "SIGKILL",
    ...(timeout ? { timeout: timeout + RUNNER_GRACE_MS } : {}),
  });
}

/** 同 spawn(command, args, options)。Windows 上按上面的找法起，.cmd / .bat 经 cmd.exe。 */
export function spawnHostCommand(command, args, options = {}) {
  if (process.platform !== "win32") return spawn(command, args, options);
  const resolved = resolveWindowsCommand(command, options.env || process.env);
  if (!resolved) return spawn(command, args, options);
  const launch = windowsLaunch(resolved, args, comspec());
  return spawn(launch.file, launch.args, { ...options, windowsVerbatimArguments: launch.verbatim, windowsHide: true });
}

/** 结束 spawnHostCommand 起的宿主。POSIX 照旧发信号；Windows 上整棵进程树一起杀（Windows 没有可转发的 SIGTERM）。 */
export function killHostProcess(child, signal) {
  if (process.platform === "win32" && child instanceof ChildProcess && child.pid && child.exitCode === null && child.signalCode === null) {
    killTree(child.pid);
    return;
  }
  child.kill(signal);
}

// 运行器：node host-command.mjs <spec JSON>。stdio 原样给宿主，宿主退出码原样带回；到点杀树，退出码 HOST_TIMEOUT_STATUS
function runAsRunner(specText) {
  const spec = JSON.parse(specText);
  let timedOut = false;
  const child = spawn(spec.file, spec.args, { stdio: "inherit", windowsVerbatimArguments: spec.verbatim, windowsHide: true });
  const timer = spec.timeoutMs > 0
    ? setTimeout(() => { timedOut = true; killTree(child.pid, spec.taskkill); }, spec.timeoutMs)
    : null;
  child.once("error", () => process.exit(127));
  child.once("exit", (code) => {
    if (timer) clearTimeout(timer);
    process.exit(timedOut ? HOST_TIMEOUT_STATUS : code ?? 1);
  });
}

if (process.argv[1] === RUNNER && process.argv.length === 3) runAsRunner(process.argv[2]);
