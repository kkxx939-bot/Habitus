// 测试只碰临时目录和假宿主：HOME、XDG_CONFIG_HOME、CODEX_HOME、DSH_HOME 都指向临时目录，PATH 只有 node 和一个假宿主 bin。
// 注册表里每个宿主命令在 bin 里都是一个记录器（只用 sh 的内建，PATH 上没有别的命令）：一调就把（命令、HOME、cwd、参数）记进日志并以 127 失败；
// fakeCodex: true 时 codex 换成 support/fake-codex.mjs（行为由 FAKE_CODEX_* 环境变量定，见那份文件）。
// Windows 没有 sh：假宿主是 <命令>.cmd 批处理（与 npm 装的宿主同一形态，走插件按 PATHEXT 找命令、经 cmd.exe 起的那条路），
// 记录器是 support/recorder.mjs，node 也是一个 node.cmd（建文件符号链接要特权）；env 另带 Windows 进程本来就有的
// SystemRoot、TEMP/TMP、PATHEXT，USERPROFILE 指向同一个临时 HOME（Windows 上 homedir() 取它，等于 POSIX 的 HOME）。
// 子进程照这份 env 起（或进程内把它传给 main / executePluginLifecycle / runPluginDoctor 的 env），宿主命令就只会落到这里
// t 是测试上下文（整个目录在它结束时删掉，假 codex 留下的进程先杀掉），整份文件共用一份时传 node:test 的 { after }
import { chmod, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { HARNESS_REGISTRY } from "../../../harnesses.mjs";

const FAKE_CODEX = fileURLToPath(new URL("./fake-codex.mjs", import.meta.url));
const RECORDER = fileURLToPath(new URL("./recorder.mjs", import.meta.url));
const WINDOWS = process.platform === "win32";

// Windows 批处理：一行起 node，%* 原样转交参数，退出码就是 node 的。路径不直接写进 .cmd：cmd 按 OEM 代码页解码批处理，
// 中文路径会被读坏；经环境变量（UTF-16）展开就不受代码页影响（npm 自己的 .cmd 用 %dp0% 也是这个道理）
const WINDOWS_SHIM_PATHS = { HABITUS_TEST_NODE: process.execPath, HABITUS_TEST_FAKE_CODEX: FAKE_CODEX, HABITUS_TEST_RECORDER: RECORDER };
function windowsShim(...nodeArguments) {
  return `@"%HABITUS_TEST_NODE%" ${nodeArguments.join(" ")} %*\r\n`;
}

async function writeHostCommands(bin, log, fakeCodex) {
  for (const command of new Set(HARNESS_REGISTRY.list().map((definition) => definition.command))) {
    const fake = command === "codex" && fakeCodex;
    if (WINDOWS) {
      await writeFile(join(bin, `${command}.cmd`), fake ? windowsShim('"%HABITUS_TEST_FAKE_CODEX%"') : windowsShim('"%HABITUS_TEST_RECORDER%"', command));
      continue;
    }
    const path = join(bin, command);
    await writeFile(path, fake ? `#!/bin/sh\nexec '${process.execPath}' '${FAKE_CODEX}' "$@"\n` : recorderScript(log));
    await chmod(path, 0o755);
  }
}

function windowsEnv(home) {
  if (!WINDOWS) return {};
  const inherited = (name) => process.env[name];
  return {
    SystemRoot: inherited("SystemRoot"),
    ComSpec: inherited("ComSpec"),
    PATHEXT: inherited("PATHEXT") || ".COM;.EXE;.BAT;.CMD",
    TEMP: tmpdir(),
    TMP: tmpdir(),
    USERPROFILE: home,
    ...WINDOWS_SHIM_PATHS,
  };
}

function recorderScript(log) {
  return [
    "#!/bin/sh",
    `printf '%s\\t%s\\t%s\\t%s\\n' "\${0##*/}" "$HOME" "$PWD" "$*" >> '${log}'`,
    "exit 127",
    "",
  ].join("\n");
}

// 解析记录器与假 codex 写的日志：[{command, home, cwd, argv}]
async function recordedCalls(log) {
  let text = "";
  try { text = await readFile(log, "utf8"); }
  catch (error) { if (error?.code !== "ENOENT") throw error; }
  return text.split("\n").filter(Boolean).map((line) => {
    const [command, home, cwd, argv] = line.split("\t");
    return { command, home, cwd, argv };
  });
}

// 假 codex 起过的长驻进程：[{role, pid}]（role 是 app-server / grandchild / features）
async function fakeCodexPids(path) {
  let text = "";
  try { text = await readFile(path, "utf8"); }
  catch (error) { if (error?.code !== "ENOENT") throw error; }
  return text.split("\n").filter(Boolean).map((line) => {
    const [role, pid] = line.split(" ");
    return { role, pid: Number(pid) };
  });
}

export async function isolatedEnv(t, { fakeCodex = false } = {}) {
  const root = await mkdtemp(join(tmpdir(), "habitus-isolated-"));
  const pids = join(root, "fake-codex-pids");
  t.after(async () => {
    // 假 codex 留下的长驻进程（忽略 SIGTERM 的 app-server、它的孙进程）先杀掉，再删目录
    for (const { pid } of await fakeCodexPids(pids)) {
      try { process.kill(pid, "SIGKILL"); } catch { /* 已经退出 */ }
    }
    await rm(root, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
  });
  const home = join(root, "home");
  const bin = join(root, "bin");
  const log = join(root, "host-calls.tsv");
  for (const directory of [home, bin, join(root, "config"), join(root, "codex-home"), join(root, "dsh-home")]) {
    await mkdir(directory, { recursive: true });
  }
  if (WINDOWS) await writeFile(join(bin, "node.cmd"), windowsShim());
  else await symlink(process.execPath, join(bin, "node"));
  await writeHostCommands(bin, log, fakeCodex);
  const env = {
    HOME: home,
    XDG_CONFIG_HOME: join(root, "config"),
    CODEX_HOME: join(root, "codex-home"),
    DSH_HOME: join(root, "dsh-home"),
    PATH: bin,
    TMPDIR: tmpdir(),
    HABITUS_PLUGIN_STATE_DIR: join(root, "state"),
    FAKE_CODEX_LOG: log,
    FAKE_CODEX_PIDS: pids,
    ...windowsEnv(home),
  };
  return { root, home, bin, log, env, calls: () => recordedCalls(log), pids: () => fakeCodexPids(pids) };
}
