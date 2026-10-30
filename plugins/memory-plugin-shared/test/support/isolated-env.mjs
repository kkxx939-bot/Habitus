// 测试只碰临时目录和假宿主：HOME、XDG_CONFIG_HOME、CODEX_HOME、DSH_HOME 都指向临时目录，PATH 只有 node 和一个假宿主 bin。
// 注册表里每个宿主命令在 bin 里都是一个记录器（只用 sh 的内建，PATH 上没有别的命令）：一调就把（命令、HOME、cwd、参数）记进日志并以 127 失败；
// fakeCodex: true 时 codex 换成 support/fake-codex.mjs（行为由 FAKE_CODEX_* 环境变量定，见那份文件）。
// 子进程照这份 env 起（或进程内把它传给 main / executePluginLifecycle / runPluginDoctor 的 env），宿主命令就只会落到这里
// t 是测试上下文（整个目录在它结束时删掉，假 codex 留下的进程先杀掉），整份文件共用一份时传 node:test 的 { after }
import { chmod, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { HARNESS_REGISTRY } from "../../../harnesses.mjs";

const FAKE_CODEX = fileURLToPath(new URL("./fake-codex.mjs", import.meta.url));

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
  await symlink(process.execPath, join(bin, "node"));
  for (const command of new Set(HARNESS_REGISTRY.list().map((definition) => definition.command))) {
    const path = join(bin, command);
    const script = command === "codex" && fakeCodex
      ? `#!/bin/sh\nexec '${process.execPath}' '${FAKE_CODEX}' "$@"\n`
      : recorderScript(log);
    await writeFile(path, script);
    await chmod(path, 0o755);
  }
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
  };
  return { root, home, bin, log, env, calls: () => recordedCalls(log), pids: () => fakeCodexPids(pids) };
}
