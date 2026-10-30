// 假 codex（真 OS 进程）：isolatedEnv({ fakeCodex: true }) 把它放进 PATH。每次调用都记进 FAKE_CODEX_LOG，
// 长驻的进程把"角色 pid"记进 FAKE_CODEX_PIDS，测试据此确认它们都已退出。行为由环境变量定：
// - 本机装了它、插件已注册（marketplace list / plugin list 都列出 Habitus），hooks 特性开着
// - FAKE_CODEX_APP_SERVER：answer（默认，两步都答、收到 SIGTERM 就退）、silent-ignore-term（不答、忽略 SIGTERM）、
//   answer-ignore-term（两步都答、忽略 SIGTERM）、grandchild（不答、收到 SIGTERM 就退，但留一个孙进程占着 stdout）
// - FAKE_CODEX_FEATURES=hang：`features list` 忽略 SIGTERM、睡 30 s 不答
import { spawn } from "node:child_process";
import { appendFileSync } from "node:fs";

const argv = process.argv.slice(2);
const env = process.env;
appendFileSync(env.FAKE_CODEX_LOG, `codex\t${env.HOME}\t${process.cwd()}\t${argv.join(" ")}\n`);
const remember = (role, pid = process.pid) => appendFileSync(env.FAKE_CODEX_PIDS, `${role} ${pid}\n`);
const stayAlive = (ms) => setTimeout(() => {}, ms);

function appServer(mode) {
  remember("app-server");
  if (mode.endsWith("ignore-term")) process.on("SIGTERM", () => {});
  else process.on("SIGTERM", () => process.exit(0));
  if (mode === "grandchild") {
    const grandchild = spawn(process.execPath, ["-e", "setTimeout(() => {}, 30000)"], { stdio: ["ignore", "inherit", "ignore"] });
    remember("grandchild", grandchild.pid);
    grandchild.unref();
  }
  let buffer = "";
  process.stdin.setEncoding("utf8").on("data", (chunk) => {
    buffer += chunk;
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const message = JSON.parse(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      if (mode === "silent-ignore-term" || mode === "grandchild") continue;
      if (message.id === 1) process.stdout.write(`${JSON.stringify({ id: 1, result: {} })}\n`);
      if (message.id === 2) process.stdout.write(`${JSON.stringify({ id: 2, result: { data: [{ cwd: env.HOME, hooks: [] }] } })}\n`);
    }
  });
  // 忽略 SIGTERM 的变体关了 stdin 也不退，只有 SIGKILL 能结束它
  stayAlive(60_000);
}

const command = argv.join(" ");
if (command === "--version") process.stdout.write("codex-cli 0.0.0-fake\n");
else if (command === "plugin marketplace list --json") {
  process.stdout.write(JSON.stringify({ marketplaces: [{ name: "habitus-local", root: `${env.HOME}/.habitus/plugin-marketplace` }] }));
} else if (command === "plugin list --json --available") {
  process.stdout.write(JSON.stringify({ installed: [{ pluginId: "habitus-memory@habitus-local", installed: true, enabled: true }] }));
} else if (command === "features list") {
  if (env.FAKE_CODEX_FEATURES === "hang") {
    remember("features");
    process.on("SIGTERM", () => {});
    stayAlive(30_000);
  } else process.stdout.write("hooks                                stable             true\n");
} else if (command === "app-server") appServer(env.FAKE_CODEX_APP_SERVER || "answer");
else process.exitCode = 127;
