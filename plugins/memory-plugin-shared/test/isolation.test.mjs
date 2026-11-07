import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readdir } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { isolatedEnv } from "./support/isolated-env.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PLUGINS = join(HERE, "..", "..");

// 整套守卫：其余插件测试全部跑在空 HOME、PATH 只有 node 与记录器的环境里。哪条测试没隔离（继承了调用方的 HOME / PATH、
// 进程内用了默认的宿主 runner）就会调到这里的记录器、或往这个 HOME 里写东西；真机上那就是调真实宿主、读写真实宿主目录
// Windows 上判残留锁会起 PowerShell 读进程启动时刻（state-store.mjs），PowerShell 把自己的缓存写到 %USERPROFILE%\AppData 下
// （隔离环境里 USERPROFILE 就是这个临时 HOME）。那是系统程序
// 自己的缓存、不是插件写的：只放过这一棵（及它的上级目录），HOME 里别的任何东西照样算违规
const POWERSHELL_CACHE = ["AppData", "Local", "Microsoft", "Windows", "PowerShell"];
const WINDOWS_ONLY_EMPTY = ["AppData\\Roaming"];

function powershellCache(entry) {
  if (process.platform !== "win32") return false;
  if (WINDOWS_ONLY_EMPTY.includes(entry)) return true;
  const parts = entry.split("\\");
  const shared = Math.min(parts.length, POWERSHELL_CACHE.length);
  return parts.slice(0, shared).every((part, index) => part === POWERSHELL_CACHE[index]);
}

test("no plugin test reaches a host command or the home directory it inherits", { timeout: 300_000 }, async (t) => {
  const isolated = await isolatedEnv(t);
  const env = { ...isolated.env };
  // 默认状态目录在 HOME 下：不替测试指好，写到默认位置的会留在这个 HOME 里
  delete env.HABITUS_PLUGIN_STATE_DIR;
  const files = (await readdir(HERE))
    .filter((name) => name.endsWith(".test.mjs") && name !== basename(fileURLToPath(import.meta.url)))
    .sort()
    .map((name) => join("memory-plugin-shared", "test", name));
  const result = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["--test", "--test-reporter=tap", ...files], {
      cwd: PLUGINS, env, stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => { output += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk) => { output += chunk; });
    child.once("error", reject);
    child.once("close", (code) => resolve({ code, output }));
  });
  assert.equal(result.code, 0, result.output.split("\n").filter((line) => /^not ok|^# (pass|fail)/.test(line.trim())).join("\n"));
  assert.match(result.output, /^# fail 0$/m);
  assert.deepEqual(await isolated.calls(), [], "a test ran a host command from the inherited PATH");
  const written = (await readdir(isolated.home, { recursive: true })).filter((entry) => !powershellCache(entry));
  assert.deepEqual(written, [], "a test wrote into the inherited HOME");
});
