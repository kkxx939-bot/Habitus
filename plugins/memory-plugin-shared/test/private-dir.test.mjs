// 插件自己的目录只给本人（private-dir.mjs）。调用口径（只对这次新建出来的最上层那一级收权、已存在的不碰）在任何平台都测；
// Windows 上真起 icacls、再用 PowerShell 读 SDDL 核对"只含当前用户 SID、禁继承、子项随它"的那条只在 Windows 跑。
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { makePrivateDirectory } from "../lib/private-dir.mjs";

async function root(t) {
  const directory = await mkdtemp(join(tmpdir(), "habitus-private-dir-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

test("on Windows only the topmost directory this call created is restricted, once; existing ones are left alone", async (t) => {
  const base = await root(t);
  const restricted = [];
  const restrict = async (path) => { restricted.push(path); };
  await makePrivateDirectory(join(base, "state", "sessions", "codex"), { platform: "win32", restrict });
  assert.deepEqual(restricted, [join(base, "state")]);
  // 再建其下一级：只有新建的那一级
  await makePrivateDirectory(join(base, "state", "sessions", "codex", "lock"), { platform: "win32", restrict });
  assert.deepEqual(restricted, [join(base, "state"), join(base, "state", "sessions", "codex", "lock")]);
  // 已存在：不碰（用户事先建好的目录不改它的 DACL）
  await mkdir(join(base, "mine"));
  await makePrivateDirectory(join(base, "mine"), { platform: "win32", restrict });
  assert.equal(restricted.length, 2);
  // POSIX：不起 icacls，靠 mkdir 的 0o700
  await makePrivateDirectory(join(base, "posix", "child"), { platform: "darwin", restrict });
  assert.equal(restricted.length, 2);
});

test("on POSIX the directories are created 0700", { skip: process.platform === "win32" && "Windows 没有 POSIX 权限位" }, async (t) => {
  const base = await root(t);
  await makePrivateDirectory(join(base, "a", "b"));
  assert.equal((await stat(join(base, "a"))).mode & 0o777, 0o700);
  assert.equal((await stat(join(base, "a", "b"))).mode & 0o777, 0o700);
});

function sddl(path) {
  const powershell = join(process.env.SystemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  return execFileSync(powershell, ["-NoProfile", "-NonInteractive", "-Command", "(Get-Acl -LiteralPath $env:HABITUS_ACL_PATH).Sddl"], {
    encoding: "utf8", env: { ...process.env, HABITUS_ACL_PATH: path }, windowsHide: true,
  }).trim();
}

test(
  "on Windows a new plugin directory and what is created inside it are private to the current user",
  { skip: process.platform !== "win32" && "只有 Windows 用 DACL" },
  async (t) => {
    const base = await root(t);
    const top = join(base, "张三 李四", "agent-plugin");
    await makePrivateDirectory(join(top, "sessions"));
    await writeFile(join(top, "sessions", "state.json"), "{}\n");
    const sid = execFileSync(join(process.env.SystemRoot, "System32", "whoami.exe"), ["/user", "/fo", "csv", "/nh"], { encoding: "utf8" })
      .trim().match(/"(S-1-[0-9-]+)"\s*$/)[1];
    // 新建的最上层：禁继承（D:P），只有一条本人完全控制、继承给子项
    const created = join(base, "张三 李四");
    assert.match(sddl(created), new RegExp(`D:P\\(A;OICI;FA;;;${sid}\\)$`), sddl(created));
    // 其下的目录与文件：只继承那一条
    for (const path of [top, join(top, "sessions"), join(top, "sessions", "state.json")]) {
      const value = sddl(path);
      const aces = value.slice(value.indexOf("D:")).match(/\([^)]*\)/g);
      assert.equal(aces.length, 1, `${path}: ${value}`);
      assert.match(aces[0], new RegExp(`;FA;;;${sid}\\)$`), `${path}: ${value}`);
    }
  },
);

// 插件自己建目录一律经 makePrivateDirectory（否则 Windows 上那一级继承 %USERPROFILE% 的 DACL）：共享库、各插件自己的脚本
// 与安装器里不再有带 recursive 的裸 mkdir。安装器暂存目录里面那一处随暂存目录继承，是唯一例外
test("plugin code creates its own directories only through makePrivateDirectory", async () => {
  const plugins = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
  const files = [
    ...(await readdir(join(plugins, "memory-plugin-shared", "lib"))).map((name) => join(plugins, "memory-plugin-shared", "lib", name)),
    join(plugins, "install-memory-plugin.mjs"),
  ];
  for (const entry of await readdir(plugins, { withFileTypes: true })) {
    if (!entry.isDirectory() || !entry.name.startsWith("habitus-memory")) continue;
    for (const name of await readdir(join(plugins, entry.name, "scripts"))) {
      if (name.endsWith(".mjs")) files.push(join(plugins, entry.name, "scripts", name));
    }
  }
  const offending = [];
  for (const file of files.filter((path) => path.endsWith(".mjs") && !path.endsWith("private-dir.mjs"))) {
    const lines = (await readFile(file, "utf8")).split("\n");
    lines.forEach((line, index) => {
      if (/\bmkdir\(.*recursive: true/.test(line) && !line.includes("dirname(destination)")) offending.push(`${file}:${index + 1}`);
    });
  }
  assert.deepEqual(offending, []);
});

