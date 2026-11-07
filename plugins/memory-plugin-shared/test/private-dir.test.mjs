// 插件自己的目录只给本人（private-dir.mjs）。调用口径（只对这次新建出来的最上层那一级收权、已存在的不碰）在任何平台都测；
// Windows 上真起 icacls、再用 PowerShell 读 SDDL 核对"只含当前用户 SID、禁继承、子项随它"的那条只在 Windows 跑。
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { makePrivateDirectory, plainWindowsPath } from "../lib/private-dir.mjs";

async function root(t) {
  const directory = await mkdtemp(join(tmpdir(), "habitus-private-dir-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

test("on Windows every directory this call created is restricted when the root is new; what is created below an existing root inherits", async (t) => {
  const base = await root(t);
  const restricted = [];
  const options = { platform: "win32", restrict: async (paths) => { restricted.push(paths); } };
  const state = join(base, "home", "state");
  // 根连同上级一起新建：这次建出来的每一级，自上而下
  await makePrivateDirectory(join(state, "sessions", "codex"), state, options);
  assert.deepEqual(restricted, [[join(base, "home"), state, join(state, "sessions"), join(state, "sessions", "codex")]]);
  // 根已在、在它下面新建（每个会话一个 outbox 目录）：随根继承，不再收
  await makePrivateDirectory(join(state, "outbox", "codex", "s1"), state, options);
  assert.equal(restricted.length, 1);
  // 只给了路径（根就是它自己）：新建就收
  await makePrivateDirectory(join(base, "marketplace-parent"), undefined, options);
  assert.deepEqual(restricted.at(-1), [join(base, "marketplace-parent")]);
  // 已存在的根：不碰（用户事先建好的目录不改它的 DACL）
  await mkdir(join(base, "mine"));
  await makePrivateDirectory(join(base, "mine"), undefined, options);
  assert.equal(restricted.length, 2);
  // POSIX：不改 DACL，靠 mkdir 的 0o700
  await makePrivateDirectory(join(base, "posix", "child"), undefined, { ...options, platform: "darwin" });
  assert.equal(restricted.length, 2);
});

test("the extended-length prefix Windows mkdir reports is dropped before comparing and restricting", () => {
  assert.equal(plainWindowsPath("\\\\?\\C:\\Users\\张三 李四\\.habitus"), "C:\\Users\\张三 李四\\.habitus");
  assert.equal(plainWindowsPath("\\\\?\\UNC\\server\\share\\x"), "\\\\server\\share\\x");
  assert.equal(plainWindowsPath("C:\\plain"), "C:\\plain");
});

test("on POSIX the directories are created 0700", { skip: process.platform === "win32" && "Windows 没有 POSIX 权限位" }, async (t) => {
  const base = await root(t);
  await makePrivateDirectory(join(base, "a", "b"));
  assert.equal((await stat(join(base, "a"))).mode & 0o777, 0o700);
  assert.equal((await stat(join(base, "a", "b"))).mode & 0o777, 0o700);
});

// 读一个路径的 DACL：是否禁继承，以及每条访问规则（SID|是否继承来的|权限|允许或拒绝）。按 SID 比，不比 SDDL 文本
// （内置 Administrator 这类账号在 SDDL 里会缩写成 LA）
function accessRules(path) {
  const powershell = join(process.env.SystemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  const script = [
    "$acl = Get-Acl -LiteralPath $env:HABITUS_ACL_PATH",
    "'protected=' + $acl.AreAccessRulesProtected",
    "'me=' + [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value",
    "$acl.Access | ForEach-Object { 'rule=' + $_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value + '|' + $_.IsInherited + '|' + $_.FileSystemRights + '|' + $_.AccessControlType }",
  ].join("; ");
  const lines = execFileSync(powershell, ["-NoProfile", "-NonInteractive", "-Command", script], {
    encoding: "utf8", env: { ...process.env, HABITUS_ACL_PATH: path }, windowsHide: true,
  }).trim().split(/\r?\n/);
  const value = (key) => lines.filter((line) => line.startsWith(`${key}=`)).map((line) => line.slice(key.length + 1));
  return { protected: value("protected")[0] === "True", me: value("me")[0], rules: value("rule") };
}

test(
  "on Windows a new plugin directory and what is created inside it are private to the current user",
  { skip: process.platform !== "win32" && "只有 Windows 用 DACL" },
  async (t) => {
    const base = await root(t);
    const created = join(base, "张三 李四");
    const top = join(created, "agent-plugin");
    const sessions = join(top, "sessions");
    await makePrivateDirectory(sessions, top);
    // 插件写文件的方式（mode 0o600）；根已在之后再建的目录（每个会话一个 outbox 目录）
    await writeFile(join(sessions, "state.json"), "{}\n", { mode: 0o600 });
    await makePrivateDirectory(join(top, "outbox", "s1"), top);
    await writeFile(join(top, "outbox", "s1", "item.json"), "{}\n", { mode: 0o600 });
    // 这次新建的每一级：禁继承，只有一条本人完全控制
    for (const path of [created, top, sessions]) {
      const acl = accessRules(path);
      assert.equal(acl.protected, true, `${path}: ${JSON.stringify(acl)}`);
      assert.deepEqual(acl.rules, [`${acl.me}|False|FullControl|Allow`], `${path}: ${JSON.stringify(acl)}`);
    }
    // 之后在里面建的文件与目录：只有继承来的本人那一条
    for (const path of [join(sessions, "state.json"), join(top, "outbox"), join(top, "outbox", "s1"), join(top, "outbox", "s1", "item.json")]) {
      const acl = accessRules(path);
      assert.deepEqual(acl.rules, [`${acl.me}|True|FullControl|Allow`], `${path}: ${JSON.stringify(acl)}`);
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

