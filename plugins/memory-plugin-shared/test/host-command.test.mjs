// 宿主命令在 Windows 上怎么找、怎么起（host-command.mjs）。找法与转义是纯函数，POSIX 上也测；
// 真起一个 .cmd 的那条只在 Windows 跑（POSIX 上 runHostCommand 就是 spawnSync，没有可测的另一支）。
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { HOST_TIMEOUT_STATUS, resolveWindowsCommand, runHostCommand, windowsLaunch } from "../../host-command.mjs";

const existing = (...paths) => (path) => paths.includes(path);

test("a bare command is looked up PATH directory by directory, each in PATHEXT order", () => {
  const env = { PATH: "C:\\first;C:\\second", PATHEXT: ".COM;.EXE;.BAT;.CMD" };
  // 先目录、后扩展名：第一个目录里的 .cmd 胜过第二个目录里的 .exe（与 cmd.exe 的找法一致）
  assert.deepEqual(
    resolveWindowsCommand("codex", env, existing("C:\\first\\codex.cmd", "C:\\second\\codex.exe")),
    { file: "C:\\first\\codex.cmd", batch: true },
  );
  assert.deepEqual(
    resolveWindowsCommand("codex", env, existing("C:\\first\\codex.cmd", "C:\\first\\codex.exe")),
    { file: "C:\\first\\codex.exe", batch: false },
  );
  assert.equal(resolveWindowsCommand("codex", env, existing()), null);
});

test("the environment is read the Windows way: Path in any case, quoted entries, a default PATHEXT", () => {
  // 进程 env 拷成普通对象后键是 Path；PATHEXT 没设用 Windows 的默认值
  const env = { Path: "\"C:\\Users\\张三 李四\\AppData\\Roaming\\npm\";C:\\other" };
  assert.deepEqual(
    resolveWindowsCommand("opencode", env, existing("C:\\Users\\张三 李四\\AppData\\Roaming\\npm\\opencode.cmd")),
    { file: "C:\\Users\\张三 李四\\AppData\\Roaming\\npm\\opencode.cmd", batch: true },
  );
});

test("only what CreateProcess or cmd.exe can start is picked, and an explicit extension or path is taken as given", () => {
  // PATHEXT 里的 .JS 交给 Windows 脚本宿主，不当宿主命令起
  const env = { PATH: "C:\\bin", PATHEXT: ".JS;.EXE" };
  assert.equal(resolveWindowsCommand("pi", env, existing("C:\\bin\\pi.js")), null);
  assert.deepEqual(resolveWindowsCommand("pi", env, existing("C:\\bin\\pi.exe")), { file: "C:\\bin\\pi.exe", batch: false });
  assert.deepEqual(resolveWindowsCommand("pi.bat", env, existing("C:\\bin\\pi.bat")), { file: "C:\\bin\\pi.bat", batch: true });
  // 带路径的不再按 PATH 找
  assert.deepEqual(
    resolveWindowsCommand("D:\\tools\\claude", env, existing("D:\\tools\\claude.exe", "C:\\bin\\claude.exe")),
    { file: "D:\\tools\\claude.exe", batch: false },
  );
});

test("an .exe starts directly; a batch file starts through cmd.exe with every argument quoted and escaped", () => {
  assert.deepEqual(
    windowsLaunch({ file: "C:\\bin\\claude.exe", batch: false }, ["plugin", "list"], "C:\\Windows\\System32\\cmd.exe"),
    { file: "C:\\bin\\claude.exe", args: ["plugin", "list"], verbatim: false },
  );
  const launch = windowsLaunch(
    { file: "C:\\Users\\张三 李四\\npm\\codex.cmd", batch: true },
    ["plugin", "marketplace", "add", "C:\\Users\\张三 李四\\m&p (1)\\", "100%"],
    "C:\\Windows\\System32\\cmd.exe",
  );
  assert.equal(launch.file, "C:\\Windows\\System32\\cmd.exe");
  assert.equal(launch.verbatim, true);
  assert.deepEqual(launch.args.slice(0, 3), ["/d", "/s", "/c"]);
  // 每个参数先加引号（结尾反斜杠加倍，免得吃掉收尾引号），再把 cmd 元字符（含引号、空格、&、括号、%）逐个 ^ 转义
  assert.equal(
    launch.args[3],
    "\"C:\\Users\\张三^ 李四\\npm\\codex.cmd ^\"plugin^\" ^\"marketplace^\" ^\"add^\" "
      + "^\"C:\\Users\\张三^ 李四\\m^&p^ ^(1^)\\\\^\" ^\"100^%^\"\"",
  );
});

test(
  "on Windows an npm-style .cmd host under a Chinese, spaced path is found and gets its arguments verbatim",
  { skip: process.platform !== "win32" && "只有 Windows 经 cmd.exe 起批处理" },
  async (t) => {
    const directory = await mkdtemp(join(tmpdir(), "habitus 宿主 命令-"));
    t.after(() => rm(directory, { recursive: true, force: true }));
    // 与 npm 的 .cmd 同样只用 %dp0% / 环境变量引用路径（批处理按 OEM 代码页解码，不直接写中文）
    await writeFile(join(directory, "echohost.cmd"), "@\"%HABITUS_TEST_NODE%\" \"%~dp0echo.mjs\" %*\r\n");
    await writeFile(join(directory, "echo.mjs"), "process.stdout.write(JSON.stringify(process.argv.slice(2)));\n");
    await writeFile(join(directory, "slowhost.cmd"), "@\"%HABITUS_TEST_NODE%\" -e \"setTimeout(() => {}, 20000)\"\r\n");
    const env = { Path: directory, SystemRoot: process.env.SystemRoot, HABITUS_TEST_NODE: process.execPath };
    const args = ["plugin", "add", "C:\\Users\\张三 李四\\m&p (1)\\", "100%"];
    const result = runHostCommand("echohost", args, { encoding: "utf8", timeout: 10_000, env });
    assert.equal(result.error, undefined);
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(result.stdout), args);
    // 到点整棵树一起杀：spawnSync 不等那个 20 秒的 node
    const started = Date.now();
    const slow = runHostCommand("slowhost", [], { encoding: "utf8", timeout: 1000, env });
    assert.equal(slow.status, HOST_TIMEOUT_STATUS);
    assert.ok(Date.now() - started < 8000, `took ${Date.now() - started} ms`);
    // 找不到的命令与 POSIX 一样是 Node 自己的 ENOENT
    assert.equal(runHostCommand("nohost", [], { env }).error?.code, "ENOENT");
  },
);
