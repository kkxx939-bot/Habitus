// 把一个目录变成"进不去写"（不能在里面新建、改名、删除）再恢复：测试注入写失败用。
// POSIX 是 chmod 0o500 / 0o700。Windows 上目录不认权限位（chmod 只碰文件的只读属性），改用 icacls 给当前用户加一条
// 可继承的拒绝 ACE（写数据/加文件、加子目录、删除、删子项），恢复时去掉；改自己文件的 DACL 不需要管理员
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, win32 } from "node:path";

function icacls(...args) {
  const executable = win32.join(process.env.SystemRoot || "C:\\Windows", "System32", "icacls.exe");
  const result = spawnSync(executable, args, { encoding: "utf8", windowsHide: true });
  if (result.error || result.status !== 0) {
    throw new Error(`icacls ${args.join(" ")} failed: ${result.error?.message || ""}${result.stdout}${result.stderr}`);
  }
}

export function denyWritesSync(path) {
  if (process.platform !== "win32") { chmodSync(path, 0o500); return; }
  // /T：也直接加到已经在里面的子项上（只加在本目录上时，现有子项不受影响，删得掉也写得进）
  icacls(path, "/deny", `${process.env.USERNAME}:(OI)(CI)(WD,AD,DE,DC)`, "/T", "/C");
}

export function allowWritesSync(path) {
  if (process.platform !== "win32") { chmodSync(path, 0o700); return; }
  icacls(path, "/remove:d", process.env.USERNAME, "/T", "/C");
}

// 这台机器上"拒写"到底拦不拦得住当前账号：建个临时目录拒写、试着在里面写一个文件。拦不住的（POSIX 的 root；Windows 上
// 带备份/还原特权的管理员，windows-latest 的 runneradmin 就是，拒绝 ACE 对它不起作用）造不出写失败，依赖它的用例只能跳过
let denialTakesEffect;

export function writeDenialTakesEffect() {
  if (denialTakesEffect !== undefined) return denialTakesEffect;
  const directory = mkdtempSync(join(tmpdir(), "habitus-deny-probe-"));
  denyWritesSync(directory);
  try {
    writeFileSync(join(directory, "probe"), "x");
    denialTakesEffect = false;
  } catch {
    denialTakesEffect = true;
  } finally {
    allowWritesSync(directory);
    rmSync(directory, { recursive: true, force: true });
  }
  return denialTakesEffect;
}

export const DENIAL_SKIP_REASON = "当前账号不受拒写限制（POSIX 的 root，或 Windows 上带备份/还原特权的管理员），造不出写失败";

