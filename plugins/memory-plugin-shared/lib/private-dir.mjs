import { execFile } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

// 插件自己的目录（状态根及其下各级、marketplace 根）只给本人。POSIX 是 mkdir 的 0o700。
// Windows 上 mode 不起作用，新目录继承上级的 DACL（%USERPROFILE% 下是本人 + SYSTEM + Administrators）：对这次新建出来的
// 最上层那一级跑一次 icacls，去掉继承、只留当前用户 SID 的完全控制并继承给子项（OI/CI），之后其下新建的文件与目录都随它，
// 与 Python 耐久层"只含当前用户 SID、禁继承"同一口径。已经存在的目录不改（与 Python 那边一样）。不需要管理员：改自己建的目录的 DACL。
export async function makePrivateDirectory(path, { platform = process.platform, restrict = restrictToOwner } = {}) {
  const created = await mkdir(path, { recursive: true, mode: 0o700 });
  if (created !== undefined && platform === "win32") await restrict(created);
  return created;
}

let currentUserSid;

function system32(name) {
  return join(process.env.SystemRoot || "C:\\Windows", "System32", name);
}

// 当前用户的 SID（whoami /user 的 CSV："域\\名","S-1-5-21-…"）；按系统目录绝对路径起，不经 PATH
async function userSid() {
  if (currentUserSid) return currentUserSid;
  const { stdout } = await execFileAsync(system32("whoami.exe"), ["/user", "/fo", "csv", "/nh"], { windowsHide: true, timeout: 10_000 });
  const sid = stdout.trim().match(/"(S-1-[0-9-]+)"\s*$/)?.[1];
  if (!sid) throw new Error(`cannot read the current user's SID from whoami: ${stdout.trim()}`);
  currentUserSid = sid;
  return sid;
}

export async function restrictToOwner(directory) {
  const sid = await userSid();
  await execFileAsync(
    system32("icacls.exe"),
    [directory, "/inheritance:r", "/grant:r", `*${sid}:(OI)(CI)F`],
    { windowsHide: true, timeout: 10_000 },
  );
}
