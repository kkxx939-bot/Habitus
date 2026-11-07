import { execFile } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

// 插件自己的目录（状态根及其下各级、marketplace 根）只给本人。POSIX 是 mkdir 的 0o700。
// Windows 上 mode 管不住访问，新目录照上级的 DACL 带进 SYSTEM、Administrators 等（windows-latest 实测 Node 新建的目录上
// 这几条是显式项，不是继承项）。所以根（root）这一级是这次新建的（或连同它的上级一起新建）时，把这次新建出来的每一级都换成
// "禁继承、只有当前用户 SID 完全控制并继承给子项"，与 Python 耐久层同一口径；根已经在、只是在它下面新建的，随根继承，不再改。
// 已经存在的根也不改（与 Python 那边一样）。不需要管理员：改自己刚建的目录的 DACL。
export async function makePrivateDirectory(path, root = path, { platform = process.platform, restrict = restrictToOwner } = {}) {
  const created = await mkdir(path, { recursive: true, mode: 0o700 });
  if (created === undefined || platform !== "win32") return created;
  // Windows 上 mkdir 回给的是 \\?\ 扩展路径，按普通路径比较
  const topmost = plainWindowsPath(created);
  const below = relative(root, topmost);
  if (below !== "" && !below.startsWith("..") && !isAbsolute(below)) return created;
  await restrict(newlyCreated(topmost, path));
  return created;
}

// 这次新建出来的各级：从最上层那一级到 path，自上而下。走到盘符根还对不上就报错，绝不去改这次没建的上级目录
function newlyCreated(topmost, path) {
  const chain = [];
  for (let current = resolve(path); ; current = dirname(current)) {
    chain.unshift(current);
    if (current === topmost) return chain;
    if (dirname(current) === current) throw new Error(`created directory ${topmost} is not an ancestor of ${path}`);
  }
}

export function plainWindowsPath(path) {
  if (path.startsWith("\\\\?\\UNC\\")) return `\\\\${path.slice(8)}`;
  if (path.startsWith("\\\\?\\")) return path.slice(4);
  return path;
}

// 新目录上可能以显式项出现的别的主体：SYSTEM、Administrators、Users、Authenticated Users、Everyone、CREATOR OWNER
const OTHER_PRINCIPALS = ["*S-1-5-18", "*S-1-5-32-544", "*S-1-5-32-545", "*S-1-5-11", "*S-1-1-0", "*S-1-3-0"];

let currentUserSid;

function system32(name) {
  return join(process.env.SystemRoot || "C:\\Windows", "System32", name);
}

// 当前用户的 SID（whoami /user 的 CSV："域\\名","S-1-5-21-…"）。icacls 与 whoami 都按系统目录绝对路径起、不依赖 PATH 与
// 其他环境变量（PowerShell 在宿主给的精简环境里会卡住，windows-latest 第 3 轮实测）
async function userSid() {
  if (currentUserSid) return currentUserSid;
  const { stdout } = await execFileAsync(system32("whoami.exe"), ["/user", "/fo", "csv", "/nh"], { windowsHide: true, timeout: 10_000 });
  const sid = stdout.trim().match(/"(S-1-[0-9-]+)"\s*$/)?.[1];
  if (!sid) throw new Error(`cannot read the current user's SID from whoami: ${stdout.trim()}`);
  currentUserSid = sid;
  return sid;
}

// 每一级：去掉继承项、把本人那条换成完全控制并继承给子项，再去掉别的主体的显式项
export async function restrictToOwner(directories) {
  const sid = await userSid();
  const icacls = system32("icacls.exe");
  const options = { windowsHide: true, timeout: 10_000 };
  for (const directory of directories) {
    await execFileAsync(icacls, [directory, "/inheritance:r", "/grant:r", `*${sid}:(OI)(CI)F`], options);
    await execFileAsync(icacls, [directory, "/remove:g", ...OTHER_PRINCIPALS], options);
  }
}
