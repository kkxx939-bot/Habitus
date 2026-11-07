import { execFile } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { join, relative, isAbsolute } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

// 插件自己的目录（状态根及其下各级、marketplace 根）只给本人。POSIX 是 mkdir 的 0o700。
// Windows 上 mode 不起作用，新目录继承上级的 DACL（%USERPROFILE% 下是本人 + SYSTEM + Administrators）。所以根（root）
// 这一级是这次新建的（或连同它的上级一起新建）时，给新建出来的最上层那一级换上"禁继承、只有当前用户 SID 完全控制并继承给
// 子项"的 DACL，与 Python 耐久层同一口径；根已经在、只是在它下面新建的，随根继承，不再改。已经存在的根也不改（与 Python 那边一样）。
// 不需要管理员：改自己刚建的目录的 DACL。
export async function makePrivateDirectory(path, root = path, { platform = process.platform, restrict = restrictToOwner } = {}) {
  const created = await mkdir(path, { recursive: true, mode: 0o700 });
  if (created === undefined || platform !== "win32") return created;
  // Windows 上 mkdir 回给的是 \\?\ 扩展路径，按普通路径比较
  const topmost = plainWindowsPath(created);
  const below = relative(root, topmost);
  if (below !== "" && !below.startsWith("..") && !isAbsolute(below)) return created;
  await restrict(topmost);
  return created;
}

export function plainWindowsPath(path) {
  if (path.startsWith("\\\\?\\UNC\\")) return `\\\\${path.slice(8)}`;
  if (path.startsWith("\\\\?\\")) return path.slice(4);
  return path;
}

// 用 .NET 的 DirectorySecurity 写一份全新的 DACL：禁继承、不保留继承来的项，只加当前用户（进程令牌里的 SID）一条
// 完全控制并继承给子项。只写 DACL 这一节，不碰属主。Set-Acl 经 SetNamedSecurityInfo 写入，会顺带传给已经建好的子项。
// PowerShell 按系统目录绝对路径起，目录经环境变量传（不拼进脚本）
const RESTRICT_SCRIPT = [
  "$ErrorActionPreference = 'Stop'",
  "$acl = New-Object System.Security.AccessControl.DirectorySecurity",
  "$acl.SetAccessRuleProtection($true, $false)",
  "$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User",
  "$acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($sid, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')))",
  "Set-Acl -LiteralPath $env:HABITUS_PRIVATE_DIRECTORY -AclObject $acl",
].join("; ");

export async function restrictToOwner(directory) {
  const powershell = join(process.env.SystemRoot || "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  await execFileAsync(powershell, ["-NoProfile", "-NonInteractive", "-Command", RESTRICT_SCRIPT], {
    env: { ...process.env, HABITUS_PRIVATE_DIRECTORY: directory },
    windowsHide: true,
    timeout: 30_000,
  });
}
