"""Windows 上 PowerShell 执行策略的 doctor 检查。

npm 全局安装会生成 ``habitus.ps1`` 和 ``habitus.cmd`` 两个启动脚本。Windows 客户端版默认的执行策略
``Restricted``（以及 ``AllSigned``）会拦下没签名的 ``habitus.ps1``，用户在 PowerShell 里敲 ``habitus`` 只看到
"禁止运行脚本"。doctor 读出当前用户的有效策略，被拦时给出不需要管理员的两条路。其他平台没有这回事，不出检查项。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable

_BLOCKING_POLICIES = frozenset({"restricted", "allsigned"})
_KNOWN_POLICIES = frozenset({"unrestricted", "remotesigned", "bypass", "default", "undefined", *_BLOCKING_POLICIES})
_CHECK_NAME = "powershell_policy"
_REMEDY = (
    "npm's habitus.ps1 will be blocked in PowerShell; without admin rights run "
    "'Set-ExecutionPolicy -Scope CurrentUser RemoteSigned', or call 'habitus.cmd' instead"
)


def judge_execution_policy(policy: str | None) -> tuple[str, str]:
    """把 ``Get-ExecutionPolicy`` 的输出判成 doctor 的 (status, detail)；``None`` 表示没读出来。"""

    if policy is None:
        return "warn", "could not read the PowerShell execution policy; if 'habitus' is blocked, use 'habitus.cmd'"
    normalized = policy.strip()
    if normalized.casefold() not in _KNOWN_POLICIES:
        return "warn", f"unrecognized PowerShell execution policy {normalized or 'empty'!r}"
    if normalized.casefold() in _BLOCKING_POLICIES:
        return "warn", f"execution policy {normalized}: {_REMEDY}"
    return "pass", f"execution policy {normalized}"


def read_execution_policy(run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> str | None:
    """问 Windows PowerShell 当前用户的有效执行策略；找不到 PowerShell 或调用失败返回 ``None``。"""

    executable = shutil.which("powershell")
    if executable is None:
        return None
    try:
        completed = run(
            [executable, "-NoProfile", "-NonInteractive", "-Command", "Get-ExecutionPolicy"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def powershell_policy_checks() -> tuple[tuple[str, str, str], ...]:
    """doctor 的检查项（名字、状态、说明）；非 Windows 返回空。"""

    if sys.platform != "win32":
        return ()
    status, detail = judge_execution_policy(read_execution_policy())
    return ((_CHECK_NAME, status, detail),)


__all__ = ["judge_execution_policy", "powershell_policy_checks", "read_execution_policy"]
