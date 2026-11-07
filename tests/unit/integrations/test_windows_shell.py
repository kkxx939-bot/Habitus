"""doctor 的 PowerShell 执行策略检查（阶段 3）：被拦时给出不需要管理员的解法，其他平台不出检查项。"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import Mock

import pytest

from habitus.integrations.local_service import windows_shell


@pytest.mark.parametrize("policy", ["Restricted", "AllSigned", " restricted\r\n"])
def test_blocking_policies_warn_with_the_no_admin_remedy(policy: str) -> None:
    status, detail = windows_shell.judge_execution_policy(policy)
    assert status == "warn"
    assert "Set-ExecutionPolicy -Scope CurrentUser RemoteSigned" in detail
    assert "habitus.cmd" in detail


@pytest.mark.parametrize("policy", ["RemoteSigned", "Unrestricted", "Bypass"])
def test_permissive_policies_pass(policy: str) -> None:
    assert windows_shell.judge_execution_policy(policy) == ("pass", f"execution policy {policy}")


@pytest.mark.parametrize("policy", [None, "", "Gibberish"])
def test_unreadable_or_unknown_policies_warn_without_claiming_a_block(policy: str | None) -> None:
    status, detail = windows_shell.judge_execution_policy(policy)
    assert status == "warn"
    assert "Set-ExecutionPolicy" not in detail


def test_reader_asks_windows_powershell_without_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="Restricted\r\n", stderr=""))
    monkeypatch.setattr(windows_shell.shutil, "which", Mock(return_value="C:\\Windows\\powershell.exe"))
    assert windows_shell.read_execution_policy(run) == "Restricted"
    command = run.call_args.args[0]
    assert command[1:] == ["-NoProfile", "-NonInteractive", "-Command", "Get-ExecutionPolicy"]
    run.return_value = subprocess.CompletedProcess([], 1, stdout="", stderr="boom")
    assert windows_shell.read_execution_policy(run) is None
    run.side_effect = subprocess.TimeoutExpired("powershell", 10)
    assert windows_shell.read_execution_policy(run) is None
    monkeypatch.setattr(windows_shell.shutil, "which", Mock(return_value=None))
    assert windows_shell.read_execution_policy(run) is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX 上没有执行策略，不出检查项")
def test_no_check_outside_windows() -> None:
    assert windows_shell.powershell_policy_checks() == ()


@pytest.mark.skipif(sys.platform != "win32", reason="真 Windows 上读出的策略必须是认得的那几种")
def test_real_windows_policy_is_read_and_judged() -> None:
    ((name, status, detail),) = windows_shell.powershell_policy_checks()
    assert name == "powershell_policy"
    assert status in {"pass", "warn"}
    assert "unrecognized" not in detail and "could not read" not in detail
