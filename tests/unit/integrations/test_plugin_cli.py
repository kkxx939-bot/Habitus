"""wheel 插件入口只负责定位资产，并完整保留生命周期命令。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from habitus.integrations.local_service import plugin_cli


@pytest.mark.parametrize("command", ["harnesses", "install", "status", "update", "remove"])
def test_plugin_cli_delegates_lifecycle_action(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "plugins"
    root.mkdir()
    (root / "install-memory-plugin.mjs").write_text("", encoding="utf-8")
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(plugin_cli.shutil, "which", Mock(return_value="/usr/bin/node"))
    monkeypatch.setattr(plugin_cli, "_plugin_root", Mock(return_value=root))
    monkeypatch.setattr(plugin_cli.subprocess, "run", run)

    with pytest.raises(SystemExit) as stopped:
        plugin_cli.main([command, "--json"])

    assert stopped.value.code == 0
    run.assert_called_once_with(
        ["/usr/bin/node", str(root / "install-memory-plugin.mjs"), command, "--json"],
        check=False,
    )


def test_plugin_cli_delegates_doctor_without_a_lifecycle_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "plugins"
    script = root / "memory-plugin-shared" / "doctor.mjs"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    run = Mock(return_value=SimpleNamespace(returncode=1))
    monkeypatch.setattr(plugin_cli.shutil, "which", Mock(return_value="/usr/bin/node"))
    monkeypatch.setattr(plugin_cli, "_plugin_root", Mock(return_value=root))
    monkeypatch.setattr(plugin_cli.subprocess, "run", run)

    with pytest.raises(SystemExit) as stopped:
        plugin_cli.main(["doctor"])

    assert stopped.value.code == 1
    run.assert_called_once_with(["/usr/bin/node", str(script)], check=False)
