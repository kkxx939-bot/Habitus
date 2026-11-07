"""``habitus-server`` 的输出编码（阶段 3）：被管道接走时，Windows 默认按本地代码页编码，中文与代码页外字符会抛错或乱码。

用 ``PYTHONIOENCODING=ascii`` 在任何平台上模拟"代码页装不下这些字符"的处境。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TEXT = "用户 张三 李四 ✓ ü"


def _run(script: str) -> subprocess.CompletedProcess[bytes]:
    environment = {**os.environ, "PYTHONIOENCODING": "ascii", "PYTHONPATH": str(REPOSITORY_ROOT)}
    return subprocess.run([sys.executable, "-c", script], capture_output=True, env=environment, check=False)


def test_console_entry_writes_utf8_even_when_the_inherited_encoding_cannot_hold_the_text() -> None:
    script = (
        "import sys\n"
        "from habitus.integrations.local_service.console import use_utf8_streams\n"
        "use_utf8_streams()\n"
        f"sys.stdout.write({_TEXT!r} + '\\n')\n"
        f"sys.stderr.write({_TEXT!r} + '\\n')\n"
    )
    completed = _run(script)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.replace(b"\r\n", b"\n") == (_TEXT + "\n").encode("utf-8")
    assert completed.stderr.replace(b"\r\n", b"\n") == (_TEXT + "\n").encode("utf-8")


def test_without_the_entry_the_same_write_fails() -> None:
    """对照：不经入口重设时同一段输出确实写不出去，说明上面那条用例测到了点子上。"""

    completed = _run(f"import sys\nsys.stdout.write({_TEXT!r})\n")
    assert completed.returncode != 0
    assert b"UnicodeEncodeError" in completed.stderr


def test_the_installed_command_goes_through_the_utf8_entry() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["habitus-server"] == "habitus.integrations.local_service.console:main"
