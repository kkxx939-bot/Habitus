"""定位随源码或 wheel 分发的 Agent 插件，并委托 Node 入口执行。"""

from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments.pop(0) if arguments else "install"
    if command not in {"doctor", "install", "status", "update", "remove"}:
        raise SystemExit("usage: m2bos-plugin [install|status|update|remove|doctor] [options]")
    node = shutil.which("node")
    if node is None:
        raise SystemExit("m2bos-plugin requires Node.js")
    root = _plugin_root()
    script = root / ("memory-plugin-shared/doctor.mjs" if command == "doctor" else "install-memory-plugin.mjs")
    delegated = arguments if command == "doctor" else [command, *arguments]
    completed = subprocess.run([node, str(script), *delegated], check=False)  # noqa: S603
    raise SystemExit(completed.returncode)


def _plugin_root() -> Path:
    source = Path(__file__).resolve().parents[2] / "plugins"
    installed = Path(sysconfig.get_path("data")) / "share" / "m2bos" / "plugins"
    for candidate in (source, installed):
        if (candidate / "install-memory-plugin.mjs").is_file():
            return candidate
    raise SystemExit("m2bOS plugin assets are not installed")


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - 由 console script 与人工诊断共用
    main()
