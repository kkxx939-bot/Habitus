"""定位随源码或 wheel 分发的 Agent 插件，并委托 Node 入口执行。"""

from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path


def run(argv: Sequence[str] | None = None) -> int:
    """执行一次 Harness 插件生命周期命令并返回退出码。"""

    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments.pop(0) if arguments else "install"
    if command in {"-h", "--help"}:
        sys.stdout.write(
            "usage: m2bos-plugin [install|status|update|remove|doctor|harnesses] "
            "[--harness ID|--host ID] [--root PATH] [--prepare-only] [--json]\n"
        )
        return 0
    if command not in {"doctor", "harnesses", "install", "status", "update", "remove"}:
        raise ValueError(
            "usage: m2bos-plugin [install|status|update|remove|doctor|harnesses] [options]"
        )
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("m2bos-plugin requires Node.js")
    root = _plugin_root()
    script = root / ("memory-plugin-shared/doctor.mjs" if command == "doctor" else "install-memory-plugin.mjs")
    delegated = arguments if command == "doctor" else [command, *arguments]
    completed = subprocess.run([node, str(script), *delegated], check=False)  # noqa: S603
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> None:
    try:
        code = run(argv)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(code)


def _plugin_root() -> Path:
    source = Path(__file__).resolve().parents[2] / "plugins"
    installed = Path(sysconfig.get_path("data")) / "share" / "m2bos" / "plugins"
    for candidate in (source, installed):
        if (candidate / "install-memory-plugin.mjs").is_file():
            return candidate
    raise SystemExit("m2bOS plugin assets are not installed")


__all__ = ["main", "run"]


if __name__ == "__main__":  # pragma: no cover - 由 console script 与人工诊断共用
    main()
