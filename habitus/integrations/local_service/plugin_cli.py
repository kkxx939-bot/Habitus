"""定位随源码或 wheel 分发的 Agent 插件，并委托 Node 入口执行。"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def run(argv: Sequence[str] | None = None) -> int:
    """执行一次 Harness 插件生命周期命令并返回退出码。"""

    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments.pop(0) if arguments else "install"
    if command in {"-h", "--help"}:
        sys.stdout.write(
            "usage: habitus-plugin [install|status|update|remove|doctor|harnesses] "
            "[--harness ID|--host ID] [--root PATH] [--prepare-only] [--json]\n"
        )
        return 0
    if command not in {"doctor", "harnesses", "install", "status", "update", "remove"}:
        raise ValueError(
            "usage: habitus-plugin [install|status|update|remove|doctor|harnesses] [options]"
        )
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("habitus-plugin requires Node.js")
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


def _plugin_root(
    *,
    source: Path | None = None,
    search_paths: Sequence[str | Path] | None = None,
) -> Path:
    # 源码树里的 plugins/ 在仓库根：本文件是 habitus/integrations/local_service/plugin_cli.py，
    # 往上三层才是仓库根（parents[0]=local_service, [1]=integrations, [2]=habitus, [3]=根）。
    source_root = Path(__file__).resolve().parents[3] / "plugins" if source is None else source
    candidates = [source_root]
    try:
        package = distribution("habitus")
    except PackageNotFoundError:
        package = None
    if package is not None:
        for item in package.files or ():
            normalized = str(item).replace("\\", "/")
            if normalized.endswith("share/habitus/plugins/install-memory-plugin.mjs"):
                candidates.append(Path(str(package.locate_file(item))).resolve().parent)
    for entry in sys.path if search_paths is None else search_paths:
        candidates.append(Path(entry).resolve() / "share" / "habitus" / "plugins")
    for candidate in candidates:
        if (candidate / "install-memory-plugin.mjs").is_file():
            return candidate
    raise SystemExit("Habitus plugin assets are not installed")


__all__ = ["main", "run"]


if __name__ == "__main__":  # pragma: no cover - 由 console script 与人工诊断共用
    main()
