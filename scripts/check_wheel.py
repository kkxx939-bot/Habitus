#!/usr/bin/env python3
"""校验构建出的 wheel 与源码树、pyproject 声明是否一致。

历史上这里是一串写死的文件名断言（"behavior/tree/store.py 必须在 wheel 里"），
每次重构目录就会过期，CI 因此红过四次。这个脚本改为按规则校验，重构目录时自动跟着变：

1. 受控包内每一个纳入版本库的 .py，都必须出现在 wheel 里（漏打包）；
2. wheel 里每一个 .py，都必须在源码树里存在（删掉的模块被陈旧构建产物带进来）；
3. [tool.setuptools.package-data] 声明的每条 glob 至少命中一个文件，且命中的文件都在 wheel 里；
4. [tool.setuptools.data-files] 声明的每条 glob 同上，落点为声明的 share 路径。

用法：python scripts/check_wheel.py [wheel 路径]；省略路径时取 dist/ 下最新的那个。
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
import zipfile
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fail(message: str, items: list[str] | None = None) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    for item in sorted(items or [])[:20]:
        print(f"    {item}", file=sys.stderr)
    if items and len(items) > 20:
        print(f"    …… 另有 {len(items) - 20} 项", file=sys.stderr)
    sys.exit(1)


def tracked_files() -> list[Path]:
    """版本库里的文件清单。用 git 而不是遍历磁盘，天然排除 habitus/benchmark/data 这类
    被 gitignore 的本地数据集——它们不进 git，CI 上也不存在。"""
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True, text=True
    ).stdout
    return [Path(name) for name in out.split("\0") if name]


def package_roots(config: dict) -> list[str]:
    """packages.find 的 include 形如 "behavior*"，取出顶层包名。"""
    patterns = config["tool"]["setuptools"]["packages"]["find"]["include"]
    return [pattern.rstrip("*") for pattern in patterns]


def main() -> None:
    if len(sys.argv) > 1:
        wheel_path = Path(sys.argv[1])
    else:
        wheels = sorted((ROOT / "dist").glob("*.whl"), key=lambda p: p.stat().st_mtime)
        if not wheels:
            fail("dist/ 下没有 wheel，先跑 python -m build --wheel")
        wheel_path = wheels[-1]

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools_config = config["tool"]["setuptools"]
    names = set(zipfile.ZipFile(wheel_path).namelist())
    roots = package_roots(config)
    tracked = tracked_files()

    # 1. 受控包内纳入版本库的 .py 都要在 wheel 里。
    expected_modules = {
        str(path)
        for path in tracked
        if path.suffix == ".py" and any(str(path).startswith(root) for root in roots)
    }
    if missing := expected_modules - names:
        fail(f"以下模块在源码树里，却没被打进 {wheel_path.name}", sorted(missing))

    # 2. 反向：wheel 里的 .py 都要在源码树里存在。防止删掉的模块被陈旧构建产物带进来。
    tracked_set = {str(path) for path in tracked}
    wheel_modules = {
        name
        for name in names
        if name.endswith(".py")
        and not name.startswith(f"{wheel_path.name.split('-')[0]}-")  # 跳过 .dist-info
    }
    if stale := wheel_modules - tracked_set:
        fail(f"{wheel_path.name} 里有源码树中不存在的模块（构建产物陈旧？）", sorted(stale))

    # 3. package-data：每条 glob 至少命中一个文件，命中的都要在 wheel 里。
    missing_data: list[str] = []
    empty_globs: list[str] = []
    for package, patterns in setuptools_config.get("package-data", {}).items():
        package_dir = Path(package.replace(".", "/"))
        for pattern in patterns:
            # 模式可以带斜杠（如 "datasets/*.json"），按相对包目录的完整路径匹配。
            matched = [
                path
                for path in tracked
                if path.is_relative_to(package_dir)
                and fnmatch(str(path.relative_to(package_dir)), pattern)
            ]
            if not matched:
                empty_globs.append(f"package-data [{package}] {pattern}")
                continue
            missing_data.extend(str(path) for path in matched if str(path) not in names)

    # 4. data-files：落点是声明的 share 路径，wheel 里带 .data/data/ 前缀。
    for target, patterns in setuptools_config.get("data-files", {}).items():
        for pattern in patterns:
            matched = sorted(ROOT.glob(pattern))
            matched = [path for path in matched if str(path.relative_to(ROOT)) in tracked_set]
            if not matched:
                empty_globs.append(f"data-files [{target}] {pattern}")
                continue
            for path in matched:
                landing = f"{target}/{path.name}"
                if not any(name.endswith(landing) for name in names):
                    missing_data.append(landing)

    if empty_globs:
        fail("以下 pyproject 声明没有命中任何文件（路径已过期？）", empty_globs)
    if missing_data:
        fail(f"以下声明的数据文件没被打进 {wheel_path.name}", missing_data)

    # 5. 反向：包目录内运行时要读的数据文件（yaml/json）如果一个都没声明就会被静默漏掉，
    #    上面第 3 条只能证明"声明了的都在"。这里补上"该在的都声明了"。
    package_dirs = {path.parent for path in tracked if path.name == "__init__.py"}
    undeclared = [
        str(path)
        for path in tracked
        if path.suffix in {".yaml", ".yml", ".json"}
        and path.parent in package_dirs
        and any(str(path).startswith(root) for root in roots)
        and str(path) not in names
    ]
    if undeclared:
        fail(
            f"以下数据文件在包目录内，却没被打进 {wheel_path.name}"
            "（在 [tool.setuptools.package-data] 里登记，或确认运行时不需要它）",
            undeclared,
        )

    print(
        f"OK {wheel_path.name}: "
        f"{len(expected_modules)} 个模块、"
        f"{len(setuptools_config.get('package-data', {}))} 组 package-data、"
        f"{len(setuptools_config.get('data-files', {}))} 组 data-files 均已覆盖"
    )


if __name__ == "__main__":
    main()
