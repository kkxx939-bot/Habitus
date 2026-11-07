"""代码注释里引用的每个「裁定 N」都必须在仓库内的裁定条文里查得到（防复发）。

为什么有这道闸：`habitus/` 的源码注释里到处用「裁定 N」标明某段逻辑的依据，但裁定条文
原先只在开发者本地的实验记录里，外部看代码的人对不上号。`docs/rulings/README.md` 把裁定的
定稿条文放进仓库当权威；这道闸扫描 `habitus/` 下全部 `.py` 里出现的每个「裁定 N」，断言每个
编号在那份文件里都有对应的 `## 裁定 N` 小节，缺任何一个就失败并报出缺哪些号——新写的注释
引用了没收录的裁定会被挡住，提醒把条文补进仓库。
"""

import re
from functools import cache
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC = REPOSITORY_ROOT / "habitus"
RULINGS_DOC = REPOSITORY_ROOT / "docs" / "rulings" / "README.md"
EXCLUDED_PATH_PARTS = frozenset({"__pycache__", "node_modules"})
# 源码里引用裁定的写法，例如「裁定 54」「裁定54」「裁定 54 第 2 条」。
REFERENCE_RE = re.compile(r"裁定\s*(\d+)")
# 条文文件里的小节标题，例如「## 裁定 54（2026-10-17）· …」。
SECTION_RE = re.compile(r"^##\s*裁定\s*(\d+)\b", re.MULTILINE)


def _source_files() -> tuple[Path, ...]:
    """``habitus/`` 下全部 ``.py``，排除 ``__pycache__`` 与 ``node_modules``。"""

    return tuple(
        sorted(
            path
            for path in SRC.rglob("*.py")
            if not EXCLUDED_PATH_PARTS.intersection(path.relative_to(REPOSITORY_ROOT).parts)
        )
    )


@cache
def _referenced_rulings() -> dict[int, list[str]]:
    """每个被引用的裁定编号 → 引用它的文件（相对仓库根），已排序去重。"""

    references: dict[int, set[str]] = {}
    for path in _source_files():
        identity = path.relative_to(REPOSITORY_ROOT).as_posix()
        for match in REFERENCE_RE.finditer(path.read_text(encoding="utf-8")):
            references.setdefault(int(match.group(1)), set()).add(identity)
    return {number: sorted(sources) for number, sources in sorted(references.items())}


@cache
def _documented_rulings() -> frozenset[int]:
    """``docs/rulings/README.md`` 里有 ``## 裁定 N`` 小节的全部编号。"""

    text = RULINGS_DOC.read_text(encoding="utf-8")
    return frozenset(int(number) for number in SECTION_RE.findall(text))


def test_rulings_doc_exists() -> None:
    """裁定条文必须在仓库里（缺文件时直接给出清晰的错误）。"""

    assert RULINGS_DOC.is_file(), f"缺少裁定条文：{RULINGS_DOC.relative_to(REPOSITORY_ROOT)}"


def test_every_referenced_ruling_is_documented() -> None:
    """``habitus/`` 里引用的每个「裁定 N」都要在 ``docs/rulings/README.md`` 有 ``## 裁定 N`` 小节。"""

    documented = _documented_rulings()
    missing = sorted(
        f"裁定 {number}: 代码引用了但 docs/rulings/README.md 没有 ## 裁定 {number} 小节"
        f"（引用处：{'、'.join(sources)}）"
        for number, sources in _referenced_rulings().items()
        if number not in documented
    )
    assert missing == [], "\n".join(missing)
