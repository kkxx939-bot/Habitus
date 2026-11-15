"""读 ``habitus/config/example.yaml`` 里列出的可调键（含注释掉的示例行），与改动前的那一份比出新增的键。

示例 YAML 的大部分分组是注释掉的（``# behavior:`` 下面一行一个 ``#   key: value  # 说明``），所以不能用 YAML 解析器读，
按行认：顶格的 ``# <组>:`` / ``<组>:`` 开一组，组下缩进两格的 ``key: value`` 是这一组的一个键。更深的嵌套键不收。
"""

from __future__ import annotations

import re
from pathlib import Path

_SECTION = re.compile(r"^#? ?(?P<name>[a-z_]+):\s*(#.*)?$")
_KEY = re.compile(r"^#?(?P<indent> +)(?P<key>[a-z_][a-z0-9_]*):\s*(?P<value>[^\s#][^#]*?)?\s*(#.*)?$")


def listed_keys(path: Path) -> dict[tuple[str, str], str]:
    """（组, 键）→ 示例里写的值（原文）。"""

    found: dict[tuple[str, str], str] = {}
    section: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        head = _SECTION.match(line)
        if head is not None and not line.startswith(("#  ", "  ")):
            section = head["name"]
            continue
        match = _KEY.match(line)
        if match is None or section is None:
            continue
        indent = len(match["indent"])
        if line.startswith("#"):
            indent -= 1  # "#   key" 是注释掉的两格缩进
        if indent != 2:
            continue
        found[(section, match["key"])] = (match["value"] or "").strip()
    return found


#: 改动前（基线 ``9f23a13`` 加第一步未提交改动）的 example.yaml 原样拷贝。
BASELINE_EXAMPLE = Path(__file__).with_name("baseline_example.yaml.txt")


#: 待定池、每晚新增、拆改、分组、锚点、合并、冷却那批键（第二步甲删掉的机制的门槛）。
RETIRED_BEHAVIOR_KEYS: dict[str, object] = {
    "kinds_recurrence_min_count": 2,
    "kinds_recurrence_min_days": 2,
    "kinds_max_pending": 5000,
    "kinds_nightly_max_new": 200,
    "kinds_recall_per_entry": 3,
    "kinds_revision_weekday": 2,
    "kinds_merge_evidence_runs": 2,
    "kinds_cooldown_days": 14,
    "kinds_max_changes_per_revision": 3,
    "kinds_anchors_per_class": 8,
    "kinds_max_anchor_pull": 0.02,
    "kinds_revision_samples_per_class": 8,
    "kinds_revision_window_days": 120,
    "kinds_grouping_min_members": 6,
    "kinds_grouping_batch_size": 60,
    "kinds_sample_max_steps": 4,
}


def added_keys(path: Path) -> dict[tuple[str, str], str]:
    """example.yaml 里比改动前多出来的（组, 键）→ 值（新键放在哪一组由实现定，所有组都比）。"""

    before = listed_keys(BASELINE_EXAMPLE)
    return {item: value for item, value in listed_keys(path).items() if item not in before}


__all__ = ["BASELINE_EXAMPLE", "RETIRED_BEHAVIOR_KEYS", "added_keys", "listed_keys"]
