"""行为管线各阶段共用的观测口径。

每个阶段从自己的产物算属性的函数放在各自包的 ``telemetry`` 模块里；这里只放共用的类别名、
时长换算和把降级/丢弃说明折成计数的规则。属性一律是计数、时长与布尔，不带行为名、主体称呼
或任何观测语义——那些是用户内容，不进可观测后端。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

OBSERVATION_CATEGORY = "behavior"

Attributes = dict[str, str | int | float | bool]

# 一条事件最多 32 个属性；按首词计数的说明来自代码里固定的几种措辞，再设一道上限兜住将来新增的措辞。
_MAX_NOTE_KINDS = 8


def seconds_between(later: datetime, earlier: datetime) -> float:
    """两个带时区时刻之间的秒数，保留到毫秒；负值（时钟抖动）按 0 记。"""

    return round(max(0.0, (later - earlier).total_seconds()), 3)


def count_notes_by_leading_token(notes: Iterable[str], *, prefix: str) -> Attributes:
    """把降级/丢弃说明按首词归类计数。

    这些说明的首词是代码里固定的类别（``subject_absent``、``continues`` 之类），后面才是带身份的
    细节；只取首词既能分出原因，又不会把身份或语义带进属性。超过上限的类别并入 ``other``。
    """

    counts: dict[str, int] = {}
    for note in notes:
        token = note.split(" ", 1)[0] if note else ""
        if not token.replace("_", "").isalnum():
            token = "other"
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    kept = dict(ranked[:_MAX_NOTE_KINDS])
    overflow = sum(count for _token, count in ranked[_MAX_NOTE_KINDS:])
    if overflow:
        kept["other"] = kept.get("other", 0) + overflow
    return {f"{prefix}{token}": count for token, count in kept.items()}


__all__ = ["OBSERVATION_CATEGORY", "Attributes", "count_notes_by_leading_token", "seconds_between"]
