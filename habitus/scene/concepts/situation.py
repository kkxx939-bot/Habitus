"""情境怎么算：日历情境上那条**机械**说明（裁定 59 第 7 条）。

情境只有日历一族：外部条件里日历的某个键取某几个值就成立（「周三」= 日历.周几∈{周三}，「工作日」= 日历.日型∈{工作日}）。
说明由外部条件包的日历机械生成（``calendar_situations``），不由模型写；情境命中由算法直接得出、不经模型。
"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.foundation.text import clean_line

MAX_VALUE_CHARS = 120


class SituationError(ValueError):
    """情境说明与自己的约束矛盾。"""


@dataclass(frozen=True)
class SituationRule:
    """一条日历情境说明：``value`` 是外部条件的日历键，``values`` 是成立的那几个取值。"""

    value: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        key = clean_line(self.value)
        if not key or len(key) > MAX_VALUE_CHARS:
            raise SituationError("a calendar situation needs a short non-empty calendar key")
        object.__setattr__(self, "value", key)
        cleaned = tuple(clean_line(item) for item in self.values)
        if not cleaned or any(not item or len(item) > MAX_VALUE_CHARS for item in cleaned):
            raise SituationError("a calendar situation names one or more values of its key")
        if len(set(cleaned)) != len(cleaned):
            raise SituationError("calendar values must be distinct")
        object.__setattr__(self, "values", cleaned)

    def payload(self) -> dict[str, object]:
        """进指纹与文档的形状。"""

        return {"value": self.value, "values": list(self.values)}

    def criterion(self) -> str:
        """给人读的一句话（也进概念文档）。"""

        return f"日历的「{self.value}」是 " + "/".join(self.values)


__all__ = ["SituationError", "SituationRule"]
