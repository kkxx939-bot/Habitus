"""一条待归类的 occurrence：看整条链的内容，不只看链头名字（设计 二）。"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.model import BehaviorKindError


def _texts(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not all(isinstance(value, str) and value.strip() for value in values):
        raise BehaviorKindError(f"{field_name} must be a tuple of non-empty text")
    return values


@dataclass(frozen=True)
class OccurrenceContent:
    """原话（链头名字）+ 每段概要 + 目标 + 步骤。"""

    name: str
    summaries: tuple[str, ...] = ()
    goal: str | None = None
    steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise BehaviorKindError("occurrence name must be non-empty text")
        _texts(self.summaries, "occurrence summaries")
        _texts(self.steps, "occurrence steps")
        if self.goal is not None and (not isinstance(self.goal, str) or not self.goal.strip()):
            raise BehaviorKindError("occurrence goal must be non-empty text or None")

    def render(self, *, max_steps: int) -> str:
        parts = [self.name]
        if self.summaries:
            parts.append(f"概要：{'；'.join(self.summaries)}")
        if self.goal:
            parts.append(f"目标：{self.goal}")
        if self.steps:
            shown = self.steps[:max_steps]
            more = f"（另 {len(self.steps) - len(shown)} 步）" if len(self.steps) > len(shown) else ""
            parts.append(f"步骤：{'／'.join(shown)}{more}")
        return "　｜".join(parts)


@dataclass(frozen=True)
class ClassifyRequest:
    """``key`` 由调用方定，结果按它返回（归约侧用链的身份）。"""

    key: str
    lane: Lane
    content: OccurrenceContent

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise BehaviorKindError("classify request key must be non-empty text")
        object.__setattr__(self, "lane", Lane(self.lane))
        if not isinstance(self.content, OccurrenceContent):
            raise BehaviorKindError("classify request content must be OccurrenceContent")


__all__ = ["ClassifyRequest", "OccurrenceContent"]
