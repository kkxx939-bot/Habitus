"""归组一天所需的输入与产出的受控数据模型。

输入只含行为树上的字段（不含原始观测）与三类**有界参照**：前 D 天的事、待用前提清单里尚未
兑现的项、今日各 kind 的上一次。产出是"事"的草稿（编号级），由刷新器换成 URI 落盘。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from habitus.scene.model import SceneLinkType, SceneRole


@dataclass(frozen=True)
class OccurrenceRow:
    """当天一条 occurrence 在归组输入里的投影（编号从 1 起、按开始瞬时排）。"""

    no: int
    uri: str
    name: str
    kind_token: str
    started_at: datetime
    last_observed_at: datetime
    status: str
    status_basis: str
    goal: str | None
    summary: str
    subjects: tuple[str, ...]
    basis: tuple[str, ...]
    # 行为树自带的短程关系，目标换成本日编号（指向本日之外的丢弃）
    links: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.no, bool) or not isinstance(self.no, int) or self.no <= 0:
            raise ValueError("occurrence row no must be a positive integer")
        for name in ("uri", "name", "kind_token", "status", "status_basis", "summary"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"occurrence row {name} must be non-empty text")
        for name in ("started_at", "last_observed_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ValueError(f"occurrence row {name} must be a timezone-aware datetime")


@dataclass(frozen=True)
class GapRow:
    started_at: datetime
    ended_at: datetime


@dataclass(frozen=True)
class SceneReference:
    """前 D 天的一件事，供跨日的 needs / results_from 指目标（编号 C1..Cn）。"""

    no: int
    uri: str
    day: date
    label: str
    started_at: datetime
    effects: tuple[str, ...]
    pending: tuple[str, ...]


@dataclass(frozen=True)
class PendingReference:
    """待用前提清单里尚未兑现的一项（编号 P1..Pn）；目标落到建立它的行为 URI。"""

    no: int
    text: str
    producer_uri: str
    producer_started_at: datetime
    scene_uri: str
    created_on: date


@dataclass(frozen=True)
class LastOccurrenceReference:
    """今日出现的某个 kind 上一次发生时的样子（只给模型看，不作边的目标）。"""

    kind_token: str
    days_ago: int
    scene_label: str | None
    # 今日属于这个 kind 的行为编号：模型只看得见行号，没有它这段参照接不上今天的记录
    today_nos: tuple[int, ...] = ()


@dataclass(frozen=True)
class SceneGroupingInput:
    day: date
    subject: str
    occurrences: tuple[OccurrenceRow, ...]
    gaps: tuple[GapRow, ...] = ()
    scene_references: tuple[SceneReference, ...] = ()
    pending_references: tuple[PendingReference, ...] = ()
    last_occurrences: tuple[LastOccurrenceReference, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.day, datetime) or not isinstance(self.day, date):
            raise TypeError("grouping input day must be a date")
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("grouping input subject must be non-empty text")
        if not self.occurrences:
            raise ValueError("grouping input must contain at least one occurrence")
        expected = tuple(range(1, len(self.occurrences) + 1))
        if tuple(row.no for row in self.occurrences) != expected:
            raise ValueError("occurrence rows must be numbered 1..N in order")
        instants = [row.started_at.astimezone(UTC) for row in self.occurrences]
        if instants != sorted(instants):
            raise ValueError("occurrence rows must be ordered by started_at")
        if tuple(item.no for item in self.scene_references) != tuple(range(1, len(self.scene_references) + 1)):
            raise ValueError("scene references must be numbered 1..N in order")
        if tuple(item.no for item in self.pending_references) != tuple(range(1, len(self.pending_references) + 1)):
            raise ValueError("pending references must be numbered 1..N in order")

    def row(self, no: int) -> OccurrenceRow:
        return self.occurrences[no - 1]


@dataclass(frozen=True)
class DraftRelation:
    """草稿里的一条边；目标恰好是四种之一（本日行为、本日情景、先前的事、待用前提的产生方）。"""

    kind: SceneLinkType
    occurrence_no: int | None = None
    scene_index: int | None = None
    reference_no: int | None = None
    pending_no: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", SceneLinkType(self.kind))
        filled = [name for name in ("occurrence_no", "scene_index", "reference_no", "pending_no") if getattr(self, name) is not None]
        if len(filled) != 1:
            raise ValueError("a draft relation must name exactly one target")


@dataclass(frozen=True)
class SceneDraft:
    """一件事的草稿：标签、成员（编号 + 角色，按开始瞬时排）、留下的改变、待用前提、边。"""

    label: str
    members: tuple[tuple[int, SceneRole], ...]
    effects: tuple[str, ...] = ()
    pending_effects: tuple[tuple[str, int], ...] = ()
    relations: tuple[DraftRelation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("scene draft label must be non-empty text")
        if not self.members:
            raise ValueError("scene draft must have at least one member")
        object.__setattr__(self, "members", tuple((no, SceneRole(role)) for no, role in self.members))

    @property
    def member_nos(self) -> tuple[int, ...]:
        return tuple(no for no, _ in self.members)


@dataclass(frozen=True)
class GroupingAssembly:
    """装配层的产物：可落盘的草稿 + 降级信号 + 未归组的行为编号。"""

    scenes: tuple[SceneDraft, ...]
    unassigned: tuple[int, ...]
    signals: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "DraftRelation",
    "GapRow",
    "GroupingAssembly",
    "LastOccurrenceReference",
    "OccurrenceRow",
    "PendingReference",
    "SceneDraft",
    "SceneGroupingInput",
    "SceneReference",
]
