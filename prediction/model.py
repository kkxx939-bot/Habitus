"""预测树的节点、地址、目标和时间边界值对象。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

_SAMPLE_ID = re.compile(r"^[0-9a-f]{64}$")
_SHARD_WIDTH = 2
_SHARD = re.compile(r"^[0-9a-f]{2}$")


class PredictionKind(str, Enum):
    """预测树中拥有独立完整样本文档的节点类型。"""

    TRANSITION = "transition"
    TRAJECTORY = "trajectory"
    CONSEQUENCE = "consequence"

    @property
    def branch_name(self) -> str:
        return {
            PredictionKind.TRANSITION: "transitions",
            PredictionKind.TRAJECTORY: "trajectories",
            PredictionKind.CONSEQUENCE: "consequences",
        }[self]

    @classmethod
    def from_branch_name(cls, value: str) -> PredictionKind:
        by_branch = {kind.branch_name: kind for kind in cls}
        try:
            return by_branch[value]
        except KeyError as exc:
            raise ValueError("unknown prediction branch") from exc


class PredictionPatternKind(str, Enum):
    """从事实样本聚合出的可检索模式图节点。"""

    STATE = "state"
    BRANCH = "branch"


class PredictionLevel(int, Enum):
    """预测树的可重建目录语义层级。"""

    ABSTRACT = 0
    OVERVIEW = 1
    DETAIL = 2

    @property
    def sidecar_filename(self) -> str:
        if self is PredictionLevel.ABSTRACT:
            return ".abstract.md"
        if self is PredictionLevel.OVERVIEW:
            return ".overview.md"
        raise ValueError("prediction L2 uses a PredictionAddress")

    @classmethod
    def from_sidecar_filename(cls, filename: object) -> PredictionLevel | None:
        if filename == ".abstract.md":
            return cls.ABSTRACT
        if filename == ".overview.md":
            return cls.OVERVIEW
        return None


class PredictionAnchorType(str, Enum):
    EVENT = "event"
    ACTION = "action"
    PHASE = "phase"


class PredictionDecisionBasis(str, Enum):
    """预测切点必须来自标签之前已经可观察的业务边界。"""

    CONTAINER_STARTED = "container_started"
    CONTAINER_OBSERVED = "container_observed"
    PREVIOUS_STEP_STARTED = "previous_step_started"
    PREVIOUS_STEP_COMPLETED = "previous_step_completed"
    PREVIOUS_STEP_ORDERED = "previous_step_ordered"
    PREVIOUS_STEP_OBSERVED = "previous_step_observed"
    TREATMENT_OBSERVED = "treatment_observed"


class PredictionTargetLevel(str, Enum):
    ACTION = "action"
    EVENT = "event"
    PHASE = "phase"
    OUTCOME = "outcome"
    TERMINAL = "terminal"


class PredictionMode(str, Enum):
    NEXT_STEP = "next_step"
    CONTINUATION = "continuation"
    CONSEQUENCE = "consequence"
    TERMINATION = "termination"


class PredictionTimePrecision(str, Enum):
    EXACT = "exact"
    BOUNDED = "bounded"
    ORDER_ONLY = "order_only"


class PredictionLabelStatus(str, Enum):
    OBSERVED = "observed"
    TERMINAL = "terminal"
    CLOSED_WITHOUT_TARGET = "closed_without_target"
    UNKNOWN = "unknown"


def is_ascii_digits(value: object) -> bool:
    return isinstance(value, str) and value.isascii() and value.isdigit()


def prediction_sample_id(value: object) -> str:
    if not isinstance(value, str) or _SAMPLE_ID.fullmatch(value) is None:
        raise ValueError("prediction sample ID must be a lowercase SHA-256 digest")
    return value


def prediction_shard(key: str) -> str:
    """取 SHA-256 身份的前两位作为物理分片目录，把单目录扇出压到 1/256。"""

    return prediction_sample_id(key)[:_SHARD_WIDTH]


def prediction_shard_segment(value: object) -> str:
    """校验一个物理分片目录名；它必须能由某个身份前缀确定性产生。"""

    if not isinstance(value, str) or len(value) != _SHARD_WIDTH or _SHARD.fullmatch(value) is None:
        raise ValueError("prediction shard directory must be lowercase hexadecimal")
    return value


@dataclass(frozen=True)
class PredictionAddress:
    """唯一映射到一个完整预测样本 Markdown 文档。"""

    kind: PredictionKind
    sample_date: date
    materialization_id: str

    def __post_init__(self) -> None:
        kind = PredictionKind(self.kind)
        if isinstance(self.sample_date, datetime) or not isinstance(self.sample_date, date):
            raise TypeError("prediction sample_date must be a date without a time")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "materialization_id", prediction_sample_id(self.materialization_id))


@dataclass(frozen=True)
class PredictionPatternAddress:
    """模式图节点地址；Branch 必须归属于一个 State。"""

    kind: PredictionPatternKind
    state_key: str
    branch_key: str | None = None

    def __post_init__(self) -> None:
        kind = PredictionPatternKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "state_key", prediction_sample_id(self.state_key))
        if kind is PredictionPatternKind.STATE:
            if self.branch_key is not None:
                raise ValueError("prediction StatePattern address forbids branch_key")
        elif self.branch_key is None:
            raise ValueError("prediction BranchPattern address requires branch_key")
        else:
            object.__setattr__(self, "branch_key", prediction_sample_id(self.branch_key))


@dataclass(frozen=True)
class PredictionDirectory:
    """严格限定于事实样本子树的目录地址。"""

    parts: tuple[str, ...] = field(default=(), compare=False)
    _identity_parts: tuple[str, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        if isinstance(self.parts, str) or not isinstance(self.parts, tuple):
            raise TypeError("prediction directory parts must be a tuple of strings")
        parts = tuple(self.parts)
        self._validate(parts)
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "_identity_parts", parts)

    @property
    def identity_parts(self) -> tuple[str, ...]:
        return self._identity_parts

    @classmethod
    def _validate(cls, parts: tuple[str, ...]) -> None:
        if not parts:
            return
        if parts[0] == "patterns":
            cls._validate_pattern_directory(parts)
            return
        if parts[0] != "samples":
            raise ValueError("prediction directory is outside the confirmed tree")
        if len(parts) == 1:
            return
        if parts[1] not in {kind.branch_name for kind in PredictionKind}:
            raise ValueError("prediction sample directory has an unknown branch")
        values = parts[2:5]
        if len(parts) > 6:
            raise ValueError("prediction dated directory is deeper than its shard level")
        if len(parts) == 6:
            prediction_shard_segment(parts[5])
        for value, width, label in zip(values, (4, 2, 2), ("year", "month", "day"), strict=False):
            if not isinstance(value, str) or len(value) != width or not is_ascii_digits(value):
                raise ValueError(f"prediction {label} directory has an invalid format")
        if values and not 1 <= int(values[0]) <= 9999:
            raise ValueError("prediction year directory is outside the calendar range")
        if len(values) >= 2 and not 1 <= int(values[1]) <= 12:
            raise ValueError("prediction month directory is outside the calendar range")
        if len(values) == 3:
            try:
                date(int(values[0]), int(values[1]), int(values[2]))
            except ValueError as exc:
                raise ValueError("prediction day directory is not a valid calendar date") from exc

    @staticmethod
    def _validate_pattern_directory(parts: tuple[str, ...]) -> None:
        """模式文档按 state_key 的前两位分片，避免单目录承载全部状态。"""

        if len(parts) == 1:
            return
        if parts[1] not in {"states", "branches"}:
            raise ValueError("prediction pattern directory is outside states or branches")
        if len(parts) == 2:
            return
        shard = prediction_shard_segment(parts[2])
        if parts[1] == "states":
            if len(parts) != 3:
                raise ValueError("StatePattern documents live directly under their shard directory")
            return
        if len(parts) > 4:
            raise ValueError("BranchPattern documents live directly under their state directory")
        if len(parts) == 4 and prediction_shard(prediction_sample_id(parts[3])) != shard:
            raise ValueError("prediction pattern shard does not match its state key")

    @classmethod
    def root(cls) -> PredictionDirectory:
        return cls()

    @classmethod
    def branch(
        cls,
        kind: PredictionKind,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        shard: str | None = None,
    ) -> PredictionDirectory:
        normalized = PredictionKind(kind)
        if year is None:
            if month is not None or day is not None:
                raise ValueError("prediction month or day requires a year")
            return cls(("samples", normalized.branch_name))
        if isinstance(year, bool) or not isinstance(year, int):
            raise TypeError("prediction directory year must be an integer")
        parts = ["samples", normalized.branch_name, f"{year:04d}"]
        if month is None:
            if day is not None:
                raise ValueError("prediction day requires a month")
            return cls(tuple(parts))
        if isinstance(month, bool) or not isinstance(month, int):
            raise TypeError("prediction directory month must be an integer")
        parts.append(f"{month:02d}")
        if day is None:
            return cls(tuple(parts))
        if isinstance(day, bool) or not isinstance(day, int):
            raise TypeError("prediction directory day must be an integer")
        parts.append(f"{day:02d}")
        if shard is None:
            return cls(tuple(parts))
        parts.append(prediction_shard_segment(shard))
        return cls(tuple(parts))

    @classmethod
    def for_address(cls, address: PredictionAddress) -> PredictionDirectory:
        if not isinstance(address, PredictionAddress):
            raise TypeError("address must be a PredictionAddress")
        sample_date = address.sample_date
        return cls.branch(
            address.kind,
            sample_date.year,
            sample_date.month,
            sample_date.day,
            prediction_shard(address.materialization_id),
        )

    @classmethod
    def for_pattern_address(cls, address: PredictionPatternAddress) -> PredictionDirectory:
        if not isinstance(address, PredictionPatternAddress):
            raise TypeError("address must be a PredictionPatternAddress")
        shard = prediction_shard(address.state_key)
        if address.kind is PredictionPatternKind.STATE:
            return cls(("patterns", "states", shard))
        return cls(("patterns", "branches", shard, address.state_key))

    def parent(self) -> PredictionDirectory | None:
        if not self.parts:
            return None
        return PredictionDirectory(self.parts[:-1])

    def lineage(self) -> tuple[PredictionDirectory, ...]:
        result: list[PredictionDirectory] = []
        current: PredictionDirectory | None = self
        while current is not None:
            result.append(current)
            current = current.parent()
        return tuple(result)


__all__ = [
    "PredictionAddress",
    "PredictionAnchorType",
    "PredictionDecisionBasis",
    "PredictionDirectory",
    "PredictionKind",
    "PredictionLabelStatus",
    "PredictionLevel",
    "PredictionMode",
    "PredictionPatternAddress",
    "PredictionPatternKind",
    "PredictionTargetLevel",
    "PredictionTimePrecision",
    "is_ascii_digits",
    "prediction_sample_id",
    "prediction_shard",
    "prediction_shard_segment",
]
