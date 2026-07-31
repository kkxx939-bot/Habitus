"""长期记忆召回热度的领域模型、纯计算函数和服务边界。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from memory.model import MemoryKind
from memory.uri import MemoryURI, MemoryURINodeType


class MemoryRecallLifecycleError(RuntimeError):
    """召回生命周期状态无法安全读取或更新。"""


class MemoryTemperature(str, Enum):
    COLD = "cold"
    WARM = "warm"
    HOT = "hot"


@dataclass(frozen=True)
class MemoryRecallLifecycleConfig:
    enabled: bool = True
    ranking_alpha: float = 0.2
    cold_threshold: float = 0.2
    hot_threshold: float = 0.6
    profile_half_life_days: float = 180.0
    preference_half_life_days: float = 90.0
    entity_half_life_days: float = 60.0
    tool_half_life_days: float = 30.0
    event_half_life_days: float = 14.0
    intention_half_life_days: float = 30.0
    sqlite_timeout_seconds: float = 5.0
    max_batch_size: int = 1_000
    max_successful_recall_count: int = 1_000_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("memory recall lifecycle enabled must be boolean")
        for name in (
            "ranking_alpha",
            "cold_threshold",
            "hot_threshold",
            "profile_half_life_days",
            "preference_half_life_days",
            "entity_half_life_days",
            "tool_half_life_days",
            "event_half_life_days",
            "intention_half_life_days",
            "sqlite_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"memory recall lifecycle {name} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"memory recall lifecycle {name} must be finite")
            object.__setattr__(self, name, number)
        if not 0.0 <= self.ranking_alpha <= 0.95:
            raise ValueError("memory recall lifecycle ranking_alpha must be between zero and 0.95")
        if self.enabled and self.ranking_alpha == 0.0:
            raise ValueError("enabled memory recall lifecycle requires a positive ranking_alpha")
        if not 0.0 <= self.cold_threshold < self.hot_threshold <= 1.0:
            raise ValueError("memory recall lifecycle temperature thresholds are invalid")
        if not 0.0 < self.cold_threshold <= 0.5 <= self.hot_threshold < 2.0 / 3.0:
            raise ValueError(
                "memory recall lifecycle thresholds must preserve new warm, first-use hot, and eventual cold states"
            )
        for name in (
            "profile_half_life_days",
            "preference_half_life_days",
            "entity_half_life_days",
            "tool_half_life_days",
            "event_half_life_days",
            "intention_half_life_days",
        ):
            if not 0.001 <= getattr(self, name) <= 36_500.0:
                raise ValueError(f"memory recall lifecycle {name} must be between 0.001 and 36500")
        if not 0.001 <= self.sqlite_timeout_seconds <= 60.0:
            raise ValueError("memory recall lifecycle sqlite_timeout_seconds must be between 0.001 and 60")
        for name, maximum in (
            ("max_batch_size", 100_000),
            ("max_successful_recall_count", 2**63 - 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"memory recall lifecycle {name} is outside its supported range")

    def half_life_days(self, kind: MemoryKind) -> float:
        normalized = MemoryKind(kind)
        return {
            MemoryKind.PROFILE: self.profile_half_life_days,
            MemoryKind.PREFERENCE: self.preference_half_life_days,
            MemoryKind.ENTITY: self.entity_half_life_days,
            MemoryKind.TOOL: self.tool_half_life_days,
            MemoryKind.EVENT: self.event_half_life_days,
            MemoryKind.INTENTION: self.intention_half_life_days,
        }[normalized]


@dataclass(frozen=True)
class MemoryRecallState:
    uri: MemoryURI
    document_revision: int
    document_created_at: datetime
    successful_recall_count: int
    last_successful_recall_at: datetime | None
    version: int

    def __post_init__(self) -> None:
        uri = _document_uri(self.uri)
        object.__setattr__(self, "uri", uri)
        for name in ("document_revision", "successful_recall_count", "version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"memory recall state {name} must be non-negative")
        if self.document_revision <= 0:
            raise ValueError("memory recall state document_revision must be positive")
        object.__setattr__(
            self,
            "document_created_at",
            _utc_timestamp(self.document_created_at, "document_created_at"),
        )
        timestamp = (
            None
            if self.last_successful_recall_at is None
            else _utc_timestamp(self.last_successful_recall_at, "last_successful_recall_at")
        )
        if (self.successful_recall_count == 0) != (timestamp is None):
            raise ValueError("memory recall state count and timestamp must describe the same lifecycle")
        if self.successful_recall_count > 0 and self.version <= 0:
            raise ValueError("persisted memory recall state must have a positive version")
        object.__setattr__(self, "last_successful_recall_at", timestamp)

    @classmethod
    def initial(
        cls,
        uri: MemoryURI,
        document_revision: int,
        document_created_at: datetime,
    ) -> MemoryRecallState:
        return cls(uri, document_revision, document_created_at, 0, None, 0)


@dataclass(frozen=True)
class MemoryRecallTarget:
    uri: MemoryURI
    document_revision: int
    document_created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _document_uri(self.uri))
        if (
            isinstance(self.document_revision, bool)
            or not isinstance(self.document_revision, int)
            or self.document_revision <= 0
        ):
            raise ValueError("memory recall target document_revision must be positive")
        object.__setattr__(
            self,
            "document_created_at",
            _utc_timestamp(self.document_created_at, "document_created_at"),
        )


@dataclass(frozen=True)
class MemoryRecallCandidate:
    target: MemoryRecallTarget
    kind: MemoryKind
    updated_at: datetime
    semantic_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.target, MemoryRecallTarget):
            raise TypeError("memory recall candidate target must be MemoryRecallTarget")
        kind = MemoryKind(self.kind)
        if self.target.uri.to_address().kind is not kind:
            raise ValueError("memory recall candidate kind does not match its URI")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "updated_at", _utc_timestamp(self.updated_at, "updated_at"))
        if self.updated_at < self.target.document_created_at:
            raise ValueError("memory recall candidate updated_at cannot precede document creation")
        object.__setattr__(self, "semantic_score", _finite_score(self.semantic_score, "semantic_score"))

    @property
    def uri(self) -> MemoryURI:
        return self.target.uri


@dataclass(frozen=True)
class MemoryRecallRanking:
    candidate: MemoryRecallCandidate
    state: MemoryRecallState
    hotness: float | None
    temperature: MemoryTemperature | None
    final_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, MemoryRecallCandidate):
            raise TypeError("memory recall ranking candidate is invalid")
        if not isinstance(self.state, MemoryRecallState):
            raise TypeError("memory recall ranking state is invalid")
        if self.candidate.uri != self.state.uri:
            raise ValueError("memory recall ranking identity does not match")
        if (self.hotness is None) != (self.temperature is None):
            raise ValueError("memory recall ranking temperature requires a hotness score")
        if self.hotness is not None:
            hotness = _finite_score(self.hotness, "hotness")
            if not 0.0 <= hotness <= 1.0:
                raise ValueError("memory recall ranking hotness must be between zero and one")
            object.__setattr__(self, "hotness", hotness)
            object.__setattr__(self, "temperature", MemoryTemperature(self.temperature))
        object.__setattr__(self, "final_score", _finite_score(self.final_score, "final_score"))


class MemoryRecallStateStore(Protocol):
    def initialize(self) -> None: ...

    def read_many(self, uris: tuple[MemoryURI, ...]) -> tuple[MemoryRecallState, ...]: ...

    def record_success(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        recalled_at: datetime,
    ) -> tuple[MemoryRecallState, ...]: ...

    def delete_many(self, uris: tuple[MemoryURI, ...]) -> int: ...


def memory_hotness(
    state: MemoryRecallState,
    *,
    updated_at: datetime,
    now: datetime,
    half_life_days: float,
) -> float:
    if not isinstance(state, MemoryRecallState):
        raise TypeError("memory hotness state must be MemoryRecallState")
    content_time = _utc_timestamp(updated_at, "updated_at")
    current = _utc_timestamp(now, "now")
    if isinstance(half_life_days, bool) or not isinstance(half_life_days, int | float):
        raise TypeError("memory hotness half_life_days must be numeric")
    half_life = float(half_life_days)
    if not math.isfinite(half_life) or half_life <= 0.0:
        raise ValueError("memory hotness half_life_days must be positive and finite")
    activity_at = content_time
    if state.last_successful_recall_at is not None:
        activity_at = max(activity_at, state.last_successful_recall_at)
    age_days = max(0.0, (current - activity_at).total_seconds() / 86_400.0)
    frequency = (state.successful_recall_count + 1.0) / (state.successful_recall_count + 2.0)
    return frequency * math.pow(2.0, -age_days / half_life)


def memory_temperature(
    hotness: float,
    *,
    cold_threshold: float,
    hot_threshold: float,
) -> MemoryTemperature:
    score = _finite_score(hotness, "hotness")
    cold = _finite_score(cold_threshold, "cold_threshold")
    hot = _finite_score(hot_threshold, "hot_threshold")
    if not 0.0 <= score <= 1.0 or not 0.0 <= cold < hot <= 1.0:
        raise ValueError("memory temperature inputs are outside their supported range")
    if score < cold:
        return MemoryTemperature.COLD
    if score > hot:
        return MemoryTemperature.HOT
    return MemoryTemperature.WARM


def lifecycle_adjusted_score(
    semantic_score: float,
    hotness: float,
    *,
    alpha: float,
) -> float:
    semantic = _finite_score(semantic_score, "semantic_score")
    heat = _finite_score(hotness, "hotness")
    weight = _finite_score(alpha, "alpha")
    if not 0.0 <= heat <= 1.0:
        raise ValueError("memory lifecycle hotness must be between zero and one")
    if not 0.0 <= weight < 1.0:
        raise ValueError("memory lifecycle alpha must be between zero and one")
    factor = 1.0 - weight * (1.0 - heat)
    return semantic * factor if semantic >= 0.0 else semantic / factor


class MemoryRecallLifecycle:
    def __init__(
        self,
        store: MemoryRecallStateStore,
        *,
        config: MemoryRecallLifecycleConfig | None = None,
    ) -> None:
        if not callable(getattr(store, "initialize", None)):
            raise TypeError("memory recall lifecycle store must implement initialize")
        if not callable(getattr(store, "read_many", None)):
            raise TypeError("memory recall lifecycle store must implement read_many")
        if not callable(getattr(store, "record_success", None)):
            raise TypeError("memory recall lifecycle store must implement record_success")
        if not callable(getattr(store, "delete_many", None)):
            raise TypeError("memory recall lifecycle store must implement delete_many")
        if config is not None and not isinstance(config, MemoryRecallLifecycleConfig):
            raise TypeError("config must be MemoryRecallLifecycleConfig")
        self.store = store
        self.config = config or MemoryRecallLifecycleConfig()

    def initialize(self) -> None:
        if not self.config.enabled:
            return
        self.store.initialize()

    def rank(
        self,
        candidates: tuple[MemoryRecallCandidate, ...],
        *,
        now: datetime,
    ) -> tuple[MemoryRecallRanking, ...]:
        values = self._candidates(candidates)
        current = _utc_timestamp(now, "now")
        if not values:
            return ()
        if not self.config.enabled:
            return tuple(
                MemoryRecallRanking(
                    candidate,
                    MemoryRecallState.initial(
                        candidate.uri,
                        candidate.target.document_revision,
                        candidate.target.document_created_at,
                    ),
                    None,
                    None,
                    candidate.semantic_score,
                )
                for candidate in values
            )
        states = {
            state.uri: state
            for state in self.store.read_many(tuple(candidate.uri for candidate in values))
        }
        rankings: list[MemoryRecallRanking] = []
        for candidate in values:
            state = states.get(candidate.uri)
            if (
                state is None
                or state.document_revision != candidate.target.document_revision
                or state.document_created_at != candidate.target.document_created_at
            ):
                state = MemoryRecallState.initial(
                    candidate.uri,
                    candidate.target.document_revision,
                    candidate.target.document_created_at,
                )
            hotness = memory_hotness(
                state,
                updated_at=candidate.updated_at,
                now=current,
                half_life_days=self.config.half_life_days(candidate.kind),
            )
            rankings.append(
                MemoryRecallRanking(
                    candidate=candidate,
                    state=state,
                    hotness=hotness,
                    temperature=memory_temperature(
                        hotness,
                        cold_threshold=self.config.cold_threshold,
                        hot_threshold=self.config.hot_threshold,
                    ),
                    final_score=lifecycle_adjusted_score(
                        candidate.semantic_score,
                        hotness,
                        alpha=self.config.ranking_alpha,
                    ),
                )
            )
        return tuple(
            sorted(
                rankings,
                key=lambda item: (-item.final_score, str(item.candidate.uri)),
            )
        )

    def record_success(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        recalled_at: datetime,
    ) -> tuple[MemoryRecallState, ...]:
        values = self._targets(targets)
        current = _utc_timestamp(recalled_at, "recalled_at")
        if not values or not self.config.enabled:
            return ()
        return self.store.record_success(values, recalled_at=current)

    def forget(self, uris: tuple[MemoryURI, ...]) -> int:
        values = _document_uris(uris, maximum=self.config.max_batch_size)
        if not values or not self.config.enabled:
            return 0
        return self.store.delete_many(values)

    def _candidates(
        self,
        values: tuple[MemoryRecallCandidate, ...],
    ) -> tuple[MemoryRecallCandidate, ...]:
        if not isinstance(values, tuple) or any(
            not isinstance(value, MemoryRecallCandidate) for value in values
        ):
            raise TypeError("memory recall lifecycle candidates must be a tuple")
        if len(values) > self.config.max_batch_size:
            raise ValueError("memory recall lifecycle candidates exceed max_batch_size")
        if len({value.uri for value in values}) != len(values):
            raise ValueError("memory recall lifecycle candidates must be unique")
        return values

    def _targets(
        self,
        values: tuple[MemoryRecallTarget, ...],
    ) -> tuple[MemoryRecallTarget, ...]:
        if not isinstance(values, tuple) or any(
            not isinstance(value, MemoryRecallTarget) for value in values
        ):
            raise TypeError("memory recall lifecycle targets must be a tuple")
        if len(values) > self.config.max_batch_size:
            raise ValueError("memory recall lifecycle targets exceed max_batch_size")
        by_uri: dict[MemoryURI, MemoryRecallTarget] = {}
        for value in values:
            previous = by_uri.get(value.uri)
            if previous is not None and previous != value:
                raise ValueError("memory recall lifecycle target revisions conflict")
            by_uri[value.uri] = value
        return tuple(sorted(by_uri.values(), key=lambda value: str(value.uri)))


def _document_uri(value: MemoryURI | str) -> MemoryURI:
    parsed = MemoryURI.parse(value)
    if parsed.node_type is not MemoryURINodeType.DOCUMENT:
        raise ValueError("memory recall lifecycle URI must identify an L2 document")
    return parsed


def _document_uris(
    values: tuple[MemoryURI, ...],
    *,
    maximum: int,
) -> tuple[MemoryURI, ...]:
    if not isinstance(values, tuple):
        raise TypeError("memory recall lifecycle URIs must be a tuple")
    if len(values) > maximum:
        raise ValueError("memory recall lifecycle URIs exceed max_batch_size")
    parsed = tuple(_document_uri(value) for value in values)
    if len(set(parsed)) != len(parsed):
        raise ValueError("memory recall lifecycle URIs must be unique")
    return parsed


def _utc_timestamp(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"memory recall lifecycle {label} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"memory recall lifecycle {label} must include a timezone")
    return value.astimezone(timezone.utc)


def _finite_score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"memory recall lifecycle {label} must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"memory recall lifecycle {label} must be finite")
    return score


__all__ = [
    "MemoryRecallCandidate",
    "MemoryRecallLifecycle",
    "MemoryRecallLifecycleConfig",
    "MemoryRecallLifecycleError",
    "MemoryRecallRanking",
    "MemoryRecallState",
    "MemoryRecallStateStore",
    "MemoryRecallTarget",
    "MemoryTemperature",
    "lifecycle_adjusted_score",
    "memory_hotness",
    "memory_temperature",
]
