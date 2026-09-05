"""长期记忆实际使用、动态冷热、COLD_2 探测和退休状态。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Protocol

from memory.model import MemoryKind
from memory.uri import MemoryURI, MemoryURINodeType


class MemoryRecallLifecycleError(RuntimeError):
    """召回生命周期状态无法安全读取或更新。"""


class MemoryTemperature(str, Enum):
    COLD_2 = "cold_2"
    COLD_1 = "cold_1"
    WARM = "warm"
    HOT = "hot"


@dataclass(frozen=True)
class MemoryRecallLifecycleConfig:
    enabled: bool = True
    ranking_alpha: float = 0.2
    cold2_threshold: float = 0.08
    cold1_threshold: float = 0.2
    hot_threshold: float = 0.6
    profile_half_life_days: float = 180.0
    preference_half_life_days: float = 90.0
    entity_half_life_days: float = 60.0
    tool_half_life_days: float = 30.0
    event_half_life_days: float = 14.0
    intention_half_life_days: float = 30.0
    profile_retire_days: float = 720.0
    preference_retire_days: float = 365.0
    entity_retire_days: float = 270.0
    tool_retire_days: float = 180.0
    event_retire_days: float = 180.0
    intention_retire_days: float = 540.0
    retire_candidate_grace_days: float = 30.0
    cold2_probe_interval_days: float = 30.0
    cold2_probe_limit: int = 2
    max_cold2_probes_per_search: int = 1
    sqlite_timeout_seconds: float = 5.0
    max_batch_size: int = 1_000
    max_useful_recall_count: int = 1_000_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("memory recall lifecycle enabled must be boolean")
        numeric = (
            "ranking_alpha",
            "cold2_threshold",
            "cold1_threshold",
            "hot_threshold",
            "profile_half_life_days",
            "preference_half_life_days",
            "entity_half_life_days",
            "tool_half_life_days",
            "event_half_life_days",
            "intention_half_life_days",
            "profile_retire_days",
            "preference_retire_days",
            "entity_retire_days",
            "tool_retire_days",
            "event_retire_days",
            "intention_retire_days",
            "retire_candidate_grace_days",
            "cold2_probe_interval_days",
            "sqlite_timeout_seconds",
        )
        for name in numeric:
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
        if not 0.0 < self.cold2_threshold < self.cold1_threshold <= 0.5:
            raise ValueError("memory recall lifecycle cold thresholds are invalid")
        if not 0.5 <= self.hot_threshold < 2.0 / 3.0:
            raise ValueError("memory recall lifecycle hot_threshold must preserve new warm and first-use hot")
        if self.cold1_threshold >= self.hot_threshold:
            raise ValueError("memory recall lifecycle cold1_threshold must be lower than hot_threshold")
        for name in (
            "profile_half_life_days",
            "preference_half_life_days",
            "entity_half_life_days",
            "tool_half_life_days",
            "event_half_life_days",
            "intention_half_life_days",
            "profile_retire_days",
            "preference_retire_days",
            "entity_retire_days",
            "tool_retire_days",
            "event_retire_days",
            "intention_retire_days",
        ):
            if not 0.001 <= getattr(self, name) <= 36_500.0:
                raise ValueError(f"memory recall lifecycle {name} must be between 0.001 and 36500")
        for name in ("retire_candidate_grace_days", "cold2_probe_interval_days"):
            if not 0.001 <= getattr(self, name) <= 36_500.0:
                raise ValueError(f"memory recall lifecycle {name} is outside its supported range")
        if not 0.001 <= self.sqlite_timeout_seconds <= 60.0:
            raise ValueError("memory recall lifecycle sqlite_timeout_seconds must be between 0.001 and 60")
        for name, minimum, maximum in (
            ("max_batch_size", 1, 100_000),
            ("max_useful_recall_count", 1, 2**63 - 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"memory recall lifecycle {name} is outside its supported range")
        if (
            isinstance(self.cold2_probe_limit, bool)
            or not isinstance(self.cold2_probe_limit, int)
            or not 1 <= self.cold2_probe_limit <= self.max_useful_recall_count
        ):
            raise ValueError("memory recall lifecycle probe observation cap is invalid")
        if (
            isinstance(self.max_cold2_probes_per_search, bool)
            or not isinstance(self.max_cold2_probes_per_search, int)
            or self.max_cold2_probes_per_search != 1
        ):
            raise ValueError("memory recall lifecycle permits at most one COLD_2 probe per search")

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

    def retire_days(self, kind: MemoryKind) -> float:
        normalized = MemoryKind(kind)
        return {
            MemoryKind.PROFILE: self.profile_retire_days,
            MemoryKind.PREFERENCE: self.preference_retire_days,
            MemoryKind.ENTITY: self.entity_retire_days,
            MemoryKind.TOOL: self.tool_retire_days,
            MemoryKind.EVENT: self.event_retire_days,
            MemoryKind.INTENTION: self.intention_retire_days,
        }[normalized]


@dataclass(frozen=True)
class MemoryRecallState:
    uri: MemoryURI
    document_revision: int
    document_created_at: datetime
    useful_recall_count: int
    last_useful_recall_at: datetime | None
    lifecycle_activity_at: datetime | None
    cold2_probe_count: int
    last_cold2_probe_at: datetime | None
    compacted_at: datetime | None
    retire_candidate_at: datetime | None
    retired_at: datetime | None
    version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _document_uri(self.uri))
        for name in ("document_revision", "useful_recall_count", "cold2_probe_count", "version"):
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
        for name in (
            "last_useful_recall_at",
            "lifecycle_activity_at",
            "last_cold2_probe_at",
            "compacted_at",
            "retire_candidate_at",
            "retired_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _utc_timestamp(value, name))
        if (self.useful_recall_count == 0) != (self.last_useful_recall_at is None):
            raise ValueError("memory recall state useful count and timestamp must describe the same lifecycle")
        if (self.cold2_probe_count == 0) != (self.last_cold2_probe_at is None):
            raise ValueError("memory recall state probe count and timestamp must describe the same lifecycle")
        if self.retire_candidate_at is not None and self.compacted_at is None:
            raise ValueError("memory recall state cannot become a retire candidate before compaction")
        if self.retired_at is not None and self.retire_candidate_at is None:
            raise ValueError("memory recall state cannot retire before its grace candidate phase")
        if self.retired_at is not None and self.compacted_at is None:
            raise ValueError("memory recall state cannot retire before compaction")
        if self.version == 0 and any(
            value is not None
            for value in (
                self.last_useful_recall_at,
                self.lifecycle_activity_at,
                self.last_cold2_probe_at,
                self.compacted_at,
                self.retire_candidate_at,
                self.retired_at,
            )
        ):
            raise ValueError("initial memory recall state cannot contain persisted lifecycle facts")

    @classmethod
    def initial(
        cls,
        uri: MemoryURI,
        document_revision: int,
        document_created_at: datetime,
    ) -> MemoryRecallState:
        return cls(
            uri,
            document_revision,
            document_created_at,
            0,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            0,
        )


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

    def record_use(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        used_at: datetime,
    ) -> tuple[MemoryRecallState, ...]: ...

    def record_probe(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        probed_at: datetime,
    ) -> tuple[MemoryRecallState, ...]: ...

    def mark_compacted(
        self,
        target: MemoryRecallTarget,
        *,
        lifecycle_activity_at: datetime,
        compacted_at: datetime,
        expected_version: int,
    ) -> MemoryRecallState: ...

    def mark_retired(
        self,
        target: MemoryRecallTarget,
        *,
        retired_at: datetime,
        expected_version: int,
    ) -> MemoryRecallState: ...

    def mark_retire_candidate(
        self,
        target: MemoryRecallTarget,
        *,
        marked_at: datetime,
        expected_version: int,
    ) -> MemoryRecallState: ...

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
    activity_at = state.lifecycle_activity_at or content_time
    if state.last_useful_recall_at is not None:
        activity_at = max(activity_at, state.last_useful_recall_at)
    age_days = max(0.0, (current - activity_at).total_seconds() / 86_400.0)
    frequency = (state.useful_recall_count + 1.0) / (state.useful_recall_count + 2.0)
    return frequency * math.pow(2.0, -age_days / half_life)


def memory_temperature(
    hotness: float,
    *,
    cold2_threshold: float,
    cold1_threshold: float,
    hot_threshold: float,
) -> MemoryTemperature:
    score = _finite_score(hotness, "hotness")
    cold2 = _finite_score(cold2_threshold, "cold2_threshold")
    cold1 = _finite_score(cold1_threshold, "cold1_threshold")
    hot = _finite_score(hot_threshold, "hot_threshold")
    if not 0.0 <= score <= 1.0 or not 0.0 < cold2 < cold1 < hot <= 1.0:
        raise ValueError("memory temperature inputs are outside their supported range")
    if score < cold2:
        return MemoryTemperature.COLD_2
    if score < cold1:
        return MemoryTemperature.COLD_1
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
        required = (
            "initialize",
            "read_many",
            "record_use",
            "record_probe",
            "mark_compacted",
            "mark_retire_candidate",
            "mark_retired",
            "delete_many",
        )
        if any(not callable(getattr(store, name, None)) for name in required):
            raise TypeError("memory recall lifecycle store does not implement the required contract")
        if config is not None and not isinstance(config, MemoryRecallLifecycleConfig):
            raise TypeError("config must be MemoryRecallLifecycleConfig")
        self.store = store
        self.config = config or MemoryRecallLifecycleConfig()

    def initialize(self) -> None:
        if self.config.enabled:
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
            if state is None or state.document_created_at != candidate.target.document_created_at:
                state = MemoryRecallState.initial(
                    candidate.uri,
                    candidate.target.document_revision,
                    candidate.target.document_created_at,
                )
            elif state.document_revision != candidate.target.document_revision:
                # 同一内容代的新业务更新立即按新鲜内容重新计算，并退出旧压缩/退休状态。
                state = MemoryRecallState(
                    candidate.uri,
                    candidate.target.document_revision,
                    candidate.target.document_created_at,
                    state.useful_recall_count,
                    state.last_useful_recall_at,
                    candidate.updated_at,
                    0,
                    None,
                    None,
                    None,
                    None,
                    state.version,
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
                        cold2_threshold=self.config.cold2_threshold,
                        cold1_threshold=self.config.cold1_threshold,
                        hot_threshold=self.config.hot_threshold,
                    ),
                    final_score=lifecycle_adjusted_score(
                        candidate.semantic_score,
                        hotness,
                        alpha=self.config.ranking_alpha,
                    ),
                )
            )
        return tuple(sorted(rankings, key=lambda item: (-item.final_score, str(item.candidate.uri))))

    def select(
        self,
        candidates: tuple[MemoryRecallCandidate, ...],
        *,
        limit: int,
        now: datetime,
    ) -> tuple[MemoryRecallRanking, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("memory recall lifecycle selection limit must be positive")
        current = _utc_timestamp(now, "now")
        ranked = self.rank(candidates, now=current)
        if not self.config.enabled:
            return ranked[:limit]
        semantic_top = tuple(
            sorted(
                (
                    item
                    for item in ranked
                    if item.state.retired_at is None
                ),
                key=lambda item: (-item.candidate.semantic_score, str(item.candidate.uri)),
            )
        )[:limit]
        probe_candidates = tuple(
            item
            for item in semantic_top
            if item.temperature is MemoryTemperature.COLD_2
            and self._probe_eligible(item.state, current)
        )[: self.config.max_cold2_probes_per_search]
        selected_probe_uris = {item.candidate.uri for item in probe_candidates}
        normal = tuple(
            item
            for item in ranked
            if item.state.retired_at is None
            and item.temperature is not MemoryTemperature.COLD_2
        )[: max(0, limit - len(probe_candidates))]
        return tuple(
            sorted(
                (*normal, *(item for item in probe_candidates if item.candidate.uri in selected_probe_uris)),
                key=lambda item: (-item.final_score, str(item.candidate.uri)),
            )
        )

    def record_use(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        used_at: datetime,
    ) -> tuple[MemoryRecallState, ...]:
        values = self._targets(targets)
        current = _utc_timestamp(used_at, "used_at")
        if not values or not self.config.enabled:
            return ()
        return self.store.record_use(values, used_at=current)

    def mark_compacted(
        self,
        target: MemoryRecallTarget,
        *,
        lifecycle_activity_at: datetime,
        compacted_at: datetime,
        expected_version: int,
    ) -> MemoryRecallState:
        if not isinstance(target, MemoryRecallTarget):
            raise TypeError("target must be MemoryRecallTarget")
        if not self.config.enabled:
            return MemoryRecallState.initial(target.uri, target.document_revision, target.document_created_at)
        return self.store.mark_compacted(
            target,
            lifecycle_activity_at=_utc_timestamp(lifecycle_activity_at, "lifecycle_activity_at"),
            compacted_at=_utc_timestamp(compacted_at, "compacted_at"),
            expected_version=expected_version,
        )

    def mark_retired(
        self,
        target: MemoryRecallTarget,
        *,
        retired_at: datetime,
        expected_version: int,
    ) -> MemoryRecallState:
        if not isinstance(target, MemoryRecallTarget):
            raise TypeError("target must be MemoryRecallTarget")
        return self.store.mark_retired(
            target,
            retired_at=_utc_timestamp(retired_at, "retired_at"),
            expected_version=expected_version,
        )

    def mark_retire_candidate(
        self,
        target: MemoryRecallTarget,
        *,
        marked_at: datetime,
        expected_version: int,
    ) -> MemoryRecallState:
        if not isinstance(target, MemoryRecallTarget):
            raise TypeError("target must be MemoryRecallTarget")
        return self.store.mark_retire_candidate(
            target,
            marked_at=_utc_timestamp(marked_at, "marked_at"),
            expected_version=expected_version,
        )

    def retirement_eligible(
        self,
        ranking: MemoryRecallRanking,
        *,
        now: datetime,
    ) -> bool:
        if not isinstance(ranking, MemoryRecallRanking):
            raise TypeError("ranking must be MemoryRecallRanking")
        current = _utc_timestamp(now, "now")
        state = ranking.state
        if (
            ranking.temperature is not MemoryTemperature.COLD_2
            or state.compacted_at is None
            or state.retired_at is not None
        ):
            return False
        activity = max(
            value
            for value in (
                ranking.candidate.updated_at,
                state.last_useful_recall_at,
                state.lifecycle_activity_at,
                state.compacted_at,
            )
            if value is not None
        )
        return current - activity >= timedelta(days=self.config.retire_days(ranking.candidate.kind))

    def retirement_grace_elapsed(
        self,
        state: MemoryRecallState,
        *,
        now: datetime,
    ) -> bool:
        if not isinstance(state, MemoryRecallState):
            raise TypeError("state must be MemoryRecallState")
        current = _utc_timestamp(now, "now")
        return state.retire_candidate_at is not None and current - state.retire_candidate_at >= timedelta(
            days=self.config.retire_candidate_grace_days
        )

    def forget(self, uris: tuple[MemoryURI, ...]) -> int:
        values = _document_uris(uris, maximum=self.config.max_batch_size)
        if not values or not self.config.enabled:
            return 0
        return self.store.delete_many(values)

    def _probe_eligible(self, state: MemoryRecallState, now: datetime) -> bool:
        if state.last_cold2_probe_at is None:
            return True
        return now - state.last_cold2_probe_at >= timedelta(days=self.config.cold2_probe_interval_days)

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
    return value.astimezone(UTC)


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
