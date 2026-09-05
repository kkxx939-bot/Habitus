"""上游行为 agent 协议到规范观测的无模型适配。

与 ``pre/conversation/adapters.py`` 同构：每个上游 agent 协议一个适配器，只做结构归一，
不做任何语义判断。新增一个上游只需注册新的适配器，不得修改既有协议的解析结果。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from habitus.behavior.observation.config import BehaviorObservationConfig
from habitus.behavior.observation.errors import (
    BehaviorObservationLimitError,
    BehaviorObservationProtocolError,
)
from habitus.behavior.observation.model import BehaviorObservation, BehaviorObservationBatch


class BehaviorObservationAdapter(Protocol):
    protocol: str

    def adapt(
        self,
        payload: object,
        *,
        observer_id: str,
        config: BehaviorObservationConfig,
    ) -> BehaviorObservationBatch: ...


class BehaviorObservationAdapterRegistry:
    """按协议名解析已注册适配器；不按上游猜测协议，也不静默回退。"""

    def __init__(self) -> None:
        self._adapters: dict[str, BehaviorObservationAdapter] = {}

    def register(self, adapter: BehaviorObservationAdapter) -> None:
        protocol = getattr(adapter, "protocol", None)
        if not isinstance(protocol, str) or not protocol or protocol != protocol.strip():
            raise ValueError("adapter protocol must be non-empty normalized text")
        normalized = protocol.lower()
        if normalized in self._adapters:
            raise ValueError(f"behavior observation protocol is already registered: {normalized}")
        if not callable(getattr(adapter, "adapt", None)):
            raise TypeError("adapter must implement adapt")
        self._adapters[normalized] = adapter

    def adapt(
        self,
        protocol: str,
        payload: object,
        *,
        observer_id: str,
        config: BehaviorObservationConfig,
    ) -> BehaviorObservationBatch:
        if not isinstance(protocol, str) or not protocol.strip():
            raise BehaviorObservationProtocolError("protocol must be non-empty text")
        normalized = protocol.strip().lower()
        adapter = self._adapters.get(normalized)
        if adapter is None:
            raise BehaviorObservationProtocolError(
                f"unsupported behavior observation protocol: {normalized}"
            )
        return adapter.adapt(payload, observer_id=observer_id, config=config)

    def protocols(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    @classmethod
    def load_default(cls) -> BehaviorObservationAdapterRegistry:
        registry = cls()
        registry.register(HabitusBehaviorObservationAdapter())
        return registry


class HabitusBehaviorObservationAdapter:
    """Habitus 自有的规范观测协议；上游 agent 直接按它组装载荷。

    尚未出现需要迁就的厂商格式，因此第一个协议就是规范形状本身。将来接入形状不同的
    上游时新增适配器，不要把它们的差异塞进这里。
    """

    protocol = "habitus_behavior_observation_v1"

    def adapt(
        self,
        payload: object,
        *,
        observer_id: str,
        config: BehaviorObservationConfig,
    ) -> BehaviorObservationBatch:
        root = _mapping(payload, "behavior observation payload")
        _require_keys(root, {"observations"}, "behavior observation payload")
        raw = root["observations"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise BehaviorObservationProtocolError("observations must be an array")
        if len(raw) > config.max_observations_per_batch:
            raise BehaviorObservationLimitError("observation batch exceeds its configured item limit")
        observations: list[BehaviorObservation] = []
        seen: set[str] = set()
        for index, item in enumerate(raw):
            observation = _observation(item, index=index, observer_id=observer_id, config=config)
            if observation.observation_id in seen:
                # 同一批里两条内容完全相同的观测（同一秒同一句——ASR 重复吐出是上游的常态，
                # benchmark 的 dirty-double-emission 正是它）：观测身份按内容派生，两条本就是
                # 一条，保留先到的即可。此前整批拒收，等于让一条脏数据废掉整份交付。
                continue
            seen.add(observation.observation_id)
            observations.append(observation)
        return BehaviorObservationBatch(observer_id=observer_id, observations=tuple(observations))


def _observation(
    value: object,
    *,
    index: int,
    observer_id: str,
    config: BehaviorObservationConfig,
) -> BehaviorObservation:
    item = _mapping(value, f"observations[{index}]")
    _require_keys(
        item,
        {
            "occurred_at",
            "available_at",
            "modality",
            "semantics",
            "participants",
            "knowledge_state",
            "confidence",
            "evidence_refs",
        },
        f"observations[{index}]",
    )
    return BehaviorObservation.create(
        observer_id=observer_id,
        occurred_at=item["occurred_at"],
        available_at=item["available_at"],
        modality=item["modality"],
        semantics=item["semantics"],
        participants=item["participants"],
        knowledge_state=item["knowledge_state"],
        confidence=item["confidence"],
        evidence_refs=item["evidence_refs"],
        config=config,
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BehaviorObservationProtocolError(f"{label} must be an object with string keys")
    return value


def _require_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown:
        raise BehaviorObservationProtocolError(f"{label} contains unsupported keys: {sorted(unknown)}")
    if missing:
        raise BehaviorObservationProtocolError(f"{label} is missing keys: {sorted(missing)}")


__all__ = [
    "BehaviorObservationAdapter",
    "BehaviorObservationAdapterRegistry",
    "HabitusBehaviorObservationAdapter",
]
