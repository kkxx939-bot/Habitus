"""Behavior Evidence & Claim Layer 的外部配置聚合。"""

from __future__ import annotations

from behavior.config import BehaviorConfig, ClaimConfig, EvidenceConfig, SourceConfig, StoreConfig
from Config.loader import construct_config, group_fields


def behavior_config_from_mapping(value: object) -> BehaviorConfig:
    """严格拒绝未知组和未知字段后构造领域配置。"""

    data = group_fields(BehaviorConfig, value, "config.behavior")
    return BehaviorConfig(
        source=construct_config(SourceConfig, data.get("source", {}), "config.behavior.source"),
        evidence=construct_config(EvidenceConfig, data.get("evidence", {}), "config.behavior.evidence"),
        claim=construct_config(ClaimConfig, data.get("claim", {}), "config.behavior.claim"),
        store=construct_config(StoreConfig, data.get("store", {}), "config.behavior.store"),
    )


__all__ = [
    "BehaviorConfig",
    "ClaimConfig",
    "EvidenceConfig",
    "SourceConfig",
    "StoreConfig",
    "behavior_config_from_mapping",
]
