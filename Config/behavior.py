"""Behavior Evidence & Claim Layer 的严格外部配置聚合。"""

from __future__ import annotations

from behavior.config import (
    BehaviorConfig,
    BehaviorEvidenceConfig,
    BehaviorStoreConfig,
    ClaimNormalizationConfig,
)
from Config.loader import construct_config, group_fields


def behavior_config_from_mapping(value: object) -> BehaviorConfig:
    data = group_fields(BehaviorConfig, value, "config.behavior")
    return BehaviorConfig(
        evidence=construct_config(
            BehaviorEvidenceConfig,
            data.get("evidence", {}),
            "config.behavior.evidence",
        ),
        normalization=construct_config(
            ClaimNormalizationConfig,
            data.get("normalization", {}),
            "config.behavior.normalization",
        ),
        store=construct_config(
            BehaviorStoreConfig,
            data.get("store", {}),
            "config.behavior.store",
        ),
    )


__all__ = [
    "BehaviorConfig",
    "BehaviorEvidenceConfig",
    "BehaviorStoreConfig",
    "ClaimNormalizationConfig",
    "behavior_config_from_mapping",
]
