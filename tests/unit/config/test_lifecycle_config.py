"""统一生命周期配置：Worker 调度只此一处（从 conversation.lifecycle 搬来），其余几样都有界。"""

from __future__ import annotations

import pytest

from habitus.config import (
    ConfigError,
    HabitusConfig,
    LifecycleConfig,
    LifecycleRetryConfig,
    LifecycleWorkerConfig,
)
from tests.unit.config.test_config_loading import valid_mapping


def test_defaults_load_when_the_section_is_absent(tmp_path) -> None:
    config = HabitusConfig.from_mapping(valid_mapping(tmp_path))
    assert config.lifecycle == LifecycleConfig()
    assert config.lifecycle.worker.maintenance_interval_seconds == 3_600
    assert config.lifecycle.retry.max_attempts == 5


def test_nested_overrides_are_read(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload["lifecycle"] = {
        "worker": {"maintenance_interval_seconds": 600, "lease_ttl_seconds": 90, "heartbeat_interval_seconds": 30.0},
        "retry": {"max_attempts": 3},
        "fallback_generations": 3,
    }
    config = HabitusConfig.from_mapping(payload).lifecycle
    assert config.worker.lease_ttl_seconds == 90
    assert config.retry.max_attempts == 3
    assert config.fallback_generations == 3


def test_worker_fields_no_longer_live_under_conversation_lifecycle(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload.setdefault("conversation", {})["lifecycle"] = {"lease_ttl_seconds": 300}
    with pytest.raises(ConfigError, match="lease_ttl_seconds"):
        HabitusConfig.from_mapping(payload)


def test_unknown_lifecycle_field_is_rejected(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload["lifecycle"] = {"half_life_days": 60}
    with pytest.raises(ConfigError, match="half_life_days"):
        HabitusConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"heartbeat_interval_seconds": 200.0, "lease_ttl_seconds": 300},  # 心跳超过租期三分之一
        {"maintenance_interval_seconds": 30},
        {"lease_ttl_seconds": True},
        {"shutdown_timeout_seconds": 0},
    ],
)
def test_bad_worker_values_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LifecycleWorkerConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backoff_base_seconds": 100.0, "backoff_cap_seconds": 10.0},
        {"max_attempts": 0},
        {"backoff_cap_seconds": float("inf")},
    ],
)
def test_bad_retry_values_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LifecycleRetryConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fallback_generations": 0},
        {"capacity_warning_share": 1.0},
        {"physical_max_backfill_days": 400},
        {"worker": {}},
    ],
)
def test_bad_top_level_values_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        LifecycleConfig(**kwargs)  # type: ignore[arg-type]
