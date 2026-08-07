from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from Config import ConfigError, M2BOSConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def mapping(tmp_path):
    payload = yaml.safe_load((REPOSITORY_ROOT / "Config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    return payload


def test_behavior_config_is_strict_typed_and_has_an_independent_root(tmp_path) -> None:
    config = M2BOSConfig.from_mapping(mapping(tmp_path))
    assert config.behavior_root == config.storage_root / "behavior"
    assert config.behavior.evidence.max_batch_size == 256
    assert config.behavior.evidence.max_correlation_refs == 32
    assert config.behavior.normalization.max_model_input_tokens == 16384
    assert not config.behavior.normalization.normalize_user_utterances
    assert config.behavior.store.max_query_limit == 500


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["behavior"].update(unknown=True),
        lambda payload: payload["behavior"]["evidence"].update(max_batch_size=True),
        lambda payload: payload["behavior"]["normalization"].update(max_model_input_tokens=64000),
        lambda payload: payload["behavior"]["store"].update(max_database_bytes=1024),
        lambda payload: payload["behavior"]["evidence"].update(allowed_lateness_seconds=30.0),
        lambda payload: payload["behavior"]["normalization"].update(min_model_confidence=0.5),
        lambda payload: payload["behavior"]["store"].update(max_accepted_claims=10),
    ],
)
def test_behavior_config_rejects_unknown_legacy_and_cross_domain_values(tmp_path, mutation) -> None:
    payload = mapping(tmp_path)
    mutation(payload)
    with pytest.raises((ConfigError, TypeError, ValueError)):
        M2BOSConfig.from_mapping(payload)
