"""语义先验伪计数表与学习器消费的合同测试。"""

from __future__ import annotations

import pytest
from test_pattern_learning import _WATER, LEARNED_AT, _transition

from prediction import PredictionPatternKind
from prediction.learning import (
    PredictionBehaviorPrior,
    PredictionLearningError,
    PredictionPatternLearner,
    behavior_branch_catalog,
    prior_entry,
)
from prediction.learning.keys import sequence_state_identity

_ROOT_IDENTITY = sequence_state_identity("action", "device_control", None)
_WATER_TARGET = {
    "target_kind": "action",
    "behavior_type": "daily_activity",
    "semantics": "倒一杯水",
    "target_refs": ["水杯"],
}


def _documents() -> tuple:
    return (
        _transition("prior-a"),
        _transition("prior-b"),
        _transition("prior-c"),
        _transition("prior-d", label=_WATER),
    )


def _root_probability(patterns, semantics: str) -> float:
    return next(
        item.fields["conditional_probability"]
        for item in patterns
        if item.kind is PredictionPatternKind.BRANCH
        and item.fields["behavior"]["semantics"] == semantics
        and item.fields["identity_material"]["branch_identity"]["target"]["semantics"] == semantics
    )


def test_prior_pseudo_counts_shift_the_root_distribution() -> None:
    documents = _documents()
    prior = PredictionBehaviorPrior(
        entries=(prior_entry(_ROOT_IDENTITY, ((_WATER_TARGET, 0.5),)),),
        strength=2.0,
    )

    plain = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    informed = PredictionPatternLearner(prior=prior).learn(documents, learned_at=LEARNED_AT)

    assert _root_probability(informed, "倒一杯水") > _root_probability(plain, "倒一杯水")
    assert _root_probability(informed, "打开空调") < _root_probability(plain, "打开空调")
    assert informed[0].fields["pattern_generation"] != plain[0].fields["pattern_generation"]


def test_unseen_prior_branches_stay_in_the_escape_mass() -> None:
    documents = _documents()
    unseen_target = {
        "target_kind": "action",
        "behavior_type": "device_control",
        "semantics": "关灯",
        "target_refs": ["灯"],
    }
    prior = PredictionBehaviorPrior(
        entries=(prior_entry(_ROOT_IDENTITY, ((unseen_target, 0.4),)),),
        strength=2.0,
    )

    plain = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    informed = PredictionPatternLearner(prior=prior).learn(documents, learned_at=LEARNED_AT)

    informed_semantics = {
        item.fields["behavior"]["semantics"]
        for item in informed
        if item.kind is PredictionPatternKind.BRANCH
    }
    assert "关灯" not in informed_semantics
    assert _root_probability(informed, "打开空调") < _root_probability(plain, "打开空调")


def test_prior_validation_and_catalog() -> None:
    with pytest.raises(PredictionLearningError, match="cannot exceed one"):
        prior_entry(_ROOT_IDENTITY, ((_WATER_TARGET, 0.7), ({"target_kind": "action", "semantics": "x"}, 0.6)))

    with pytest.raises(PredictionLearningError, match="positive finite"):
        PredictionBehaviorPrior(strength=0.0)

    catalog = behavior_branch_catalog(_documents())
    assert set(catalog) == {"打开空调", "倒一杯水"}
    assert catalog["倒一杯水"]["target_refs"] == ["水杯"]
