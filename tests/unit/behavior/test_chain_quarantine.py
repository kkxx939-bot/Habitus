"""链的隔离：副本、顶替的「没读懂」判断、撤原判断；中途崩溃重来是幂等的。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.reduction.chains import assemble_chains
from habitus.behavior.reduction.quarantine import ChainQuarantine
from habitus.behavior.reduction.record import parse_judgement_record
from habitus.infrastructure.lifecycle import Quarantine
from tests.unit.behavior.test_reduction_runner import OBS_A, OBS_C, at, judgement_record, record_id


def _setup(tmp_path):
    store = BehaviorJudgementStore(tmp_path)
    quarantine = Quarantine(tmp_path / "lifecycle", owner="behavior-reduction-test", clock=lambda: datetime.now(UTC))
    records = [
        judgement_record(
            name,
            behavior="洗手",
            started_at=at(start),
            last_observed_at=at(start + 20),
            evidence_ready_at=at(start + 22),
            observation_ids=(observation,),
            source_refs=("s" * 64,),
            summary="洗手",
        )
        for name, start, observation in (("first", 10, OBS_A.observation_id), ("second", 40, OBS_C.observation_id))
    ]
    for record in records:
        store.put_payload(record)
    chain = assemble_chains(tuple(parse_judgement_record(store.read(r["judgement_id"])) for r in records[:1])).chains[0]
    return store, ChainQuarantine(store, quarantine), chain


def test_isolating_a_chain_replaces_it_with_an_unreadable_judgement(tmp_path) -> None:
    store, quarantine, chain = _setup(tmp_path)
    replacement = quarantine.isolate_chain(chain, "behavior name is not addressable")
    assert store.read(record_id("first")) is None
    unreadable = parse_judgement_record(store.read(replacement))
    assert unreadable.behavior is None
    assert unreadable.observation_ids == (OBS_A.observation_id,)
    assert unreadable.started_at == at(10)
    assert store.read(record_id("second")) is not None  # 别的判断不动


def test_isolating_twice_after_a_crash_is_idempotent(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, quarantine, chain = _setup(tmp_path)
    original = store.discard

    def crash(_identity):
        raise OSError("crashed before the originals were removed")

    monkeypatch.setattr(store, "discard", crash)
    with pytest.raises(OSError):
        quarantine.isolate_chain(chain, "x")
    monkeypatch.setattr(store, "discard", original)
    first = quarantine.isolate_chain(chain, "x")
    assert store.read(record_id("first")) is None
    assert len(quarantine.quarantine.items()) == 1
    assert store.read(first) is not None


def test_isolating_a_record_removes_it_and_keeps_a_copy(tmp_path) -> None:
    store, quarantine, _chain = _setup(tmp_path)
    raw = store.read(record_id("second"))
    assert raw is not None
    quarantine.isolate_record(raw, "judgement record is not parseable")
    assert store.read(record_id("second")) is None
    (item,) = quarantine.quarantine.items()
    assert item.meta == {"kind": "judgement"}
