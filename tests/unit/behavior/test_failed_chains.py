"""失败的链（裁定 67）：判断进每晚重试的堆、顶上一条「没读懂」判断、撤原判断；中途崩溃重来是幂等的；当晚重试又失败的只撤下。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.reduction.chains import assemble_chains
from habitus.behavior.reduction.failed import FailedChains, unreadable_record
from habitus.behavior.reduction.ledger import BehaviorReductionLedger
from habitus.behavior.reduction.record import parse_judgement_record
from habitus.behavior.retry import RetryBacklogStore, RetryCause, RetryKind
from tests.unit.behavior.test_reduction_runner import OBS_A, OBS_C, at, judgement_record, record_id

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


def _setup(tmp_path):
    store = BehaviorJudgementStore(tmp_path)
    backlog = RetryBacklogStore(tmp_path, max_items=10)
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
    failed = FailedChains(
        store, ledger=BehaviorReductionLedger(tmp_path / "reduction"), clock=lambda: NOW, backlog=backlog
    )
    return store, backlog, failed, chain


def test_a_failed_chain_goes_to_the_backlog_and_is_replaced_by_an_unreadable_judgement(tmp_path) -> None:
    store, backlog, failed, chain = _setup(tmp_path)
    replacement, _admission = failed.fail_chain(chain, consumed=frozenset())
    assert replacement is not None and store.read(record_id("first")) is None
    unreadable = parse_judgement_record(store.read(replacement))
    assert unreadable.behavior is None and unreadable.observation_ids == (OBS_A.observation_id,)
    assert store.read(record_id("second")) is not None  # 别的判断不动
    (item,) = backlog.snapshot().items
    assert (item.kind, item.cause, item.day) == (RetryKind.CHAIN, RetryCause.DETERMINISTIC, at(10).date())
    assert [record["judgement_id"] for record in item.payload["records"]] == [record_id("first")]


def test_setting_aside_twice_after_a_crash_is_idempotent(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, backlog, failed, chain = _setup(tmp_path)
    original = store.discard

    def crash(_identity):
        raise OSError("crashed before the originals were removed")

    monkeypatch.setattr(store, "discard", crash)
    with pytest.raises(OSError):
        failed.fail_chain(chain, consumed=frozenset())
    monkeypatch.setattr(store, "discard", original)
    first, _admission = failed.fail_chain(chain, consumed=frozenset())
    assert first is not None and store.read(first) is not None
    assert store.read(record_id("first")) is None and backlog.count() == 1


def test_a_chain_failing_again_on_retry_is_only_removed(tmp_path) -> None:
    """它的「没读懂」已经归约过（消费账本里有）：不再造第二条、不再进堆（否则账本撞车、每晚无限推迟）。"""

    probe_store, _backlog, _failed, chain = _setup(tmp_path / "probe")
    records = [probe_store.read(record_id("first"))]
    done = frozenset({unreadable_record(chain.chain_digest, records)["judgement_id"]})
    store, backlog, failed, chain = _setup(tmp_path / "retry")
    assert failed.fail_chain(chain, consumed=done) == (None, None)
    assert store.read(record_id("first")) is None and backlog.count() == 0
    assert store.read(unreadable_record(chain.chain_digest, records)["judgement_id"]) is None


def test_dropping_a_record_removes_it(tmp_path) -> None:
    store, backlog, failed, _chain = _setup(tmp_path)
    raw = store.read(record_id("second"))
    assert raw is not None
    failed.drop_record(raw)
    assert store.read(record_id("second")) is None and backlog.count() == 0


def test_a_set_aside_chain_is_restored_judged_recovered_by_the_ledger_and_withdrawn_if_left_over(tmp_path) -> None:
    """裁定 68 第 5 条：载荷格式只在这里——放回、归约之后按消费账本认成没成；上一晚中途被杀剩下的、放回了还没归约的撤下。"""

    from habitus.behavior.reduction.ledger import BehaviorReductionEntry

    store, backlog, failed, chain = _setup(tmp_path)
    failed.fail_chain(chain, consumed=frozenset())
    (item,) = backlog.snapshot().items
    assert failed.restore(item) and store.read(record_id("first")) is not None
    assert not failed.recovered(item)
    failed.withdraw(item)
    assert store.read(record_id("first")) is None  # 没归约就撤下：不让它白天落进已经跑过批的那一天
    assert failed.restore(item)
    failed.ledger.append(
        BehaviorReductionEntry("c" * 64, "occurrence", "behavior://x", (record_id("first"),), NOW.isoformat(), "v")
    )
    assert failed.recovered(item)
    failed.withdraw(item)
    assert store.read(record_id("first")) is not None  # 已经归约消费了的不撤（释放归归约自己管）
    broken = type(item)(item.item_id, item.kind, item.cause, item.day, item.entered_at, {"records": []})
    assert not failed.restore(broken)
