"""归约纯函数层：待归约判断的分拣、封口、payload 物化的机械规则。"""

from __future__ import annotations

from datetime import UTC, timedelta
from typing import Any

import pytest

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.reduction import (
    BehaviorReductionEntry,
    BehaviorReductionError,
    BehaviorReductionLedger,
    ReducibleJudgement,
    gap_payload,
    occurrence_payload,
    parse_judgement_record,
    pending_judgements,
    seal_horizon,
    sealed_gaps,
    sealed_indexes,
)
from habitus.foundation.integrity import canonical_digest
from tests.unit.behavior.reduction_fixtures import (
    SUBJECT,
    at,
    judgement_record,
    observation,
    record_id,
)
from tests.unit.kind_ids import kind_id

OBS_A = observation(18, "人走到水池边打开水龙头")
OBS_B = observation(40, "人打肥皂搓手")
OBS_C = observation(62, "人冲水擦干")
OBS_INDEX = {item.observation_id: item for item in (OBS_A, OBS_B, OBS_C)}
SOURCE = ("e" * 64,)


def parse(record: dict) -> ReducibleJudgement:
    return parse_judgement_record(record)


def wash(seed: str = "wash", **overrides: Any) -> dict:
    values: dict[str, Any] = dict(
        behavior="洗手",
        started_at=at(18),
        last_observed_at=at(62),
        evidence_ready_at=at(64),
        observation_ids=(OBS_A.observation_id, OBS_B.observation_id, OBS_C.observation_id),
        source_refs=SOURCE,
        goal="清洁双手",
        summary="在水池边洗手，冲水擦干结束",
        status="completed",
        status_basis="observed",
        basis=(
            ("打开水龙头打肥皂搓手", (OBS_A.observation_id, OBS_B.observation_id)),
            ("冲水擦干", (OBS_C.observation_id,)),
        ),
    )
    values.update(overrides)
    return judgement_record(seed, **values)


def error_record(seed: str = "blur", *, offset: int = 120, **overrides: Any) -> dict:
    values: dict[str, Any] = dict(
        behavior=None,
        started_at=at(offset),
        last_observed_at=at(offset),
        evidence_ready_at=at(offset + 2),
        observation_ids=(observation(offset, "画面模糊").observation_id,),
        source_refs=SOURCE,
    )
    values.update(overrides)
    return judgement_record(seed, **values)


# --- 分拣 -------------------------------------------------------------------------------


def test_pending_splits_readable_and_empty_judgements_and_skips_consumed_ones(tmp_path) -> None:
    """可读判断按行为时刻排（每条一条 occurrence），空判断按成立时刻排（每条一段空白）；账本里有的不再归约。"""

    store = BehaviorJudgementStore(tmp_path / "judgements")
    ledger = BehaviorReductionLedger(tmp_path / "reduction")
    later = wash("later", started_at=at(300), last_observed_at=at(320), evidence_ready_at=at(30))
    earlier = wash("earlier", evidence_ready_at=at(400))
    consumed = wash("consumed", started_at=at(500), last_observed_at=at(520), evidence_ready_at=at(522))
    for record in (later, earlier, consumed, error_record("blur-b", offset=200), error_record("blur-a")):
        store.put_payload(record)
    ledger.append(
        BehaviorReductionEntry(
            chain_digest=canonical_digest({"judgement_ids": [record_id("consumed")]}),
            kind="occurrence",
            uri="behavior://occurrences/x",
            judgement_ids=(record_id("consumed"),),
            staged_at=at(600).isoformat(),
            reduction_version="behavior_reduction_v1",
        )
    )

    pending = pending_judgements(store, ledger)

    assert [item.judgement_id for item in pending.occurrences] == [record_id("earlier"), record_id("later")]
    assert [item.judgement_id for item in pending.gaps] == [record_id("blur-a"), record_id("blur-b")]
    assert pending.quarantined == ()


# --- 封口 -------------------------------------------------------------------------------


def test_seal_horizon_takes_the_earlier_of_wall_clock_and_queue_frontier() -> None:
    now = at(7200)
    lookback = 3_600.0
    unblocked = seal_horizon(now=now, frontier_cutoff=None, lookback_seconds=lookback)
    assert unblocked == (now - timedelta(seconds=3600)).astimezone(UTC)
    # 还没写成判断的观测更早时，封口视界必须跟着它走。
    frontier = at(300)
    blocked = seal_horizon(now=now, frontier_cutoff=frontier, lookback_seconds=lookback)
    assert blocked == (frontier - timedelta(seconds=3600)).astimezone(UTC)


def test_a_judgement_seals_once_its_evidence_leaves_the_window() -> None:
    early = parse(wash("early"))
    late = parse(wash("late", started_at=at(4000), last_observed_at=at(4100), evidence_ready_at=at(5000)))
    horizon = seal_horizon(now=at(7200), frontier_cutoff=None, lookback_seconds=3_600.0)
    assert sealed_indexes((early, late), horizon) == (0,)
    fully_out = seal_horizon(now=at(12_000), frontier_cutoff=None, lookback_seconds=3_600.0)
    assert sealed_indexes((early, late), fully_out) == (0, 1)


def test_gaps_wait_out_the_window_before_materialising() -> None:
    blur = parse_judgement_record(error_record())
    inside = seal_horizon(now=at(600), frontier_cutoff=None, lookback_seconds=3_600.0)
    assert sealed_gaps([blur], inside) == ()
    outside = seal_horizon(now=at(7200), frontier_cutoff=None, lookback_seconds=3_600.0)
    assert sealed_gaps([blur], outside) == (blur,)


# --- payload 物化 -----------------------------------------------------------------------


def test_occurrence_payload_is_the_judgement_verbatim() -> None:
    record = parse(wash())
    payload = occurrence_payload(
        record, name="洗手", original_name=None, kind_token=kind_id("洗手"), observations=OBS_INDEX
    )
    assert payload["name"] == "洗手"
    assert payload["started_at"] == at(18).isoformat(timespec="microseconds")
    assert payload["status"] == "completed" and payload["status_basis"] == "observed"
    assert payload["last_observed_at"] == at(62).isoformat(timespec="microseconds")
    assert payload["occurred_on"] == "2026-08-16"
    assert payload["goal"] == "清洁双手"
    assert payload["summary"] == "在水池边洗手，冲水擦干结束"
    # 溯源只留节点身份（原料发布即释放）：所消费判断身份集合的摘要
    assert payload["chain_digest"] == record.chain_digest == canonical_digest({"judgement_ids": [record_id("wash")]})
    assert payload["reminded"] is False and payload["place"] is None


def test_onset_available_at_converts_evidence_instant_to_the_local_offset() -> None:
    payload = occurrence_payload(
        parse(wash()), name="洗手", original_name=None, kind_token=kind_id("洗手"), observations=OBS_INDEX
    )
    # evidence_ready_at 存储为 UTC；进树换算成行为开始时刻的本地偏移，瞬时不变。
    assert payload["onset_available_at"] == at(64).isoformat(timespec="microseconds")
    assert payload["onset_available_at"].endswith("+08:00")


def test_basis_step_times_are_materialised_from_observations() -> None:
    payload = occurrence_payload(
        parse(wash()), name="洗手", original_name=None, kind_token=kind_id("洗手"), observations=OBS_INDEX
    )
    first, second = payload["basis"]
    assert first["started_at"] == at(18).isoformat(timespec="microseconds")
    assert first["ended_at"] == at(40).isoformat(timespec="microseconds")
    assert first["available_at"] == at(42).isoformat(timespec="microseconds")
    assert second["started_at"] == second["ended_at"] == at(62).isoformat(timespec="microseconds")


def test_basis_step_missing_observation_fails_loudly() -> None:
    with pytest.raises(BehaviorReductionError, match="no longer stored"):
        occurrence_payload(parse(wash()), name="洗手", original_name=None, kind_token=kind_id("洗手"), observations={})


def test_subjects_keep_first_seen_order_without_repeats() -> None:
    together = parse(wash(subjects=(SUBJECT, "家庭成员B", SUBJECT)))
    payload = occurrence_payload(
        together, name="洗手", original_name=None, kind_token=kind_id("洗手"), observations=OBS_INDEX
    )
    assert payload["subjects"] == [SUBJECT, "家庭成员B"]


def test_gap_payload_is_verbatim_and_zero_duration_is_legal() -> None:
    record = parse_judgement_record(error_record())
    payload = gap_payload(record, chain_digest="a" * 64)
    assert payload["gap_kind"] == "没读懂"
    assert payload["started_at"] == payload["ended_at"]  # 单观测段：起止同刻，是事实不是矛盾
    assert payload["chain_digest"] == "a" * 64


def test_zero_offset_records_are_rejected_at_parse_time() -> None:
    """零偏移与树同口径在门口现形，不许走到 stage 最深处才炸。"""

    bad = wash(started_at=at(18).astimezone(UTC), last_observed_at=at(40).astimezone(UTC))
    with pytest.raises(BehaviorReductionError, match="non-zero local offset"):
        parse(bad)
