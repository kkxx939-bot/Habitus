"""归约纯函数层：组链、封口、payload 物化的机械规则。"""

from __future__ import annotations

from datetime import timedelta, timezone
from typing import Any

import pytest

from behavior.reduction import (
    BehaviorReductionError,
    ReducibleJudgement,
    assemble_chains,
    gap_payload,
    occurrence_payload,
    parse_judgement_record,
    seal_horizon,
    sealed_chain_indexes,
    sealed_gaps,
)
from tests.unit.behavior.reduction_fixtures import (
    SUBJECT,
    at,
    judgement_record,
    observation,
    record_id,
)

OBS_A = observation(18, "人走到水池边打开水龙头")
OBS_B = observation(40, "人打肥皂搓手")
OBS_C = observation(62, "人冲水擦干")
OBS_INDEX = {item.observation_id: item for item in (OBS_A, OBS_B, OBS_C)}
SOURCE = ("e" * 64,)


def parse(record: dict) -> ReducibleJudgement:
    return parse_judgement_record(record)


def wash_head(**overrides: Any) -> dict:
    values: dict[str, Any] = dict(
        behavior="洗手",
        started_at=at(18),
        last_observed_at=at(40),
        evidence_ready_at=at(42),
        observation_ids=(OBS_A.observation_id, OBS_B.observation_id),
        source_refs=SOURCE,
        goal=None,
        summary="在水池边洗手",
        status="ongoing",
        status_basis="observation_lost",
        basis=(("打开水龙头打肥皂搓手", (OBS_A.observation_id, OBS_B.observation_id)),),
    )
    values.update(overrides)
    return judgement_record("head", **values)


def wash_tail(**overrides: Any) -> dict:
    values: dict[str, Any] = dict(
        behavior="继续洗手",
        started_at=at(62),
        last_observed_at=at(62),
        evidence_ready_at=at(64),
        observation_ids=(OBS_C.observation_id,),
        source_refs=SOURCE,
        goal="清洁双手",
        summary="冲水擦干结束",
        status="completed",
        status_basis="observed",
        basis=(("冲水擦干", (OBS_C.observation_id,)),),
        relations=(("continues", record_id("head")),),
    )
    values.update(overrides)
    return judgement_record("tail", **values)


def unreadable_record(seed: str = "blur", *, offset: int = 120, **overrides: Any) -> dict:
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


# --- 组链 -------------------------------------------------------------------------------


def test_continues_merges_into_one_chain_ordered_by_behavior_time() -> None:
    assembly = assemble_chains([parse(wash_tail()), parse(wash_head())])
    assert len(assembly.chains) == 1
    chain = assembly.chains[0]
    assert [item.judgement_id for item in chain.view] == [record_id("head"), record_id("tail")]
    assert chain.head.behavior == "洗手"
    assert chain.tail.status == "completed"


def test_supersedes_replaces_in_place_and_inherits_chain_structure() -> None:
    """jB 替换链头 jA 后，jC 的 continues 穿透到 jB——链不因修正断开，全史随链消费。"""

    corrected = judgement_record(
        "corrected",
        behavior="洗手",
        started_at=at(18),
        last_observed_at=at(40),
        evidence_ready_at=at(50),
        observation_ids=(OBS_A.observation_id, OBS_B.observation_id),
        source_refs=SOURCE,
        summary="看清了，是在洗手",
        status="ongoing",
        status_basis="observation_lost",
        relations=(("supersedes", record_id("head")),),
    )
    assembly = assemble_chains([parse(wash_head()), parse(corrected), parse(wash_tail())])
    assert len(assembly.chains) == 1
    chain = assembly.chains[0]
    assert [item.judgement_id for item in chain.view] == [
        record_id("corrected"),
        record_id("tail"),
    ]
    assert [item.judgement_id for item in chain.superseded] == [record_id("head")]
    assert {item.judgement_id for item in chain.consumed} == {
        record_id("head"),
        record_id("corrected"),
        record_id("tail"),
    }


def test_supersedes_claims_an_unreadable_stretch_instead_of_leaving_a_gap() -> None:
    blur = unreadable_record()
    claimer = wash_head(relations=(("supersedes", record_id("blur")),))
    assembly = assemble_chains([parse(blur), parse(claimer)])
    assert assembly.gaps == ()
    assert [item.judgement_id for item in assembly.absorbed_unreadable] == [record_id("blur")]
    # 被认领的空白随链消费——否则替换关系随链消费后，下一轮它会被错当成无主空白。
    assert record_id("blur") in {item.judgement_id for item in assembly.chains[0].consumed}


def test_unclaimed_unreadable_stays_a_gap_and_cannot_anchor_edges() -> None:
    blur = unreadable_record()
    follower = wash_head(relations=(("continues", record_id("blur")),))
    assembly = assemble_chains([parse(blur), parse(follower)])
    assert [item.judgement_id for item in assembly.gaps] == [record_id("blur")]
    assert any("unreadable" in note for note in assembly.dropped_edges)


def test_concurrent_declared_on_the_earlier_side_is_flipped_not_dropped() -> None:
    """并行是对称关系、融合允许任一边声明——声明在先开始那条上也不许丢，翻挂到晚链。"""

    later = judgement_record(
        "later",
        behavior="擦桌子",
        started_at=at(200),
        last_observed_at=at(230),
        evidence_ready_at=at(232),
        observation_ids=(OBS_C.observation_id,),
        source_refs=SOURCE,
        summary="边洗手边擦桌子",
    )
    early_declarer = wash_head(relations=(("concurrent_with", record_id("later")),))
    assembly = assemble_chains([parse(early_declarer), parse(later)])
    later_index = next(
        index
        for index, chain in enumerate(assembly.chains)
        if chain.head.judgement_id == record_id("later")
    )
    earlier_index = 1 - later_index
    assert assembly.cross_links_of(later_index) == (("concurrent_with", earlier_index),)
    assert assembly.cross_links_of(earlier_index) == ()
    assert not any("concurrent_with" in note for note in assembly.dropped_edges)


def test_results_from_cannot_point_at_a_later_chain() -> None:
    """results_from 有方向（结果指向原因），指向更晚链的边机械作废并留信号。"""

    later = judgement_record(
        "later",
        behavior="擦桌子",
        started_at=at(200),
        last_observed_at=at(230),
        evidence_ready_at=at(232),
        observation_ids=(OBS_C.observation_id,),
        source_refs=SOURCE,
        summary="顺手擦了桌子",
    )
    backwards = wash_head(relations=(("results_from", record_id("later")),))
    assembly = assemble_chains([parse(backwards), parse(later)])
    assert all(assembly.cross_links_of(index) == () for index in range(2))
    assert any("starts later" in note for note in assembly.dropped_edges)


def test_duplicate_identities_are_rejected() -> None:
    with pytest.raises(BehaviorReductionError, match="duplicate"):
        assemble_chains([parse(wash_head()), parse(wash_head())])


# --- 封口 -------------------------------------------------------------------------------


def test_seal_horizon_takes_the_earlier_of_wall_clock_and_queue_frontier() -> None:
    now = at(7200)
    lookback = 3_600.0
    unblocked = seal_horizon(now=now, frontier_cutoff=None, lookback_seconds=lookback)
    assert unblocked == (now - timedelta(seconds=3600)).astimezone(timezone.utc)
    # 队列积压的旧段还能引用旧判断——前沿更早时，封口视界必须跟着前沿走。
    frontier = at(300)
    blocked = seal_horizon(now=now, frontier_cutoff=frontier, lookback_seconds=lookback)
    assert blocked == (frontier - timedelta(seconds=3600)).astimezone(timezone.utc)


def test_a_chain_linking_an_unsealed_target_is_deferred_to_the_fixpoint() -> None:
    early_target = wash_head(evidence_ready_at=at(5000))  # 仍在窗口内，未封口
    late_source = judgement_record(
        "late",
        behavior="擦桌子",
        started_at=at(200),
        last_observed_at=at(230),
        evidence_ready_at=at(232),
        observation_ids=(OBS_C.observation_id,),
        source_refs=SOURCE,
        summary="顺手擦了桌子",
        relations=(("results_from", record_id("head")),),
    )
    # late 自身已出窗，但它引用的链没封——推迟，不带着悬空引用落盘。
    assembly = assemble_chains([parse(early_target), parse(late_source)])
    horizon = seal_horizon(now=at(7200), frontier_cutoff=None, lookback_seconds=3_600.0)
    assert sealed_chain_indexes(assembly, horizon) == ()

    fully_out = seal_horizon(now=at(12_000), frontier_cutoff=None, lookback_seconds=3_600.0)
    assert len(sealed_chain_indexes(assembly, fully_out)) == 2


def test_gaps_wait_out_the_window_before_materialising() -> None:
    blur = parse_judgement_record(unreadable_record())
    inside = seal_horizon(now=at(600), frontier_cutoff=None, lookback_seconds=3_600.0)
    assert sealed_gaps([blur], inside) == ()
    outside = seal_horizon(now=at(7200), frontier_cutoff=None, lookback_seconds=3_600.0)
    assert sealed_gaps([blur], outside) == (blur,)


# --- payload 物化 -----------------------------------------------------------------------


def build_chain():
    assembly = assemble_chains([parse(wash_head()), parse(wash_tail())])
    return assembly.chains[0]


def test_occurrence_payload_takes_identity_from_head_and_ending_from_tail() -> None:
    payload = occurrence_payload(
        build_chain(),
        name="洗手",
        original_name=None,
        kind_token="洗手",
        observations=OBS_INDEX,
    )
    assert payload["name"] == "洗手"  # 链头原话；链尾"继续洗手"不参与身份
    assert payload["started_at"] == at(18).isoformat(timespec="microseconds")
    assert payload["status"] == "completed" and payload["status_basis"] == "observed"
    assert payload["last_observed_at"] == at(62).isoformat(timespec="microseconds")
    assert payload["occurred_on"] == "2026-08-16"
    assert payload["goal"] == "清洁双手"  # 链内非空 goal 去重拼接；一致时就是那一个
    assert payload["summary"] == "在水池边洗手；冲水擦干结束"
    assert payload["judgement_ids"] == [record_id("head"), record_id("tail")]
    assert payload["reminded"] is False and payload["place"] is None


def test_onset_available_at_converts_evidence_instant_to_the_local_offset() -> None:
    payload = occurrence_payload(
        build_chain(), name="洗手", original_name=None, kind_token="洗手", observations=OBS_INDEX
    )
    # 链头 evidence_ready_at 存储为 UTC；进树换算成链头行为时刻的本地偏移，瞬时不变。
    assert payload["onset_available_at"] == at(42).isoformat(timespec="microseconds")
    assert payload["onset_available_at"].endswith("+08:00")


def test_basis_step_times_are_materialised_from_observations() -> None:
    payload = occurrence_payload(
        build_chain(), name="洗手", original_name=None, kind_token="洗手", observations=OBS_INDEX
    )
    first, second = payload["basis"]
    assert first["started_at"] == at(18).isoformat(timespec="microseconds")
    assert first["ended_at"] == at(40).isoformat(timespec="microseconds")
    assert first["available_at"] == at(42).isoformat(timespec="microseconds")
    assert second["started_at"] == second["ended_at"] == at(62).isoformat(timespec="microseconds")


def test_basis_step_missing_observation_fails_loudly() -> None:
    with pytest.raises(BehaviorReductionError, match="no longer stored"):
        occurrence_payload(
            build_chain(), name="洗手", original_name=None, kind_token="洗手", observations={}
        )


def test_subjects_merge_across_the_chain_in_first_seen_order() -> None:
    together = wash_tail(subjects=(SUBJECT, "家庭成员B"))
    assembly = assemble_chains([parse(wash_head()), parse(together)])
    payload = occurrence_payload(
        assembly.chains[0], name="洗手", original_name=None, kind_token="洗手", observations=OBS_INDEX
    )
    assert payload["subjects"] == [SUBJECT, "家庭成员B"]


def test_gap_payload_is_verbatim_and_zero_duration_is_legal() -> None:
    record = parse_judgement_record(unreadable_record())
    payload = gap_payload(record)
    assert payload["gap_kind"] == "没读懂"
    assert payload["started_at"] == payload["ended_at"]  # 单观测段：起止同刻，是事实不是矛盾
    assert payload["judgement_ids"] == [record_id("blur")]


def test_a_superseded_members_cross_edges_are_dropped_with_a_signal() -> None:
    """被替换判断的跨链边随判断作废——机械丢弃必须留信号，不许静默蒸发。"""

    other = judgement_record(
        "other",
        behavior="擦桌子",
        started_at=at(5),
        last_observed_at=at(10),
        evidence_ready_at=at(12),
        observation_ids=(OBS_C.observation_id,),
        source_refs=SOURCE,
        summary="先擦了桌子",
    )
    vague = wash_head(relations=(("concurrent_with", record_id("other")),))
    corrected = judgement_record(
        "corrected",
        behavior="洗手",
        started_at=at(18),
        last_observed_at=at(40),
        evidence_ready_at=at(60),
        observation_ids=(OBS_A.observation_id, OBS_B.observation_id),
        source_refs=SOURCE,
        summary="看清了，是在洗手",
        relations=(("supersedes", record_id("head")),),
    )
    assembly = assemble_chains([parse(other), parse(vague), parse(corrected)])
    assert any("superseded" in note for note in assembly.dropped_edges)
    # 被替换者声明的并行不过继给替换者
    wash_index = assembly.chain_of[record_id("corrected")]
    assert assembly.cross_links_of(wash_index) == ()


def test_a_supersedes_cycle_stays_visible_and_unconsumed() -> None:
    """supersedes 成环是产物内部矛盾：不吞掉、不消费，每轮报出直到有人处置。"""

    a = wash_head(relations=(("supersedes", record_id("b")),))
    b = judgement_record(
        "b",
        behavior="洗手",
        started_at=at(18),
        last_observed_at=at(40),
        evidence_ready_at=at(50),
        observation_ids=(OBS_A.observation_id,),
        source_refs=SOURCE,
        summary="另一条洗手",
        relations=(("supersedes", record_id("head")),),
    )
    assembly = assemble_chains([parse(a), parse(b)])
    assert assembly.chains == ()
    assert any("cyclic" in note for note in assembly.dropped_edges)


def test_zero_offset_records_are_rejected_at_parse_time() -> None:
    """零偏移与树同口径在门口现形，不许走到 stage 最深处才炸。"""

    from datetime import timezone as _tz

    bad = wash_head(
        started_at=at(18).astimezone(_tz.utc),
        last_observed_at=at(40).astimezone(_tz.utc),
    )
    with pytest.raises(BehaviorReductionError, match="non-zero local offset"):
        parse(bad)


def test_divergent_goals_are_all_preserved_in_order() -> None:
    """链内前后段声明了不同目标：原封不动按序保留（用户裁定），不替模型二选一、不置空丢失。"""

    tail = wash_tail(goal="准备吃饭")
    head = wash_head(goal="清洁双手", basis=(("打开水龙头搓手", (OBS_A.observation_id,)),))
    assembly = assemble_chains([parse(head), parse(tail)])
    payload = occurrence_payload(
        assembly.chains[0], name="洗手", original_name=None, kind_token="洗手", observations=OBS_INDEX
    )
    assert payload["goal"] == "清洁双手；准备吃饭"


def test_a_correction_inherits_the_chain_position_it_replaced() -> None:
    """新判断即新链头（规格字面）：修正者继承被替换者的位置，时间修正不会让行为改名。"""

    tail = wash_tail()  # 「继续洗手」19:31:02 起，continues head
    corrected = judgement_record(
        "corrected",
        behavior="洗手",
        started_at=at(30),  # 修正：其实 19:30:30 才开始——仍早于延续段，合法
        last_observed_at=at(40),
        evidence_ready_at=at(50),
        observation_ids=(OBS_A.observation_id, OBS_B.observation_id),
        source_refs=SOURCE,
        summary="修正了开始时间",
        status="ongoing",
        status_basis="observation_lost",
        relations=(("supersedes", record_id("head")),),
    )
    assembly = assemble_chains([parse(wash_head()), parse(corrected), parse(tail)])
    chain = assembly.chains[0]
    assert chain.head.judgement_id == record_id("corrected")  # 坐进 head 的位置
    assert chain.head.behavior == "洗手"
    assert chain.head.started_at == at(30)


def test_a_correction_crossing_its_own_continuation_is_quarantined() -> None:
    """矛盾链不进树：延续段不可能早于事件自己的开始——隔离、留信号、不消费。"""

    tail = wash_tail()  # 延续段 19:31:02 起
    corrected = judgement_record(
        "corrected",
        behavior="洗手",
        started_at=at(70),  # 修正把开始时间改到延续段（62s）之后——自相矛盾
        last_observed_at=at(80),
        evidence_ready_at=at(90),
        observation_ids=(OBS_A.observation_id,),
        source_refs=SOURCE,
        summary="时间倒挂的修正",
        status="ongoing",
        status_basis="observation_lost",
        relations=(("supersedes", record_id("head")),),
    )
    assembly = assemble_chains([parse(wash_head()), parse(corrected), parse(tail)])
    assert assembly.chains == ()
    assert any("quarantined" in note for note in assembly.dropped_edges)


def test_flipped_concurrent_seals_both_sides_together() -> None:
    """并行边由早链声明、翻挂到晚链后：早链不许独走——否则边的唯一载体随链消费，
    下一轮晚链落盘时边已无声蒸发（两路评审独立抓到的跨轮接缝）。"""

    late = judgement_record(
        "late",
        behavior="洗衣",
        started_at=at(120),
        last_observed_at=at(200),
        evidence_ready_at=at(5000),  # 晚链未出窗
        observation_ids=(OBS_C.observation_id,),
        source_refs=SOURCE,
        summary="洗衣还在继续",
        status="ongoing",
        status_basis="observation_lost",
    )
    early_declarer = wash_head(relations=(("concurrent_with", record_id("late")),))
    assembly = assemble_chains([parse(early_declarer), parse(late)])
    horizon = seal_horizon(now=at(7200), frontier_cutoff=None, lookback_seconds=3_600.0)
    assert sealed_chain_indexes(assembly, horizon) == ()  # 早链已出窗也必须等宿主

    both_out = seal_horizon(now=at(12_000), frontier_cutoff=None, lookback_seconds=3_600.0)
    assert len(sealed_chain_indexes(assembly, both_out)) == 2  # 同批一起走，边不丢


def test_a_correction_moving_a_mid_segment_before_the_head_is_quarantined() -> None:
    """镜像矛盾：修正把**中段**的开始时间改到早于链头——严格位置继承下同样是
    "延续早于事件开始"的产物矛盾，与改晚方向同一处置（隔离、留信号、不换头改名）。"""

    tail = wash_tail()  # 中段 19:31:02，continues head(19:30:18)
    corrected_mid = judgement_record(
        "corrected-mid",
        behavior="擦手",
        started_at=at(5),  # 修正把中段时间改到链头之前
        last_observed_at=at(10),
        evidence_ready_at=at(90),
        observation_ids=(OBS_C.observation_id,),
        source_refs=SOURCE,
        summary="其实更早就在擦手",
        status="completed",
        status_basis="observed",
        relations=(("supersedes", record_id("tail")),),
    )
    assembly = assemble_chains([parse(wash_head()), parse(tail), parse(corrected_mid)])
    assert assembly.chains == ()  # 不换头、不改名——整链隔离
    assert any("quarantined" in note for note in assembly.dropped_edges)
    assert record_id("corrected-mid") in assembly.quarantined_ids
