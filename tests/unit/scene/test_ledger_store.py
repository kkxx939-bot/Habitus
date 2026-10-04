"""④ 账本的模型与存储：承诺/结算/提醒的自洽约束、机会快照、JSON 回读、add-only、按假设分目录。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from habitus.scene.hypotheses import Aspect
from habitus.scene.ledger import (
    Claim,
    ClaimRef,
    Coverage,
    Intervention,
    InterventionResponse,
    LedgerError,
    LedgerSchemaError,
    LedgerStore,
    LedgerStoreError,
    Opportunity,
    OpportunityPass,
    OpportunitySnapshot,
    Outcome,
    Response,
    Settlement,
    WindowSpan,
)
from habitus.scene.ledger.codec import (
    LedgerCodecError,
    decode_claim,
    decode_settlement,
    encode_claim,
    encode_settlement,
)
from habitus.scene.occurrences import ConceptHit
from tests.unit.scene.fixtures import DAY1, DAY2, DAY3, at
from tests.unit.scene.ledger_fixtures import (
    BOOKING_TO_BALL,
    LATE_TO_BREAKFAST,
    NOW,
    daily_snapshot,
    opportunity,
    uri_for,
)

TRIGGER = uri_for(DAY1, "就寝", 2, 10)
SNAPSHOT = daily_snapshot("早餐", at(DAY1, 2, 10), 3)


def claim(**overrides) -> Claim:
    values = dict(
        hypothesis_identity=LATE_TO_BREAKFAST.identity,
        hypothesis_fingerprint=LATE_TO_BREAKFAST.fingerprint,
        aspect=Aspect.PROBABILITY,
        trigger_uri=TRIGGER,
        antecedent_hits=(ConceptHit("晚睡", "轻"),),
        antecedent_uris=(TRIGGER,),
        situation_snapshot=("周末",),
        control=SNAPSHOT,
        created_at=NOW,
    )
    values.update(overrides)
    return Claim(**values)


# ── 模型 ──────────────────────────────────────────────────────────────────────


def test_opportunity_snapshots_are_ascending_and_index_arrivals() -> None:
    """七h：02:10 的锚，早餐的第 1 个窗口是当天 06:15–09:15 那个峰；窗口之间的到来不归任何窗口（2026-10-01 窗口账）。"""

    first = SNAPSHOT.at(1)
    assert first is not None and first.at == at(DAY1, 7, 45) and first.span == WindowSpan(at(DAY1, 6, 15), at(DAY1, 9, 15)) and first.probability == 0.88
    assert SNAPSHOT.at(2).at == at(DAY2, 7, 45) and SNAPSHOT.at(4) is None  # type: ignore[union-attr]
    assert SNAPSHOT.index_of(at(DAY1, 5, 0)) is None and SNAPSHOT.index_of(at(DAY1, 11, 0)) is None  # 窗口之外
    assert SNAPSHOT.index_of(at(DAY1, 7, 30)) == 1 and SNAPSHOT.index_of(at(DAY2, 8, 0)) == 2
    assert SNAPSHOT.expected_count(1, 2) == pytest.approx(1.76) and SNAPSHOT.expected_count(2, 3) is None
    with pytest.raises(LedgerError, match="do not overlap"):
        OpportunitySnapshot("gen-1", (opportunity(DAY1, 7, 45, 0.88, 90), opportunity(DAY1, 8, 0, 0.5, 90)))
    with pytest.raises(LedgerError, match="at least one"):
        OpportunitySnapshot("gen-1", ())
    with pytest.raises(LedgerError, match="\\[0, 1\\]"):
        opportunity(DAY1, 7, 45, 1.5, 90)
    assert opportunity(DAY1, 7, 45, 0.0, 90).probability == 0.0  # 窗口里曲线质量为零是合法的对照；None 才是"没曲线"
    with pytest.raises(LedgerError, match="inside its span"):
        Opportunity(at(DAY1, 12, 0), WindowSpan(at(DAY1, 6, 0), at(DAY1, 9, 0)), 0.5)
    with pytest.raises(LedgerError, match="counts from 1"):
        SNAPSHOT.at(0)


def test_a_claim_is_self_consistent() -> None:
    item = claim()
    assert item.anchor == at(DAY1, 2, 10) and item.ref == ClaimRef(LATE_TO_BREAKFAST.identity, DAY1, item.trigger_address.identity_name)
    assert item.consequent == "早餐"
    # 锚正在其中的峰算第 1 次机会（晚起正是晚在这个峰上）；**已经结束**的峰才不合法。
    inside = OpportunitySnapshot("gen-1", (opportunity(DAY1, 1, 30, 0.5, 60),))  # 00:30–02:30 含锚 02:10
    assert claim(control=inside).control is inside
    with pytest.raises(LedgerError, match="has not ended by the anchor"):
        claim(control=OpportunitySnapshot("gen-1", (opportunity(DAY1, 0, 30, 0.5, 30),)))  # 00:00–01:00，锚之前就结束了
    with pytest.raises(LedgerError, match="include the trigger"):
        claim(antecedent_uris=(uri_for(DAY1, "别的", 1, 0),))
    with pytest.raises(LedgerError, match="occurrence document"):
        claim(trigger_uri="behavior://gaps/2026/08/15/没读懂--20260815T020000000000+0800.md")
    with pytest.raises(LedgerError, match="disagrees with its hypothesis identity"):
        claim(aspect=Aspect.COUNT)
    with pytest.raises(LedgerError, match="repeats a concept"):
        claim(situation_snapshot=("周末", "周末"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(outcome=Outcome.OCCURRED), "names when and by which"),
        (dict(outcome=Outcome.CENSORED, observed_at=at(DAY1, 7, 0), fulfilling_uri=uri_for(DAY1, "早餐", 7, 0)), "saw no consequent"),
        (dict(outcome=Outcome.CENSORED, count=1), "counted settlements only"),
        (dict(outcome=Outcome.COUNTED, count=1, reason="x"), "censored settlements only"),
        (dict(outcome=Outcome.OBSERVED, observed_at=at(DAY1, 7, 0), fulfilling_uri=uri_for(DAY1, "早餐", 7, 0), latency_hours=3.0), "occurred settlements only"),
        (dict(outcome=Outcome.OCCURRED, observed_at=at(DAY1, 7, 0), fulfilling_uri=uri_for(DAY1, "早餐", 7, 0)), "occurred settlements only"),  # 缺 latency
        (dict(outcome=Outcome.COUNTED, count=1, opportunity_index=1), "opportunity_index belongs"),
        (dict(outcome=Outcome.CENSORED, passes=(OpportunityPass(at(DAY1, 7, 45), True), OpportunityPass(at(DAY1, 7, 45), False))), "ascending"),
    ],
)
def test_a_settlement_matches_its_outcome(kwargs, message) -> None:
    with pytest.raises(LedgerError, match=message):
        Settlement(ref=claim().ref, settled_at=NOW, **kwargs)


def test_settlement_shapes_that_are_allowed() -> None:
    ref = claim().ref
    occurred = Settlement(ref, Outcome.OCCURRED, NOW, observed_at=at(DAY2, 7, 50), fulfilling_uri=uri_for(DAY2, "早餐", 7, 50), latency_hours=29.7, opportunity_index=2, passes=(OpportunityPass(at(DAY1, 7, 45), True),))
    assert occurred.observed_passes == 1 and not occurred.is_right_censored
    censored = Settlement(ref, Outcome.CENSORED, NOW, passes=(OpportunityPass(at(DAY1, 7, 45), True), OpportunityPass(at(DAY2, 7, 45), False)))
    assert censored.observed_passes == 1 and censored.is_right_censored
    Settlement(ref, Outcome.CENSORED, NOW, reason="生命周期关闭")
    Settlement(ref, Outcome.OBSERVED, NOW, observed_at=at(DAY1, 9, 40), fulfilling_uri=uri_for(DAY1, "起床", 9, 40), opportunity_index=1)
    Settlement(ref, Outcome.ABSENT, NOW, opportunity_index=1)
    Settlement(ref, Outcome.COUNTED, NOW, count=0)
    with pytest.raises(LedgerError):
        Coverage(1.2)


# ── 编解码 ────────────────────────────────────────────────────────────────────


def test_claim_json_round_trips_and_keeps_the_local_offset() -> None:
    item = claim()
    raw = encode_claim(item)
    assert b'"at":"2026-08-15T07:45:00.000000+08:00"' in raw  # 机会是本地时刻，不折成 UTC
    assert decode_claim(raw, expected=item.ref) == item
    with pytest.raises(LedgerCodecError, match="canonical form"):
        decode_claim(raw.replace(b'"probability":0.88', b'"probability":0.880'), expected=item.ref)
    with pytest.raises(LedgerCodecError, match="identity of its address"):
        decode_claim(raw, expected=ClaimRef(BOOKING_TO_BALL.identity, DAY1, item.ref.leaf))
    settlement = Settlement(item.ref, Outcome.CENSORED, NOW, passes=(OpportunityPass(at(DAY1, 7, 45), True),), reason="x")
    assert decode_settlement(encode_settlement(settlement), expected=item.ref) == settlement


# ── 存储 ──────────────────────────────────────────────────────────────────────


def test_store_files_by_hypothesis_and_records_are_add_only(tmp_path) -> None:
    store = LedgerStore(tmp_path / "scene")
    item = claim()
    path = store.write_claim(item)
    assert path == tmp_path / "scene" / "ledger" / "早餐" / "晚睡--probability--1" / "claims" / "2026" / "08" / "15" / f"{item.ref.leaf}.json"
    store.write_claim(item)  # 同内容重放是空操作
    with pytest.raises(LedgerStoreError, match="add-only"):
        store.write_claim(claim(control=None))  # 同地址不同内容：机会快照事后不改
    assert store.claim_exists(item.ref) and store.read_claim(item.ref) == item
    assert store.claims_for(LATE_TO_BREAKFAST.identity) == (item,) and store.open_claims(LATE_TO_BREAKFAST.identity) == (item,)
    assert store.hypotheses() == (LATE_TO_BREAKFAST.identity,)

    settlement = Settlement(item.ref, Outcome.OCCURRED, NOW, observed_at=at(DAY1, 7, 0), fulfilling_uri=uri_for(DAY1, "早餐", 7, 0), latency_hours=4.83, opportunity_index=1)
    store.write_settlement(settlement)
    assert store.read_settlement(item.ref) == settlement and store.open_claims(LATE_TO_BREAKFAST.identity) == ()
    assert store.settlements_for(LATE_TO_BREAKFAST.identity) == (settlement,)
    with pytest.raises(LedgerStoreError, match="add-only"):
        store.write_settlement(Settlement(item.ref, Outcome.CENSORED, NOW))
    orphan = claim(trigger_uri=uri_for(DAY2, "就寝", 1, 0), antecedent_uris=(uri_for(DAY2, "就寝", 1, 0),), control=daily_snapshot("早餐", at(DAY2, 1, 0), 3))
    with pytest.raises(LedgerStoreError, match="claim on disk first"):
        store.write_settlement(Settlement(orphan.ref, Outcome.CENSORED, NOW))


def test_claims_are_listed_in_anchor_order_across_days(tmp_path) -> None:
    store = LedgerStore(tmp_path / "scene")
    later = claim(trigger_uri=uri_for(DAY2, "就寝", 1, 0), antecedent_uris=(uri_for(DAY2, "就寝", 1, 0),), control=daily_snapshot("早餐", at(DAY2, 1, 0), 3))
    store.write_claim(later)
    store.write_claim(claim())
    assert [item.anchor for item in store.claims_for(LATE_TO_BREAKFAST.identity)] == [at(DAY1, 2, 10), at(DAY2, 1, 0)]


def test_interventions_point_at_a_claim_and_responses_are_recorded_separately(tmp_path) -> None:
    """提醒是承诺之后的事：承诺不动，提醒另记；回应又是提醒之后的事：提醒不动，回应另记、读时 join。

    不这样分，add-only 下先写 ``response=None`` 再写回应就是同地址不同内容——"发了但他没理"进不了盘。
    """

    store = LedgerStore(tmp_path / "scene")
    item = claim()
    reminder = Intervention(item.ref, reminded_at=at(DAY1, 7, 0), recorded_at=NOW)
    with pytest.raises(LedgerStoreError, match="claim on disk first"):
        store.record_intervention(reminder)
    store.write_claim(item)
    store.record_intervention(reminder)
    store.record_intervention(Intervention(item.ref, reminded_at=at(DAY1, 7, 30), recorded_at=NOW, note="再提一次"))
    found = store.interventions_for(item.ref)
    assert [i.reminded_at for i in found] == [at(DAY1, 7, 0), at(DAY1, 7, 30)] and found[1].note == "再提一次"
    assert store.responses_for(item.ref) == ()  # 发了、还没理
    answer = InterventionResponse(item.ref, reminded_at=at(DAY1, 7, 30), responded_at=at(DAY1, 7, 32), response=Response.REJECTED, note="说不饿")
    with pytest.raises(LedgerStoreError, match="intervention on disk first"):
        store.record_response(InterventionResponse(item.ref, reminded_at=at(DAY1, 9, 0), responded_at=at(DAY1, 9, 1), response=Response.ACCEPTED))
    store.record_response(answer)
    assert store.responses_for(item.ref) == (answer,) and store.read_claim(item.ref) == item  # 承诺一个字没动
    with pytest.raises(LedgerError, match="after its reminder"):
        InterventionResponse(item.ref, reminded_at=at(DAY1, 7, 30), responded_at=at(DAY1, 7, 0), response=Response.ACCEPTED)


def test_voiding_a_day_withdraws_claims_and_settlements_but_keeps_the_intervention_data(tmp_path) -> None:
    """改概念定义重映射某天之后，那天开的承诺按旧命中记的，要撤了重开；提醒与回应是唯一的干预数据，不撤。"""

    store = LedgerStore(tmp_path / "scene")
    item = claim()
    other = claim(trigger_uri=uri_for(DAY1, "又睡", 3, 0), antecedent_uris=(uri_for(DAY1, "又睡", 3, 0),), control=daily_snapshot("早餐", at(DAY1, 3, 0), 3))
    later = claim(trigger_uri=uri_for(DAY2, "就寝", 1, 0), antecedent_uris=(uri_for(DAY2, "就寝", 1, 0),), control=daily_snapshot("早餐", at(DAY2, 1, 0), 3))
    for each in (item, other, later):
        store.write_claim(each)
    store.write_settlement(Settlement(item.ref, Outcome.CENSORED, NOW, reason="x"))
    store.record_intervention(Intervention(item.ref, reminded_at=at(DAY1, 7, 0), recorded_at=NOW))
    day_dir = store.claim_path(item.ref).parent
    (day_dir / ".DS_Store").write_bytes(b"\x00")
    (day_dir / f".{item.ref.leaf}.json.{'0' * 32}.tmp").write_bytes(b"half")

    assert store.retain_only(LATE_TO_BREAKFAST.identity, DAY1, frozenset({other.ref.leaf})) == (item.ref.leaf,)
    assert store.claims_for(LATE_TO_BREAKFAST.identity) == (other, later) and store.read_settlement(item.ref) is None
    assert len(store.interventions_for(item.ref)) == 1  # 干预数据留着
    assert not (day_dir / ".DS_Store").exists() and not any(day_dir.glob("*.tmp"))  # 噪音顺手清掉
    assert store.void_day(LATE_TO_BREAKFAST.identity, DAY1) == (other.ref.leaf,) and store.claims_for(LATE_TO_BREAKFAST.identity) == (later,)
    assert store.void_day(LATE_TO_BREAKFAST.identity, DAY1) == ()
    # 重开同一条触发的承诺：提醒自然接上。
    store.write_claim(item)
    assert len(store.interventions_for(item.ref)) == 1


def test_voiding_settlements_that_read_a_remapped_day_leaves_the_claims_standing(tmp_path) -> None:
    """评审 B3：``void_day`` 撤的是**那天开的**账；靠那天的记录结出来、但开在别的天的结算它撤不到。

    重映射 DAY2 之后，DAY1 那条承诺的结算是脏的（DAY2 可能多出/少了一条早餐），而结算 add-only 改不了。
    ``void_settlements_touching`` 只撤结算、**承诺不动**，下一夜按新记录重结。
    """

    store = LedgerStore(tmp_path / "scene")
    # DAY1 开的承诺，快照铺三天（DAY1–DAY3）：它读过 DAY2 的记录。
    spanning = claim(control=daily_snapshot("早餐", at(DAY1, 2, 10), 3))
    # DAY3 开的承诺：锚在 DAY2 之后，没读过那天。
    later = claim(trigger_uri=uri_for(DAY3, "就寝", 1, 0), antecedent_uris=(uri_for(DAY3, "就寝", 1, 0),), control=daily_snapshot("早餐", at(DAY3, 1, 0), 3))
    for each in (spanning, later):
        store.write_claim(each)
        store.write_settlement(Settlement(each.ref, Outcome.CENSORED, NOW, reason="等过了机会没来"))

    voided = store.void_settlements_touching(LATE_TO_BREAKFAST.identity, DAY2)
    assert voided == (spanning.ref,)
    assert store.read_settlement(spanning.ref) is None and store.read_settlement(later.ref) is not None
    # 承诺一条没动：两条都还在盘上，被撤了结算的那条重新变成"开着"，下一夜重结。
    assert store.claims_for(LATE_TO_BREAKFAST.identity) == (spanning, later)
    assert store.open_claims(LATE_TO_BREAKFAST.identity) == (spanning,)
    assert store.void_settlements_touching(LATE_TO_BREAKFAST.identity, DAY2) == ()
    # 没有快照的承诺（要不到对照）：锚之后的哪一天都算它读过，宁可多重结一次。
    blind = claim(trigger_uri=uri_for(DAY1, "又睡", 3, 0), antecedent_uris=(uri_for(DAY1, "又睡", 3, 0),), control=None)
    store.write_claim(blind)
    store.write_settlement(Settlement(blind.ref, Outcome.CENSORED, NOW, reason="生命周期关掉"))
    # DAY3 的账：没快照那条（锚在 DAY1）与 DAY3 开的那条（快照从 DAY3 铺起）都读过它，按 (天, 叶名) 升序返回。
    assert store.void_settlements_touching(LATE_TO_BREAKFAST.identity, DAY3) == (blind.ref, later.ref)


def test_noise_is_cleared_on_the_write_path_and_path_problems_are_not_reported_as_conflicts(tmp_path) -> None:
    """噪音（``.DS_Store`` / 崩溃遗留的 ``.tmp``）读时跳过、**写时**清：读路径清会删掉并发写者正在 link 的临时文件，
    而未结算的承诺是给预测层在线读的（直接读账本）。"""

    store = LedgerStore(tmp_path / "scene")
    item = claim()
    store.write_claim(item)
    day_dir = store.claim_path(item.ref).parent
    (day_dir / ".DS_Store").write_bytes(b"\x00")
    assert store.claims_for(LATE_TO_BREAKFAST.identity) == (item,) and (day_dir / ".DS_Store").exists()  # 读不删
    store.write_claim(item)  # 同内容重放：写路径清噪音
    assert not (day_dir / ".DS_Store").exists()
    # 父目录被一个同名文件占住：报"目录建不了"，不报"add-only 冲突"。
    other = claim(trigger_uri=uri_for(DAY2, "就寝", 1, 0), antecedent_uris=(uri_for(DAY2, "就寝", 1, 0),), control=daily_snapshot("早餐", at(DAY2, 1, 0), 3))
    blocker = store.claim_path(other.ref).parent
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(LedgerStoreError) as caught:
        store.write_claim(other)
    assert "add-only" not in str(caught.value)


def test_a_settlement_must_be_one_its_claims_aspect_allows(tmp_path) -> None:
    """``Settlement`` 自己不知道 aspect，所以只有存储层拦得住：一条 COUNTED 写给概率承诺会被收下之后在读侧静默蒸发。"""

    store = LedgerStore(tmp_path / "scene")
    item = claim()
    store.write_claim(item)
    with pytest.raises(LedgerStoreError, match="cannot settle as counted"):
        store.write_settlement(Settlement(item.ref, Outcome.COUNTED, NOW, count=3))
    with pytest.raises(LedgerStoreError, match="cannot settle as observed"):
        store.write_settlement(Settlement(item.ref, Outcome.OBSERVED, NOW, observed_at=NOW, fulfilling_uri=item.trigger_uri))
    store.write_settlement(Settlement(item.ref, Outcome.CENSORED, NOW, reason="关掉"))


def test_an_old_schema_record_is_not_reported_as_corrupt(tmp_path) -> None:
    """旧口径 ≠ 损坏（评审 B9）：拒绝读是对的（没有版本迁移，定义变了整个重算），但**原因要分得开**。

    混成一句"corrupt"的后果：一条旧文件让这条假设的 claims_for / open_claims / settle_due_all 全抛错、
    夜批停在第一条，而运维照着"损坏"去查磁盘。
    """

    import json

    store = LedgerStore(tmp_path / "scene")
    item = claim()
    path = store.write_claim(item)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "scene_ledger_v1"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(LedgerSchemaError, match="has to be recomputed"):
        store.read_claim(item.ref)
    with pytest.raises(LedgerSchemaError):
        store.claims_for(item.ref.hypothesis_identity)


def test_store_refuses_non_canonical_directories_and_corrupt_records(tmp_path) -> None:
    store = LedgerStore(tmp_path / "scene")
    item = claim()
    path = store.write_claim(item)
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(LedgerStoreError, match="corrupt"):
        store.read_claim(item.ref)
    (tmp_path / "scene" / "ledger" / "Breakfast").mkdir()
    with pytest.raises(LedgerStoreError, match="canonical identity"):
        store.hypotheses()
    with pytest.raises(LedgerStoreError, match="'<consequent>/<leaf>'"):
        store.claims_for("晚睡--probability")
    assert store.claims_for(BOOKING_TO_BALL.identity) == ()
    assert timedelta(0) == timedelta()  # 保持导入有用


def test_voiding_deletes_settlements_first_so_a_crash_cannot_leave_a_settlement_covering_a_reopened_claim(tmp_path) -> None:
    """撤账撤到一半崩了：剩下"有承诺没结算"（还开着，下一夜自愈），而不是孤儿结算。

    反过来的话，重映射后重开的同触发承诺一写进去就带着一条按**旧命中**结出来的结算：``open_claims`` 按
    ``settlement_exists`` 判开闭 → 它永不进结算，``load_account`` 还把旧结算配给新承诺（评审 B1 实测）。
    """

    store = LedgerStore(tmp_path / "scene")
    item = claim()
    store.write_claim(item)
    store.write_settlement(Settlement(item.ref, Outcome.CENSORED, NOW, reason="按旧命中结的"))
    # 模拟崩在两段之间：``retain_only`` 先删结算，所以"删了结算、还没删承诺"是崩溃点的真实状态。
    store._discard(store.settlement_path(item.ref))  # noqa: SLF001 - 就是要复现那个中间状态
    assert store.claim_exists(item.ref) and store.read_settlement(item.ref) is None
    assert store.open_claims(LATE_TO_BREAKFAST.identity) == (item,)  # 还开着 → 下一夜重结，可自愈
    assert store.orphan_settlements(LATE_TO_BREAKFAST.identity) == ()

    # 另一种崩法（结算留着、承诺没了）留下的是毒：重开时必须拒，不能静默盖章。
    store.write_settlement(Settlement(item.ref, Outcome.CENSORED, NOW, reason="按旧命中结的"))
    store._discard(store.claim_path(item.ref))  # noqa: SLF001
    assert store.orphan_settlements(LATE_TO_BREAKFAST.identity) == (item.ref,)
    with pytest.raises(LedgerStoreError, match="voiding was interrupted"):
        store.write_claim(item)
    store.retain_only(LATE_TO_BREAKFAST.identity, DAY1, frozenset())  # 撤干净
    store.write_claim(item)
    assert store.open_claims(LATE_TO_BREAKFAST.identity) == (item,)


def test_open_claims_only_decodes_the_ones_still_open(tmp_path) -> None:
    """开没开按两棵日目录的文件名差集判：一年 40,440 条承诺时不该把整本账解码一遍再逐条 stat（评审 B4）。"""

    store = LedgerStore(tmp_path / "scene")
    settled = claim()
    later = claim(trigger_uri=uri_for(DAY2, "就寝", 1, 0), antecedent_uris=(uri_for(DAY2, "就寝", 1, 0),), control=daily_snapshot("早餐", at(DAY2, 1, 0), 3))
    store.write_claim(settled)
    store.write_claim(later)
    store.write_settlement(Settlement(settled.ref, Outcome.CENSORED, NOW, reason="x"))
    reads: list[str] = []
    original = store.read_claim

    def counting(ref):
        reads.append(ref.leaf)
        return original(ref)

    store.read_claim = counting  # type: ignore[method-assign]
    assert store.open_claims(LATE_TO_BREAKFAST.identity) == (later,)
    assert reads == [later.ref.leaf]  # 已结算的那条一次都没解码


def test_a_claims_anchor_and_ref_are_computed_once(tmp_path) -> None:
    """锚、地址、ref 在构造时各算一次：读侧每条承诺要问锚上百次（逐条算块），每次重解析 URI 会让一条假设 199 秒
    （评审 B4b 实测 207 万次解析）。"""

    item = claim()
    assert item.trigger_address is item.trigger_address and item.ref is item.ref
    assert item.anchor is item.anchor and item.anchor == at(DAY1, 2, 10)
