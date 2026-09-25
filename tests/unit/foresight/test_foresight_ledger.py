"""结算账本：只记带时窗的「会」、同一次即将发生只记一条；结算按承诺自己的槽宽对时窗；门是纯计数。"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.judge import CandidateVerdict, Judgement
from habitus.foresight.ledger import (
    LEDGER_SCHEMA_VERSION,
    Settlement,
    claims_from,
    decode_claim,
    decode_settlement,
    encode_claim,
    encode_settlement,
    settle,
    verified_count,
    verified_counts,
)
from tests.unit.foresight.fixtures import MONDAY, Ground, at

NOW = MONDAY + timedelta(days=28)
SETTLED_AT = datetime(2026, 9, 1, 3, 0, tzinfo=UTC)


def ground_for(tmp_path) -> Ground:
    ground = Ground(tmp_path, now=at(NOW, 23, 0))
    for week in range(4):
        day = MONDAY + timedelta(days=7 * week)
        ground.record(day, "收拾球包", 18, 40, kind="收拾球包")
        play = ground.record(day, "打球", 19, 0, kind="打球", lasts_minutes=60)
        if week < 2:
            ground.associate(play, kind="打球", context=f"第 {week + 1} 周", situation="周一下班后自己去")
    return ground


def judgement_at(pack, *verdicts: CandidateVerdict, judged_at: datetime | None = None) -> Judgement:
    return Judgement(
        judged_at=judged_at or pack.moment.at.astimezone(UTC),
        generation=pack.generation,
        moment=pack.moment,
        verdicts=tuple(sorted(verdicts, key=lambda v: v.kind_token)),
        day_state="正常",
        day_note=None,
        judge_version="test-judge",
    )


def verdict(kind: str, result: str, window: tuple[int, int] | None, basis: tuple[str, ...]) -> CandidateVerdict:
    return CandidateVerdict(kind_token=kind, verdict=result, window=window, next=(), basis=basis, note="")


def test_only_promises_with_a_window_and_a_cited_card_enter_the_ledger(tmp_path) -> None:
    """「不会」「说不准」没有可核对的时点；说不出时窗的「会」同样核对不了——三者都不进账。"""

    pack = ground_for(tmp_path).pack(at(NOW, 19, 5))
    play = next(item for item in pack.expanded if item.kind_token == "打球")
    cards = play.background.cards
    claims = claims_from(
        pack,
        judgement_at(
            pack,
            verdict("打球", "会", (76, 78), (cards[0].uri, cards[1].uri)),
            verdict("收拾球包", "不会", None, ()),
        ),
    )
    assert [(c.kind_token, c.window, c.slot, c.slot_minutes) for c in claims] == [("打球", (76, 78), 76, 15)]
    (claim,) = claims
    assert claim.situations == ("周一下班后自己去",)  # 两张卡同一情形，按规范身份去重
    assert claim.basis == (cards[0].uri, cards[1].uri) and claim.numbers is play.numbers
    assert claim.generation == pack.generation and claim.judge_version == "test-judge" and claim.conditions == ()
    assert claims_from(pack, judgement_at(pack, verdict("打球", "说不准", None, (cards[0].uri,)))) == ()
    assert claims_from(pack, judgement_at(pack, verdict("打球", "会", None, (cards[0].uri,)))) == ()


def test_the_conditions_the_asked_keys_and_the_source_version_are_all_stamped(tmp_path) -> None:
    """三样一起存：只看答案分不清"那天这个源离线"与"从来没有这个键"，也分不清是谁按什么口径答的。"""

    pack = ground_for(tmp_path).pack(at(NOW, 19, 5))
    play = next(item for item in pack.expanded if item.kind_token == "打球")
    (claim,) = claims_from(
        pack,
        judgement_at(pack, verdict("打球", "会", (76, 78), (play.background.cards[0].uri,))),
        conditions=(("天气.天象", "晴"),),
        condition_keys=("天气.天象", "天气.温度"),
        facts_version="weather_v1",
    )
    assert claim.conditions == (("天气.天象", "晴"),)  # 温度那天答不出来：问过、没答
    assert claim.condition_keys == ("天气.天象", "天气.温度") and claim.facts_version == "weather_v1"
    assert decode_claim(encode_claim(claim)) == claim
    # 承诺不许答一个没问过的键，也不许收下乱序或重复的条件——第三刀按键取份额，重键取到哪个都对不上。
    with pytest.raises(ForesightError, match="did not ask"):
        replace(claim, conditions=(("天气.湿度", "高"),))
    with pytest.raises(ForesightError, match="sorted by key"):
        replace(claim, conditions=(("天气.温度", "31"), ("天气.天象", "晴")), condition_keys=("天气.天象", "天气.温度"))


def test_an_abstention_carrying_a_window_does_not_reach_the_ledger(tmp_path) -> None:
    """schema 允许「说不准」带时窗（说不出就填 null），账本不能因此炸掉整拍。"""

    pack = ground_for(tmp_path).pack(at(NOW, 19, 5))
    play = next(item for item in pack.expanded if item.kind_token == "打球")
    assert claims_from(pack, judgement_at(pack, verdict("打球", "说不准", (76, 80), (play.background.cards[0].uri,)))) == ()


def test_the_same_upcoming_occurrence_is_recorded_once_but_a_later_window_is_its_own_promise(tmp_path) -> None:
    ground = ground_for(tmp_path)
    pack = ground.pack(at(NOW, 19, 5))
    play = next(item for item in pack.expanded if item.kind_token == "打球")
    basis = (play.background.cards[0].uri,)
    (first,) = claims_from(pack, judgement_at(pack, verdict("打球", "会", (76, 78), basis)))
    # 下一槽同样的话、或略微挪动的时窗：与已有的那条相交，说的是同一次，不重记。
    later = ground.pack(at(NOW, 19, 20))
    assert claims_from(later, judgement_at(later, verdict("打球", "会", (76, 78), basis)), open_claims=(first,)) == ()
    assert claims_from(later, judgement_at(later, verdict("打球", "会", (77, 79), basis)), open_claims=(first,)) == ()
    # 改口说晚上那次：不相交，是另一条承诺。
    (second,) = claims_from(later, judgement_at(later, verdict("打球", "会", (84, 86), basis)), open_claims=(first,))
    assert second.window == (84, 86) and second.claim_id != first.claim_id


def test_settlement_reads_the_first_occurrence_after_the_claim_against_its_own_window(tmp_path) -> None:
    ground = ground_for(tmp_path)
    pack = ground.pack(at(NOW, 18, 50))
    play = next(item for item in pack.expanded if item.kind_token == "打球")
    basis = (play.background.cards[0].uri,)

    def claim(window: tuple[int, int], *, spoken: datetime | None = None):
        spoken = spoken or at(NOW, 18, 50)
        judgement = judgement_at(pack, verdict("打球", "会", window, basis), judged_at=spoken.astimezone(UTC))
        (item,) = claims_from(pack, judgement)
        return item

    ground.record(NOW, "打球", 19, 10, kind="打球")  # 第 76 槽
    rows = ground.cache().day(NOW).rows
    verified = settle(claim((76, 77)), rows, settled_at=SETTLED_AT)
    assert (verified.outcome, verified.slot_offset, verified.verified) == ("验证", None, True)
    assert verified.occurrence_uri and "打球" in verified.occurrence_uri
    late = settle(claim((74, 75)), rows, settled_at=SETTLED_AT)
    assert (late.outcome, late.slot_offset, late.verified) == ("偏离", 1, False)
    early = settle(claim((78, 80)), rows, settled_at=SETTLED_AT)
    assert (early.outcome, early.slot_offset) == ("偏离", -2)  # 早于窗：负数，账本只记事实
    missed = settle(claim((77, 78), spoken=at(NOW, 19, 20)), rows, settled_at=SETTLED_AT)
    assert (missed.outcome, missed.occurrence_uri) == ("落空", None)


def test_a_claim_is_settled_with_the_slot_width_it_was_made_under(tmp_path) -> None:
    """槽号配上槽宽才有意义：换参数重建的新一代不能把旧承诺核对错。"""

    ground = ground_for(tmp_path)
    pack = ground.pack(at(NOW, 18, 50))
    play = next(item for item in pack.expanded if item.kind_token == "打球")
    judgement = judgement_at(pack, verdict("打球", "会", (76, 77), (play.background.cards[0].uri,)))
    (claim,) = claims_from(pack, judgement)
    ground.record(NOW, "打球", 19, 40, kind="打球")  # 15 分钟槽下是第 78 槽，30 分钟槽下是第 39 槽
    item = settle(claim, ground.cache().day(NOW).rows, settled_at=SETTLED_AT)
    assert claim.slot_minutes == 15 and (item.outcome, item.slot_offset) == ("偏离", 1)


def test_the_gate_counts_verified_promises_per_kind_and_situation() -> None:
    def row(kind: str, outcome: str, situations: tuple[str, ...], offset: int | None = None) -> Settlement:
        return Settlement(
            claim_id=f"{kind}-{outcome}-{'|'.join(situations)}-{offset}",
            kind_token=kind,
            day=NOW,
            situations=situations,
            outcome=outcome,
            occurrence_uri=None if outcome == "落空" else "behavior://x",
            slot_offset=offset,
            settled_at=SETTLED_AT,
        )

    rows = (
        row("打球", "验证", ("周一下班后自己去",)),
        row("打球", "验证", ("周一下班后自己去", "和朋友约好")),
        row("打球", "偏离", ("周一下班后自己去",), offset=2),
        row("打球", "验证", ()),
        row("洗澡", "落空", ()),
    )
    assert verified_counts(rows) == {("打球", "周一下班后自己去"): 2, ("打球", "和朋友约好"): 1, ("打球", ""): 1}
    assert verified_count(rows, "打球", "周一下班后自己去") == 2 and verified_count(rows, "洗澡") == 0


def test_records_round_trip_through_the_codec(tmp_path) -> None:
    ground = ground_for(tmp_path)
    pack = ground.pack(at(NOW, 19, 5))
    play = next(item for item in pack.expanded if item.kind_token == "打球")
    (claim,) = claims_from(pack, judgement_at(pack, verdict("打球", "会", (76, 78), (play.background.cards[0].uri,))))
    assert decode_claim(encode_claim(claim)) == claim
    item = settle(claim, ground.cache().day(NOW).rows, settled_at=SETTLED_AT)
    assert decode_settlement(encode_settlement(item)) == item
    with pytest.raises(ForesightError, match="malformed"):
        decode_claim({"schema_version": LEDGER_SCHEMA_VERSION, "claim_id": "x"})
    # 旧版本的记录不猜着读：第二刀给承诺加了条件三件套，v1 的文件按 v1 的形状去解只会读出半条。
    with pytest.raises(ForesightError, match="unknown ledger schema"):
        decode_claim({**encode_claim(claim), "schema_version": "foresight_ledger_v1"})


def test_the_shapes_refuse_self_contradiction() -> None:
    with pytest.raises(ForesightError, match="exactly when"):
        Settlement(claim_id="a", kind_token="打球", day=NOW, situations=(), outcome="验证", occurrence_uri=None, slot_offset=None, settled_at=SETTLED_AT)
    with pytest.raises(ForesightError, match="slot_offset"):
        Settlement(claim_id="a", kind_token="打球", day=NOW, situations=(), outcome="验证", occurrence_uri="u", slot_offset=2, settled_at=SETTLED_AT)
