"""历史卡与此刻场景：同一种材料的两头。

卡：前后 ±k 槽的序列分成之前 / 这次 / 之后。
此刻：到此刻为止的序列，未封口的补进来、与已封口的不重、窗外的不要；今天做过什么全天计。
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from habitus.foresight import EvidencePack, ForesightError, UnsealedRow, history_card
from habitus.scene.views import context_view
from tests.unit.foresight.fixtures import MONDAY, Ground, at
from tests.unit.kind_ids import kind_id

NOW = MONDAY + timedelta(days=28)


def unsealed_row(name: str | None, started, *, kind: str | None, lasts_minutes: int = 5) -> UnsealedRow:
    return UnsealedRow(
        name=name,
        kind_token=kind,
        started_at=started,
        last_observed_at=started + timedelta(minutes=lasts_minutes),
        summary=None if name is None else f"{name}的摘要",
    )


def weekly_ground(tmp_path) -> Ground:
    """三个周一 19:00 打球（一个半小时）；每次之前收拾球包、之后洗澡。"""

    ground = Ground(tmp_path, now=at(NOW, 23, 0))
    for week in range(3):
        day = MONDAY + timedelta(days=7 * week)
        ground.record(day, "收拾球包", 18, 40, kind="收拾球包")
        ground.record(day, "打球", 19, 0, kind="打球", lasts_minutes=90)
        ground.record(day, "洗澡", 20, 50, kind="洗澡")
        ground.record(day, "开电脑", 21, 40, kind="开电脑")  # 20:30 + 3 槽 = 21:15 之后，不在卡里
    return ground


def test_a_card_splits_the_flow_into_before_this_and_after(tmp_path) -> None:
    ground = weekly_ground(tmp_path)
    pack = ground.pack(at(NOW, 19, 5))
    assert [item.kind_token for item in pack.expanded] == sorted(
        [kind_id("打球"), kind_id("收拾球包")]
    )  # 18:40 在 ±3 槽里，洗澡不在
    candidate = next(item for item in pack.expanded if item.kind_token == kind_id("打球"))
    cards = candidate.background.cards
    assert [card.at.date() for card in cards] == [MONDAY + timedelta(days=7 * week) for week in range(3)]
    first = cards[0]
    assert [row.name for row in first.before] == ["收拾球包"]
    assert first.flow[first.own_index].name == "打球"
    assert [row.name for row in first.after] == ["洗澡"]  # 之后段从最后所见 20:30 起再 +3 槽，开电脑 21:40 不在
    assert first.layer == "slot"


def test_a_card_must_find_itself_in_its_flow(tmp_path) -> None:
    ground = weekly_ground(tmp_path)
    cache = ground.cache()
    play = ground.record(NOW, "打球", 19, 0, kind="打球")
    view = context_view(play, cache, window_days=30)
    card = history_card(view, "slot", cache, slot_minutes=15, half_width=2)
    assert card.uri == play and card.before == () and card.after == ()
    assert card.own.day_count == 1 and card.own.last_observed_at == at(NOW, 19, 10)
    with pytest.raises(ForesightError, match="unknown shrinkage layer"):
        history_card(view, "somewhere", cache, slot_minutes=15, half_width=2)


def test_the_now_scene_stops_at_this_moment_and_folds_in_the_unsealed(tmp_path) -> None:
    """此刻 19:05、k=3 → 窗口 [18:15, 19:05]。树上有 18:40 的收拾球包；判断存储里三条：
    一条在窗内且树上没有（补进来）、一条与树上同名同时刻（不重）、一条在窗外（不要）。"""

    ground = weekly_ground(tmp_path)
    ground.record(NOW, "收拾球包", 18, 40, kind="收拾球包")
    ground.record(NOW, "吃早饭", 8, 0, kind="吃早饭")
    ground.record(NOW, "打游戏", 19, 30, kind="打游戏")  # 此刻之后，不在
    unsealed = (
        unsealed_row("出门", at(NOW, 18, 55), kind=kind_id("出门")),
        unsealed_row("收拾球包", at(NOW, 18, 40), kind=kind_id("收拾球包")),
        unsealed_row("换衣服", at(NOW, 18, 0), kind=None),
        unsealed_row(None, at(NOW, 18, 45), kind=None, lasts_minutes=3),  # 没读懂的一段，在窗内
        unsealed_row("喝水", at(NOW, 9, 0), kind=kind_id("喝水")),  # 今天早上、还没归约：不进流，但"今天做过"要有它
        unsealed_row(None, at(NOW, 9, 30), kind=None, lasts_minutes=5),  # 早上没读懂的一段：今天的空白
        unsealed_row("散步", at(NOW - timedelta(days=1), 9, 0), kind=kind_id("散步")),  # 昨天的：不是今天
    )
    pack = ground.pack(at(NOW, 19, 5), unsealed=unsealed)
    now = pack.now
    assert now.since == at(NOW, 18, 15)
    assert [row.name for row in now.flow] == ["收拾球包"]
    assert [row.name for row in now.unsealed] == ["出门"]
    assert now.done_today == {kind_id("出门"): 1, kind_id("吃早饭"): 1, kind_id("喝水"): 1, kind_id("收拾球包"): 1}
    assert now.gaps == ()
    assert [(gap.started_at, gap.ended_at, gap.kind) for gap in now.unsealed_gaps] == [
        (at(NOW, 9, 30), at(NOW, 9, 35), "没读懂"),
        (at(NOW, 18, 45), at(NOW, 18, 48), "没读懂"),
    ]


def test_the_now_scene_fingerprint_follows_the_scene_not_the_clock(tmp_path) -> None:
    """同一槽内钟在走不算场景在变；多一条未封口、多一条树上的行、多一段空白都算。"""

    ground = weekly_ground(tmp_path)
    ground.record(NOW, "收拾球包", 18, 40, kind="收拾球包")
    base = ground.pack(at(NOW, 19, 5)).now.fingerprint
    assert ground.pack(at(NOW, 19, 12)).now.fingerprint == base
    assert (
        ground.pack(
            at(NOW, 19, 5), unsealed=(unsealed_row("出门", at(NOW, 18, 55), kind=kind_id("出门")),)
        ).now.fingerprint
        != base
    )
    assert (
        ground.pack(at(NOW, 19, 5), unsealed=(unsealed_row(None, at(NOW, 18, 45), kind=None),)).now.fingerprint != base
    )
    ground.gap(NOW, 12, 0, 13, 0)
    assert ground.pack(at(NOW, 19, 5)).now.fingerprint != base


def test_the_now_scene_lists_todays_gaps_up_to_now(tmp_path) -> None:
    ground = weekly_ground(tmp_path)
    ground.gap(NOW, 12, 0, 13, 0)
    ground.gap(NOW, 21, 0, 22, 0)  # 此刻之后
    now = ground.pack(at(NOW, 19, 5)).now
    assert [(gap.started_at.hour, gap.ended_at.hour) for gap in now.gaps] == [(12, 13)]


def test_candidates_enter_by_evidence_near_this_slot_not_by_a_count(tmp_path) -> None:
    """每个周一 8:00 吃早饭：它在周一有曲线，所以 19:05 也是候选，但本槽 / 邻域 / 跨周几三层都没有
    出处日，只列名字与数字；打球三层有出处，摊开。"""

    ground = weekly_ground(tmp_path)
    for week in range(3):
        ground.record(MONDAY + timedelta(days=7 * week), "吃早饭", 8, 0, kind="吃早饭")
    pack = ground.pack(at(NOW, 19, 5))
    assert [item.kind_token for item in pack.candidates] == sorted(
        [kind_id("吃早饭"), kind_id("开电脑"), kind_id("打球"), kind_id("收拾球包"), kind_id("洗澡")]
    )
    assert [item.kind_token for item in pack.expanded] == sorted([kind_id("打球"), kind_id("收拾球包")])
    breakfast = next(item for item in pack.named_only if item.kind_token == kind_id("吃早饭"))
    assert breakfast.background.cards == () and breakfast.provenance.all_day.days != ()
    assert pack.generation == "test-generation" and pack.config_digest == ground.tree().config_digest


def test_a_pack_keeps_its_candidates_sorted_and_its_now_at_its_own_moment(tmp_path) -> None:
    ground = weekly_ground(tmp_path)
    pack = ground.pack(at(NOW, 19, 5))
    with pytest.raises(ForesightError, match="unique and sorted"):
        EvidencePack(
            generation=pack.generation,
            config_digest=pack.config_digest,
            slot_minutes=pack.slot_minutes,
            moment=pack.moment,
            now=pack.now,
            candidates=tuple(reversed(pack.candidates)),
            labels=pack.labels,
        )
    other = ground.pack(at(NOW, 20, 5))
    with pytest.raises(ForesightError, match="own moment"):
        EvidencePack(
            generation=pack.generation,
            config_digest=pack.config_digest,
            slot_minutes=pack.slot_minutes,
            moment=pack.moment,
            now=other.now,
            candidates=pack.candidates,
            labels=pack.labels,
        )


def test_a_card_on_another_weekday_lands_in_the_cross_weekday_layer(tmp_path) -> None:
    """周三 19:30 那次落在跨周几层：卡按四层的出处日取，标它落在的最内层。"""

    ground = weekly_ground(tmp_path)
    wednesday = ground.record(MONDAY + timedelta(days=2), "打球", 19, 30, kind="打球", lasts_minutes=90)
    candidate = next(item for item in ground.pack(at(NOW, 19, 5)).expanded if item.kind_token == kind_id("打球"))
    card = next(card for card in candidate.background.cards if card.uri == wednesday)
    assert card.layer == "cross_weekday"
