"""按账精确压实（26 号清单 4.8；裁定 61 第 7 条）：只压三条都满足的；压实段只留偏差多重集与机会间隔（权重归单元，不再折
累计值）；结算读压实的偏差当更早的历史；编解码往返。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.foresight import account_key, ranked_account
from habitus.foresight.compaction import compact, compactable
from habitus.foresight.ledger import AccountFact
from habitus.foresight.ledger.compaction import decode_compaction, encode_compaction

CST = timezone(timedelta(hours=8))
START = datetime(2026, 1, 1, 9, 0, tzinfo=CST)
ACCOUNT = account_key("s-k0002", "日子", "rel")


def history(pattern: str, *, start: datetime = START, prefix: str = "d") -> list[AccountFact]:
    """每天一次机会：``h`` 准时、``e`` 早到 30 分钟、``m`` 没做（过期待决）、``p`` 提醒后才做、``v`` 作废、``u`` 观测空白、
    ``c`` 后补（强度 0.4）。"""

    found: list[AccountFact] = []
    for day, mark in enumerate(pattern):
        at, identity = start + timedelta(days=day), f"{prefix}{day:03d}"
        if mark == "h":
            found.append(AccountFact(ACCOUNT, at, True, 1.0, "准时", identity, offset_minutes=0))
        elif mark == "e":
            found.append(AccountFact(ACCOUNT, at, True, 0.5, "早到", identity, offset_minutes=-30))
        elif mark == "m":
            found.append(AccountFact(ACCOUNT, at, False, 1.0, "过期待决", identity))
        elif mark == "p":
            found.append(AccountFact(ACCOUNT, at, True, 0.0, "提醒后发生", identity))
        elif mark == "v":
            found.append(AccountFact(ACCOUNT, at, False, 0.0, "作废", identity))
        elif mark == "u":
            found.append(AccountFact(ACCOUNT, at, False, 0.0, "过期待决", identity))
        elif mark == "c":
            found.append(AccountFact(ACCOUNT, at, False, 0.4, "后补", identity))
    return found


def compacted(facts: list[AccountFact], *, now: datetime, window: int, rounds: int = 1):  # type: ignore[no-untyped-def]
    """压 ``rounds`` 次（每次之后再往后压一段），每次都经过编解码往返（落盘再读回）。"""

    base = None
    for _ in range(rounds):
        account = ranked_account(facts, visible_at=now, base=base)
        plan = compactable(account, open_day=None, diagnosis_since=now.date(), window=window)
        if plan.boundary is None:
            break
        base = decode_compaction(encode_compaction(compact(account, plan, table=None)))
        window = max(1, window - 7)
    return base


def test_only_what_is_settled_older_than_diagnosis_and_outside_the_window_is_compacted() -> None:
    facts = history("h" * 60)
    now = START + timedelta(days=60)
    account = ranked_account(facts, visible_at=now)
    by_window = compactable(account, open_day=None, diagnosis_since=now.date(), window=35)
    assert by_window.boundary == (START + timedelta(days=24), "d024")  # 60 次里留最近 35 次
    by_open = compactable(account, open_day=(START + timedelta(days=10)).date(), diagnosis_since=now.date(), window=35)
    assert by_open.boundary == (START + timedelta(days=9), "d009")  # 第 10 天还没结完：之后的一笔都不压
    by_diagnosis = compactable(account, open_day=None, diagnosis_since=(START + timedelta(days=5)).date(), window=35)
    assert by_diagnosis.boundary == (START + timedelta(days=4), "d004")
    assert compactable(account, open_day=START.date(), diagnosis_since=now.date(), window=35).boundary is None
    assert all(fact.at <= START + timedelta(days=24) for fact in by_window.facts)
    with pytest.raises(ValueError):
        compactable(account, open_day=None, diagnosis_since=now.date(), window=0)
    with pytest.raises(ValueError):
        compact(account, by_window.__class__(None, ()), table=None)


def test_void_and_unobserved_opportunities_do_not_move_the_boundary() -> None:
    """作废、观测空白不是机会：窗口按机会数，它们不占名额。"""

    facts = history("h" * 40 + "uuvu" + "u" * 10 + "mmmm")
    now = START + timedelta(days=200)
    account = ranked_account(facts, visible_at=now)
    plan = compactable(account, open_day=None, diagnosis_since=now.date(), window=35)
    assert plan.boundary == (START + timedelta(days=8), "d008")  # 44 次机会里留最近 35 次
    base = compacted(facts, now=now, window=35)
    assert base is not None and base.through == plan.boundary
    remaining = [fact for fact in facts if not base.covers(fact.at, fact.expectation_id)]
    assert len(remaining) == len(facts) - 9


def test_the_compaction_keeps_the_offsets_and_the_gaps_and_accumulates_across_rounds() -> None:
    facts = history("hehhe" * 10)
    now = START + timedelta(days=200)
    base = compacted(facts, now=now, window=5)
    assert base is not None
    assert base.through == (START + timedelta(days=44), "d044")  # 50 次里留最近 5 次
    assert dict(base.offsets) == {0: 27, 30: 18}  # 前 45 天：27 次准时、18 次早到 30 分钟
    assert base.gap_count == 44 and base.longest_gap == 1.0 and base.last_day == date(2026, 2, 14)
    assert base.day_types == (None,)
    assert sorted(base.offset_history()) == [0] * 27 + [30] * 18
    # 读的一方忽略压实段以内残留的文件；同一窗口再压一次没有可压的
    remaining = [fact for fact in facts if not base.covers(fact.at, fact.expectation_id)]
    again = ranked_account(facts, visible_at=now, base=base)
    assert again.facts == ranked_account(remaining, visible_at=now, base=base).facts
    assert compactable(again, open_day=None, diagnosis_since=now.date(), window=5).boundary is None
    # 窗口收窄再压一段：接着上一段累计
    plan = compactable(again, open_day=None, diagnosis_since=now.date(), window=1)
    further = decode_compaction(encode_compaction(compact(again, plan, table=None)))
    assert further.through == (START + timedelta(days=48), "d048")
    assert dict(further.offsets) == {0: 30, 30: 19} and further.gap_count == 48
    assert further.last_day == date(2026, 2, 18)


def test_the_compaction_codec_refuses_other_schemas_and_contradictions() -> None:
    from habitus.foresight.errors import ForesightError

    facts = history("h" * 50)
    base = compacted(facts, now=START + timedelta(days=100), window=5)
    assert base is not None
    raw = encode_compaction(base)
    assert decode_compaction(raw) == base
    with pytest.raises(ForesightError, match="schema"):
        decode_compaction({**raw, "schema_version": "foresight_account_compaction_v1"})
    with pytest.raises(ForesightError, match="offsets"):
        decode_compaction({**raw, "offsets": [[30, 1], [0, 2]]})
    with pytest.raises(ForesightError, match="gaps"):
        decode_compaction({**raw, "gap_count": 0, "longest_gap": 3.0})


def test_settlement_reads_the_compacted_offsets_as_earlier_history() -> None:
    """窗外发生的强度拿这一次在全部历史偏差里的位置算：压实了的那一段的偏差照样算进历史。"""

    from habitus.foresight.facts import settlement_fact
    from habitus.foresight.ledger import Expectation, Settlement

    formed = START + timedelta(days=300)
    item = Expectation(
        expectation_id="now",
        kind_token="s-k0002",
        day=formed.date(),
        formed_at=formed,
        source="日子",
        trigger="rel",
        trigger_instance="behavior://now.md",
        window=(formed, formed + timedelta(minutes=30)),
        deadline=formed + timedelta(hours=2),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=formed,
    )
    late = Settlement("now", "s-k0002", formed.date(), formed + timedelta(hours=3), "晚了但做了", "behavior://x.md", 40)
    alone = settlement_fact(item, late, known=())
    assert alone.strength == 1.0  # 没有历史：只有方向
    with_history = settlement_fact(item, late, known=(), earlier_offsets=(0, 0, 0, 60))
    assert with_history.strength == pytest.approx(0.25)  # 历史四次里一次偏得不比这次少
