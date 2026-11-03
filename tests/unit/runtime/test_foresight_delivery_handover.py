"""依据没说成，成员接着说（裁定 92 第 2 条②）：依据记了「宿主没取」「不可达」，或停在「等时窗末」过了最晚时限，被它「单元已选」
吸收、自己时限还在的成员到了时窗末照常走递送；依据说出口了、或自己决定不说（已发生……）的，成员照旧被吸收、不再问。同一刻只接手
一条；「同类已说」与名额不把同一刻里没说成的那几条算进去，别的单元照算；结算时说出口的成员自己写那一笔、依据照旧按没说过。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from habitus.foresight.ledger import Delivery, Expectation
from habitus.runtime.foresight_delivery import JudgeMaterials
from habitus.runtime.foresight_settlement import Asked
from tests.unit.runtime.test_foresight_delivery import DAY0, Outlet, earn, expectation, stage


def at(minutes: int) -> datetime:
    return DAY0 + timedelta(minutes=minutes)


def reasons(ledger, identity: str) -> list[str]:  # type: ignore[no-untyped-def]
    return [
        item.reason
        for item in sorted(ledger.deliveries_on(DAY0.date()), key=lambda item: item.decided_at)
        if item.expectation_id == identity
    ]


def moment(tmp_path, *, members: tuple[tuple[str, str, int], ...] = (("later", "clock", 4),), **overrides):  # type: ignore[no-untyped-def]
    """依据「now」09:10–09:40、最晚时限 11:00；成员（身份, 触发, 最晚时限几小时）09:30 形成、时窗 09:40–10:10，和依据连成一刻、
    还没到时窗——09:16 那一拍记「单元已选」引依据。"""

    run, ledger = stage(tmp_path, **overrides)
    earn(ledger, trigger="rel")
    ledger.record_expectation(expectation("now", DAY0, trigger="rel"))
    for identity, trigger, hours in members:
        earn(ledger, trigger=trigger)
        ledger.record_expectation(expectation(identity, at(30), trigger=trigger, deadline_hours=hours))
    asyncio.run(run.run_once(at(16)))
    assert reasons(ledger, "now") == ["等时窗末"]
    assert all(reasons(ledger, identity) == ["单元已选"] for identity, _trigger, _hours in members)
    return run, ledger


def test_a_member_speaks_once_its_basis_was_never_claimed(tmp_path) -> None:
    """依据「待取」到最晚时限没人取：时限还在的成员接手——同一拍记「待取」、进可取清单；依据那条「待取」不算「同类已说」。"""

    outlet = Outlet()
    run, ledger = moment(tmp_path, outlet=outlet)
    asyncio.run(run.run_once(at(41)))
    assert reasons(ledger, "now") == ["等时窗末", "待取"]
    over = asyncio.run(run.run_once(at(121)))  # 11:01：依据过了最晚时限
    assert reasons(ledger, "now") == ["等时窗末", "待取", "宿主没取"]
    assert reasons(ledger, "later") == ["单元已选", "待取"]
    assert over.open_requests == 1 and [row.request_id for row in outlet.published[-1]] == ["later"]
    assert asyncio.run(run.run_once(at(122))).decided == 0  # 接手只一次
    # 说出口了：结算时成员按自己的说出时刻结、反馈由它写；依据照旧没说过
    ledger.record_delivery(DAY0.date(), Delivery("later", at(130), 1, True, "说了", 1.0, "话", via="codex", turn="c1"))
    made = ledger.deliveries_on(DAY0.date())
    assert Asked.of(made, "later") == Asked(at(130), own=True, basis="later")
    assert Asked.of(made, "now") == Asked(None)


def test_a_basis_that_was_spoken_keeps_its_members_subsumed(tmp_path) -> None:
    run, ledger = moment(tmp_path)
    asyncio.run(run.run_once(at(41)))
    ledger.record_delivery(DAY0.date(), Delivery("now", at(50), 1, True, "说了", 1.0, "话", via="codex", turn="c1"))
    for minutes in (121, 200, 239):
        assert asyncio.run(run.run_once(at(minutes))).decided == 0
    assert reasons(ledger, "later") == ["单元已选"]
    made = ledger.deliveries_on(DAY0.date())
    assert Asked.of(made, "later") == Asked(at(50), basis="now")  # 成员照旧按依据的说出时刻结


def test_a_basis_that_chose_not_to_speak_keeps_its_members_subsumed(tmp_path) -> None:
    """依据自己决定不说（做了）不是"没说成"：那是单元替这一刻做的决定。"""

    run, ledger = moment(tmp_path)
    done = JudgeMaterials(None, lambda uris: {}, happened=lambda item: item.expectation_id == "now")
    asyncio.run(run.run_once(at(41), done))
    assert reasons(ledger, "now") == ["等时窗末", "已发生"]
    for minutes in (121, 239):
        assert asyncio.run(run.run_once(at(minutes))).decided == 0
    assert reasons(ledger, "later") == ["单元已选"]


def test_an_unreachable_basis_lets_a_member_speak_once_a_host_is_back(tmp_path) -> None:
    """依据时窗末没有宿主活着记「不可达」；成员到了自己的时窗末，宿主回来了就照常「待取」。"""

    outlet = Outlet(alive=False)
    run, ledger = moment(tmp_path, outlet=outlet)
    asyncio.run(run.run_once(at(41)))
    assert reasons(ledger, "now") == ["等时窗末", "不可达"]
    assert asyncio.run(run.run_once(at(55))).decided == 0  # 成员的时窗 10:10 才末
    outlet.is_alive = True
    asyncio.run(run.run_once(at(71)))
    assert reasons(ledger, "later") == ["单元已选", "待取"]


def test_a_basis_stuck_past_its_deadline_hands_over(tmp_path) -> None:
    """停过机：依据的时窗末那几拍都没跑、停在「等时窗末」过了最晚时限——也是没说成。"""

    run, ledger = moment(tmp_path)
    asyncio.run(run.run_once(at(130)))  # 09:16 之后直接到 11:10
    assert reasons(ledger, "now") == ["等时窗末"]
    assert reasons(ledger, "later") == ["单元已选", "待取"]


def test_only_one_member_takes_over_and_the_next_one_only_after_it_failed_too(tmp_path) -> None:
    run, ledger = moment(tmp_path, members=(("zz", "zz", 4), ("aa", "aa", 6)))
    asyncio.run(run.run_once(at(41)))
    asyncio.run(run.run_once(at(121)))
    first, second = ("zz", "aa") if reasons(ledger, "zz")[-1] == "待取" else ("aa", "zz")
    assert reasons(ledger, first) == ["单元已选", "待取"] and reasons(ledger, second) == ["单元已选"]
    assert first == "zz"  # 同下界按账取（``units.basis``）
    asyncio.run(run.run_once(at(200)))
    assert reasons(ledger, second) == ["单元已选"]  # 接手的那条还等着宿主：别的不接
    asyncio.run(run.run_once(at(271)))  # 13:31：zz 也过了最晚时限
    assert reasons(ledger, "zz") == ["单元已选", "待取", "宿主没取"]
    assert reasons(ledger, "aa") == ["单元已选", "待取"]


def test_another_moment_of_the_same_kind_already_spoken_still_counts(tmp_path) -> None:
    """不算进去的只是同一刻里没说成的那几条：同一类今天别的单元说过了，接手的成员照样「同类已说」。"""

    run, ledger = moment(tmp_path)
    asyncio.run(run.run_once(at(41)))
    other = expectation("other", DAY0 - timedelta(hours=2), trigger="rel")
    ledger.record_expectation(other)
    ledger.record_delivery(other.day, Delivery("other", at(60), 1, True, "说了", 1.0, "话", via="codex", turn="c2"))
    asyncio.run(run.run_once(at(121)))
    assert reasons(ledger, "later") == ["单元已选", "同类已说"]


def test_the_daily_cap_does_not_count_the_moment_that_was_never_said_but_counts_the_rest(tmp_path) -> None:
    run, ledger = moment(tmp_path)  # 名额 2
    earn(ledger, kind="s-k0003")
    ledger.record_expectation(expectation("k3", DAY0, kind="s-k0003"))
    asyncio.run(run.run_once(at(17)))
    asyncio.run(run.run_once(at(41)))
    assert reasons(ledger, "now")[-1] == "待取" and reasons(ledger, "k3")[-1] == "待取"  # 名额占满
    asyncio.run(run.run_once(at(121)))
    assert reasons(ledger, "later") == ["单元已选", "待取"]  # 依据那条没说成、不占
    # 别的类今天又说了一句：不算同一刻那条，名额照样满了，接手的成员「名额已满」
    run2, ledger2 = moment(tmp_path / "b")
    for kind in ("s-k0003", "s-k0004"):
        earn(ledger2, kind=kind)
        ledger2.record_expectation(expectation(f"x-{kind}", DAY0, kind=kind))
    asyncio.run(run2.run_once(at(17)))
    asyncio.run(run2.run_once(at(41)))
    assert reasons(ledger2, "x-s-k0003")[-1] == "待取" and reasons(ledger2, "x-s-k0004")[-1] == "名额已满"
    late: Expectation = expectation("x-late", DAY0, kind="s-k0005")
    ledger2.record_expectation(late)
    ledger2.record_delivery(late.day, Delivery("x-late", at(60), 1, True, "说了", 1.0, "话", via="c", turn="t"))
    asyncio.run(run2.run_once(at(121)))
    assert reasons(ledger2, "later") == ["单元已选", "名额已满"]


def test_a_question_recognised_late_still_counts_as_said(tmp_path) -> None:
    """转写晚到：宿主没取已记下之后，出口才认出时限前那次提问、补一条「说了」（decided_at 是提问时刻，早于「宿主没取」）。
    依据其实说出口了：成员不接手（只看最近一条会把它读成没说成）。"""

    run, ledger = moment(tmp_path)
    asyncio.run(run.run_once(at(41)))
    asyncio.run(run.run_once(at(119)))  # 时限前一拍：还在等
    ledger.record_delivery(DAY0.date(), Delivery("now", at(121), 1, False, "宿主没取", 0.0))
    ledger.record_delivery(DAY0.date(), Delivery("now", at(110), 1, True, "说了", 1.0, "话", via="codex", turn="c1"))
    asyncio.run(run.run_once(at(122)))
    assert reasons(ledger, "later") == ["单元已选"]


def test_a_member_past_its_own_deadline_does_not_take_over_and_does_not_block_the_next(tmp_path) -> None:
    """成员自己的时限已过就不接手，也不占住这一刻：时限还在的那条接手（zz 时限 10:30，依据 11:00 才没说成）。"""

    run, ledger = moment(tmp_path, members=(("zz", "zz", 1), ("aa", "aa", 6)))
    asyncio.run(run.run_once(at(41)))
    asyncio.run(run.run_once(at(121)))
    assert reasons(ledger, "zz") == ["单元已选"]
    assert reasons(ledger, "aa") == ["单元已选", "待取"]
