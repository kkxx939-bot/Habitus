"""上级让给还在等时窗末的下级（Q20；裁定 101 第 8 条、裁定 103、裁定 92 第 2 条②）：

上级到了时窗末，同一件事有下级还没到自己的时窗末（时窗还没开、或只记了「等时窗末」），说中的范围重叠，下级到时窗末时上级的最晚时限
还在，而且下级此刻能开口（节点在用、这回没做、此刻算出来在提醒档）——上级这一拍不决定，等下级决定之后再决定：下级说了，上级记
「上下级已说」；下级没说成或决定不说，上级照常往下走。下级不能开口（候选关系、巧合）、被压回影子的，上级不等，照常说。

入口是 ``DeliveryStage.run_once``，每 15 分钟一拍；断言落在账上每条预期的递送原因和时刻、可取清单上。共同摆法：上级 P 时窗
09:10–09:40，下级 C 时窗 11:00–11:15，最晚时限都是当天 24:00，说中的范围重叠，两个单元都在提醒档，宿主活着。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from habitus.foresight import Seen
from habitus.foresight.delivery import done_since
from habitus.foresight.hierarchy import KindHierarchy
from habitus.foresight.judge import Judge
from habitus.foresight.ledger import Delivery, Expectation, UnitFeedback, UnitLevel
from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities, TreeReading
from habitus.model_client import ChatClient, ModelResponse, StructuredChatClient
from habitus.runtime.foresight_delivery import DeliveryStage, JudgeMaterials
from tests.unit.foresight.scripted_model import ScriptedProvider, model_config
from tests.unit.hierarchy_downstream.site import WEEKLY as WORLD_CHILD
from tests.unit.hierarchy_downstream.site import WRITE_DOC as WORLD_PARENT
from tests.unit.hierarchy_downstream.site import World
from tests.unit.hierarchy_downstream.site import at as world_at
from tests.unit.hierarchy_downstream.site import expectation as world_expectation
from tests.unit.runtime.test_foresight_delivery import DAY0, EVIDENCE, LOSS, Outlet, earn, seeded_of, stage

PARENT, CHILD = "s-k0109", "s-k0280"  # 做报表（上级）、做渠道日报（下级）
OTHER = "s-k0200"
KINDS = KindHierarchy({PARENT: frozenset({CHILD})})
DAY = DAY0.date()
MIDNIGHT = DAY0.replace(hour=0) + timedelta(days=1)
RELATION = "rel-child"


def clock(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return DAY0.replace(hour=hour, minute=minute, second=second)


def made(
    identity: str,
    kind: str,
    window: tuple[datetime, datetime],
    *,
    deadline: datetime = MIDNIGHT,
    observed: datetime | None = None,
    trigger: str = "rel",
    source: str = "日子",
    instance: str | None = None,
) -> Expectation:
    formed = observed if observed is not None else clock(0)
    return Expectation(
        expectation_id=identity,
        kind_token=kind,
        day=DAY,
        formed_at=formed,
        source=source,
        trigger=trigger,
        trigger_instance=instance if instance is not None else f"behavior://{identity}.md",
        window=window,
        deadline=deadline,
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=formed,
    )


def parent_item(**overrides: Any) -> Expectation:
    return made("p", PARENT, (clock(9, 10), clock(9, 40)), **overrides)


def child_item(**overrides: Any) -> Expectation:
    window = overrides.pop("window", (clock(11), clock(11, 15)))
    return made("c", CHILD, window, **overrides)


def site(
    tmp_path: Path, *items: Expectation, unearned: Sequence[str] = (), **overrides: Any
) -> tuple[Any, Any, Outlet]:
    """一份账本：``items`` 记上，它们的账都到提醒档（``unearned`` 里的类除外）。"""

    outlet = Outlet()
    run, ledger = stage(tmp_path, outlet=outlet, tick_seconds=900.0, **overrides)
    earned: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.kind_token, item.trigger, item.source)
        if item.kind_token not in unearned and key not in earned:
            earn(ledger, kind=item.kind_token, trigger=item.trigger, source=item.source)
            earned.add(key)
        ledger.record_expectation(item)
    return run, ledger, outlet


def materials(
    now: datetime,
    *,
    kinds: KindHierarchy = KINDS,
    table: OpportunityTable | None = None,
    rows: Sequence[Seen] = (),
) -> JudgeMaterials:
    seen = tuple(row for row in rows if row.started_at <= now)
    return JudgeMaterials(
        table,
        lambda uris: {},
        happened=lambda item: done_since(item, seen, now=now, kinds=kinds),
        seen=seen,
        kinds=kinds,
    )


def walk(run: Any, start: datetime, end: datetime, **options: Any) -> None:
    """从 ``start`` 到 ``end``（含）每 15 分钟一拍。"""

    now = start
    while now <= end:
        asyncio.run(run.run_once(now, materials(now, **options)))
        now += timedelta(minutes=15)


def book(ledger: Any, *identities: str) -> dict[str, list[tuple[str, str]]]:
    """账上每条预期的（递送原因, 决定时刻），按决定先后。"""

    found: dict[str, list[tuple[str, str]]] = {identity: [] for identity in identities}
    for item in sorted(ledger.deliveries_on(DAY), key=lambda delivery: delivery.decided_at):
        if item.expectation_id in found:
            found[item.expectation_id].append((item.reason, item.decided_at.strftime("%H:%M")))
    return found


def listed(outlet: Outlet) -> list[str]:
    return [row.request_id for row in outlet.published[-1]]


def said(identity: str, moment: datetime) -> Delivery:
    return Delivery(identity, moment, 1, True, "说了", 1.0, "题目", via="codex", turn=f"turn-{identity}")


def relation(*, status: str = "established", reading: TreeReading | None = None) -> RelationOpportunities:
    return RelationOpportunities(
        key=RELATION,
        antecedent="c-a",
        consequent="c-b",
        antecedent_label="前因",
        consequent_label="后果",
        antecedent_classes=(OTHER,),
        consequent_classes=(CHILD,),
        segment="chain",
        status=status,
        upward=True,
        items=(),
        reading=reading,
    )


def table_with(item: RelationOpportunities) -> OpportunityTable:
    return OpportunityTable("session", DAY, DAY, 15, 3, (item,), ())


# ── 复现：上级这一拍不决定，等下级 ────────────────────────────────────────────


def test_q20_1_the_parent_waits_for_a_child_whose_window_has_not_opened(tmp_path: Path) -> None:
    """Q20-1（P23 形状）：P 到时窗末时 C 的时窗还没开、账上一条决定都没有：P 09:45 不记决定；C 11:00 等时窗末、11:30 待取；
    P 11:30 上下级已说。说出去的是最细的那一条。"""

    run, ledger, outlet = site(tmp_path, parent_item(), child_item())
    walk(run, clock(9), clock(11, 45))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("上下级已说", "11:30")],
        "c": [("等时窗末", "11:00"), ("待取", "11:30")],
    }
    assert listed(outlet) == ["c"]


def test_q20_1_on_the_assembled_foresight_layer_the_finest_one_is_said(tmp_path: Path) -> None:
    """Q20-1（组合根装出来的预测层）：词表上真实的上下级（写文档 → 写周报），预测拍走到递送，结果同上。"""

    day = world_at(DAY, 0).date()
    world = World(tmp_path)
    world.use(WORLD_PARENT, WORLD_CHILD)
    for kind in (WORLD_PARENT, WORLD_CHILD):
        world.earn(kind, before=world_at(day, 0))
    world.host_alive(world_at(day, 8))
    world.ledger.record_expectation(
        world_expectation("p", WORLD_PARENT, world_at(day, 0), window=(550, 580), deadline_minutes=24 * 60 - 1)
    )
    world.ledger.record_expectation(
        world_expectation("c", WORLD_CHILD, world_at(day, 0), window=(660, 675), deadline_minutes=24 * 60 - 1)
    )
    ticker = world.ticker()
    moment = world_at(day, 9)
    while moment <= world_at(day, 11, 45):
        world.tick(moment, ticker)
        moment += timedelta(minutes=15)
    made_reasons = world.deliveries(day)
    assert (made_reasons["p"], made_reasons["c"]) == (["等时窗末", "上下级已说"], ["等时窗末", "待取"])


def test_q20_2_the_parent_waits_for_a_child_already_waiting_for_its_window_end(tmp_path: Path) -> None:
    """Q20-2：C 已记「等时窗末」（时窗 09:30–10:10）：C 10:15 待取；P 10:15 上下级已说。"""

    run, ledger, outlet = site(tmp_path, parent_item(), child_item(window=(clock(9, 30), clock(10, 10))))
    walk(run, clock(9), clock(10, 30))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("上下级已说", "10:15")],
        "c": [("等时窗末", "09:30"), ("待取", "10:15")],
    }
    assert listed(outlet) == ["c"]


def test_q20_3_when_he_did_the_child_neither_of_them_is_said(tmp_path: Path) -> None:
    """Q20-3（实盘 04-07）：P、C 用实盘的时窗，他 11:50 做了 C：P 11:00 不记决定；12:00 P、C 都「已发生」；没有「说了」。"""

    parent = made("p", PARENT, (clock(10, 15, 59), clock(10, 54, 22)))
    child = made("c", CHILD, (clock(11, 39, 26), clock(11, 54, 26)))
    run, ledger, outlet = site(tmp_path, parent, child)
    rows = (Seen(clock(11, 50), clock(11, 55), CHILD, "behavior://done.md"),)
    walk(run, clock(10), clock(12, 15), rows=rows)
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "10:30"), ("已发生", "12:00")],
        "c": [("等时窗末", "11:45"), ("已发生", "12:00")],
    }
    assert not any(item.spoken for item in ledger.deliveries_on(DAY))
    assert listed(outlet) == []


def test_q20_9_a_child_whose_antecedent_failed_does_not_block_the_parent(tmp_path: Path) -> None:
    """Q20-9（前因不成立）：C 是事件预期，到时窗末核出前因不成立：C 11:30 前因不成立；P 同一拍 11:30 待取（P 没被堵死）。"""

    started = clock(10, 30)
    child = child_item(trigger=RELATION, source="事件", instance=started.astimezone(UTC).isoformat())
    run, ledger, outlet = site(tmp_path, parent_item(), child)
    rows = (Seen(started, started, "s-k0299", "behavior://antecedent.md"),)  # 那一行归成了别的类
    walk(run, clock(9), clock(11, 45), table=table_with(relation()), rows=rows)
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("待取", "11:30")],
        "c": [("等时窗末", "11:00"), ("前因不成立", "11:30")],
    }
    assert listed(outlet) == ["p"]


class _OnlyParentLooksRight(ScriptedProvider):
    """模型替身：下级那个候选答不像，其余答像。"""

    def __init__(self) -> None:
        super().__init__([])

    async def complete_async(self, request: Any) -> ModelResponse:
        self.calls += 1
        verdicts: list[dict[str, object]] = []
        order: list[int] = []
        for line in (request.request.messages[-1].content or "").splitlines():
            if not line.startswith("## 候选 "):
                continue
            number = int(line.removeprefix("## 候选 ").split("：", 1)[0])
            like = CHILD not in line
            verdicts.append(
                {
                    "candidate": number,
                    "like": "yes" if like else "no",
                    "like_basis": 0,
                    "text": "该做了" if like else "",
                }
            )
            if like:
                order.append(number)
        body = json.dumps({"verdicts": verdicts, "order": order}, ensure_ascii=False)
        return ModelResponse(content=body, model=self.model, provider=self.provider_name, finish_reason="stop")


def _cap_is_full(ledger: Any) -> None:
    other = made("other", OTHER, (clock(6), clock(6, 30)), deadline=clock(7))
    ledger.record_expectation(other)
    ledger.record_delivery(DAY, said("other", clock(6, 50)))


def _same_kind_said(ledger: Any) -> None:
    """同一类今天早些时候说过一句；那一条说中的范围（00:00–07:00）与 P、C（08:00 起）不重叠，所以不算"上下级已说"。"""

    other = made("other", CHILD, (clock(6), clock(6, 30)), deadline=clock(7))
    ledger.record_expectation(other)
    ledger.record_delivery(DAY, said("other", clock(6, 50)))


@pytest.mark.parametrize(
    ("arrange", "options", "child_reason", "parent_reason"),
    [
        (_cap_is_full, {"daily_cap": 1}, "名额已满", "名额已满"),
        (_same_kind_said, {}, "同类已说", "待取"),
        (None, {"judge": True}, "模型判不像", "待取"),
    ],
    ids=["名额已满", "同类已说", "模型判不像"],
)
def test_q20_12_a_child_that_decides_not_to_speak_releases_the_parent_in_the_same_tick(
    tmp_path: Path,
    arrange: Callable[[Any], None] | None,
    options: dict[str, Any],
    child_reason: str,
    parent_reason: str,
) -> None:
    """Q20-12：C 到时窗末记了不开口的原因（名额已满、同类已说、模型判不像）：P 不再等，同一拍 11:30 照常往下决定。"""

    extra = dict(options)
    if extra.pop("judge", False):
        client = StructuredChatClient(ChatClient(model_config(), _OnlyParentLooksRight()), validation_retries=1)
        extra["judge"] = Judge(client, retries=0, delay_seconds=0.0)
    observed = clock(8)
    run, ledger, _outlet = site(tmp_path, parent_item(observed=observed), child_item(observed=observed), **extra)
    if arrange is not None:
        arrange(ledger)
    walk(run, clock(9), clock(11, 45))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), (parent_reason, "11:30")],
        "c": [("等时窗末", "11:00"), (child_reason, "11:30")],
    }


def test_q20_18_a_member_that_took_over_also_waits_for_the_finer_child(tmp_path: Path) -> None:
    """Q20-18：P 是接手的成员（依据 B 没说成），有 Q20-1 那样的下级：接手的 P 不记决定，到 C 决定后再决定。"""

    basis = made("b", PARENT, (clock(9, 5), clock(9, 20)), deadline=clock(9, 30), trigger="b")
    member = made("p", PARENT, (clock(9, 20), clock(9, 40)), trigger="p")
    run, ledger, outlet = site(tmp_path, basis, member, child_item())
    walk(run, clock(9), clock(11, 45))
    assert book(ledger, "b", "p", "c") == {
        "b": [("等时窗末", "09:15"), ("待取", "09:30"), ("宿主没取", "09:45")],
        "p": [("单元已选", "09:15"), ("上下级已说", "11:30")],
        "c": [("等时窗末", "11:00"), ("待取", "11:30")],
    }
    assert listed(outlet) == ["c"]


def test_q20_19_while_waiting_the_ledger_shows_only_the_wait_and_a_restart_changes_nothing(tmp_path: Path) -> None:
    """Q20-19：P 等待期间账上只有那条「等时窗末」；进程重启（新建的 ``DeliveryStage`` 读同一份账本）后照样继续等，结果相同。"""

    run, ledger, outlet = site(tmp_path, parent_item(), child_item())
    walk(run, clock(9), clock(10))
    assert book(ledger, "p", "c") == {"p": [("等时窗末", "09:15")], "c": []}
    assert listed(outlet) == []
    restarted = DeliveryStage(
        ledger,
        tolerance=LOSS,
        evidence=EVIDENCE,
        daily_cap=2,
        tick_seconds=900.0,
        outlet=outlet,
        history=seeded_of(ledger).store,
    )
    walk(restarted, clock(10, 15), clock(11, 45))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("上下级已说", "11:30")],
        "c": [("等时窗末", "11:00"), ("待取", "11:30")],
    }


# ── 不变：下级不能开口、被压回影子的，上级不等 ────────────────────────────────


def test_q20_4_a_child_pushed_to_shadow_by_feedback_is_not_waited_for_even_if_its_stored_level_is_one(
    tmp_path: Path,
) -> None:
    """Q20-4（专门用例，实盘 04-13）：C 账上没有决定、账上存的单元档是 1，但提醒反馈已把单元压到影子（现算的档是 0）。
    要按现算的档判：C 此刻不能开口，P 不等，照常 09:45 待取；C 到 11:00 记影子，账上存的档这时才变成 0。"""

    run, ledger, outlet = site(tmp_path, parent_item(), child_item())
    ledger.record_unit_level(UnitLevel(CHILD, clock(0) - timedelta(days=9), 1))
    for offset in range(1, 9):
        moment = clock(9) - timedelta(days=offset)
        ledger.record_feedback(UnitFeedback(CHILD, moment, False, f"said-{offset}"))
    stored = ledger.unit_level_of(CHILD)
    assert stored is not None and stored.level == 1  # 账上存的档还是 1
    assert book(ledger, "c") == {"c": []}  # 账上没有决定
    walk(run, clock(9), clock(10))
    assert book(ledger, "p", "c") == {"p": [("等时窗末", "09:15"), ("待取", "09:45")], "c": []}
    assert listed(outlet) == ["p"]
    walk(run, clock(10, 15), clock(11))
    assert book(ledger, "c") == {"c": [("影子", "11:00")]}
    latest = ledger.unit_level_of(CHILD)
    assert latest is not None and latest.level == 0


def test_q20_5_a_child_whose_reading_is_below_the_line_is_not_waited_for(tmp_path: Path) -> None:
    """Q20-5：C 的单元档 0（读数没过线）：P 09:45 待取；C 到自己的时窗记影子。"""

    run, ledger, outlet = site(tmp_path, parent_item(), child_item(), unearned=(CHILD,))
    walk(run, clock(9), clock(11))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("待取", "09:45")],
        "c": [("影子", "11:00")],
    }
    assert listed(outlet) == ["p"]


def test_q20_6_a_child_already_recorded_as_shadow_is_not_waited_for(tmp_path: Path) -> None:
    """Q20-6（实盘 04-24）：C 已记「影子」：P 照常 09:45 待取。"""

    child = child_item(window=(clock(9, 30), clock(10, 10)))
    run, ledger, outlet = site(tmp_path, parent_item(), child, unearned=(CHILD,))
    walk(run, clock(9), clock(10))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("待取", "09:45")],
        "c": [("影子", "09:30")],
    }
    assert listed(outlet) == ["p"]


@pytest.mark.parametrize(
    "voiceless",
    [relation(status="candidate"), relation(reading=TreeReading(0.5, 0.45, large_enough=False))],
    ids=["Q20-7 候选关系", "Q20-8 巧合"],
)
def test_q20_7_and_8_a_child_that_cannot_speak_is_not_waited_for(
    tmp_path: Path, voiceless: RelationOpportunities
) -> None:
    """Q20-7、Q20-8（不能开口）：C 是候选关系、或是巧合（效应不够大）：账再高也不能开口，P 不等，09:45 待取；C 到时窗记影子。"""

    run, ledger, outlet = site(tmp_path, parent_item(), child_item(trigger=RELATION, source="事件"))
    walk(run, clock(9), clock(11), table=table_with(voiceless))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("待取", "09:45")],
        "c": [("影子", "11:00")],
    }
    assert listed(outlet) == ["p"]


def test_q20_7_extra_a_voiceless_child_does_not_stand_in_for_a_child_that_can_speak(tmp_path: Path) -> None:
    """Q20-7 补（不能开口的摆在前面）：先到的下级 V 是候选关系、本来不能开口，后到的下级 C 能开口：V 不堵 C，P 等的是 C——
    C 11:30 待取，P 11:30 上下级已说。"""

    voiceless = made("v", "s-k0281", (clock(9, 30), clock(9, 35)), trigger=RELATION, source="事件")
    kinds = KindHierarchy({PARENT: frozenset({CHILD, "s-k0281"})})
    run, ledger, outlet = site(tmp_path, parent_item(), voiceless, child_item())
    walk(run, clock(9), clock(11, 45), kinds=kinds, table=table_with(relation(status="candidate")))
    assert book(ledger, "p", "v", "c") == {
        "p": [("等时窗末", "09:15"), ("上下级已说", "11:30")],
        "v": [("影子", "09:30")],
        "c": [("等时窗末", "11:00"), ("待取", "11:30")],
    }
    assert listed(outlet) == ["c"]


# ── 不变：已被宿主取走的、别的形状 ────────────────────────────────────────────


def test_q20_10_a_child_already_taken_by_the_host_silences_the_later_parent(tmp_path: Path) -> None:
    """Q20-10（已被取走）：C 先到时窗末、已「待取」并被宿主取走，P 后到：P 记「上下级已说」。"""

    taken: set[str] = set()
    parent = made("p", PARENT, (clock(10), clock(10, 30)))
    child = child_item(window=(clock(9), clock(9, 30)))
    run, ledger, _outlet = site(tmp_path, parent, child, claimed=taken.__contains__)
    walk(run, clock(9), clock(9, 45))
    taken.add("c")
    walk(run, clock(10), clock(10, 45))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "10:00"), ("上下级已说", "10:45")],
        "c": [("等时窗末", "09:00"), ("待取", "09:45")],
    }


def test_q20_11_a_parent_already_taken_silences_a_child_formed_afterwards(tmp_path: Path) -> None:
    """Q20-11（已被取走）：P 已「待取」并被取走之后，C 才形成：C 记「上下级已说」。"""

    taken: set[str] = set()
    run, ledger, _outlet = site(tmp_path, parent_item(), claimed=taken.__contains__)
    earn(ledger, kind=CHILD, trigger="rel")
    walk(run, clock(9), clock(9, 45))
    taken.add("p")
    ledger.record_expectation(child_item(window=(clock(10, 15), clock(10, 30)), observed=clock(10)))
    walk(run, clock(10), clock(10, 45))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("待取", "09:45")],
        "c": [("等时窗末", "10:15"), ("上下级已说", "10:45")],
    }


def test_q20_13_a_child_done_before_its_window_means_the_parent_is_done_too(tmp_path: Path) -> None:
    """Q20-13：他在 C 时窗开之前做了 C：P 到时窗末记「已发生」，不等、也不说。"""

    run, ledger, outlet = site(tmp_path, parent_item(), child_item())
    rows = (Seen(clock(9, 20), clock(9, 25), CHILD, "behavior://done.md"),)
    walk(run, clock(9), clock(11), rows=rows)
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("已发生", "09:45")],
        "c": [("已发生", "11:00")],
    }
    assert listed(outlet) == []


def test_q20_14_ranges_that_do_not_overlap_are_said_separately(tmp_path: Path) -> None:
    """Q20-14：说中的范围不重叠（P 最晚时限 10:00，C 10:30 才起算）：各说各的。"""

    parent = parent_item(deadline=clock(10))
    child = child_item(window=(clock(10, 40), clock(10, 55)), observed=clock(10, 30))
    run, ledger, _outlet = site(tmp_path, parent, child)
    walk(run, clock(9), clock(11))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("待取", "09:45"), ("宿主没取", "10:15")],
        "c": [("等时窗末", "10:45"), ("待取", "11:00")],
    }


def test_q20_15_a_child_closing_after_the_parents_deadline_is_not_waited_for(tmp_path: Path) -> None:
    """Q20-15：C 的时窗末晚于 P 的最晚时限（P 最晚时限 10:30）：等不到，P 照常 09:45 待取。"""

    run, ledger, outlet = site(tmp_path, parent_item(deadline=clock(10, 30)), child_item())
    walk(run, clock(9), clock(10))
    assert book(ledger, "p", "c") == {"p": [("等时窗末", "09:15"), ("待取", "09:45")], "c": []}
    assert listed(outlet) == ["p"]


def test_q20_16_siblings_are_said_separately(tmp_path: Path) -> None:
    """Q20-16：兄弟节点（不是上下级）：各说各的。"""

    first, second = "s-k0201", "s-k0202"
    siblings = KindHierarchy({OTHER: frozenset({first, second})})
    run, ledger, _outlet = site(
        tmp_path,
        made("p", first, (clock(9, 10), clock(9, 40))),
        made("c", second, (clock(11), clock(11, 15))),
    )
    walk(run, clock(9), clock(11, 30), kinds=siblings)
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("待取", "09:45")],
        "c": [("等时窗末", "11:00"), ("待取", "11:30")],
    }


def test_q20_17_when_both_close_in_the_same_tick_the_finer_one_is_said(tmp_path: Path) -> None:
    """Q20-17：同一拍 P、C 都到时窗末：C 待取；P 上下级已说。"""

    run, ledger, outlet = site(tmp_path, parent_item(), child_item(window=(clock(9, 10), clock(9, 40))))
    walk(run, clock(9), clock(9, 45))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("上下级已说", "09:45")],
        "c": [("等时窗末", "09:15"), ("待取", "09:45")],
    }
    assert listed(outlet) == ["c"]


# ── Q20-20：下级时窗终点 + 一拍 < 上级最晚时限，上级才等 ──────────────────────


@pytest.mark.parametrize(
    "child_start",
    [(clock(10), clock(10, 20)), (clock(10), clock(10, 15))],
    ids=["Q20-20 不到一拍", "Q20-20 正好一拍"],
)
def test_q20_20_a_child_closing_too_close_to_the_parents_deadline_is_not_waited_for(
    tmp_path: Path, child_start: tuple[datetime, datetime]
) -> None:
    """Q20-20（前两种摆法）：P 最晚时限 10:30，C 时窗终点到它不到一拍（10:20）或加一拍正好等于它（10:15）：下级到时窗末之后的
    第一拍，宿主已经取不走上级的话（出口从最晚时限那一刻起不给取），等不到，P 在自己的时窗末 09:45 照常记「待取」。"""

    run, ledger, outlet = site(tmp_path, parent_item(deadline=clock(10, 30)), child_item(window=child_start))
    walk(run, clock(9), clock(9, 45))
    assert book(ledger, "p", "c") == {"p": [("等时窗末", "09:15"), ("待取", "09:45")], "c": []}
    assert listed(outlet) == ["p"]


def test_q20_20_a_child_closing_more_than_a_tick_before_the_deadline_is_waited_for(tmp_path: Path) -> None:
    """Q20-20（第三种摆法）：C 时窗终点 10:10，加一拍早于 P 最晚时限 10:30：P 等；C 10:15 记「待取」，P 同一拍记「上下级已说」。"""

    child = child_item(window=(clock(9, 50), clock(10, 10)))
    run, ledger, outlet = site(tmp_path, parent_item(deadline=clock(10, 30)), child)
    walk(run, clock(9), clock(10, 15))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("上下级已说", "10:15")],
        "c": [("等时窗末", "10:00"), ("待取", "10:15")],
    }
    assert listed(outlet) == ["c"]


def test_q20_20_when_the_child_ends_up_silent_the_parent_has_already_spoken(tmp_path: Path) -> None:
    """Q20-20（第四种摆法）：第一种摆法里 P 没等、09:45 记「待取」并问出口了；C 后来决定不说（10:30 记「上下级已说」）：
    这件事当天有人说过，说的是 P。"""

    child = child_item(window=(clock(10), clock(10, 20)))
    run, ledger, _outlet = site(tmp_path, parent_item(deadline=clock(10, 30)), child)
    walk(run, clock(9), clock(9, 45))
    ledger.record_delivery(DAY, said("p", clock(9, 50)))
    walk(run, clock(10), clock(10, 30))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("待取", "09:45"), ("说了", "09:50")],
        "c": [("等时窗末", "10:00"), ("上下级已说", "10:30")],
    }
    assert [item.expectation_id for item in ledger.deliveries_on(DAY) if item.spoken] == ["p"]


# ── Q20-21 到 Q20-23：下级这一拍决定不了时，上级等不等 ────────────────────────


def _unreadable_child_at(
    now: datetime, unreadable_at: datetime, reread: Callable[[Expectation], Awaitable[bool]]
) -> JudgeMaterials:
    """这一拍的材料：只有 ``unreadable_at`` 那一拍，C 说中的范围里有「没读懂」。"""

    plain = materials(now)
    return JudgeMaterials(
        plain.table,
        plain.rows_for,
        happened=plain.happened,
        unobserved=lambda item: now == unreadable_at and item.expectation_id == "c",
        reread=reread,
        seen=plain.seen,
        kinds=plain.kinds,
    )


def test_q20_21_the_parent_keeps_waiting_while_the_child_is_deferred_by_a_reread(tmp_path: Path) -> None:
    """Q20-21：C 到时窗末那一拍（11:30）范围里有「没读懂」、当场重读读出了内容，这一拍不决定：P 在 11:30 也不决定；
    11:45 C 记「待取」，P 同一拍记「上下级已说」。"""

    run, ledger, outlet = site(tmp_path, parent_item(), child_item())
    reread_at = clock(11, 30)
    asked: list[str] = []

    async def recovered(item: Expectation) -> bool:
        asked.append(item.expectation_id)
        return item.expectation_id == "c"

    now = clock(9)
    while now <= clock(11, 45):
        asyncio.run(run.run_once(now, _unreadable_child_at(now, reread_at, recovered)))
        if now == reread_at:
            assert book(ledger, "p", "c") == {"p": [("等时窗末", "09:15")], "c": [("等时窗末", "11:00")]}
        now += timedelta(minutes=15)
    assert asked == ["c"]
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("上下级已说", "11:45")],
        "c": [("等时窗末", "11:00"), ("待取", "11:45")],
    }
    assert listed(outlet) == ["c"]


def test_q20_22_a_child_first_seen_too_late_to_decide_in_time_is_not_waited_for(tmp_path: Path) -> None:
    """Q20-22：C 时窗 10:02–10:10 整段落在两拍之间，P、C 最晚时限都是 10:29。10:15 那一拍 C 才第一次被看到，要到 10:30 才能
    决定，那时两条都过了时限：P 不能再等，10:15 照常记「待取」。"""

    deadline = clock(10, 29)
    child = child_item(window=(clock(10, 2), clock(10, 10)), deadline=deadline)
    run, ledger, outlet = site(tmp_path, parent_item(deadline=deadline), child)
    walk(run, clock(9), clock(10, 15))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("待取", "10:15")],
        "c": [("等时窗末", "10:15")],
    }
    assert listed(outlet) == ["p"]


def test_q20_23_the_same_narrow_child_is_waited_for_when_the_parents_deadline_is_late_enough(tmp_path: Path) -> None:
    """Q20-23：同样窄的 C（10:02–10:10），P 最晚时限 11:00：P 10:15 还让着，C 10:30 记「待取」，P 10:30 记「上下级已说」。"""

    child = child_item(window=(clock(10, 2), clock(10, 10)))
    run, ledger, outlet = site(tmp_path, parent_item(deadline=clock(11)), child)
    walk(run, clock(9), clock(10, 30))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("上下级已说", "10:30")],
        "c": [("等时窗末", "10:15"), ("待取", "10:30")],
    }
    assert listed(outlet) == ["c"]


def _unreadable_everywhere(now: datetime, asked: list[tuple[datetime, str]]) -> JudgeMaterials:
    """这一拍的材料：每条预期说中的范围里都有「没读懂」；当场重读记下这一拍问到了谁，什么也读不出来。"""

    async def nothing_found(item: Expectation) -> bool:
        asked.append((now, item.expectation_id))
        return False

    plain = materials(now)
    return JudgeMaterials(
        plain.table,
        plain.rows_for,
        happened=plain.happened,
        unobserved=lambda item: True,
        reread=nothing_found,
        seen=plain.seen,
        kinds=plain.kinds,
    )


def test_q20_24_a_parent_giving_way_to_its_child_is_not_reread_while_it_waits(tmp_path: Path) -> None:
    """Q20-24：当场重读要花模型调用，只给这一拍真要决定的预期用。P 时窗末 09:45、C 时窗末 11:15，两条的范围里都有「没读懂」，
    重读什么也读不出来：P 让着 C 的那些拍（09:45 到 11:15）不为它重读，账上只有那条「等时窗末」；11:30 那一拍 P、C 各问一次。"""

    run, ledger, _outlet = site(tmp_path, parent_item(), child_item())
    asked: list[tuple[datetime, str]] = []
    now = clock(9)
    while now <= clock(11, 15):
        asyncio.run(run.run_once(now, _unreadable_everywhere(now, asked)))
        now += timedelta(minutes=15)
    assert [identity for _moment, identity in asked if identity == "p"] == []
    assert book(ledger, "p") == {"p": [("等时窗末", "09:15")]}
    asked.clear()
    last = clock(11, 30)
    asyncio.run(run.run_once(last, _unreadable_everywhere(last, asked)))
    assert sorted(identity for _moment, identity in asked) == ["c", "p"]


# ── Q20-25、Q20-26：下级账上只有「单元已选」；下级的节点不在用 ────────────────


def test_q20_25_the_parent_waits_for_a_child_that_was_only_subsumed_so_far(tmp_path: Path) -> None:
    """Q20-25：下级同一类两条：b 是候选关系（不能开口），09:30 记影子并把 m 收进单元；m 能开口，时窗 11:00–11:15，账上只有
    「单元已选」。P 时窗 10:00–10:05：10:15 不决定；m 11:30 接手记「待取」，P 11:30 记「上下级已说」，可取清单上是 m。"""

    parent = made("p", PARENT, (clock(10), clock(10, 5)))
    voiceless = made("b", CHILD, (clock(9, 30), clock(9, 40)), trigger=RELATION, source="事件")
    member = made("m", CHILD, (clock(11), clock(11, 15)), trigger="m")
    run, ledger, outlet = site(tmp_path, parent, voiceless, member)
    walk(run, clock(9, 15), clock(11, 30), table=table_with(relation(status="candidate")))
    assert book(ledger, "p", "b", "m") == {
        "p": [("等时窗末", "10:00"), ("上下级已说", "11:30")],
        "b": [("影子", "09:30")],
        "m": [("单元已选", "09:30"), ("待取", "11:30")],
    }
    assert listed(outlet) == ["m"]


def test_q20_26_a_child_whose_node_left_use_is_not_waited_for(tmp_path: Path) -> None:
    """Q20-26：下级的节点此刻不在用：它不会再开口，P 不等，照常在自己的时窗末 09:45 记「待取」。"""

    kinds = KindHierarchy({PARENT: frozenset({CHILD})}, in_use=frozenset({PARENT}))
    run, ledger, outlet = site(tmp_path, parent_item(), child_item())
    walk(run, clock(9), clock(9, 45), kinds=kinds)
    assert book(ledger, "p", "c") == {"p": [("等时窗末", "09:15"), ("待取", "09:45")], "c": []}
    assert listed(outlet) == ["p"]


# ── Q20-27：三层，中间那层在让，上面那层也跟着让 ───────────────────────────────

GRAND = "s-k0107"
THREE_LAYERS = KindHierarchy({GRAND: frozenset({PARENT, CHILD}), PARENT: frozenset({CHILD})})


def _three_layers(tmp_path: Path, child: Expectation | None) -> tuple[Any, Any, Outlet]:
    grand = made("g", GRAND, (clock(9, 10), clock(9, 40)), deadline=clock(10, 10))
    parent = parent_item()
    return site(tmp_path, grand, parent, *(() if child is None else (child,)))


def test_q20_27_a_parent_yielding_to_its_child_makes_the_grandparent_yield_too(tmp_path: Path) -> None:
    """Q20-27（裁定 101 第 8 条）：三层 G → P → C。G、P 时窗 09:10–09:40，G 最晚时限 10:10；C 时窗 11:00–11:15。同一拍里 P 正在
    让着 C，G 也跟着让：09:45 起 G、P 都不进可取清单；11:30 C 记「待取」、P 同一拍记「上下级已说」；整天 G 没有「待取」「说了」，
    只有 C 那一条说出去。"""

    run, ledger, outlet = _three_layers(tmp_path, child_item())
    now = clock(9)
    while now <= clock(12):
        asyncio.run(run.run_once(now, materials(now, kinds=THREE_LAYERS)))
        if now >= clock(9, 45):
            assert not {"g", "p"} & set(listed(outlet)), now
        now += timedelta(minutes=15)
    found = book(ledger, "g", "p", "c")
    assert found["c"] == [("等时窗末", "11:00"), ("待取", "11:30")]
    assert found["p"] == [("等时窗末", "09:15"), ("上下级已说", "11:30")]
    assert not {"待取", "说了"} & {reason for reason, _at in found["g"]}
    assert [item.expectation_id for item in ledger.deliveries_on(DAY) if item.reason == "待取" or item.spoken] == ["c"]


def test_q20_27_control_with_a_child_that_cannot_speak_the_middle_layer_is_said(tmp_path: Path) -> None:
    """Q20-27 对照：C 换成不能开口的（候选关系）：没人可让，P 09:45 照常记「待取」，G 记「上下级已说」，和不带 C 时相同。"""

    voiceless = child_item(trigger=RELATION, source="事件")
    run, ledger, _outlet = _three_layers(tmp_path / "voiceless", voiceless)
    walk(run, clock(9), clock(9, 45), kinds=THREE_LAYERS, table=table_with(relation(status="candidate")))
    alone, alone_ledger, _alone_outlet = _three_layers(tmp_path / "alone", None)
    walk(alone, clock(9), clock(9, 45), kinds=THREE_LAYERS)
    expected = {
        "g": [("等时窗末", "09:15"), ("上下级已说", "09:45")],
        "p": [("等时窗末", "09:15"), ("待取", "09:45")],
    }
    assert book(ledger, "g", "p") == book(alone_ledger, "g", "p") == expected
