"""递送那一拍（裁定 48–51、56、58）：到了时窗的预期按行为单元拢在一起，每个单元决定一次说不说，
记进账本（说没说都记、记原因）。

单元（裁定 56，``runtime.foresight_units``）：同一个行为这一刻的全部输入——到了时窗的那几条预期，加上说中的范围
和它们连成一片、还没到时窗的那几条——拢成一个单元；单元读数 = 这一组在场时历史上下一步来了多少，按依据那本账的线过线。
依据那一条照下面的流程走（时窗开 / 时窗末、模型、名额）；其余几条记「单元已选」并引依据——它们照旧各自结算，各前因的权重
（账）仍各自随结算调。同一刻后来才形成的同一类预期（说中的范围和已决定过的依据相连）也并进它，不另起一个单元。历史是换代时按
那一代语义树重标过的那一份（``runtime.foresight_history``），预测拍不扫账本。

档不存成状态：每次算出来——边的级（``foresight.ladder.climb``，从单元历史里这条边的那几刻读）与单元的档
（``foresight.units.unit_level``：单元读数、单元的提醒反馈、单元的设置）分开，各自只在变的时候记一条（账的级、``UnitLevel``）
给滞回与可见面用。预测拍不扫账本：边从换代重标过的单元历史读，提醒反馈与设置从 ``units/<行为>/`` 读（裁定 61）。

一条预期到了时窗决定一次（时窗与这一拍的区间有交集就算到了：一拍一槽，时窗比一槽短的也落得上），账上已有决定的不再决定；
只有「等时窗末」不是终局——到时窗末还没做再说（「该来了还没来」是提醒的说话时机；不分急缓，裁定 58）。时窗末那一拍
同样按拍区间算：这一拍的区间 (上一拍, 此刻] 与 (时窗终点, 最晚时限] 有交集、此刻已过终点（终点到最晚时限短于一拍也落得上）；
终点与最晚时限重合时是越过终点的那一拍，开时窗那一拍就已越过终点时是它的下一拍——拍点照常走，「等时窗末」一定有下文。
中间停过机、(终点, 最晚时限] 整段落在没跑的那几拍里的不再决定（过了时限的不再说，结算照常结它）。到时窗末那一拍
按依据与它当时一并决定的那几条重算一次单元（窗里设了「以后别提了」就不说了），这时才问模型、才写那句话：
话里的"今天""这会儿"说的是开口那一刻。上级和下级各凭自己的账决定，不互相等（裁定 105）。
到了时窗时这件事已经做了记「已发生」；同一类今天已经说过一句的记「同类已说」；没有宿主活着的记「不可达」（最近
一段没有宿主来取过提醒就不问模型、不写话，结算按没说过处理）。每日上限按决定的日子数，宿主取得走的「待取」也占名额（取不走、没取走的不占，裁定 104）。
记哪一种原因、先后怎么排是纯规则（``foresight.delivery``）；这里分拨、取材料、问模型、落账。

模型只在有宿主活着时问（没人来取，问了也说不出口）：像不像、先说哪个、一句话——只问这一拍真要开口的（提醒档、时窗末还没做、
同一类今天还没说过、名额没满）。一拍一个小包。两遍都说不像的记「模型判不像」；像的按模型的先后占名额、用模型的话；模型不可用
按算法顺序、模板措辞。
依据没说成，成员接着说（裁定 92）：依据的话没对人说出口（记了「不可达」「宿主没取」，或停在「等时窗末」「待取」
过了最晚时限；或依据本来就不能开口——候选关系、巧合，单元只能从它里面取、记了影子，裁定 56；``foresight.delivery.unsaid``），
被它「单元已选」吸收、自己最晚时限还在、之后再没决定过的成员接手这一刻：按同一组输入
重算单元、从这几条里选一条（``units.basis``），它到了自己的时窗末照常走时窗末那一套（做了、影子、同类已说、不可达、名额、待取……）。
同一刻只接手一条：已经接手的那条还没说成（待取、等宿主）或自己决定了不说，别的成员不再接手；它也没说成才轮到下一条。「同类已说」与
每日上限不把同一刻里没说成的那几条算进去（同一刻只说一次，它们一句也没说出口）；别的单元照算。结算不变：成员说出口了按它自己的说出
时刻结、反馈由它写一笔，依据那条照旧按没说过结。
送达在本层之外（``runtime.foresight_outlet``）：这里产出的是"递送请求 + 档 + 话"，记「待取」、每拍把可取的请求清单交给出口；
「说了」由出口在转写里认出提问后追加。「待取」之后每拍只做三件事（``runtime.foresight_withdrawal``）：做了「已发生」、过时限「宿主没取」、依据失效撤下。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta

from habitus.foresight import Evidence, Seen, Standing, Tolerance
from habitus.foresight.delivery import on_closing, on_opening
from habitus.foresight.hierarchy import FLAT, KindHierarchy
from habitus.foresight.judge import Judge, Judgement, UnitInput, Verdict
from habitus.foresight.ledger import AWAITING_HOST, Delivery, Expectation
from habitus.foresight.opportunities import OpportunityTable
from habitus.foresight.units import UNIT, UNIT_CITE, voiced
from habitus.runtime.foresight_delivery_due import DueScan, scan_due
from habitus.runtime.foresight_delivery_facts import circumstances, returned
from habitus.runtime.foresight_delivery_judge import ask, ordered_closing, text_for, unit_inputs
from habitus.runtime.foresight_delivery_standing import StandingReader, TickReadings
from habitus.runtime.foresight_delivery_tally import Tally, spoken_today
from habitus.runtime.foresight_delivery_views import AccountView, DeliveryViews, OmissionView
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_outlet import OpenRequest, OutletPort
from habitus.runtime.foresight_units import Unit, UnitPlanner, UnitView
from habitus.runtime.foresight_withdrawal import awaiting_outcome


@dataclass(frozen=True)
class DeliveryReport:
    at: datetime
    decided: int
    spoken: int
    by_reason: dict[str, int] = field(default_factory=dict)
    by_level: dict[int, int] = field(default_factory=dict)
    #: 模型这一拍问了几个候选（像不像）、调了几次、留下的信号。
    judged: int = 0
    calls: int = 0
    signals: tuple[str, ...] = ()
    judge_failed: bool = False
    #: 这一拍有几本账换了档。
    level_changes: int = 0
    #: 账本里读不出来、被跳过的文件数。
    unreadable: int = 0
    #: 这一拍末尾交给出口的可取请求数（「待取」还没被问出口、也没到最晚时限的）。
    open_requests: int = 0


@dataclass(frozen=True)
class JudgeMaterials:
    """模型小包要的材料，由组合根按这一拍准备：机会表（历史那几次）、按出处取行（原话）、这件事这回是不是已经做了
    （做了就不递：那句话说的事他刚做完；结算会记准时 / 早到）。"""

    table: OpportunityTable | None
    rows_for: Callable[[Iterable[str]], Mapping[str, Seen]]
    happened: Callable[[Expectation], bool] = lambda item: False
    #: 当地日历对某一天的说法（进小包给模型判像不像；没有日历数据时恒为空）。
    day_note: Callable[[date], str | None] = lambda day: None
    #: 这条预期说中的范围里（起算 → 此刻）有没有「没读懂」（裁定 67、68）。
    unobserved: Callable[[Expectation], bool] = lambda item: False
    #: 时窗末用得着就当场重读那段「没读懂」（裁定 68）：读出了什么为真——这一拍不决定，下一拍按读出的再判；
    #: 没接（预测层单独跑、没有重试堆）就是 None，有「没读懂」直接记「说不准」。
    reread: Callable[[Expectation], Awaitable[bool]] | None = None
    #: 这一拍读到的行（最近几天已封口的与还没封口、现场归上类的）：开口前再核事件触发的前因从这里按实例找那一行
    #: （与结算同一个找法，``ledger.settle.event_antecedent_holds``）。
    seen: Sequence[Seen] = ()
    since: datetime | None = None  # 区间起点（``foresight_tick.delivery_since``）；None 是此刻 − 1 拍
    #: 这一拍开头取的上下级：节点在不在用、"这回做了"（下级做了对它是"不知道"，这一次不说；裁定 105）都按它。
    kinds: KindHierarchy = FLAT


class DeliveryStage:
    def __init__(
        self,
        ledger: ForesightLedgerStore,
        *,
        tolerance: Tolerance,
        evidence: Evidence,
        daily_cap: int,
        judge: Judge | None = None,
        tick_seconds: float = 900.0,
        outlet: OutletPort | None = None,
        unit_prior_share: float = 1 / 3,
        history: HistoryStore | None = None,
        lane: str = "session",
        claimed: Callable[[str], bool] = lambda request_id: False,
    ) -> None:
        if daily_cap < 1:
            raise ValueError("daily_cap must be a positive integer")
        if tick_seconds <= 0:
            raise ValueError("tick_seconds must be a positive number")
        self.ledger = ledger
        self.tolerance = tolerance
        self.evidence = evidence
        self.daily_cap, self.judge = daily_cap, judge
        self.tick = timedelta(seconds=float(tick_seconds))
        #: 提醒出口（没有 = 没有宿主活着：提醒档到时窗末都「不可达」，也不问模型）；``claimed``：出口上这条请求取走过没有。
        self.outlet, self.claimed = outlet, claimed
        self.units = UnitPlanner(history, evidence=evidence, prior_share=unit_prior_share, lane=lane)
        self.readings = StandingReader(ledger, self.units, tolerance=tolerance, evidence=evidence)
        #: 只读可见面（账、单元、漏报）：边的位置与递送引擎同一个读法。
        self.views = DeliveryViews(ledger, standing=self.readings.standing)

    async def run_once(self, now: datetime, materials: JudgeMaterials | None = None) -> DeliveryReport:
        if now.utcoffset() is None:
            raise ValueError("now must be a timezone-aware datetime")
        table = materials.table if materials is not None else None
        since = (materials.since if materials is not None else None) or now.astimezone(UTC) - self.tick
        found = scan_due(self.ledger, now, since, voiced=lambda item: voiced(table, item), claimed=self.claimed)
        tick = self.readings.for_tick(now, table)
        plan = _plan(found, tick)
        tally = Tally(
            now,
            spoken_today(self.ledger, now),
            alive=self.outlet is not None and self.outlet.alive(now),
            same_moment=plan.moment_of,
        )
        self._carry_over(tally, found, plan, materials, now)
        judgement = await self._decide(found, plan, tally, materials, now)
        changes = self.readings.record_levels(tick.accounts, plan.every, now)
        open_requests = self._publish(found, tally, materials, now)
        return _report(now, tally, judgement, changes, len(self.ledger.take_unreadable()), open_requests)

    def _carry_over(
        self, tally: Tally, found: DueScan, plan: _Plan, materials: JudgeMaterials | None, now: datetime
    ) -> None:
        """不用开口就能落的三拨：「待取」的下文、单元替成员一并决定的、并进已决定过的单元的。"""

        for item, pending in found.awaiting:
            self._record(tally, item, awaiting_outcome(self.claimed, item, pending, materials, now))
        for unit in plan.units:
            for member in unit.members:
                self._subsume(tally, member, unit.basis.expectation_id, unit.level, unit)
        for item, cited, level in found.joins:
            self._subsume(tally, item, cited, level)

    async def _decide(
        self, found: DueScan, plan: _Plan, tally: Tally, materials: JudgeMaterials | None, now: datetime
    ) -> Judgement | None:
        dues = (*found.closing, *plan.handed)
        deferred = await self._reread(dues, materials)  # 当场重读：推迟到下一拍的这一拍不决定
        closing = tuple(pair for pair in dues if pair[0].expectation_id not in deferred)
        speaking = tuple(item for item, _pending in closing if now <= item.deadline)  # 过了时限的不说、不问模型
        judgement = await self._judge(speaking, plan.standings, materials, now, tally, plan.inputs)
        for item, pending in ordered_closing(closing, judgement):
            verdict = judgement.verdicts.get(item.expectation_id) if judgement is not None else None
            unit = plan.closing_units[item.expectation_id]
            self._record(tally, item, self._closing(item, pending, unit, verdict, materials, tally, plan.inputs))
        for unit in plan.units:
            self._record(tally, unit.basis, self._opening(unit, materials, tally))
        return judgement

    def _publish(self, found: DueScan, tally: Tally, materials: JudgeMaterials | None, now: datetime) -> int:
        """这一拍末尾把可取的请求清单交给出口（只换这一支的）：之前「待取」还没终局的，加上这一拍刚记「待取」的。"""

        if self.outlet is None:
            return 0
        rows = [
            OpenRequest(
                item.expectation_id,
                item.kind_token,
                pending.text or "",
                item.deadline,
                item.formed_at,
                pending.level,
                item.day,
                self.units.lane,  # 这一支：「说了」与回话写回这一支的账本
            )
            for item, pending in (*found.awaiting, *tally.awaiting)
            if item.expectation_id not in tally.withdrawn
            if now <= item.deadline and pending.text and not (materials is not None and materials.happened(item))
        ]
        return self.outlet.publish(rows, now=now)

    # ── 一条预期的决定（原因的先后在 ``foresight.delivery``，这里只取材料）────────────────

    def _opening(self, unit: Unit, materials: JudgeMaterials | None, tally: Tally) -> Delivery:
        item = unit.basis
        facts = circumstances(item, materials, self.ledger, self.evidence)
        delivery = on_opening(item, unit.standing, now=tally.now, facts=facts)
        return replace(delivery, loss_readings=unit.readings_for(item))  # 单元这一拍的读数（裁定 114，只给尺子看）

    def _closing(
        self,
        item: Expectation,
        pending: Delivery,
        unit: Unit,
        verdict: Verdict | None,
        materials: JudgeMaterials | None,
        tally: Tally,
        inputs: Mapping[str, Sequence[UnitInput]],
    ) -> Delivery:
        spoken_today, spoken_kinds = tally.spoken_for(item.expectation_id)
        delivery = on_closing(
            item,
            pending,
            unit.standing,
            now=tally.now,
            facts=circumstances(item, materials, self.ledger, self.evidence),
            kept=verdict.kept if verdict is not None else None,
            evidence=verdict.evidence if verdict is not None else (),
            spoken_kinds=spoken_kinds,
            alive=tally.alive,
            spoken_today=spoken_today,
            daily_cap=self.daily_cap,
            text=lambda: text_for(item, verdict, materials, tally.now, inputs),
        )
        return replace(delivery, loss_readings=unit.readings_for(item))  # 时窗末重算的那一次（裁定 114）

    @staticmethod
    async def _reread(
        closing: Sequence[tuple[Expectation, Delivery]], materials: JudgeMaterials | None
    ) -> frozenset[str]:
        """时窗末要说、还没看到他做、范围里有「没读懂」的，先当场重读那一段（裁定 68）；读出了什么的这一拍不决定。"""

        if materials is None or materials.reread is None:
            return frozenset()
        found: set[str] = set()
        for item, _pending in closing:
            if materials.happened(item) or not materials.unobserved(item):
                continue
            if await materials.reread(item):
                found.add(item.expectation_id)
        return frozenset(found)

    def _subsume(self, tally: Tally, item: Expectation, basis_id: str, level: int, unit: Unit | None = None) -> None:
        """同一刻同一个行为的另一个输入：单元替它一并决定，记「单元已选」并引依据。这一拍算了单元（``unit``）的带上读数；
        并进已决定过的单元的没有算，不带（"没量到"，裁定 114）。"""

        cited = (f"{UNIT_CITE}{basis_id}",)
        delivery = Delivery(item.expectation_id, tally.now, level, False, UNIT, 0.0, evidence=cited)
        self._record(tally, item, replace(delivery, loss_readings=None if unit is None else unit.readings_for(item)))

    def _record(self, tally: Tally, item: Expectation, delivery: Delivery | None) -> None:
        if delivery is None or not self.ledger.record_delivery(item.day, delivery):
            return
        tally.count(delivery, item)
        if delivery.reason == AWAITING_HOST:
            tally.awaiting.append((item, delivery))

    # ── 模型 ────────────────────────────────────────────────────────────────

    async def _judge(
        self,
        due: tuple[Expectation, ...],
        standings: Mapping[str, Standing],
        materials: JudgeMaterials | None,
        now: datetime,
        tally: Tally,
        inputs: Mapping[str, Sequence[UnitInput]],
    ) -> Judgement | None:
        """这一拍真要开口的那几个交模型（提醒档、还没做、同一类今天还没说过、名额没满）；没有模型、没有材料、没有宿主活着就不问。"""

        if self.judge is None or materials is None or not tally.alive:
            return None
        asked = tuple(
            item
            for item in due
            if standings[item.expectation_id].level >= 1
            and tally.room_for(item, self.daily_cap)
            and not materials.happened(item)
            and not returned(item, standings[item.expectation_id], materials, self.ledger, self.evidence)
        )
        if not asked:
            return None
        levels = {item.expectation_id: standings[item.expectation_id].level for item in asked}
        return await ask(self.judge, asked, levels, materials, now, inputs)

    # ── 可见面的只读口（``DeliveryViews``）──────────────────────────────────

    def standing_of(self, expectation: Expectation, now: datetime, table: OpportunityTable | None = None) -> Standing:
        """这条预期那条边此刻的位置（运维用）。"""

        return self.readings.standing(expectation.account, expectation.kind_token, now, table)

    def account_views(
        self, now: datetime, table: OpportunityTable | None = None, *, recent: int = 5
    ) -> tuple[AccountView, ...]:
        """全部账此刻的样子。"""

        return self.views.account_views(now, table, recent=recent)

    def unit_views(self, now: datetime, table: OpportunityTable | None) -> tuple[UnitView, ...]:
        """全部行为单元此刻的样子。"""

        return self.views.unit_views(now, table)

    def omission_views(self, now: datetime, *, days: int = 30) -> tuple[OmissionView, ...]:
        """最近 ``days`` 天的漏报按类汇总。"""

        return self.views.omission_views(now, days=days)


@dataclass(frozen=True)
class _Plan:
    """这一拍要决定的单元：时窗开的、时窗末按依据与它当时一并决定的那几条重算的、成员接手的。"""

    units: tuple[Unit, ...]
    closing_units: Mapping[str, Unit]
    handed: tuple[tuple[Expectation, Delivery], ...]
    #: 接手那一刻的全部输入（「同类已说」与每日上限不把同一刻里没说成的那几条算进去）。
    moment_of: Mapping[str, frozenset[str]]
    every: tuple[Unit, ...]
    standings: Mapping[str, Standing]
    inputs: Mapping[str, tuple[UnitInput, ...]]


def _plan(found: DueScan, tick: TickReadings) -> _Plan:
    units = tick.plan(found.opened, found.waiting)
    # 时窗末：按依据与它当时一并决定的那几条重算单元；只按依据那本账自己的档判的话，几个前因一起才稳的轻事
    # 时窗开时说该说、时窗末又成了影子
    closing_units = {
        item.expectation_id: tick.unit_of((item, *found.members.get(item.expectation_id, ())), (item.expectation_id,))
        for item, _pending in found.closing
    }
    # 依据没说成：同一刻里接手的那条到了自己的时窗末，按同一组输入重算单元、照时窗末那一套走（``due`` 只有到了时窗末的成员）
    handed: list[tuple[Expectation, Delivery]] = []
    moment_of: dict[str, frozenset[str]] = {}
    for group, due in found.handovers:
        unit = tick.unit_of(group, due)
        closing_units[unit.basis.expectation_id] = unit
        handed.append((unit.basis, found.subsumed[unit.basis.expectation_id]))
        moment_of[unit.basis.expectation_id] = frozenset(item.expectation_id for item in group)
    every = (*units, *closing_units.values())
    return _Plan(
        units=units,
        closing_units=closing_units,
        handed=tuple(handed),
        moment_of=moment_of,
        every=every,
        standings={unit.basis.expectation_id: unit.standing for unit in every},
        inputs={unit.basis.expectation_id: unit_inputs(unit) for unit in every},
    )


def _report(
    now: datetime, tally: Tally, judgement: Judgement | None, changes: int, unreadable: int, open_requests: int
) -> DeliveryReport:
    return DeliveryReport(
        now,
        tally.decided,
        tally.spoken,
        tally.by_reason,
        tally.by_level,
        judged=0 if judgement is None else len(judgement.verdicts) if not judgement.failed else len(judgement.order),
        calls=0 if judgement is None else judgement.calls,
        signals=() if judgement is None else tuple(judgement.signals),
        judge_failed=judgement is not None and judgement.failed,
        level_changes=changes,
        unreadable=unreadable,
        open_requests=open_requests,
    )


__all__ = ["AccountView", "DeliveryReport", "DeliveryStage", "JudgeMaterials", "OmissionView"]
