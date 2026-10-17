"""结算这一拍（夜批，排在预测树重建之后）：对着行为树给已到最晚时限的预期一个结果，给过期待决的补裁决，给定稿日查漏报。

"那天定稿了"只有归约说了算（``BehaviorReductionRunner.closed_days``）。一条预期在它最晚时限那天定稿之后才结算，
结算只做一次；过期待决的在之后每晚看一次有没有补做（这本账下一条预期形成之前：每个机会都建预期之后，同一本账的下一条
就是这条触发自己的下一次机会，裁定 49）。漏报按定稿日查一次，查过就标记。一天里的预期都结了、过期待决的裁决再也不会变，
就标这一天结完（``mark_final``），之后每晚只读还要看的那一截，账本再长也一样。

结算与裁决落账时顺手往这本账上写事实（裁定 48、50、51、58）：**预测轴**——准时 / 早到 / 晚了但做了是好、过期待决是坏，
后补再写一条打折的坏覆盖它（补做那天离上一次发生隔了几天，在这一类历史间隔里的位置）；提醒后才做的是中性（提醒的功劳不算
习惯本身）。强度按这本账自己的历史分布算（``foresight.strength``）：偏差在历史偏差里的位置、补做那天离上次多久在这一类历史
间隔里的位置。**提醒组**——说了之后做了是好、说了没做（落空）是坏；**扣发组**——随机扣下没说之后做了是好、没做是坏；这两组
单位权重，提醒的效果是两组完成率之差。
账本里读不出来的文件跳过、条数进观测，不让一个坏文件停掉整晚。

词表迁移做到一半时整拍不做（树上的行一半新编号一半旧编号），下一拍再来。结算失败自己留观测，不许拖着后面一夜不跑。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Protocol, runtime_checkable

from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.foresight import Evidence, Standing, Tolerance, climb, current_facts
from habitus.foresight.causes import AGAIN, CITE
from habitus.foresight.diagnosis import POSTPONED, Diagnosis, Followed, MissFacts, Settled, diagnose
from habitus.foresight.ladder import HELD_OUT, PREDICTION, REMINDED, REVISIONS
from habitus.foresight.ledger import (
    FACT_AXES,
    AccountFact,
    Delivery,
    Expectation,
    Ruling,
    Settlement,
    antecedent_holds,
    omissions,
    rule_overdue,
    settle,
)
from habitus.foresight.opportunities import OpportunityTable
from habitus.foresight.strength import MIN_HISTORY, catch_up_strength, gaps_of, offset_strength
from habitus.foresight.units import UNIT
from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.scene import DayTypeCalendar
from habitus.scene.views import DayIndexCache, FlowRow
from habitus.series.reader import admitted

#: 前因不成立时记的事实种类（权重 0）。
VOID = "作废"
#: 随机扣下没说的递送原因。
HELD = "随机扣发"


@dataclass(frozen=True)
class Asked:
    """这条预期递送上发生过什么：第一次说出口的时刻（没说过是 None）、有没有被随机扣下。"""

    spoken_at: datetime | None
    held: bool

    @staticmethod
    def of(deliveries: Iterable[Delivery], expectation_id: str) -> Asked:
        mine = [item for item in deliveries if item.expectation_id == expectation_id]
        spoken = [item.decided_at for item in mine if item.spoken]
        return Asked(min(spoken) if spoken else None, any(item.reason == HELD for item in mine))


@runtime_checkable
class BehaviorReadSide(Protocol):
    """结算要读的那棵行为树、主体与日历（预测层的装配器就是这个形状）。"""

    @property
    def behavior_tree(self) -> BehaviorTree: ...

    @property
    def subject(self) -> str: ...

    @property
    def calendar(self) -> DayTypeCalendar: ...


@dataclass(frozen=True)
class SettlementReport:
    settled: int
    outcomes: dict[str, int]
    rulings: int
    omissions: int
    #: 有未结算的预期、但还没定稿的日子。
    pending_days: tuple[date, ...]


class SettlementStage:
    def __init__(
        self,
        ledger: ForesightLedgerStore,
        read_side: BehaviorReadSide,
        *,
        closed_days: Callable[[], Iterable[date]],
        current: Callable[[str], Sequence[str]],
        migrating: Callable[[], bool],
        evidence: Evidence,
        opportunities: OpportunityStore | None = None,
        lane: str = "session",
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
        tolerance: Tolerance | None = None,
    ) -> None:
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        if not isinstance(read_side, BehaviorReadSide):
            raise TypeError("read_side must expose the behaviour tree, subject and calendar")
        if not callable(closed_days) or not callable(current) or not callable(migrating):
            raise TypeError("closed_days, current and migrating must be callables")
        self.ledger = ledger
        self.read_side = read_side
        self.closed_days = closed_days
        # 编号 → 现在对应哪些在用编号（词表的 ``current_ids``）：预期记的是形成时的编号，结算时词表可能已拆改
        self.current = current
        self.migrating = migrating
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence")
        self.evidence = evidence
        if opportunities is not None and not isinstance(opportunities, OpportunityStore):
            raise TypeError("opportunities must be an OpportunityStore or None")
        # 机会表的日历给补做强度用（这一类历史上相邻两次隔几天）；没有就只有方向
        self.opportunities = opportunities
        self.lane = lane
        self.observer: Observer = observer if observer is not None else NullObserver()
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        if tolerance is not None and not isinstance(tolerance, Tolerance):
            raise TypeError("tolerance must be a Tolerance or None")
        #: 诊断要知道每本账的线（预算内那一关）；没给就不在夜批观测里出诊断。
        self.tolerance = tolerance

    def run_once(self) -> SettlementReport:
        started = time.monotonic()
        try:
            report = self._run()
        except Exception as exc:
            self._observe(ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started)
            raise
        diagnosis = self._nightly_diagnosis()
        self._observe(
            ObservationStatus.SUCCESS,
            {
                **diagnosis,
                "settled": report.settled,
                "rulings": report.rulings,
                "omissions": report.omissions,
                "pending_days": len(report.pending_days),
                "unreadable": len(self.ledger.take_unreadable()),
                **{f"outcome_{key}": value for key, value in report.outcomes.items()},
            },
            started,
        )
        return report

    def _run(self) -> SettlementReport:
        days = self.ledger.open_days()
        if self.migrating():
            waiting = tuple(day for day in days if self.ledger.unsettled_expectations(day))
            return SettlementReport(settled=0, outcomes={}, rulings=0, omissions=0, pending_days=waiting)
        closed = frozenset(self.closed_days())
        cache = DayIndexCache(
            self.read_side.behavior_tree,
            subject=self.read_side.subject,
            admits=admitted,
            calendar=self.read_side.calendar,
        )
        now = self._clock()
        outcomes: dict[str, int] = {}
        rulings = 0
        pending: list[date] = []
        formed = self._formed_by_account(days)
        table = self.opportunities.latest(self.lane) if self.opportunities is not None else None
        last_closed = max(closed) if closed else None
        waiting_rulings: set[date] = set()
        for day in days:
            for item in self.ledger.unsettled_expectations(day):
                if item.deadline.date() not in closed:
                    if day not in pending:
                        pending.append(day)
                    continue
                asked = Asked.of(self.ledger.deliveries_on(day), item.expectation_id)
                result = settle(
                    item,
                    _rows(cache, item.formed_at.date(), item.deadline.date()),
                    spoken_at=asked.spoken_at,
                    settled_at=now,
                    current=self.current,
                )
                # 先写事实再写结算：中间断了，重来时这条还没结、照结一遍，事实的身份由预期与形成时刻定、重写幂等；
                # 反过来的顺序会让结过的这条再也不被看到，事实永远补不上。前因不成立的这次机会不存在：只记作废（权重 0）
                # 断在两次写之间、重来时：上次写下的原判就是这次机会的事实，不再算一遍（这期间账上又多了别的事实，
                # 重算出的强度可能不同，同一身份写不同内容会让整晚卡死）
                if not self._settled_facts_exist(item):
                    for fact in self._facts_for(
                        item, result, asked=asked, cache=cache, table=table, known_by=item.deadline
                    ):
                        self.ledger.append_fact(fact)
                self.ledger.record_settlement(result)
                outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
            count, undecided = self._rule(day, cache, last_closed, now, formed, table)
            rulings += count
            if undecided:
                waiting_rulings.add(day)
        found = self._omissions(closed, cache, now, days)
        for day in days:
            if day not in pending and day not in waiting_rulings and not self.ledger.unsettled_expectations(day):
                self.ledger.mark_final(day, at=now)
        return SettlementReport(
            settled=sum(outcomes.values()),
            outcomes=outcomes,
            rulings=rulings,
            omissions=found,
            pending_days=tuple(pending),
        )

    def _rule(
        self,
        day: date,
        cache: DayIndexCache,
        last_closed: date | None,
        now: datetime,
        formed: dict[str, list[datetime]],
        table: OpportunityTable | None,
    ) -> tuple[int, bool]:
        """这一天里结成「过期待决」、还没有裁决的预期：最晚时限之后到看得见的尽头（这本账下一条预期形成之前）有没有补做。
        返回（这一拍裁了几条，还有没有以后可能被裁的）。以后再也裁不出不同结果的不再等：下一次机会已经形成而且那天已定稿，
        或者离上次发生已经隔了不短于这一类最长的正常间隔（那之后补做的强度是 1，和过期待决一样坏）。"""

        settled = {s.expectation_id for s in self.ledger.settlements_on(day)}
        ruled = {r.expectation_id for r in self.ledger.rulings_on(day)}
        by_id = {item.expectation_id: item for item in self.ledger.expectations_on(day)}
        # 过期待决按账上现在的那条预测轴事实认，不按结算记录：重结可能把它改成了别的（作废、准时），也可能改回了过期待决
        overdue = {
            expectation_id
            for expectation_id in settled - ruled
            if expectation_id in by_id
            and (fact := self._current_prediction(by_id[expectation_id])) is not None
            and fact.kind == "过期待决"
        }
        if not overdue:
            return 0, False
        count = 0
        waiting = False
        for expectation_id in sorted(overdue):
            item = by_id[expectation_id]
            if last_closed is None or last_closed <= item.deadline.date():
                waiting = waiting or not _ruling_moot(item, table)
                continue
            before = next((at for at in formed.get(item.account, ()) if at > item.deadline), None)
            ruling = rule_overdue(
                item, _rows(cache, item.deadline.date(), last_closed), before=before, ruled_at=now, current=self.current
            )
            if ruling is not None:
                self.ledger.append_fact(self._catch_up_fact(item, ruling, table))
                if self.ledger.record_ruling(day, ruling):
                    count += 1
                continue
            if not _ruling_settled(item, before, last_closed, table):
                waiting = True
        return count, waiting

    def _facts_for(
        self,
        item: Expectation,
        result: Settlement,
        *,
        asked: Asked,
        cache: DayIndexCache,
        table: OpportunityTable | None,
        known_by: datetime,
    ) -> tuple[AccountFact, ...]:
        """一次结算往账上写哪几条事实。前因不成立：这次机会不存在，只记作废（权重 0）；窗口里有观测空白又没看到发生：他可能做了、
        我们没看到，记中性（裁定 54）；不是这条自己的提醒引出的、可那次发生是别的账的提醒引出来的：不是这本账的自然命中，记一笔强度 0 的
        「提醒后发生」（做了，但不算证据）；其余按结果与强度（历史只取 ``known_by`` 之前已经写下的）。"""

        if not self._antecedent_holds(item, cache, table):
            return (AccountFact(item.account, item.formed_at, PREDICTION, False, 0.0, VOID, item.expectation_id),)
        if result.outcome == "过期待决" and _unobserved(item, cache):
            return (AccountFact(item.account, item.formed_at, PREDICTION, False, 0.0, "过期待决", item.expectation_id),)
        if result.outcome != "提醒后发生" and result.occurrence_uri is not None and _prompted(cache, result.occurrence_uri):
            return (AccountFact(item.account, item.formed_at, PREDICTION, True, 0.0, "提醒后发生", item.expectation_id),)
        return self._settlement_facts(item, result, asked=asked, known_by=known_by)

    def resettle(self, days: Iterable[date]) -> int:
        """修了 bug 之后：用当前代码把这些日子里已结的预期重结一遍，和账上现有的不一样的追加改判（机会的时刻不变，写下的时刻取
        此刻，后写的覆盖先写的；旧记录不改，裁定 54）。新结果少了一条轴（比如原来说了没做、现在作废）就给那条轴补一笔强度 0 的，
        把旧的盖掉。只碰三条轴上的结算事实，不碰后补、回话这类之后才有的改判。改回过期待决的日子若已标结完，重新打开，
        让之后每晚照常看有没有补做。返回改了几条预期。"""

        cache = DayIndexCache(
            self.read_side.behavior_tree,
            subject=self.read_side.subject,
            admits=admitted,
            calendar=self.read_side.calendar,
        )
        table = self.opportunities.latest(self.lane) if self.opportunities is not None else None
        now = self._clock()
        revised = 0
        for day in sorted(set(days)):
            settled = {item.expectation_id for item in self.ledger.settlements_on(day)}
            deliveries = self.ledger.deliveries_on(day)
            reopen = changed_here = False
            for item in self.ledger.expectations_on(day):
                if item.expectation_id not in settled:
                    continue
                asked = Asked.of(deliveries, item.expectation_id)
                result = settle(
                    item,
                    _rows(cache, item.formed_at.date(), item.deadline.date()),
                    spoken_at=asked.spoken_at,
                    settled_at=now,
                    current=self.current,
                )
                fresh = {
                    fact.axis: fact
                    for fact in self._facts_for(
                        item,
                        result,
                        asked=asked,
                        cache=cache,
                        table=table,
                        known_by=now,
                    )
                }
                latest = {
                    fact.axis: fact
                    for fact in current_facts(self.ledger.facts_of(item.account))
                    if fact.expectation_id == item.expectation_id
                }
                prediction = fresh[PREDICTION]
                changed = False
                for axis in FACT_AXES:
                    old, new = latest.get(axis), fresh.get(axis)
                    if old is not None and old.kind in _LATER_RULINGS:
                        continue
                    if new is None:
                        if old is None or old.strength == 0.0:
                            continue
                        # 这条轴这次不该有事实了：补一笔同种类、强度 0 的盖掉旧的
                        new = replace(old, good=False, strength=0.0, kind=prediction.kind, offset_minutes=None)
                    if old is not None and _same(old, new):
                        continue
                    self.ledger.append_fact(replace(new, at=item.formed_at, written_at=now))
                    changed = True
                if changed:
                    revised += 1
                    changed_here = True
                    if prediction.kind == "过期待决" and prediction.strength > 0.0:
                        reopen = True
            if reopen and self.ledger.is_final(day):
                self.ledger.reopen(day)
            elif changed_here:
                self.ledger.forget_explained(day)  # 改判后的那条不带原因：让找原因那一拍再找一次
        return revised

    # ── 前因 ────────────────────────────────────────────────────────────────

    def _antecedent_holds(self, item: Expectation, cache: DayIndexCache, table: OpportunityTable | None) -> bool:
        """事件触发：前因那条记录（按开始时刻找，封口后的样子）现在还属于这条关系的前因类吗。"""

        if item.source != "事件":
            return True
        relation = next((r for r in table.relations if r.key == item.trigger), None) if table is not None else None
        classes = relation.antecedent_classes if relation is not None else None
        try:
            started = datetime.fromisoformat(item.trigger_instance).astimezone(item.formed_at.tzinfo)
        except ValueError:
            return True
        moment = started.astimezone(UTC)
        rows = [row for offset in (0, -1, 1) for row in cache.day(started.date() + timedelta(days=offset)).rows]
        # 先找开始时刻对得上的；封口时前因的开始可能被更正挪了一点，就找跨过那一刻的那一行（行为有先后、不会两行同时）
        found = next((row.kind_token for row in rows if row.at.astimezone(UTC) == moment), None)
        if found is None:
            found = next(
                (
                    row.kind_token
                    for row in rows
                    if row.at.astimezone(UTC) <= moment <= row.last_observed_at.astimezone(UTC)
                ),
                None,
            )
        return antecedent_holds(item, found, classes, current=self.current)

    def _current_prediction(self, item: Expectation) -> AccountFact | None:
        """这条预期此刻在预测轴上算数的那一条（后写的覆盖先写的）。"""

        mine = [fact for fact in self.ledger.facts_of(item.account) if fact.expectation_id == item.expectation_id]
        return next((fact for fact in current_facts(mine) if fact.axis == PREDICTION), None)

    def _settled_facts_exist(self, item: Expectation) -> bool:
        """这条预期的原判（结算时写的、不是改判）已经写过了。"""

        return any(
            fact.expectation_id == item.expectation_id and fact.written_at is None
            for fact in self.ledger.facts_of(item.account)
        )

    # ── 账上的事实 ──────────────────────────────────────────────────────────

    def _settlement_facts(
        self, item: Expectation, result: Settlement, *, asked: Asked, known_by: datetime
    ) -> tuple[AccountFact, ...]:
        """预测轴一笔，说过的再在提醒组记一笔、随机扣下的在扣发组记一笔（裁定 58 第 5–7 条）：提醒组与扣发组单位权重；
        提醒后才做的在预测轴是中性（强度 0）——提醒的功劳不算这个习惯本身，提醒有没有用看提醒组与扣发组之差。

        事实的时刻是这次机会形成的时刻（不是结算那一刻）：遗忘按机会算，「这次不用」压回之前形成的机会不算压回之后的证据。
        历史分布只看这本账在这次机会**之前**的预期（之后的机会拿来比这一次是用未来解释过去），且只认 ``known_by`` 之前已经写下的
        （结算时取最晚时限：哪一晚结、断没断过重来，算出的都是同一份；重结时取此刻：用修过的那一份）。"""

        account = item.account
        at = item.formed_at
        known = [fact for fact in self.ledger.facts_of(account) if fact.precedence <= known_by]
        prior = tuple(
            fact for fact in current_facts(known) if fact.expectation_id != item.expectation_id and fact.at < at
        )
        outcome = result.outcome
        if outcome in ("准时", "早到", "晚了但做了"):
            offset = 0 if outcome == "准时" else int(result.offset_minutes or 0)
            history = [
                fact.offset_minutes or 0
                for fact in prior
                if fact.axis == PREDICTION and fact.good and fact.kind in ("准时", "早到", "晚了但做了")
            ]
            natural = AccountFact(
                account,
                at,
                PREDICTION,
                True,
                offset_strength(offset, history),
                outcome,
                item.expectation_id,
                offset_minutes=offset,
            )
            if asked.held:  # 扣下没说，他自己做了
                return (natural, AccountFact(account, at, HELD_OUT, True, 1.0, "扣发后发生", item.expectation_id))
            return (natural,)
        if outcome == "提醒后发生":
            delay = int(result.offset_minutes or 0)
            return (
                AccountFact(account, at, PREDICTION, True, 0.0, outcome, item.expectation_id, offset_minutes=delay),
                AccountFact(account, at, REMINDED, True, 1.0, outcome, item.expectation_id, offset_minutes=delay),
            )
        found = [AccountFact(account, at, PREDICTION, False, 1.0, "过期待决", item.expectation_id)]
        if asked.spoken_at is not None:
            found.append(AccountFact(account, at, REMINDED, False, 1.0, "落空", item.expectation_id))
        elif asked.held:
            found.append(AccountFact(account, at, HELD_OUT, False, 1.0, "扣发后落空", item.expectation_id))
        return tuple(found)

    @staticmethod
    def _catch_up_fact(item: Expectation, ruling: Ruling, table: OpportunityTable | None) -> AccountFact:
        """后补：覆盖那条过期待决的坏，按"补做那天离上一次发生隔了几天"在这一类历史间隔里的位置打折
        （隔得比他最短的正常间隔还短就不算坏；已经隔了一个正常间隔那么久，就和没做一样坏）。"""

        delay = int(ruling.delay_minutes or 0)
        return AccountFact(
            item.account,
            item.formed_at,
            PREDICTION,
            False,
            _catch_up_strength(item, delay, table),
            "后补",
            item.expectation_id,
            offset_minutes=delay,
            written_at=ruling.ruled_at,
        )

    def _formed_by_account(self, days: Sequence[date]) -> dict[str, list[datetime]]:
        """账 → 这本账从最早一个还要看的日子起全部预期形成的时刻（升序），一晚读一遍：裁「后补」时找这本账的下一次机会。"""

        found: dict[str, list[datetime]] = {}
        if not days:
            return found
        for day in self.ledger.days_with_expectations():
            if day < days[0]:
                continue
            for item in self.ledger.expectations_on(day):
                found.setdefault(item.account, []).append(item.formed_at)
        for times in found.values():
            times.sort()
        return found

    def _omissions(self, closed: frozenset[date], cache: DayIndexCache, now: datetime, days: Sequence[date]) -> int:
        """每个定稿日查一次漏报：那天的行里发生前没有在时预期的。在时的预期可能形成在前几天（次日、第 2–7 天、长的时窗），
        从还要看的日子里取最晚时限不早于这一天的（结完的日子里的预期最晚时限都早于还没查过的定稿日）。"""

        found = 0
        for day in sorted(closed):
            if self.ledger.scanned(day):
                continue
            rows = cache.day(day).rows
            covering = [
                item
                for earlier in days
                if earlier <= day
                for item in self.ledger.expectations_on(earlier)
                if item.deadline.date() >= day
            ]
            items = omissions(rows, covering, noticed_at=now, current=self.current)
            for omission in items:
                self.ledger.record_omission(omission)
            self.ledger.mark_scanned(day, scanned_at=now, found=len(items))
            found += len(items)
        return found

    # ── 诊断（只读，裁定 54） ────────────────────────────────────────────────

    def _nightly_diagnosis(self) -> dict[str, int]:
        """夜批观测里带最近一段说错的提醒按错因的条数、校准表里标出来的切片数；诊断失败不许拖垮结算。"""

        if self.tolerance is None:
            return {}
        try:
            found = self.diagnose(now=self._clock())
        except Exception:  # noqa: BLE001 - 诊断只读、只是观测，坏了不影响结算
            return {"diagnosis_failed": 1}
        counts = {f"wrong_{cause}": value for cause, value in found.counts(spoken_only=True).items() if value}
        return {
            **counts,
            "wrong_within_budget": found.within_budget(spoken_only=True),
            "calibration_off": sum(1 for row in found.calibration if row.off),
            "held_again": sum(item.held for item in found.adjustments.values()),
            "held_again_right": sum(item.right for item in found.adjustments.values()),
            "substitutions_flagged": sum(1 for row in found.substitutions if row.flagged),
        }

    def diagnose(self, *, now: datetime, days: int = 30) -> Diagnosis:
        """最近 ``days`` 天里结过的预期：每一次落空归到哪一关（``foresight.diagnosis``），外加按切片的校准表。只读。"""

        if self.tolerance is None:
            raise ValueError("diagnosis needs the tolerance of each account's line")
        if isinstance(days, bool) or not isinstance(days, int) or days < 1:
            raise ValueError("days must be a positive integer")
        cache = DayIndexCache(
            self.read_side.behavior_tree,
            subject=self.read_side.subject,
            admits=admitted,
            calendar=self.read_side.calendar,
        )
        table = self.opportunities.latest(self.lane) if self.opportunities is not None else None
        listed = {relation.key for relation in table.relations} if table is not None else None
        start = now.date() - timedelta(days=days)
        accounts: dict[str, tuple[AccountFact, ...]] = {}
        misses: list[MissFacts] = []
        settled_rows: list[Settled] = []
        held: list[tuple[str, str]] = []  # （因此没说的那条预期, 引用的那次落空）
        followed: list[Followed] = []
        for day in self.ledger.days_with_expectations():
            if day < start:
                continue
            outcomes = {item.expectation_id: item.outcome for item in self.ledger.settlements_on(day)}
            deliveries = self.ledger.deliveries_on(day)
            spoken = {item.expectation_id for item in deliveries if item.spoken}
            subsumed = {item.expectation_id for item in deliveries if item.reason == UNIT}
            held.extend(
                (item.expectation_id, cite.removeprefix(CITE))
                for item in deliveries
                if item.reason == AGAIN
                for cite in item.evidence
                if cite.startswith(CITE)
            )
            for item in self.ledger.expectations_on(day):
                outcome = outcomes.get(item.expectation_id)
                if outcome is None:
                    continue
                facts = accounts.get(item.account)
                if facts is None:
                    facts = accounts.setdefault(item.account, current_facts(self.ledger.facts_of(item.account)))
                mine = [fact for fact in facts if fact.expectation_id == item.expectation_id]
                # 这一次现在算什么，按账上现在算数的那条预测轴事实（重结、后补都改的是它），不按当初的结算记录
                now_as = next((fact for fact in mine if fact.axis == PREDICTION), None)
                if now_as is None:
                    continue
                voided = now_as.kind == VOID
                neutral = (now_as.kind == "过期待决" and now_as.strength == 0.0) or any(
                    fact.kind == "已做过" for fact in mine
                )
                name = _slice(item)
                visible_miss = now_as.kind == "后补" or (now_as.kind == "过期待决" and now_as.strength > 0.0)
                if not voided and (now_as.kind in _NATURAL_HITS or visible_miss):
                    followed.append(Followed(item.account, self._next_kind(item, cache), missed=visible_miss))
                if not voided and not neutral and now_as.kind != "提醒后发生" and item.opportunities > 0:
                    settled_rows.append(
                        Settled(name, item.implementation, item.hits / item.opportunities, now_as.kind in _NATURAL_HITS)
                    )
                if not (now_as.kind in ("过期待决", "后补") or (voided and outcome == "过期待决")):
                    continue
                if item.expectation_id in subsumed:
                    continue  # 单元替它一并决定过：这一刻的落空记在依据那条上，一刻只算一次
                misses.append(
                    self._miss(
                        item,
                        name,
                        spoken=item.expectation_id in spoken,
                        voided=voided,
                        neutral=neutral,
                        standing=self._standing_at(item.account, facts, item.deadline),
                        cache=cache,
                        listed=listed,
                        found=POSTPONED if now_as.kind == "后补" else (now_as.cause.kind if now_as.cause else None),
                    )
                )
        return diagnose(
            misses, settled_rows, alpha=self.evidence.alpha, held=self._held(held, accounts), followed=followed
        )

    def _next_kind(self, item: Expectation, cache: DayIndexCache) -> str | None:
        """时窗开始之后他接着做的第一件别的事的类（到最晚时限；时窗开始前就结束了的不算；与找原因的"下一步"同一个口径）。"""

        kinds = {item.kind_token, *self.current(item.kind_token)}
        start, end = item.window[0].astimezone(UTC), item.deadline.astimezone(UTC)
        found = [
            row
            for row in _rows(cache, item.window[0].date(), item.deadline.date())
            if row.kind_token not in kinds
            and _classified(row.kind_token)
            and row.at.astimezone(UTC) <= end
            and row.last_observed_at.astimezone(UTC) >= start
        ]
        first = min(found, key=lambda row: (row.at.astimezone(UTC), row.uri), default=None)
        return None if first is None else first.kind_token

    def _held(
        self, held: Sequence[tuple[str, str]], accounts: dict[str, tuple[AccountFact, ...]]
    ) -> tuple[tuple[str, bool], ...]:
        """每一次"上次的原因今天又在、不说"：（那条原因的种类, 他果然没做）。还没结的不算。"""

        found = []
        for expectation_id, cited in held:
            day = self.ledger.day_of(expectation_id)
            item = next(
                (e for e in self.ledger.expectations_on(day) if e.expectation_id == expectation_id), None
            ) if day is not None else None
            if item is None:
                continue
            facts = accounts.get(item.account)
            if facts is None:
                facts = accounts.setdefault(item.account, current_facts(self.ledger.facts_of(item.account)))
            outcome = next((f for f in facts if f.expectation_id == expectation_id and f.axis == PREDICTION), None)
            cause = next(
                (f.cause for f in facts if f.expectation_id == cited and f.axis == PREDICTION and f.cause is not None),
                None,
            )
            if outcome is None or cause is None or outcome.kind in (VOID, "提醒后发生"):
                continue
            found.append((cause.kind, outcome.kind not in _NATURAL_HITS))
        return tuple(found)

    def _standing_at(self, account: str, facts: tuple[AccountFact, ...], at: datetime) -> Standing:
        """这次落空那一刻（最晚时限）这本账的位置：账上那时已有的机会（含这一次），按那时生效的重轻、设置与上一次的级。
        拿此刻的位置去判一个月前的落空，过了断点又恢复的会被说成预算内，正在断的会把之前的也拖成说不清。"""

        assert self.tolerance is not None
        return climb(
            facts,
            tolerance=self.tolerance,
            evidence=self.evidence,
            now=at,
            severity=_latest_by(self.ledger.severities_of(account), at, lambda item: item.judged_at),
            setting=_latest_by(self.ledger.settings_of(account), at, lambda item: item.at),
            previous=_latest_by(self.ledger.levels_of(account), at, lambda item: item.at),
        )

    def _miss(
        self,
        item: Expectation,
        name: str,
        *,
        spoken: bool,
        voided: bool,
        neutral: bool,
        standing: Standing,
        cache: DayIndexCache,
        listed: set[str] | None,
        found: str | None = None,
    ) -> MissFacts:
        kinds = {item.kind_token, *self.current(item.kind_token)}
        window_rows = _rows(cache, item.observed_from.date(), item.deadline.date())
        abandoned = any(
            row.kind_token in kinds and row.status == "abandoned" and item.covers(row.at.astimezone(UTC))
            for row in window_rows
        )
        today_note = cache.day(item.day).day_note
        history = [_basis_row(cache, uri) for uri in item.basis]
        notes = {cache.day(row.at.date()).day_note for row in history if row is not None}
        return MissFacts(
            expectation_id=item.expectation_id,
            account=item.account,
            kind_token=item.kind_token,
            day=item.day,
            slice=name,
            spoken=spoken,
            voided=voided,
            unseen=neutral,
            abandoned=abandoned,
            broken=standing.break_at is not None and item.formed_at >= standing.break_at,
            within_budget=standing.prediction.lower >= standing.threshold,
            relation_gone=item.source == "事件" and listed is not None and item.trigger not in listed,
            reclassified=tuple(self.current(item.kind_token)) != (item.kind_token,),
            antecedent_absent=any(a.upward and not a.present for a in item.antecedents),
            day_type_differs=today_note is not None and bool(notes) and today_note not in notes,
            notice_driven=any(row is not None and row.prompted_by == "host_agent" for row in history),
            found=found,
        )

    def _observe(
        self, status: ObservationStatus, attributes: dict[str, str | int | float | bool], started: float
    ) -> None:
        try:
            self.observer.record(
                ObservationEvent(
                    category="foresight",
                    operation="settlement",
                    status=status,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes=attributes,
                )
            )
        except Exception:  # noqa: BLE001 - 观测不许影响结算
            pass


def _last_before(item: Expectation, table: OpportunityTable | None) -> tuple[date | None, tuple[int, ...]]:
    """这条预期之前这一类最后一次发生在哪天，与这一类历史上相邻两次隔几天（机会表的日历；没有表就是空）。"""

    calendar = table.kind(item.kind_token) if table is not None else None
    if calendar is None:
        return None, ()
    gaps = gaps_of([entry.day.toordinal() for entry in calendar.days])
    earlier = [entry.day for entry in calendar.days if entry.first_at < item.observed_from]
    return (max(earlier) if earlier else None), gaps


def _catch_up_strength(item: Expectation, delay_minutes: int, table: OpportunityTable | None) -> float:
    last, gaps = _last_before(item, table)
    if last is None:
        return 1.0
    caught = (item.deadline + timedelta(minutes=delay_minutes)).date()
    return catch_up_strength(float(max(0, (caught - last).days)), gaps)


def _ruling_settled(item: Expectation, before: datetime | None, last_closed: date, table: OpportunityTable | None) -> bool:
    """这条过期待决以后还会不会被裁出不同的结果：下一次机会已经形成且那天已定稿（之后的发生归下一次），或者从上一次发生到
    已定稿的尽头已经隔了不短于最长的正常间隔（再补做强度也是 1），或者没有间隔可比（强度恒为 1）。"""

    if before is not None and last_closed >= before.date():
        return True
    if _ruling_moot(item, table):
        return True
    last, gaps = _last_before(item, table)
    assert last is not None
    return (last_closed - last).days >= max(gaps)


def _ruling_moot(item: Expectation, table: OpportunityTable | None) -> bool:
    """没有上一次或没有间隔可比：补做的强度恒为 1，裁不裁账上都一样坏（裁了只多一个「后补」的名字）。"""

    last, gaps = _last_before(item, table)
    return last is None or len(gaps) < MIN_HISTORY


_NATURAL_HITS = frozenset({"准时", "早到", "晚了但做了"})
#: 结算之后才有的改判（后补、回话）：重结不盖它们。
_LATER_RULINGS = REVISIONS | {"这次不用"}


def _slice(item: Expectation) -> str:
    """校准表的切片：触发的种类（钟面按周几是一种，事件按跨度段）。"""

    if item.source == "钟面":
        return "钟面"
    if item.source == "事件":
        return f"事件:{item.trigger.rsplit('|', 1)[-1]}"
    return f"日子:{item.trigger}"


def _basis_row(cache: DayIndexCache, uri: str) -> FlowRow | None:
    try:
        day = BehaviorURI.parse(uri).to_address().occurred_on
    except (TypeError, ValueError):
        return None
    return cache.day(day).by_uri.get(uri)


def _unobserved(item: Expectation, cache: DayIndexCache) -> bool:
    """这条预期说中的范围（起算 → 最晚时限）里有没有观测空白：那段没看到，"没做"就说不准。"""

    start, end = item.observed_from.astimezone(UTC), item.deadline.astimezone(UTC)
    day = item.observed_from.date()
    while day <= item.deadline.date():
        for gap in cache.day(day).gaps:
            if gap.started_at.astimezone(UTC) < end and gap.ended_at.astimezone(UTC) > start:
                return True
        day += timedelta(days=1)
    return False


def _latest_by[T](items: Iterable[T], at: datetime, moment: Callable[[T], datetime]) -> T | None:
    earlier = [item for item in items if moment(item) <= at]
    return max(earlier, key=moment) if earlier else None


def _classified(token: str) -> bool:
    """成了类的编号（形如 ``s-k0013``）；「待定」「非事件」这些占位标记不是一件可比的事。"""

    _lane, separator, rest = token.partition("-k")
    return bool(separator) and rest.isdigit()


def _same(old: AccountFact, new: AccountFact) -> bool:
    return (old.kind, old.good, round(old.strength, 9), old.offset_minutes) == (
        new.kind,
        new.good,
        round(new.strength, 9),
        new.offset_minutes,
    )


def _prompted(cache: DayIndexCache, uri: str) -> bool:
    """这次发生是系统提醒引出来的（归约封口时盖了 reminded，``prompted_by`` 就是 system）。"""

    row = _basis_row(cache, uri)
    return row is not None and row.prompted_by == "system"


def _rows(cache: DayIndexCache, first: date, last: date) -> tuple[FlowRow, ...]:
    rows: list[FlowRow] = []
    cursor = first
    while cursor <= last:
        rows.extend(cache.day(cursor).rows)
        cursor += timedelta(days=1)
    return tuple(rows)


__all__ = ["BehaviorReadSide", "SettlementReport", "SettlementStage"]
