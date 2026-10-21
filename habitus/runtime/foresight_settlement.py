"""结算这一拍（夜批的第一步，排在找原因之前，见 ``runtime.nightly``；没接语义树时挂在只重建之后）：对着行为树给已到最晚时限的预期一个结果，给过期待决的补裁决，给定稿日查漏报。

"那天定稿了"只有归约说了算（``BehaviorReductionRunner.closed_days``）。一条预期在它最晚时限那天定稿之后才结算，
结算只做一次；过期待决的在之后每晚看一次有没有补做（这本账下一条预期形成之前：每个机会都建预期之后，同一本账的下一条
就是这条触发自己的下一次机会，裁定 49）。漏报按定稿日查一次，查过就标记。一天里的预期都结了、过期待决的裁决再也不会变，
就标这一天结完（``mark_final``），之后每晚只读还要看的那一截，账本再长也一样。

结算与裁决落账时顺手往这本账上写事实（裁定 48、50、51、58、61）：准时 / 早到 / 晚了但做了是好、过期待决是坏，
后补再写一条打折的坏覆盖它（补做那天离上一次发生隔了几天，在这一类历史间隔里的位置）；提醒后才做的是中性（提醒的功劳不算
习惯本身）。强度按这本账自己的历史分布算（``foresight.strength``）：偏差在历史偏差里的位置、补做那天离上次多久在这一类历史
间隔里的位置。**提醒反馈**挂在行为单元上、一个行为一份：依据那条说出口的预期结算时写一笔——做了是好、没做（没回应或回了负）
是坏、回了中又没做的不写（``foresight.feedback``）；同一单元一并决定的几条按依据的说出时刻结算（「单元已选」引它）。
账本里读不出来的文件跳过、条数进观测，不让一个坏文件停掉整晚。

词表迁移做到一半时整拍不做（树上的行一半新编号一半旧编号），下一拍再来。结算失败自己留观测，不许拖着后面一夜不跑。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Protocol, runtime_checkable

from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.conditions import DayTypeCalendar
from habitus.foresight import Evidence, Standing, Tolerance, climb, current_facts
from habitus.foresight.clock import REVISIONS, is_opportunity
from habitus.foresight.diagnosis import Diagnosis, MissFacts, Settled, diagnose
from habitus.foresight.facts import catch_up_fact, ruling_moot, ruling_settled, same_fact, settlement_fact, slice_of
from habitus.foresight.feedback import reminder_feedback
from habitus.foresight.history import UnitHistory
from habitus.foresight.ledger import (
    NATURAL_HITS,
    VOID,
    AccountFact,
    Delivery,
    Expectation,
    Response,
    Settlement,
    antecedent_holds,
    omissions,
    relation_backed,
    rule_overdue,
    settle,
)
from habitus.foresight.opportunities import OpportunityTable
from habitus.foresight.units import UNIT, UNIT_CITE
from habitus.foundation.lifecycle import covers
from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.scene.views import DayIndexCache, FlowRow
from habitus.series.reader import admitted

#: 诊断看最近几天结过的预期；重建"落空那一刻的账"时再往前看几天（最晚时限在那一刻之后的预期那时还没结出来）。
#: 两者之和是诊断够得着的最早一天：账本压实只压它之前的（``runtime.foresight_compaction``）。
DIAGNOSIS_DAYS = 30
DEADLINE_LOOKBACK_DAYS = 30
DIAGNOSIS_REACH_DAYS = DIAGNOSIS_DAYS + DEADLINE_LOOKBACK_DAYS


@dataclass(frozen=True)
class Asked:
    """这条预期递送上发生过什么：第一次说出口的时刻（没说过是 None），以及是不是它自己说的（依据）——单元替它一并决定的
    （「单元已选」）按依据的说出时刻结算，但提醒反馈只由依据那条写一笔（裁定 61 第 1 条："同一单元一起开口的几条按同一个
    说出时刻结算"）。"""

    spoken_at: datetime | None
    own: bool = False

    @staticmethod
    def of(
        deliveries: Iterable[Delivery],
        expectation_id: str,
        *,
        of_basis: Callable[[str], Iterable[Delivery]] | None = None,
    ) -> Asked:
        """``deliveries`` 是这条预期所在那一天的递送；``of_basis`` 按依据的身份给依据那一天的递送——单元跨天（次日段的事件
        预期与第二天的日子 / 情境预期拢成一个单元）时依据的「说了」不在成员这一天下。"""

        items = tuple(deliveries)
        spoken = [item.decided_at for item in items if item.expectation_id == expectation_id and item.spoken]
        if spoken:
            return Asked(min(spoken), own=True)
        cited = {
            cite.removeprefix(UNIT_CITE)
            for item in items
            if item.expectation_id == expectation_id and item.reason == UNIT
            for cite in item.evidence
            if cite.startswith(UNIT_CITE)
        }
        pool = list(items)
        if of_basis is not None:
            for basis in sorted(cited):
                pool.extend(of_basis(basis))
        by_basis = [item.decided_at for item in pool if item.expectation_id in cited and item.spoken]
        return Asked(min(by_basis) if by_basis else None)


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
        history: HistoryStore | None = None,
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
        if history is not None and not isinstance(history, HistoryStore):
            raise TypeError("history must be a HistoryStore or None")
        #: 单元历史（夜里重标过的快照）：诊断重建"落空那一刻这条边的位置"从它读（裁定 61：权重归单元）；没给就按没有历史算。
        self.history = history
        self._facts_cache: dict[str, list[AccountFact]] | None = None

    def run_once(self) -> SettlementReport:
        started = time.monotonic()
        try:
            with self._caching_facts():
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
                asked = Asked.of(self.ledger.deliveries_on(day), item.expectation_id, of_basis=self._deliveries_of)
                result = settle(
                    item,
                    _rows(cache, item.formed_at.date(), item.deadline.date()),
                    spoken_at=asked.spoken_at,
                    settled_at=now,
                    current=self.current,
                )
                # 先写事实再写结算：中间断了，重来时这条还没结、照结一遍，事实的身份由预期与形成时刻定、重写幂等；
                # 反过来的顺序会让结过的这条再也不被看到，事实永远补不上。前因不成立的这次机会不存在：只记作废（权重 0）
                # 断在两次写之间、重来时：上次写下的那一笔不再算一遍（这期间账上又多了别的事实，重算出的强度可能不同，
                # 同一身份写不同内容会让整晚卡死）
                if not self._settled(item):
                    self._append(self._fact_for(item, result, cache=cache, table=table, known_by=item.deadline))
                # 提醒反馈：只有依据那条写；这次机会不算数（观测空白、前因不成立）的不写——不知道他做没做就不是误提醒
                # （裁定 54）；已经写过的不再算一遍（这期间来了回应会改变内容，同一身份不同内容会让整晚卡死）
                fact = self._current_fact(item)
                if asked.own and fact is not None and is_opportunity(fact) and not self._feedback_written(item):
                    feedback = reminder_feedback(
                        item, result, spoken_at=asked.spoken_at, responses=self._responses(day, item.expectation_id)
                    )
                    if feedback is not None:
                        self.ledger.record_feedback(feedback)
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
            and (fact := self._current_fact(by_id[expectation_id])) is not None
            and fact.kind == "过期待决"
        }
        if not overdue:
            return 0, False
        count = 0
        waiting = False
        for expectation_id in sorted(overdue):
            item = by_id[expectation_id]
            if last_closed is None or last_closed <= item.deadline.date():
                waiting = waiting or not ruling_moot(item, table)
                continue
            before = next((at for at in formed.get(item.account, ()) if at > item.deadline), None)
            ruling = rule_overdue(
                item, _rows(cache, item.deadline.date(), last_closed), before=before, ruled_at=now, current=self.current
            )
            if ruling is not None:
                self._append(catch_up_fact(item, ruling, table))
                if self.ledger.record_ruling(day, ruling):
                    count += 1
                continue
            if not ruling_settled(item, before, last_closed, table):
                waiting = True
        return count, waiting

    def _fact_for(
        self,
        item: Expectation,
        result: Settlement,
        *,
        cache: DayIndexCache,
        table: OpportunityTable | None,
        known_by: datetime,
    ) -> AccountFact:
        """一次结算往账上写哪一笔事实。前因不成立：这次机会不存在，只记作废（权重 0）；窗口里有观测空白又没看到发生：他可能做了、
        我们没看到，记中性（裁定 54）；不是这条自己的提醒引出的、可那次发生是别的账的提醒引出来的：不是这本账的自然命中，记一笔强度 0 的
        「提醒后发生」（做了，但不算证据）；其余按结果与强度（历史只取 ``known_by`` 之前已经写下的）。"""

        if not self._antecedent_holds(item, cache, table):
            return AccountFact(item.account, item.formed_at, False, 0.0, VOID, item.expectation_id)
        if result.outcome == "过期待决" and _unobserved(item, cache):
            return AccountFact(item.account, item.formed_at, False, 0.0, "过期待决", item.expectation_id)
        if (
            result.outcome != "提醒后发生"
            and result.occurrence_uri is not None
            and _prompted(cache, result.occurrence_uri)
        ):
            return AccountFact(item.account, item.formed_at, True, 0.0, "提醒后发生", item.expectation_id)
        known = [fact for fact in self._facts(item.account) if fact.precedence <= known_by]
        base = self.ledger.compaction_of(item.account)
        earlier = base.offset_history() if base is not None else ()
        return settlement_fact(item, result, known=known, earlier_offsets=earlier)

    def _responses(self, day: date, expectation_id: str) -> tuple[Response, ...]:
        return tuple(item for item in self.ledger.responses_on(day) if item.expectation_id == expectation_id)

    def _deliveries_of(self, expectation_id: str) -> tuple[Delivery, ...]:
        """一条预期（依据）所在那一天的递送；不知道它在哪一天就是空。"""

        day = self.ledger.day_of(expectation_id)
        return self.ledger.deliveries_on(day) if day is not None else ()

    def _feedback_written(self, item: Expectation) -> bool:
        return any(found.expectation_id == item.expectation_id for found in self.ledger.feedback_of(item.kind_token))

    def resettle(self, days: Iterable[date]) -> int:
        """修了 bug 之后：用当前代码把这些日子里已结的预期重结一遍，和账上现有的不一样的追加改判（机会的时刻不变，写下的时刻取
        此刻，后写的覆盖先写的；旧记录不改，裁定 54）。只碰结算事实，不碰后补这类之后才有的改判，也不碰单元的提醒反馈。
        改回过期待决的日子若已标结完，重新打开，让之后每晚照常看有没有补做。返回改了几条预期。"""

        cache = DayIndexCache(
            self.read_side.behavior_tree,
            subject=self.read_side.subject,
            admits=admitted,
            calendar=self.read_side.calendar,
        )
        table = self.opportunities.latest(self.lane) if self.opportunities is not None else None
        now = self._clock()
        with self._caching_facts():
            return self._resettle(days, cache, table, now)

    def _resettle(
        self, days: Iterable[date], cache: DayIndexCache, table: OpportunityTable | None, now: datetime
    ) -> int:
        revised = 0
        for day in sorted(set(days)):
            settled = {item.expectation_id for item in self.ledger.settlements_on(day)}
            deliveries = self.ledger.deliveries_on(day)
            reopen = False
            for item in self.ledger.expectations_on(day):
                if item.expectation_id not in settled or self._compacted(item):
                    continue  # 重结只作用于没压实的窗口：压实的那一段已经并进累计值，不会再变
                asked = Asked.of(deliveries, item.expectation_id, of_basis=self._deliveries_of)
                result = settle(
                    item,
                    _rows(cache, item.formed_at.date(), item.deadline.date()),
                    spoken_at=asked.spoken_at,
                    settled_at=now,
                    current=self.current,
                )
                fresh = self._fact_for(item, result, cache=cache, table=table, known_by=now)
                old = self._current_fact(item)
                if old is not None and (old.kind in _LATER_RULINGS or same_fact(old, fresh)):
                    continue
                self._append(replace(fresh, at=item.formed_at, written_at=now))
                revised += 1
                if fresh.kind == "过期待决" and fresh.strength > 0.0:
                    reopen = True
            if reopen and self.ledger.is_final(day):
                self.ledger.reopen(day)
        return revised

    def _compacted(self, item: Expectation) -> bool:
        base = self.ledger.compaction_of(item.account)
        return base is not None and base.covers(item.formed_at, item.expectation_id)

    # ── 一拍里每本账的事实只读一次 ─────────────────────────────────────────────

    @contextmanager
    def _caching_facts(self) -> Iterator[None]:
        """一晚的结算里一本账的事实只从盘上读一次，之后写下的顺手加进缓存（2026-10-17 四方评审：原来每条预期把这本账整本读
        两三遍，一晚的量是条数 × 事实数）。出了这一拍就不留。"""

        self._facts_cache = {}
        try:
            yield
        finally:
            self._facts_cache = None

    def _facts(self, account: str) -> tuple[AccountFact, ...]:
        if self._facts_cache is None:
            return self.ledger.facts_of(account)
        if account not in self._facts_cache:
            self._facts_cache[account] = list(self.ledger.facts_of(account))
        return tuple(self._facts_cache[account])

    def _append(self, fact: AccountFact) -> None:
        if self.ledger.append_fact(fact) and self._facts_cache is not None and fact.account in self._facts_cache:
            self._facts_cache[fact.account].append(fact)

    # ── 前因 ────────────────────────────────────────────────────────────────

    def _antecedent_holds(self, item: Expectation, cache: DayIndexCache, table: OpportunityTable | None) -> bool:
        """事件触发：前因那条记录（按开始时刻找，封口后的样子）现在还属于这条关系的前因类吗。情境触发：按最新的日历，
        那一天还处在这个日历情境里吗（放假数据后来补上了，原来按名义当成工作日的那天就作废）。关系已不在表里说不清，不判作废。"""

        relation = table.relation(item.trigger) if table is not None else None
        if item.source == "情境":
            if relation is None or relation.calendar_key is None or table is None:
                return True
            today = dict(self.read_side.calendar.for_expectations(item.day))
            return relation.calendar_holds(today, official=table.unseen_day_types == "official")
        if item.source != "事件":
            return True
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

    def _current_fact(self, item: Expectation) -> AccountFact | None:
        """这条预期此刻算数的那一笔（后写的覆盖先写的）。"""

        mine = [fact for fact in self._facts(item.account) if fact.expectation_id == item.expectation_id]
        return next(iter(current_facts(mine)), None)

    def _settled(self, item: Expectation) -> bool:
        """这条预期的原判（结算时写的、不是改判）已经写下了。"""

        return any(
            fact.expectation_id == item.expectation_id and fact.written_at is None for fact in self._facts(item.account)
        )

    # ── 账上的事实 ──────────────────────────────────────────────────────────

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
        if not closed:
            return 0
        # 只走"查过漏报"水位以上的日子（26 号清单 6.11；定稿不按顺序，所以水位按连续前缀推）
        start = min(closed)
        watermark = self.ledger.scan_watermark(start=start)
        for day in sorted(closed):
            if covers(watermark, day):
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
            watermark = self.ledger.mark_scanned(day, start=start)
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
        }

    def diagnose(self, *, now: datetime, days: int = DIAGNOSIS_DAYS) -> Diagnosis:
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
        lane = table.lane if table is not None else self.lane
        history = self.history.latest(lane) if self.history is not None else None
        misses: list[MissFacts] = []
        settled_rows: list[Settled] = []
        for day in self.ledger.days_with_expectations():
            if day < start:
                continue
            outcomes = {item.expectation_id: item.outcome for item in self.ledger.settlements_on(day)}
            deliveries = self.ledger.deliveries_on(day)
            spoken = {item.expectation_id for item in deliveries if item.spoken}
            subsumed = {item.expectation_id for item in deliveries if item.reason == UNIT}
            for item in self.ledger.expectations_on(day):
                outcome = outcomes.get(item.expectation_id)
                if outcome is None:
                    continue
                facts = accounts.get(item.account)
                if facts is None:
                    facts = accounts.setdefault(item.account, current_facts(self.ledger.facts_of(item.account)))
                # 这一次现在算什么，按账上现在算数的那笔事实（重结、后补都改的是它），不按当初的结算记录
                now_as = next((fact for fact in facts if fact.expectation_id == item.expectation_id), None)
                if now_as is None:
                    continue
                voided = now_as.kind == VOID
                neutral = now_as.kind == "过期待决" and now_as.strength == 0.0
                name = slice_of(item)
                if not voided and not neutral and now_as.kind != "提醒后发生" and item.opportunities > 0:
                    settled_rows.append(
                        Settled(name, item.implementation, item.hits / item.opportunities, now_as.kind in NATURAL_HITS)
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
                        standing=self._standing_at(item, history),
                        cache=cache,
                        listed=listed,
                        postponed=now_as.kind == "后补",
                    )
                )
        return diagnose(misses, settled_rows, alpha=self.evidence.alpha)

    def _standing_at(self, item: Expectation, history: UnitHistory | None) -> Standing:
        """这次落空那一刻（最晚时限）这条边的位置：单元历史（``history``，夜里按当晚语义树重标的最晚一晚，一次诊断只读一回）里
        这条边在那一刻之前的那几刻，按那时生效的设置与上一次的级。拿此刻的位置去判一个月前的落空，过了断点又恢复的会被说成
        预算内，正在断的会把之前的也拖成说不清；按刻的时刻截到那一刻（``edge_reading(visible_at=)``），之后的刻不算。"""

        assert self.tolerance is not None
        at = item.deadline
        moments = history.of(item.kind_token) if history is not None else ()
        return climb(
            moments,
            item.account,
            now=at,
            tolerance=self.tolerance,
            evidence=self.evidence,
            setting=_latest_by(self.ledger.settings_of(item.kind_token), at, lambda found: found.at),
            previous=_latest_by(self.ledger.levels_of(item.account), at, lambda found: found.at),
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
        postponed: bool = False,
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
            relation_gone=relation_backed(item.source) and listed is not None and item.trigger not in listed,
            reclassified=tuple(self.current(item.kind_token)) != (item.kind_token,),
            antecedent_absent=any(a.upward and not a.present for a in item.antecedents),
            day_type_differs=today_note is not None and bool(notes) and today_note not in notes,
            notice_driven=any(row is not None and row.prompted_by == "host_agent" for row in history),
            postponed=postponed,
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


#: 结算之后才有的改判（后补）：重结不盖它们。
_LATER_RULINGS = REVISIONS


def _basis_row(cache: DayIndexCache, uri: str) -> FlowRow | None:
    try:
        day = BehaviorURI.parse(uri).to_address().occurred_on
    except (TypeError, ValueError):
        return None
    return cache.day(day).by_uri.get(uri)


def _unobserved(item: Expectation, cache: DayIndexCache) -> bool:
    """这条预期说中的范围（起算 → 最晚时限）里有没有观测空白：那段没看到，"没做"就说不准。"""

    return cache.has_gap(item.observed_from, item.deadline)


def _latest_by[T](items: Iterable[T], at: datetime, moment: Callable[[T], datetime]) -> T | None:
    earlier = [item for item in items if moment(item) <= at]
    return max(earlier, key=moment) if earlier else None


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
