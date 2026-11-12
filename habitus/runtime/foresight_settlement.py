"""结算这一拍（换代里重试收尾之后、读序列之前，见 ``runtime.generation``）：对着行为树给已到最晚时限的预期一个结果，给过期待决的补裁决，给定稿日查漏报。

"那天定稿了"只有归约说了算（``BehaviorReductionRunner.closed_days``）。一条预期在它最晚时限那天定稿之后才结算，
结算只做一次；过期待决的在之后每一代看一次有没有补做（这本账下一条预期开始看之前：每个机会都建预期之后，同一本账的下一条
就是这条触发自己的下一次机会，裁定 49）。漏报按定稿日查一次，查过就标记。一天里的预期都结了、过期待决的裁决再也不会变，
就标这一天结完（``mark_final``），之后每一代只读还要看的那一截，账本再长也一样。

结算与裁决落账时顺手往这本账上写事实（裁定 48、50、51、58、61）：准时 / 早到 / 晚了但做了是好、过期待决是坏，
后补再写一条打折的坏覆盖它（补做那天离上一次发生隔了几天，在这一类历史间隔里的位置）；提醒后才做的是中性（提醒的功劳不算
习惯本身）。强度按这本账自己的历史分布算（``foresight.strength``）：偏差在历史偏差里的位置、补做那天离上次多久在这一类历史
间隔里的位置。**提醒反馈**挂在行为单元上、一个行为一份：依据那条说出口的预期结算时写一笔——做了是好、没做（没回应或回了负）
是坏、回了中又没做的不写、回了「已做」又没看到做的不写且账上那一笔记中性（裁定 91 第 3 条②；``foresight.feedback``）；同一单元一并决定的几条按依据的说出时刻结算（「单元已选」引它）。
账本里读不出来的文件跳过、条数进观测，不让一个坏文件停掉整代。

词表迁移做到一半时整拍不做（树上的行一半新编号一半旧编号），下一拍再来。结算是换代里尽力而为的一步：失败自己留观测，下一代补结
（幂等），不挡今天这一代。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Protocol, runtime_checkable

from habitus.behavior.tree import BehaviorTree
from habitus.conditions import DayTypeCalendar
from habitus.foresight import Evidence, Tolerance, current_facts
from habitus.foresight.clock import REVISIONS, is_opportunity
from habitus.foresight.diagnosis import (
    DIAGNOSIS_DAYS,
    Diagnosis,
)
from habitus.foresight.facts import (
    catch_up_fact,
    ruling_moot,
    ruling_settled,
    same_fact,
    settlement_fact,
    short_circuit_fact,
)
from habitus.foresight.feedback import reminder_feedback, said_done
from habitus.foresight.ledger import (
    AccountFact,
    AntecedentRow,
    Delivery,
    Expectation,
    Response,
    Settlement,
    antecedent_holds_when_settling,
    omissions,
    rule_overdue,
    settle,
)
from habitus.foresight.opportunities import OpportunityTable
from habitus.foresight.units import UNIT, UNIT_CITE
from habitus.foundation.lifecycle import covers
from habitus.foundation.observability import (
    NullObserver,
    ObservationStatus,
    Observer,
    record_observation,
)
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_settlement_diagnosis import SettlementDiagnosis
from habitus.runtime.foresight_tree_rows import basis_row, rows_between
from habitus.scene.views import DayIndexCache
from habitus.series.model import silent_day_unwatched
from habitus.series.reader import admitted


@dataclass(frozen=True)
class Asked:
    """这条预期递送上发生过什么：第一次说出口的时刻（没说过是 None），以及是不是它自己说的（依据）——单元替它一并决定的
    （「单元已选」）按依据的说出时刻结算，但提醒反馈只由依据那条写一笔（裁定 61 第 1 条："同一单元一起开口的几条按同一个
    说出时刻结算"）。"""

    spoken_at: datetime | None
    own: bool = False
    #: 说出口的那条预期的身份（它自己，或单元替它一并决定时最早说出口的那条依据）：他对那句提醒的回话记在它名下。没说过是 None。
    basis: str | None = None

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
            return Asked(min(spoken), own=True, basis=expectation_id)
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
        by_basis = sorted(
            (item.decided_at, item.expectation_id) for item in pool if item.expectation_id in cited and item.spoken
        )
        return Asked(by_basis[0][0], basis=by_basis[0][1]) if by_basis else Asked(None)


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
    #: 这一拍新写的提醒反馈条数（只给观测）。
    feedback: int = 0


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
        finished: Callable[[date], bool | None] | None = None,
    ) -> None:
        self.ledger = ledger
        self.read_side = read_side
        self.closed_days = closed_days
        # 编号 → 现在对应哪些在用编号（词表的 ``current_ids``）：预期记的是形成时的编号，结算时词表可能已拆改
        self.current = current
        self.migrating = migrating
        self.evidence = evidence
        # 机会表的日历给补做强度用（这一类历史上相邻两次隔几天）；没有就只有方向
        self.opportunities = opportunities
        self.lane = lane
        self.observer: Observer = observer if observer is not None else NullObserver()
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        #: 诊断要知道每本账的线（预算内那一关）；没给就不在结算观测里出诊断。
        self.tolerance = tolerance
        #: 单元历史（换代里重标过的快照）：诊断重建"落空那一刻这条边的位置"从它读（裁定 61：权重归单元）；没给就按没有历史算。
        self.history = history
        #: 换代的完成标记记的"这一天的换代做完了没有"（``runtime.generation_marks.GenerationMarks.finished``）：没有这一代的
        #: 那天后补只看到它开始为止（裁定 94 第 9 条）；没给就不认那天。
        self.finished = finished
        self._facts_cache: dict[str, list[AccountFact]] | None = None
        #: 只读诊断（裁定 54）：结算观测与运维读它；每本账的线每次从 ``tolerance`` 现读。
        self.diagnosis = SettlementDiagnosis(
            ledger,
            read_side,
            evidence=evidence,
            current=current,
            opportunities=opportunities,
            lane=lane,
            history=history,
        )

    def run_once(self) -> SettlementReport:
        started = time.monotonic()
        try:
            with self._caching_facts():
                report = self._run()
        except Exception as exc:
            self._observe(ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started)
            raise
        diagnosis = self._diagnosis_attributes()
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
        rulings = feedback_written = 0
        pending: list[date] = []
        opened = self._opened_by_account(days)
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
                    rows_between(cache, item.observed_from.date(), item.deadline.date()),
                    spoken_at=asked.spoken_at,
                    settled_at=now,
                    current=self.current,
                )
                # 先写事实再写结算：中间断了，重来时这条还没结、照结一遍，事实的身份由预期与形成时刻定、重写幂等；
                # 反过来的顺序会让结过的这条再也不被看到，事实永远补不上。前因不成立的这次机会不存在：只记作废（权重 0）
                # 断在两次写之间、重来时：上次写下的那一笔不再算一遍（这期间账上又多了别的事实，重算出的强度可能不同，
                # 同一身份写不同内容会让整代卡死）
                if not self._settled(item):
                    self._append(
                        self._fact_for(item, result, asked=asked, cache=cache, table=table, known_by=item.deadline)
                    )
                # 提醒反馈：只有依据那条写；这次机会不算数（观测空白、前因不成立）的不写——不知道他做没做就不是误提醒
                # （裁定 54）；已经写过的不再算一遍（这期间来了回应会改变内容，同一身份不同内容会让整代卡死）
                fact = self._current_fact(item)
                if asked.own and fact is not None and is_opportunity(fact) and not self._feedback_written(item):
                    feedback = reminder_feedback(
                        item, result, spoken_at=asked.spoken_at, responses=self._responses(day, item.expectation_id)
                    )
                    if feedback is not None:
                        feedback_written += self.ledger.record_feedback(feedback)
                self.ledger.record_settlement(result)
                outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
            count, undecided = self._rule(day, cache, last_closed, now, opened, table)
            rulings += count
            if undecided:
                waiting_rulings.add(day)
        found = self._omissions(closed, cache, now, days)
        for day in days:
            if day not in pending and day not in waiting_rulings and not self.ledger.unsettled_expectations(day):
                self.ledger.mark_final(day, at=now)
        return SettlementReport(
            sum(outcomes.values()), outcomes, rulings, found, tuple(pending), feedback=feedback_written
        )

    def _rule(
        self,
        day: date,
        cache: DayIndexCache,
        last_closed: date | None,
        now: datetime,
        opened: dict[str, list[datetime]],
        table: OpportunityTable | None,
    ) -> tuple[int, bool]:
        """这一天里结成「过期待决」、还没有裁决的预期：最晚时限之后到看得见的尽头（这本账下一条预期开始看之前）有没有补做。
        返回（这一拍裁了几条，还有没有以后可能被裁的）。以后再也裁不出不同结果的不再等：下一次机会已经形成而且那天已定稿，
        或者离上次发生已经隔了不短于这一类最长的正常间隔（那之后补做的强度是 1，和过期待决一样坏）。"""

        settled = {s.expectation_id for s in self.ledger.settlements_on(day)}
        ruled = {r.expectation_id for r in self.ledger.rulings_on(day)}
        by_id = {item.expectation_id: item for item in self.ledger.expectations_on(day)}
        # 过期待决按账上现在的那条预测轴事实认，不按结算记录：重结可能把它改成了别的（作废、准时），也可能改回了过期待决。
        # 强度 0 的是没在看的中性（``facts.short_circuit_fact``：观测空白、会话 lane 整天没记录）：谈不上抓到缺失，不进待裁——
        # 之后做了不改判成「后补」（裁定 54 第 3 条③、裁定 89 第 2 条；与 ``_resettle`` 只在强度 > 0 时重开同一口径）
        overdue = {
            expectation_id
            for expectation_id in settled - ruled
            if expectation_id in by_id
            and (fact := self._current_fact(by_id[expectation_id])) is not None
            and fact.kind == "过期待决"
            and fact.strength > 0.0
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
            # 下一次机会从它开始看的那一刻（``observed_from``）起算：日子预期从当天零点起看、正好等于前一条的最晚时限，零点
            # 之后做的归这一条，不再记成上一条的「后补」（裁定 94 第 1 条，D4）
            before = next((at for at in opened.get(item.account, ()) if at >= item.deadline), None)
            # 没有这一代的那天按看不到：后补只看到它开始为止（裁定 94 第 9 条）——那天没有日子预期，不能把那天正常做的那次
            # 裁成上一条的「后补」
            stopped = None if self.finished is None else first_stopped_day(item.deadline, last_closed, self.finished)
            if stopped is not None and (before is None or stopped < before):
                before = stopped
            ruling = rule_overdue(
                item,
                rows_between(cache, item.deadline.date(), last_closed),
                before=before,
                ruled_at=now,
                current=self.current,
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
        asked: Asked,
        cache: DayIndexCache,
        table: OpportunityTable | None,
        known_by: datetime,
    ) -> AccountFact:
        """一次结算往账上写哪一笔事实：四条短路（作废、观测空白的中性、回了「已做」的中性、别的账的提醒引出的）的规则在
        ``foresight.facts.short_circuit_fact``，这里只把它要的四件事从树上、机会表与回话判好；都不是就按结果与强度（历史只取
        ``known_by`` 之前已经写下的）。"""

        uri = result.occurrence_uri
        short = short_circuit_fact(
            item,
            result,
            antecedent_holds=self._antecedent_holds(item, cache, table),
            unobserved=self._unobserved(item, cache),
            said_done=self._said_done(asked),
            prompted=uri is not None and _prompted(cache, uri),
        )
        if short is not None:
            return short
        known = [fact for fact in self._facts(item.account) if fact.precedence <= known_by]
        base = self.ledger.compaction_of(item.account)
        earlier = base.offset_history() if base is not None else ()
        return settlement_fact(item, result, known=known, earlier_offsets=earlier)

    def _unobserved(self, item: Expectation, cache: DayIndexCache) -> bool:
        """这条说中的范围里有没有没在看的：上游报的空白，或（会话 lane）一整天没有会话的日子——后者等同上游报了全天「未观测」
        （裁定 89 第 2 条，``series.model.silent_day_unwatched``）。递送那一拍不看这一条（裁定 68 第 2 条：白天还不知道这天会不会空着）。"""

        if cache.has_gap(item.observed_from, item.deadline):
            return True
        return silent_day_unwatched(self.lane) and cache.silent_within(item.observed_from, item.deadline)

    def _said_done(self, asked: Asked) -> bool:
        """说给他的那句提醒（这条自己的，或单元一并决定时依据那条的）他回了「已做」（裁定 91 第 3 条②）。"""

        if asked.basis is None:
            return False
        day = self.ledger.day_of(asked.basis)
        return day is not None and said_done(self._responses(day, asked.basis))

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
        改回过期待决的日子若已标结完，重新打开，让之后每一代照常看有没有补做。返回改了几条预期。"""

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
                    rows_between(cache, item.observed_from.date(), item.deadline.date()),
                    spoken_at=asked.spoken_at,
                    settled_at=now,
                    current=self.current,
                )
                fresh = self._fact_for(item, result, asked=asked, cache=cache, table=table, known_by=now)
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
        """一次结算里一本账的事实只从盘上读一次，之后写下的顺手加进缓存（2026-10-17 四方评审：原来每条预期把这本账整本读
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
        """这次机会的前因还成不成立（规则在 ``ledger.settle.antecedent_holds_when_settling``）：日历按最新的读，行从树上读。"""

        return antecedent_holds_when_settling(
            item,
            table.relation(item.trigger) if table is not None else None,
            official=table is not None and table.unseen_day_types == "official",
            calendar_on=self.read_side.calendar.for_expectations,
            rows_on=lambda day: tuple(
                AntecedentRow(row.at, row.last_observed_at, row.kind_token) for row in cache.day(day).rows
            ),
            current=self.current,
        )

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

    def _opened_by_account(self, days: Sequence[date]) -> dict[str, list[datetime]]:
        """账 → 这本账从最早一个还要看的日子起全部预期开始看的时刻（``observed_from``，升序），一晚读一遍：裁「后补」时找这本账
        的下一次机会。"""

        found: dict[str, list[datetime]] = {}
        if not days:
            return found
        for day in self.ledger.days_with_expectations():
            if day < days[0]:
                continue
            for item in self.ledger.expectations_on(day):
                found.setdefault(item.account, []).append(item.observed_from)
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

    def _diagnosis_attributes(self) -> dict[str, int]:
        """结算观测里带最近一段说错的提醒按错因的条数、校准表里标出来的切片数；诊断失败不许拖垮结算。"""

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
        """最近 ``days`` 天里结过的预期：每一次落空归到哪一关，外加按切片的校准表。只读（``SettlementDiagnosis``）。"""

        return self.diagnosis.diagnose(now=now, days=days, tolerance=self.tolerance)

    def _observe(
        self, status: ObservationStatus, attributes: dict[str, str | int | float | bool], started: float
    ) -> None:
        record_observation(self.observer, "foresight", "settlement", status, started=started, attributes=attributes)


#: 结算之后才有的改判（后补）：重结不盖它们。
_LATER_RULINGS = REVISIONS


def _prompted(cache: DayIndexCache, uri: str) -> bool:
    """这次发生是系统提醒引出来的（归约封口时盖了 reminded，``prompted_by`` 就是 system）。"""

    row = basis_row(cache, uri)
    return row is not None and row.prompted_by == "system"


def first_stopped_day(
    deadline: datetime, last_closed: date, finished: Callable[[date], bool | None]
) -> datetime | None:
    """最晚时限之后、到最后一个定稿日为止，第一个没有这一代的日子从哪一刻开始（主体本地零点，取最晚时限的偏移）；没有是 None。

    读的是换代的完成标记（``finished``）：那天没有标记——必须成的那几步没成（读序列、机会表、日子预期、完成标记）、一整天都没
    做成、没开机都一样，那天都没有日子预期（裁定 94 第 9 条）。标记够不着的那几天（None）不认。"""

    day = deadline.date()
    while day <= last_closed:
        start = datetime.combine(day, datetime.min.time(), tzinfo=deadline.tzinfo)
        if start >= deadline and finished(day) is False:
            return start
        day += timedelta(days=1)
    return None


__all__ = ["BehaviorReadSide", "SettlementReport", "SettlementStage", "first_stopped_day"]
