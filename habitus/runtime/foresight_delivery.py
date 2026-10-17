"""递送那一拍（方案第二节、第四节后半；裁定 48–51、56、58）：到了时窗的预期按**行为单元**拢在一起，每个单元决定一次说不说，
记进账本（说没说都记、记原因）。

**单元**（裁定 56 第 12 条，``runtime.foresight_units``）：同一个行为这一刻的全部输入——到了时窗的那几条预期，加上说中的范围
和它们连成一片、还没到时窗的那几条——拢成一个单元；单元读数 = 这一组在场时历史上下一步来了多少，按依据那本账的线过线。
依据那一条照下面的流程走（时窗开 / 时窗末、模型、名额）；其余几条记「单元已选」并引依据——它们照旧各自结算，各前因的权重
（账）仍各自随结算调。

档不存成状态：每次从这本账的事实折出来（``foresight.ladder.climb``），只在变的时候记一条"级的变化"给滞回与可见面用。
一拍只读到了时窗那几本账的 ``accounts/<账>/`` 目录，不扫整本账（裁定 48：账加按账索引）。

一条预期到了时窗决定一次（时窗与这一拍的区间有交集就算到了：一拍一槽，时窗比一槽短的也落得上），账上已有决定的不再决定；
只有「等时窗末」不是终局——轻的事到时窗末还没做再说（「该来了还没来」是提醒的说话时机；「重」的时窗一开就说）。到时窗末那一拍
重新折一遍这本账的档（窗里回了「这次不用」或设了「以后别提了」就不说了），这时才问模型、才写那句话：话里的"今天""这会儿"
说的是开口那一刻。
到了时窗时这件事已经做了记「已发生」；同一类今天已经说过一句的记「同类已说」；没有送达渠道的记「不可达」（裁定 47：接上渠道
之前提醒档都是它，结算按没说过处理）。每日上限按决定的日子数。

模型只在有渠道时问（一级都送不到，问了也说不出口）：
- **重轻**按账判一次（裁定 48）：这本账已有过至少一次结算、还没判过或已到期，就在它到时窗那一拍判；判了落账、到期
  （``severity_review_days``）重判；没判过、模型不可用按轻。
- **像不像、先说哪个、一句话**：只问这一拍真要开口的——提醒档、还没做、同一类今天还没说过、名额没满；重的在时窗开时问，
  轻的在时窗末问。一拍一个小包。两遍都说不像的记「模型判不像」；像的按模型的先后占名额、用模型的话；模型不可用按算法顺序、
  模板措辞。
送达渠道在本层之外：这里产出的是"递送决定 + 档 + 话"。
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from habitus.foresight import Evidence, Seen, Standing, Tolerance, climb, current_facts, decide
from habitus.foresight.causes import AGAIN, CITE, cause_holds
from habitus.foresight.judge import (
    Judge,
    Judgement,
    Package,
    UnitInput,
    Verdict,
    build_package,
    plain_text,
    sources_of,
)
from habitus.foresight.ladder import PREDICTION
from habitus.foresight.ledger import (
    AccountFact,
    AccountLevel,
    AccountSeverity,
    Delivery,
    Expectation,
    antecedent_holds,
    parse_account,
)
from habitus.foresight.opportunities import OpportunityTable, TreeReading
from habitus.foresight.units import UNIT, UNIT_CITE
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_units import Unit, UnitPlanner, UnitView

#: 不是终局的递送原因：之后还会再决定一次。
_PENDING = "等时窗末"

_NATURAL_HITS = frozenset({"准时", "早到", "晚了但做了"})


@dataclass(frozen=True)
class DeliveryReport:
    at: datetime
    decided: int
    spoken: int
    by_reason: dict[str, int] = field(default_factory=dict)
    by_level: dict[int, int] = field(default_factory=dict)
    #: 模型这一拍问了几个候选（像不像）、判了几本账的重轻、调了几次、留下的信号。
    judged: int = 0
    severities: int = 0
    calls: int = 0
    signals: tuple[str, ...] = ()
    judge_failed: bool = False
    #: 这一拍有几本账换了档。
    level_changes: int = 0
    #: 账本里读不出来、被跳过的文件数。
    unreadable: int = 0


@dataclass(frozen=True)
class JudgeMaterials:
    """模型小包要的材料，由组合根按这一拍准备：机会表（历史那几次）、按出处取行（原话）、这件事这回是不是已经做了
    （做了就不递：那句话说的事他刚做完；结算会记准时 / 早到）。"""

    table: OpportunityTable | None
    rows_for: Callable[[Iterable[str]], Mapping[str, Seen]]
    happened: Callable[[Expectation], bool] = lambda item: False
    #: 当地日历对某一天的说法（进小包给模型判像不像；没有日历数据时恒为空）。
    day_note: Callable[[date], str | None] = lambda day: None
    #: 最近几天到此刻开始过的事（核"上次落空的原因今天又在不在"用，裁定 55）。
    seen: tuple[Seen, ...] = ()


@dataclass(frozen=True)
class AccountView:
    """可见面读的一本账（裁定 50）：是什么、此刻在哪一档、两条轴的读数、重轻、设置、最近几条事实。"""

    account: str
    standing: Standing
    severity: AccountSeverity | None
    facts: int
    recent: tuple[str, ...]


@dataclass(frozen=True)
class OmissionView:
    """可见面里"还没看懂的事"（裁定 50）：一类行为最近一段里发生了、而发生前没有任何一条在时预期的次数。
    ``with_expectation`` 是其中那天这一类有过别的预期的次数（时窗没对上）；其余是根本没有预期（还没认出规律）。"""

    kind_token: str
    count: int
    with_expectation: int
    latest_uri: str


class DeliveryStage:
    def __init__(
        self,
        ledger: ForesightLedgerStore,
        *,
        tolerance: Tolerance,
        evidence: Evidence,
        daily_cap: int,
        severity_review_days: float,
        holdout_rate: float = 0.0,
        holdout_rate_heavy: float = 0.0,
        draw: Callable[[], float] | None = None,
        judge: Judge | None = None,
        tick_seconds: float = 900.0,
        reachable: bool = False,
        unit_prior_strength: float = 8.0,
    ) -> None:
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        if not isinstance(tolerance, Tolerance) or not isinstance(evidence, Evidence):
            raise TypeError("tolerance must be a Tolerance and evidence an Evidence")
        if isinstance(daily_cap, bool) or not isinstance(daily_cap, int) or daily_cap < 1:
            raise ValueError("daily_cap must be a positive integer")
        for name, rate in (("holdout_rate", holdout_rate), ("holdout_rate_heavy", holdout_rate_heavy)):
            if isinstance(rate, bool) or not isinstance(rate, int | float) or not 0.0 <= rate < 1.0:
                raise ValueError(f"{name} must be a number in [0, 1)")
        if holdout_rate_heavy > holdout_rate:
            raise ValueError("a graver matter is not held out more often than a light one")
        if isinstance(tick_seconds, bool) or not isinstance(tick_seconds, int | float) or tick_seconds <= 0:
            raise ValueError("tick_seconds must be a positive number")
        if (
            isinstance(severity_review_days, bool)
            or not isinstance(severity_review_days, int | float)
            or severity_review_days <= 0
        ):
            raise ValueError("severity_review_days must be a positive number")
        if judge is not None and not isinstance(judge, Judge):
            raise TypeError("judge must be a Judge or None")
        if not isinstance(reachable, bool):
            raise TypeError("reachable must be a boolean")
        self.ledger = ledger
        self.tolerance = tolerance
        self.evidence = evidence
        self.daily_cap = daily_cap
        self.severity_review = timedelta(days=float(severity_review_days))
        #: 随机扣发的比例，最轻、最重两级（中间按级线性插；裁定 58 第 7 条：重的也扣、比例更小）。
        self.holdout_rate = float(holdout_rate)
        self.holdout_rate_heavy = float(holdout_rate_heavy)
        self._draw = draw if draw is not None else random.random  # noqa: S311 - 扣发抽样不是安全用途
        self.judge = judge
        self.tick = timedelta(seconds=float(tick_seconds))
        #: 有没有送达渠道（没有 = 提醒档都「不可达」，也不问模型）。
        self.reachable = reachable
        self.units = UnitPlanner(ledger, evidence=evidence, prior_strength=unit_prior_strength)

    async def run_once(self, now: datetime, materials: JudgeMaterials | None = None) -> DeliveryReport:
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise ValueError("now must be a timezone-aware datetime")
        found = self._due(now)
        due, closing, waiting = found.opened, found.closing, found.waiting
        judged, calls, signals, severity_failed, judged_accounts = await self._severities(due, materials, now)
        # 这一拍新判的优先，没有就读账上已有的（之前判的、人改的）
        severities = {
            item.account: judged.get(item.account) or self.ledger.severity_of(item.account) for item in (*due, *waiting)
        }
        cached: dict[str, Standing] = {}

        def standing_of(item: Expectation) -> Standing:
            if item.account not in cached:
                cached[item.account] = self._standing(item.account, now, severities.get(item.account))
            return cached[item.account]

        units = self.units.plan(
            due,
            waiting,
            standing_of=standing_of,
            previous_level=lambda account: level.level if (level := self.ledger.level_of(account)) is not None else None,
            table=materials.table if materials is not None else None,
            now=now,
        )
        opened = tuple(unit.basis for unit in units)
        standings = {unit.basis.account: unit.standing for unit in units}
        for item, _pending in closing:
            if item.account not in standings:
                standings[item.account] = self._standing(item.account, now, None)
        spoken_today, spoken_kinds = self._spoken_today(now)
        tally = _Tally(now, spoken_today, spoken_kinds)
        for unit in units:
            for member in unit.members:
                self._record(
                    tally,
                    member,
                    Delivery(
                        member.expectation_id, now, unit.level, False, UNIT, 0.0, evidence=(f"{UNIT_CITE}{unit.basis.expectation_id}",)
                    ),
                )
        inputs: dict[str, tuple[UnitInput, ...]] = {unit.basis.expectation_id: _unit_inputs(unit) for unit in units}
        for item, _pending in closing:
            together = (item, *found.members.get(item.expectation_id, ()))
            inputs[item.expectation_id] = tuple(
                UnitInput(member, standing_of(member).prediction.k, standing_of(member).prediction.n) for member in together
            )
        speaking = (
            *(item for item in opened if _urgent(severities.get(item.account), now)),
            *(item for item, _pending in closing),
        )
        judgement = await self._judge(speaking, standings, materials, now, tally, inputs)
        for item, pending in closing:
            verdict = judgement.verdicts.get(item.expectation_id) if judgement is not None else None
            severity = self.ledger.severity_of(item.account)
            self._record(
                tally,
                item,
                self._closing(item, pending, standings[item.account], severity, verdict, materials, tally, inputs),
            )
        for item in self._ordered(opened, judgement):
            verdict = judgement.verdicts.get(item.expectation_id) if judgement is not None else None
            standing = standings[item.account]
            severity = severities.get(item.account)
            self._record(tally, item, self._opening(item, standing, severity, verdict, materials, tally, inputs))
        changes = self._record_levels(standings, now)
        if judgement is not None:
            calls += judgement.calls
            signals = (*signals, *judgement.signals)
        return DeliveryReport(
            now,
            tally.decided,
            tally.spoken,
            tally.by_reason,
            tally.by_level,
            judged=0
            if judgement is None
            else len(judgement.verdicts)
            if not judgement.failed
            else len(judgement.order),
            severities=judged_accounts,
            calls=calls,
            signals=signals,
            judge_failed=severity_failed or (judgement is not None and judgement.failed),
            level_changes=changes,
            unreadable=len(self.ledger.take_unreadable()),
        )

    # ── 一条预期的决定 ──────────────────────────────────────────────────────

    def _opening(
        self,
        item: Expectation,
        standing: Standing,
        severity: AccountSeverity | None,
        verdict: Verdict | None,
        materials: JudgeMaterials | None,
        tally: _Tally,
        inputs: Mapping[str, Sequence[UnitInput]],
    ) -> Delivery:
        """时窗开了：按账的档决定；重的按模型的答案当场说，轻的记「等时窗末」（那时再问模型、再写话）。"""

        now = tally.now
        level = standing.level
        evidence = verdict.evidence if verdict is not None else ()
        if materials is not None and materials.happened(item):
            return Delivery(item.expectation_id, now, level, False, "已发生", 0.0, evidence=evidence)
        if level == 0:
            return Delivery(item.expectation_id, now, 0, False, "影子", 0.0, evidence=evidence)
        if not self.reachable:
            return Delivery(item.expectation_id, now, level, False, "不可达", 0.0, evidence=evidence)
        if not _urgent(severity, now) and now <= item.window[1]:
            # 该来了还没来才说：到时窗末再决定、再问模型
            return Delivery(item.expectation_id, now, level, False, _PENDING, 0.0)
        return self._speak(item, level, severity, verdict, materials, tally, inputs)

    def _closing(
        self,
        item: Expectation,
        pending: Delivery,
        standing: Standing,
        severity: AccountSeverity | None,
        verdict: Verdict | None,
        materials: JudgeMaterials | None,
        tally: _Tally,
        inputs: Mapping[str, Sequence[UnitInput]],
    ) -> Delivery:
        """时窗末到了的轻的事：还没做、这本账此刻还在提醒档就说；做了记「已发生」；窗里被压回影子或关了账的记「影子」。"""

        now = tally.now
        if materials is not None and materials.happened(item):
            return Delivery(item.expectation_id, now, pending.level, False, "已发生", 0.0)
        if standing.level == 0:
            return Delivery(item.expectation_id, now, 0, False, "影子", 0.0)
        return self._speak(item, standing.level, severity, verdict, materials, tally, inputs)

    def _speak(
        self,
        item: Expectation,
        level: int,
        severity: AccountSeverity | None,
        verdict: Verdict | None,
        materials: JudgeMaterials | None,
        tally: _Tally,
        inputs: Mapping[str, Sequence[UnitInput]],
    ) -> Delivery:
        now = tally.now
        evidence = verdict.evidence if verdict is not None else ()
        if not self._antecedent_holds(item, materials):
            return Delivery(item.expectation_id, now, level, False, "前因不成立", 0.0, evidence=evidence)
        again = self._cause_again(item, materials, now)
        if again is not None:
            return Delivery(item.expectation_id, now, level, False, AGAIN, 0.0, evidence=(f"{CITE}{again}",))
        if verdict is not None and not verdict.kept:
            return Delivery(item.expectation_id, now, level, False, "模型判不像", 0.0, evidence=evidence)
        if item.kind_token in tally.spoken_kinds:
            return Delivery(item.expectation_id, now, level, False, "同类已说", 0.0, evidence=evidence)
        return decide(
            item,
            level,
            now=now,
            spoken_today=tally.spoken_today,
            daily_cap=self.daily_cap,
            holdout_rate=self._holdout(severity, now),
            draw=self._draw(),
            text=self._text(item, verdict, materials, now, inputs),
            evidence=evidence,
        )

    @staticmethod
    def _antecedent_holds(item: Expectation, materials: JudgeMaterials | None) -> bool:
        """开口之前再核一遍事件触发的前因：它可能是没封口时建的，封口后归成了别的类、或成了非事件（找不到了）。"""

        if item.source != "事件" or materials is None or materials.table is None:
            return True
        relation = next((r for r in materials.table.relations if r.key == item.trigger), None)
        seen = materials.rows_for((item.trigger_instance,)).get(item.trigger_instance)
        return antecedent_holds(
            item,
            seen.kind_token if seen is not None else None,
            relation.antecedent_classes if relation is not None else None,
            current=lambda token: (token,),
        )

    def _cause_again(self, item: Expectation, materials: JudgeMaterials | None, now: datetime) -> str | None:
        """这本账最近的落空里，原因此刻对这条预期又成立的那一次（预期身份）；没有是 None。

        "最近"借遗忘的半衰期（生命周期统一设定之前各层都借它）。一条原因被用过、而因此没说的那一次他其实自己做了，
        这条原因就错了，不再用（下一次的答案验这次调整）。"""

        if materials is None or materials.table is None:
            return None
        table = materials.table
        horizon = timedelta(days=self.evidence.half_life_days)
        facts = current_facts(self.ledger.facts_of(item.account))
        causes = sorted(
            (
                fact
                for fact in facts
                if fact.axis == PREDICTION
                and fact.cause is not None
                and fact.kind == "过期待决"
                and fact.expectation_id != item.expectation_id
                and now - horizon <= fact.at <= now
            ),
            key=lambda fact: fact.at,
            reverse=True,
        )
        for fact in causes:
            assert fact.cause is not None
            if self._disproved(fact, facts):
                continue
            if cause_holds(
                fact.cause,
                item,
                materials.seen,
                now=now,
                lane=table.lane,
                slot_minutes=table.slot_minutes,
                window_slots=table.window_slots,
                day_note=materials.day_note,
            ):
                return fact.expectation_id
        return None

    def _disproved(self, cause: AccountFact, facts: Iterable[AccountFact]) -> bool:
        """这条原因用过、而因此没说的那一次他自己做了。"""

        cited = f"{CITE}{cause.expectation_id}"
        for fact in facts:
            if fact.axis != PREDICTION or fact.kind not in _NATURAL_HITS or fact.at <= cause.at:
                continue
            day = self.ledger.day_of(fact.expectation_id)
            if day is None:
                continue
            if any(
                delivery.expectation_id == fact.expectation_id and delivery.reason == AGAIN and cited in delivery.evidence
                for delivery in self.ledger.deliveries_on(day)
            ):
                return True
        return False

    def _holdout(self, severity: AccountSeverity | None, now: datetime) -> float:
        """漏一次比错提一次代价大的（c > 1）用较小的扣发比例，其余用较大的（裁定 58 第 7 条：都扣，重的扣得少）。"""

        return self.holdout_rate_heavy if _urgent(severity, now) else self.holdout_rate

    def _record(self, tally: _Tally, item: Expectation, delivery: Delivery) -> None:
        if not self.ledger.record_delivery(item.day, delivery):
            return
        tally.count(delivery, item.kind_token)

    # ── 账 ──────────────────────────────────────────────────────────────────

    def standing_of(self, expectation: Expectation, now: datetime) -> Standing:
        """这条预期所在那本账此刻的位置（运维与回放用）。"""

        return self._standing(expectation.account, now, None)

    def _standing(self, account: str, now: datetime, judged: AccountSeverity | None) -> Standing:
        return climb(
            self.ledger.facts_of(account),
            tolerance=self.tolerance,
            evidence=self.evidence,
            now=now,
            severity=judged if judged is not None else self.ledger.severity_of(account),
            setting=self.ledger.setting_of(account),
            previous=self.ledger.level_of(account),
        )

    def _record_levels(self, standings: Mapping[str, Standing], now: datetime) -> int:
        """档变了才记一条（滞回要知道上一次在哪一档；可见面要知道哪天升的）。"""

        changes = 0
        for account, standing in standings.items():
            previous = self.ledger.level_of(account)
            if (previous.level if previous is not None else 0) != standing.level:
                if self.ledger.record_level(AccountLevel(account, now, standing.level)):
                    changes += 1
        return changes

    def account_views(self, now: datetime, *, recent: int = 5) -> tuple[AccountView, ...]:
        """全部账此刻的样子（可见面的只读口；可见面本身在层外）。"""

        views = []
        for account in self.ledger.accounts():
            facts = self.ledger.facts_of(account)
            severity = self.ledger.severity_of(account)
            standing = self._standing(account, now, severity)
            tail = tuple(
                f"{item.at:%m-%d} {item.axis}{'好' if item.good else '坏'} {item.kind}" for item in facts[-recent:]
            )
            views.append(AccountView(account, standing, severity, len(facts), tail))
        return tuple(views)

    def unit_views(self, now: datetime, table: OpportunityTable | None) -> tuple[UnitView, ...]:
        """全部行为单元此刻的样子（可见面的只读口）：每个行为的每个输入——账此刻的位置、语义树读数、两个数对不对得上。"""

        by_kind: dict[str, list[tuple[str, Standing, TreeReading | None]]] = {}
        for account in self.ledger.accounts():
            kind, source, trigger = parse_account(account)
            standing = self._standing(account, now, self.ledger.severity_of(account))
            relation = (
                next((item for item in table.relations if item.key == trigger), None)
                if table is not None and source == "事件"
                else None
            )
            by_kind.setdefault(kind, []).append((account, standing, relation.reading if relation is not None else None))
        return UnitPlanner.views(by_kind)

    def omission_views(self, now: datetime, *, days: int = 30) -> tuple[OmissionView, ...]:
        """最近 ``days`` 天的漏报按类汇总，次数多的在前（可见面的只读口）。"""

        found: dict[str, list[tuple[datetime, str, bool]]] = {}
        for offset in range(days, -1, -1):
            for item in self.ledger.omissions_on(now.date() - timedelta(days=offset)):
                found.setdefault(item.kind_token, []).append(
                    (item.noticed_at, item.occurrence_uri, item.had_expectation)
                )
        views = [
            OmissionView(
                kind_token=kind,
                count=len(items),
                with_expectation=sum(1 for _at, _uri, had in items if had),
                latest_uri=max(items)[1],
            )
            for kind, items in found.items()
        ]
        return tuple(sorted(views, key=lambda view: (-view.count, view.kind_token)))

    # ── 模型 ────────────────────────────────────────────────────────────────

    async def _severities(
        self, due: tuple[Expectation, ...], materials: JudgeMaterials | None, now: datetime
    ) -> tuple[dict[str, AccountSeverity], int, tuple[str, ...], bool, int]:
        """到了时窗的账里还没判过或已到期的：判一次轻重缓急、落账。没有渠道或模型就不判。"""

        found: dict[str, AccountSeverity] = {}
        if self.judge is None or materials is None or not self.reachable:
            return found, 0, (), False, 0
        asked: dict[str, Expectation] = {}
        for item in due:
            if item.account in asked or item.account in found:
                continue
            current = self.ledger.severity_of(item.account)
            if current is not None and current.current(now):
                found[item.account] = current
                continue
            # 第一次到了该问时就判，不等这本账结算过一次（一年几次的事等一次结算就是几个月，裁定 58 第 1 条）
            asked[item.account] = item
        if not asked:
            return found, 0, (), False, 0
        package = self._package(tuple(asked.values()), materials, now, {})
        result = await self.judge.judge_severity(package)
        if result.failed:
            return found, result.calls, result.signals, True, 0
        for account, item in asked.items():
            verdict = result.verdicts.get(item.expectation_id)
            if verdict is None:
                continue
            record = AccountSeverity(account, now, verdict[0], verdict[1], now + self.severity_review)
            self.ledger.record_severity(record)
            found[account] = record
        return found, result.calls, result.signals, False, len(asked)

    async def _judge(
        self,
        due: tuple[Expectation, ...],
        standings: Mapping[str, Standing],
        materials: JudgeMaterials | None,
        now: datetime,
        tally: _Tally,
        inputs: Mapping[str, Sequence[UnitInput]],
    ) -> Judgement | None:
        """这一拍真要开口的那几个交模型（提醒档、还没做、同一类今天还没说过、名额没满）；没有模型、没有材料、没有渠道就不问。"""

        if self.judge is None or materials is None or not self.reachable:
            return None
        if tally.spoken_today >= self.daily_cap:
            return None
        asked = tuple(
            item
            for item in due
            if standings[item.account].level >= 1
            and item.kind_token not in tally.spoken_kinds
            and not materials.happened(item)
        )
        if not asked:
            return None
        levels = {item.expectation_id: standings[item.account].level for item in asked}
        return await self.judge.judge(self._package(asked, materials, now, levels, inputs))

    @staticmethod
    def _package(
        items: tuple[Expectation, ...],
        materials: JudgeMaterials,
        now: datetime,
        levels: Mapping[str, int],
        inputs: Mapping[str, Sequence[UnitInput]] | None = None,
    ) -> Package:
        rows = materials.rows_for(sources_of(items, materials.table, units=inputs))
        return build_package(
            items, materials.table, rows, now=now, levels=levels, day_note=materials.day_note, units=inputs
        )

    @staticmethod
    def _ordered(due: tuple[Expectation, ...], judgement: Judgement | None) -> tuple[Expectation, ...]:
        """模型排的先后在前（名额先给它们），其余按时窗。"""

        if judgement is None:
            return due
        by_id = {item.expectation_id: item for item in due}
        first = [by_id[identity] for identity in judgement.order if identity in by_id]
        rest = [item for item in due if item.expectation_id not in judgement.order]
        return tuple(first + rest)

    @staticmethod
    def _text(
        item: Expectation,
        verdict: Verdict | None,
        materials: JudgeMaterials | None,
        now: datetime,
        inputs: Mapping[str, Sequence[UnitInput]],
    ) -> str:
        if verdict is not None and verdict.text:
            return verdict.text
        table = materials.table if materials is not None else None
        rows = materials.rows_for(sources_of((item,), table, units=inputs)) if materials is not None else {}
        package = build_package((item,), table, rows, now=now, levels={}, units=inputs)
        return plain_text(package.candidates[0])

    # ── 读账 ────────────────────────────────────────────────────────────────

    def _due(self, now: datetime) -> _Due:
        """三拨：时窗与这一拍的区间（上一拍到此刻）有交集、还没结算、账上还没有决定的（开了）；决定过「等时窗末」、此刻过了
        时窗终点、还没到最晚时限的（该来了还没来）；还没到时窗、没决定过的（单元从这里拉同一刻的另外几个输入）。过了时窗终点
        还没决定过的不再递：那是结算去记的事。另记每条依据替哪几条一并决定过（「单元已选」引它）：时窗末开口时小包与话照样
        带上同一刻的全部输入。"""

        opened: list[Expectation] = []
        closing: list[tuple[Expectation, Delivery]] = []
        waiting: list[Expectation] = []
        members: dict[str, list[Expectation]] = {}
        instant = now.astimezone(UTC)
        since = instant - self.tick
        for day in self.ledger.open_days():
            if day > now.date():
                continue
            decisions: dict[str, list[Delivery]] = {}
            for delivery in self.ledger.deliveries_on(day):
                decisions.setdefault(delivery.expectation_id, []).append(delivery)
            for item in self.ledger.unsettled_expectations(day):
                made = decisions.get(item.expectation_id, [])
                for cite in (cited for made_one in made if made_one.reason == UNIT for cited in made_one.evidence):
                    members.setdefault(cite.removeprefix(UNIT_CITE), []).append(item)
                start, end = (item.window[0].astimezone(UTC), item.window[1].astimezone(UTC))
                if not made:
                    if start <= instant and end > since:
                        opened.append(item)
                    elif start > instant:
                        waiting.append(item)
                    continue
                pending = [d for d in made if d.reason == _PENDING]
                if pending and len(pending) == len(made) and end < instant <= item.deadline.astimezone(UTC):
                    closing.append((item, pending[-1]))
        return _Due(tuple(opened), tuple(closing), tuple(waiting), {key: tuple(value) for key, value in members.items()})

    def _spoken_today(self, now: datetime) -> tuple[int, frozenset[str]]:
        """今天（按决定的日子）已经说过几次、说过哪些类。说出口的递送挂在预期所在的日子下：今天决定的，那条预期还没结（结算
        在它最晚时限那天定稿之后），所以只看还没结完的日子。"""

        today = now.date()
        spoken = 0
        kinds: set[str] = set()
        for day in self.ledger.open_days():
            deliveries = [item for item in self.ledger.deliveries_on(day) if item.spoken]
            if not deliveries:
                continue
            by_id = {item.expectation_id: item.kind_token for item in self.ledger.expectations_on(day)}
            for delivery in deliveries:
                if delivery.decided_at.astimezone(now.tzinfo).date() == today:
                    spoken += 1
                    kinds.add(by_id.get(delivery.expectation_id, ""))
        return spoken, frozenset(kinds)


@dataclass(frozen=True)
class _Due:
    opened: tuple[Expectation, ...]
    closing: tuple[tuple[Expectation, Delivery], ...]
    waiting: tuple[Expectation, ...]
    #: 依据的预期身份 → 它在「单元已选」里替哪几条一并决定过。
    members: Mapping[str, tuple[Expectation, ...]]


def _unit_inputs(unit: Unit) -> tuple[UnitInput, ...]:
    """单元的全部输入给小包：每条预期与它那本账预测轴上（遗忘之后的）说中几次、几次机会。"""

    return tuple(
        UnitInput(item.expectation, item.standing.prediction.k, item.standing.prediction.n) for item in unit.inputs
    )


def _urgent(severity: AccountSeverity | None, now: datetime) -> bool:
    """漏一次比错提一次代价大（此刻生效的代价比 c > 1；过期的按没判过）：时窗一开就说，不等时窗末。1 是好处等于代价的临界点。"""

    return severity is not None and severity.current(now) and severity.cost_ratio > 1.0


class _Tally:
    """这一拍的计数：决定了几条、说了几条、今天说过几次、今天说过哪些类。"""

    def __init__(self, now: datetime, spoken_today: int, spoken_kinds: frozenset[str]) -> None:
        self.now = now
        self.spoken_today = spoken_today
        self.spoken_kinds = set(spoken_kinds)
        self.decided = 0
        self.spoken = 0
        self.by_reason: dict[str, int] = {}
        self.by_level: dict[int, int] = {}

    def count(self, delivery: Delivery, kind_token: str) -> None:
        self.decided += 1
        self.by_reason[delivery.reason] = self.by_reason.get(delivery.reason, 0) + 1
        self.by_level[delivery.level] = self.by_level.get(delivery.level, 0) + 1
        if delivery.spoken:
            self.spoken += 1
            self.spoken_today += 1
            self.spoken_kinds.add(kind_token)


__all__ = ["AccountView", "DeliveryReport", "DeliveryStage", "JudgeMaterials", "OmissionView"]
