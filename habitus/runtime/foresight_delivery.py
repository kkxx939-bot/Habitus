"""递送那一拍（方案第二节、第四节后半；裁定 48–51、56、58）：到了时窗的预期按**行为单元**拢在一起，每个单元决定一次说不说，
记进账本（说没说都记、记原因）。

**单元**（裁定 56 第 12 条，``runtime.foresight_units``）：同一个行为这一刻的全部输入——到了时窗的那几条预期，加上说中的范围
和它们连成一片、还没到时窗的那几条——拢成一个单元；单元读数 = 这一组在场时历史上下一步来了多少，按依据那本账的线过线。
依据那一条照下面的流程走（时窗开 / 时窗末、模型、名额）；其余几条记「单元已选」并引依据——它们照旧各自结算，各前因的权重
（账）仍各自随结算调。同一刻后来才形成的同一类预期（说中的范围和已决定过的依据相连）也并进它，不另起一个单元。历史是夜里按当晚
语义树重标过的那一份（``runtime.foresight_history``），白天不扫账本。

档不存成状态：每次算出来——边的级（``foresight.ladder.climb``，从单元历史里这条边的那几刻读）与单元的档
（``foresight.units.unit_level``：单元读数、单元的提醒反馈、单元的设置）分开，各自只在变的时候记一条（账的级、``UnitLevel``）
给滞回与可见面用。白天不扫账本：边从夜里重标过的单元历史读，提醒反馈与设置从 ``units/<行为>/`` 读（裁定 61）。

一条预期到了时窗决定一次（时窗与这一拍的区间有交集就算到了：一拍一槽，时窗比一槽短的也落得上），账上已有决定的不再决定；
只有「等时窗末」不是终局——到时窗末还没做再说（「该来了还没来」是提醒的说话时机；这一版不分急缓，裁定 58 第 1 条）。到时窗末那一拍
按依据与它当时一并决定的那几条重算一次单元（窗里设了「以后别提了」就不说了），这时才问模型、才写那句话：
话里的"今天""这会儿"说的是开口那一刻。
到了时窗时这件事已经做了记「已发生」；同一类今天已经说过一句的记「同类已说」；没有宿主活着的记「不可达」（方案 32 第 3.3 条：最近
一段没有宿主来取过提醒就不问模型、不写话，结算按没说过处理）。每日上限按决定的日子数，「待取」也占名额。

模型只在有宿主活着时问（没人来取，问了也说不出口）：**像不像、先说哪个、一句话**——只问这一拍真要开口的（提醒档、时窗末还没做、
同一类今天还没说过、名额没满）。一拍一个小包。两遍都说不像的记「模型判不像」；像的按模型的先后占名额、用模型的话；模型不可用
按算法顺序、模板措辞。
送达在本层之外（``runtime.foresight_outlet``）：这里产出的是"递送请求 + 档 + 话"，记「待取」、每拍把可取的请求清单交给出口；
「说了」由出口在转写里认出提问后追加。「待取」之后每拍只做两件事：做了追加「已发生」、过了最晚时限还没人问出口追加「宿主没取」。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from habitus.foresight import Evidence, Seen, Standing, Tolerance, climb, decide
from habitus.foresight.absence import returned_after_absence
from habitus.foresight.feedback import feedback_reading
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
from habitus.foresight.ledger import (
    AWAITING_HOST,
    HOST_NEVER_CLAIMED,
    RETURNED,
    AccountLevel,
    Delivery,
    Expectation,
    UnitLevel,
    antecedent_holds,
    parse_account,
    relation_backed,
)
from habitus.foresight.opportunities import OpportunityTable, TreeReading
from habitus.foresight.reliability import Reliability
from habitus.foresight.units import UNIT, UNIT_CITE
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_outlet import OpenRequest, OutletPort
from habitus.runtime.foresight_units import Unit, UnitPlanner, UnitView

#: 不是终局的递送原因：之后还会再决定一次。
_PENDING = "等时窗末"
#: 优先于久别的设置：人说了"以后都提醒我"，就不因为久别回来而先看一次。
_ALWAYS = "以后都提醒我"


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


@dataclass(frozen=True)
class AccountView:
    """可见面读的一本账（裁定 50）：是什么、此刻在哪一档、两条轴的读数、设置、最近几条事实。"""

    account: str
    standing: Standing
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
        judge: Judge | None = None,
        tick_seconds: float = 900.0,
        outlet: OutletPort | None = None,
        unit_prior_share: float = 1 / 3,
        history: HistoryStore | None = None,
        lane: str = "session",
    ) -> None:
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        if not isinstance(tolerance, Tolerance) or not isinstance(evidence, Evidence):
            raise TypeError("tolerance must be a Tolerance and evidence an Evidence")
        if isinstance(daily_cap, bool) or not isinstance(daily_cap, int) or daily_cap < 1:
            raise ValueError("daily_cap must be a positive integer")
        if isinstance(tick_seconds, bool) or not isinstance(tick_seconds, int | float) or tick_seconds <= 0:
            raise ValueError("tick_seconds must be a positive number")
        if judge is not None and not isinstance(judge, Judge):
            raise TypeError("judge must be a Judge or None")
        if outlet is not None and not (
            callable(getattr(outlet, "alive", None)) and callable(getattr(outlet, "publish", None))
        ):
            raise TypeError("outlet must implement alive() and publish()")
        self.ledger = ledger
        self.tolerance = tolerance
        self.evidence = evidence
        self.daily_cap = daily_cap
        self.judge = judge
        self.tick = timedelta(seconds=float(tick_seconds))
        #: 提醒出口（没有 = 没有宿主活着：提醒档到时窗末都「不可达」，也不问模型）。
        self.outlet = outlet
        self.units = UnitPlanner(history, evidence=evidence, prior_share=unit_prior_share, lane=lane)

    async def run_once(self, now: datetime, materials: JudgeMaterials | None = None) -> DeliveryReport:
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise ValueError("now must be a timezone-aware datetime")
        found = self._due(now)
        table = materials.table if materials is not None else None
        calls = 0
        signals: tuple[str, ...] = ()
        cached: dict[str, Standing] = {}
        feedbacks: dict[str, Reliability | None] = {}
        settings: dict[str, str | None] = {}

        def standing_of(item: Expectation) -> Standing:
            if item.account not in cached:
                cached[item.account] = self._standing(item.account, item.kind_token, now, table)
            return cached[item.account]

        def previous_unit(kind: str) -> int | None:
            level = self.ledger.unit_level_of(kind)
            return level.level if level is not None else None

        def feedback_of(kind: str) -> Reliability | None:
            if kind not in feedbacks:
                feedbacks[kind] = self._feedback(kind, table)
            return feedbacks[kind]

        def setting_of(kind: str) -> str | None:
            if kind not in settings:
                settings[kind] = self._setting(kind, now)
            return settings[kind]

        units = self.units.plan(
            found.opened,
            found.waiting,
            standing_of=standing_of,
            previous_level=previous_unit,
            feedback_of=feedback_of,
            setting_of=setting_of,
            table=table,
            now=now,
        )
        # 时窗末：按依据与它当时一并决定的那几条重算单元（2026-10-17 四方评审：原来改按依据那本账自己的档判，几个前因一起才稳的
        # 轻事时窗开时单元说该说、时窗末又成了影子）
        closing_units = {
            item.expectation_id: self.units.unit_for(
                (item, *found.members.get(item.expectation_id, ())),
                due=(item.expectation_id,),
                standing_of=standing_of,
                previous_level=previous_unit,
                feedback_of=feedback_of,
                setting_of=setting_of,
                table=table,
                now=now,
            )
            for item, _pending in found.closing
        }
        every = (*units, *closing_units.values())
        standings = {unit.basis.expectation_id: unit.standing for unit in every}
        inputs: dict[str, tuple[UnitInput, ...]] = {unit.basis.expectation_id: _unit_inputs(unit) for unit in every}
        spoken_today, spoken_kinds = self._spoken_today(now)
        tally = _Tally(now, spoken_today, spoken_kinds, alive=self.outlet is not None and self.outlet.alive(now))
        for item, pending in found.awaiting:
            self._record(tally, item, self._awaiting(item, pending, materials, now))
        for unit in units:
            for member in unit.members:
                self._subsume(tally, member, unit.basis.expectation_id, unit.level)
        for item, cited, level in found.joins:
            self._subsume(tally, item, cited, level)
        speaking = tuple(item for item, _pending in found.closing)
        judgement = await self._judge(speaking, standings, materials, now, tally, inputs)
        for item, pending in self._ordered_closing(found.closing, judgement):
            verdict = judgement.verdicts.get(item.expectation_id) if judgement is not None else None
            self._record(
                tally,
                item,
                self._closing(item, pending, standings[item.expectation_id], verdict, materials, tally, inputs),
            )
        for unit in units:
            self._record(
                tally, unit.basis, self._opening(unit.basis, standings[unit.basis.expectation_id], materials, tally)
            )
        changes = self._record_levels(cached, every, now)
        if judgement is not None:
            calls += judgement.calls
            signals = (*signals, *judgement.signals)
        open_requests = self._publish(found, tally, materials, now)
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
            calls=calls,
            signals=signals,
            judge_failed=judgement is not None and judgement.failed,
            level_changes=changes,
            unreadable=len(self.ledger.take_unreadable()),
            open_requests=open_requests,
        )

    def _publish(self, found: _Due, tally: _Tally, materials: JudgeMaterials | None, now: datetime) -> int:
        """这一拍末尾把可取的请求清单交给出口：之前「待取」还没终局的，加上这一拍刚记「待取」的。"""

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
            )
            for item, pending in (*found.awaiting, *tally.awaiting)
            if now <= item.deadline and pending.text and not (materials is not None and materials.happened(item))
        ]
        return self.outlet.publish(rows, now=now)

    # ── 一条预期的决定 ──────────────────────────────────────────────────────

    def _opening(
        self, item: Expectation, standing: Standing, materials: JudgeMaterials | None, tally: _Tally
    ) -> Delivery:
        """时窗开了：按档决定；依据那本账久别回来的记「久别回来」；提醒档一律记「等时窗末」——到时窗末还没做再问模型、再写话（这一版不分急缓，
        裁定 58 第 1 条）。有没有宿主活着到时窗末才看：时窗开时没人、末了打开了 Codex 的那条照样要问。"""

        now = tally.now
        level = standing.level
        if materials is not None and materials.happened(item):
            return Delivery(item.expectation_id, now, level, False, "已发生", 0.0)
        if level == 0:
            return Delivery(item.expectation_id, now, 0, False, "影子", 0.0)
        if self._returned(item, standing, materials):
            return Delivery(item.expectation_id, now, level, False, RETURNED, 0.0)
        return Delivery(item.expectation_id, now, level, False, _PENDING, 0.0)

    def _closing(
        self,
        item: Expectation,
        pending: Delivery,
        standing: Standing,
        verdict: Verdict | None,
        materials: JudgeMaterials | None,
        tally: _Tally,
        inputs: Mapping[str, Sequence[UnitInput]],
    ) -> Delivery:
        """时窗末到了：还没做、此刻还在提醒档就说；做了记「已发生」；窗里被压回影子或关了账的记「影子」；久别回来的记「久别回来」
        （时窗开时判过一次；跨夜的时窗换了机会表，这里再判一次）。"""

        now = tally.now
        if materials is not None and materials.happened(item):
            return Delivery(item.expectation_id, now, pending.level, False, "已发生", 0.0)
        if standing.level == 0:
            return Delivery(item.expectation_id, now, 0, False, "影子", 0.0)
        if self._returned(item, standing, materials):
            return Delivery(item.expectation_id, now, standing.level, False, RETURNED, 0.0)
        return self._speak(item, standing.level, verdict, materials, tally, inputs)

    def _speak(
        self,
        item: Expectation,
        level: int,
        verdict: Verdict | None,
        materials: JudgeMaterials | None,
        tally: _Tally,
        inputs: Mapping[str, Sequence[UnitInput]],
    ) -> Delivery:
        now = tally.now
        evidence = verdict.evidence if verdict is not None else ()
        if not self._antecedent_holds(item, materials):
            return Delivery(item.expectation_id, now, level, False, "前因不成立", 0.0, evidence=evidence)
        if verdict is not None and not verdict.kept:
            return Delivery(item.expectation_id, now, level, False, "模型判不像", 0.0, evidence=evidence)
        if item.kind_token in tally.spoken_kinds:
            return Delivery(item.expectation_id, now, level, False, "同类已说", 0.0, evidence=evidence)
        if not tally.alive:
            return Delivery(item.expectation_id, now, level, False, "不可达", 0.0, evidence=evidence)
        return decide(
            item,
            level,
            now=now,
            spoken_today=tally.spoken_today,
            daily_cap=self.daily_cap,
            text=self._text(item, verdict, materials, now, inputs),
            evidence=evidence,
            awaiting_host=True,
        )

    @staticmethod
    def _awaiting(
        item: Expectation, pending: Delivery, materials: JudgeMaterials | None, now: datetime
    ) -> Delivery | None:
        """「待取」之后每拍只做两件事：做了追加「已发生」；过了最晚时限还没人问出口追加「宿主没取」（终局，结算按没说过处理）。"""

        if materials is not None and materials.happened(item):
            return Delivery(item.expectation_id, now, pending.level, False, "已发生", 0.0)
        if now > item.deadline:
            return Delivery(item.expectation_id, now, pending.level, False, HOST_NEVER_CLAIMED, 0.0)
        return None

    def _returned(self, item: Expectation, standing: Standing, materials: JudgeMaterials | None) -> bool:
        """依据那本账久别回来：这一次只记影子，不改账的级、不写单元的档（26 号清单 3.3）；「以后都提醒我」优先。"""

        if standing.setting == _ALWAYS:
            return False
        table = materials.table if materials is not None else None
        return returned_after_absence(
            self.ledger.facts_of(item.account), item, table, self.ledger.compaction_of(item.account)
        )

    @staticmethod
    def _antecedent_holds(item: Expectation, materials: JudgeMaterials | None) -> bool:
        """开口之前再核一遍事件触发的前因：它可能是没封口时建的，封口后归成了别的类、或成了非事件（找不到了）。"""

        if item.source != "事件" or materials is None or materials.table is None:
            return True
        relation = materials.table.relation(item.trigger)
        seen = materials.rows_for((item.trigger_instance,)).get(item.trigger_instance)
        return antecedent_holds(
            item,
            seen.kind_token if seen is not None else None,
            relation.antecedent_classes if relation is not None else None,
            current=lambda token: (token,),
        )

    def _subsume(self, tally: _Tally, item: Expectation, basis_id: str, level: int) -> None:
        """同一刻同一个行为的另一个输入：单元替它一并决定，记「单元已选」并引依据。"""

        self._record(
            tally,
            item,
            Delivery(item.expectation_id, tally.now, level, False, UNIT, 0.0, evidence=(f"{UNIT_CITE}{basis_id}",)),
        )

    def _record(self, tally: _Tally, item: Expectation, delivery: Delivery | None) -> None:
        if delivery is None or not self.ledger.record_delivery(item.day, delivery):
            return
        tally.count(delivery, item.kind_token)
        if delivery.reason == AWAITING_HOST:
            tally.awaiting.append((item, delivery))

    # ── 账 ──────────────────────────────────────────────────────────────────

    def standing_of(self, expectation: Expectation, now: datetime, table: OpportunityTable | None = None) -> Standing:
        """这条预期那条边此刻的位置（运维与回放用）。"""

        return self._standing(expectation.account, expectation.kind_token, now, table)

    def _standing(self, account: str, kind: str, now: datetime, table: OpportunityTable | None) -> Standing:
        """一条边此刻的位置：从这个行为的单元历史（夜里重标过的最晚一晚）里签名含它的那几刻读。"""

        history = self.units.history_of(table)
        return climb(
            history.of(kind) if history is not None else (),
            account,
            now=now,
            tolerance=self.tolerance,
            evidence=self.evidence,
            setting=self.ledger.setting_of(kind),
            previous=self.ledger.level_of(account),
        )

    def _feedback(self, kind: str, table: OpportunityTable | None) -> Reliability | None:
        """单元的提醒反馈读数：反馈那几笔，加上单元历史里他没做的那几刻推时钟（``feedback.feedback_reading``）。"""

        facts = self.ledger.feedback_of(kind)
        if not facts:
            return None
        history = self.units.history_of(table)
        return feedback_reading(
            facts,
            moments=history.labelled(kind) if history is not None else (),
            alpha=self.evidence.alpha,
            half_life=self.evidence.half_life,
            window=self.evidence.window,
        )

    def _setting(self, kind: str, now: datetime) -> str | None:
        setting = self.ledger.setting_of(kind)
        if setting is None or setting.at > now or setting.setting == "照常":
            return None
        return setting.setting

    def _record_levels(self, accounts: Mapping[str, Standing], units: Iterable[Unit], now: datetime) -> int:
        """档变了才记一条：账的级按这本账自己从事实折出的档（滞回要知道上一次在哪一档；可见面要知道哪天升的），
        行为单元的档另记（``UnitLevel``，按行为）——两者分开，账自己折档时的滞回不读单元的档。"""

        changes = 0
        for account, standing in accounts.items():
            previous = self.ledger.level_of(account)
            if (previous.level if previous is not None else 0) != standing.level:
                if self.ledger.record_level(AccountLevel(account, now, standing.level)):
                    changes += 1
        latest: dict[str, int] = {}
        for unit in units:
            latest[unit.kind_token] = unit.level
        for kind, level in latest.items():
            previous_unit = self.ledger.unit_level_of(kind)
            if (previous_unit.level if previous_unit is not None else 0) != level:
                if self.ledger.record_unit_level(UnitLevel(kind, now, level)):
                    changes += 1
        return changes

    def account_views(
        self, now: datetime, table: OpportunityTable | None = None, *, recent: int = 5
    ) -> tuple[AccountView, ...]:
        """全部账此刻的样子（可见面的只读口；可见面本身在层外）：边的位置从单元历史读，流水里最近几笔给人看。"""

        views = []
        for account in self.ledger.accounts():
            facts = self.ledger.facts_of(account)
            kind, _source, _trigger = parse_account(account)
            standing = self._standing(account, kind, now, table)
            tail = tuple(f"{item.at:%m-%d} {'好' if item.good else '坏'} {item.kind}" for item in facts[-recent:])
            views.append(AccountView(account, standing, len(facts), tail))
        return tuple(views)

    def unit_views(self, now: datetime, table: OpportunityTable | None) -> tuple[UnitView, ...]:
        """全部行为单元此刻的样子（可见面的只读口）：每个行为的每个输入——边此刻的位置、语义树读数、两个数对不对得上。"""

        by_kind: dict[str, list[tuple[str, Standing, TreeReading | None]]] = {}
        for account in self.ledger.accounts():
            kind, source, trigger = parse_account(account)
            standing = self._standing(account, kind, now, table)
            relation = table.relation(trigger) if table is not None and relation_backed(source) else None
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

    async def _judge(
        self,
        due: tuple[Expectation, ...],
        standings: Mapping[str, Standing],
        materials: JudgeMaterials | None,
        now: datetime,
        tally: _Tally,
        inputs: Mapping[str, Sequence[UnitInput]],
    ) -> Judgement | None:
        """这一拍真要开口的那几个交模型（提醒档、还没做、同一类今天还没说过、名额没满）；没有模型、没有材料、没有宿主活着就不问。"""

        if self.judge is None or materials is None or not tally.alive:
            return None
        if tally.spoken_today >= self.daily_cap:
            return None
        asked = tuple(
            item
            for item in due
            if standings[item.expectation_id].level >= 1
            and item.kind_token not in tally.spoken_kinds
            and not materials.happened(item)
            and not self._returned(item, standings[item.expectation_id], materials)
        )
        if not asked:
            return None
        levels = {item.expectation_id: standings[item.expectation_id].level for item in asked}
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
    def _ordered_closing(
        closing: tuple[tuple[Expectation, Delivery], ...], judgement: Judgement | None
    ) -> tuple[tuple[Expectation, Delivery], ...]:
        """模型排的先后在前（名额先给它们），其余按时窗。"""

        if judgement is None:
            return closing
        by_id = {item.expectation_id: (item, pending) for item, pending in closing}
        first = [by_id[identity] for identity in judgement.order if identity in by_id]
        rest = [pair for pair in closing if pair[0].expectation_id not in judgement.order]
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
        """四拨：时窗与这一拍的区间（上一拍到此刻）有交集、还没结算、账上还没有决定的（开了）；决定过「等时窗末」、此刻过了
        时窗终点、还没到最晚时限的（该来了还没来）；还没到时窗、没决定过的（单元从这里拉同一刻的另外几个输入）；最近一条决定是
        「待取」、还没说出口的（等宿主）。过了时窗终点还没决定过的不再递：那是结算去记的事。另记每条依据替哪几条一并决定过
        （「单元已选」引它）：时窗末开口时小包与话照样带上同一刻的全部输入。"""

        fresh: list[Expectation] = []
        closing: list[tuple[Expectation, Delivery]] = []
        awaiting: list[tuple[Expectation, Delivery]] = []
        waiting: list[Expectation] = []
        members: dict[str, list[Expectation]] = {}
        #: 已经作为依据决定过、还没结的（类 → (预期, 最近那次决定的档)）：后来才形成的同一刻的输入并进它，不另起一个单元。
        bases: dict[str, list[tuple[Expectation, int]]] = {}
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
                        fresh.append(item)
                    elif start > instant:
                        waiting.append(item)
                    continue
                if all(made_one.reason != UNIT for made_one in made):
                    bases.setdefault(item.kind_token, []).append((item, made[-1].level))
                pending = [d for d in made if d.reason == _PENDING]
                if pending and len(pending) == len(made) and end < instant <= item.deadline.astimezone(UTC):
                    closing.append((item, pending[-1]))
                latest = max(made, key=lambda d: d.decided_at)
                if latest.reason == AWAITING_HOST and not any(d.spoken for d in made):
                    awaiting.append((item, latest))  # 等宿主来取：做了记「已发生」、过时限记「宿主没取」
        # 同一刻被拆成几个单元（2026-10-17 四方评审）：先到时窗的那条（日子、情境）已在时窗开时决定过，同一刻后来才形成的事件那条并进它
        opened: list[Expectation] = []
        joins: list[tuple[Expectation, str, int]] = []
        for item in fresh:
            base = next(
                (
                    (other, level)
                    for other, level in bases.get(item.kind_token, ())
                    if item.observed_from < other.deadline and item.deadline > other.observed_from
                ),
                None,
            )
            if base is None:
                opened.append(item)
            else:
                joins.append((item, base[0].expectation_id, base[1]))
                members.setdefault(base[0].expectation_id, []).append(item)
        return _Due(
            tuple(opened),
            tuple(closing),
            tuple(waiting),
            {key: tuple(value) for key, value in members.items()},
            tuple(joins),
            tuple(awaiting),
        )

    def _spoken_today(self, now: datetime) -> tuple[int, frozenset[str]]:
        """今天（按决定的日子）已经说过几次、说过哪些类——「待取」也算（话已经写好、等着问出口），同一条预期「待取」之后又来
        「说了」只算一次。说出口的递送挂在预期所在的日子下：今天决定的，那条预期还没结（结算在它最晚时限那天定稿之后），所以只看
        还没结完的日子。"""

        today = now.date()
        counted: set[str] = set()
        kinds: set[str] = set()
        for day in self.ledger.open_days():
            deliveries = [
                item for item in self.ledger.deliveries_on(day) if item.spoken or item.reason == AWAITING_HOST
            ]
            if not deliveries:
                continue
            by_id = {item.expectation_id: item.kind_token for item in self.ledger.expectations_on(day)}
            for delivery in deliveries:
                if (
                    delivery.decided_at.astimezone(now.tzinfo).date() == today
                    and delivery.expectation_id not in counted
                ):
                    counted.add(delivery.expectation_id)
                    kinds.add(by_id.get(delivery.expectation_id, ""))
        return len(counted), frozenset(kinds)


@dataclass(frozen=True)
class _Due:
    opened: tuple[Expectation, ...]
    closing: tuple[tuple[Expectation, Delivery], ...]
    waiting: tuple[Expectation, ...]
    #: 依据的预期身份 → 它在「单元已选」里替哪几条一并决定过。
    members: Mapping[str, tuple[Expectation, ...]]
    #: 这一拍刚到时窗、并进一个已经决定过的单元的：（这条, 依据的身份, 依据那次决定的档）。
    joins: tuple[tuple[Expectation, str, int], ...] = ()
    #: 最近一条决定是「待取」、还没说出口的：（这条, 那条「待取」）。
    awaiting: tuple[tuple[Expectation, Delivery], ...] = ()


def _unit_inputs(unit: Unit) -> tuple[UnitInput, ...]:
    """单元的全部输入给小包：每条预期与它那本账预测轴上（遗忘之后的）说中几次、几次机会。"""

    return tuple(
        UnitInput(item.expectation, item.standing.prediction.k, item.standing.prediction.n) for item in unit.inputs
    )


class _Tally:
    """这一拍的计数：决定了几条、说了几条、今天说过几次、今天说过哪些类。"""

    def __init__(self, now: datetime, spoken_today: int, spoken_kinds: frozenset[str], *, alive: bool = False) -> None:
        self.now = now
        self.spoken_today = spoken_today
        self.spoken_kinds = set(spoken_kinds)
        #: 有没有宿主活着（这一拍开头看一次）。
        self.alive = alive
        self.decided = 0
        self.spoken = 0
        self.by_reason: dict[str, int] = {}
        self.by_level: dict[int, int] = {}
        #: 这一拍刚记「待取」的（拍末交给出口）。
        self.awaiting: list[tuple[Expectation, Delivery]] = []

    def count(self, delivery: Delivery, kind_token: str) -> None:
        self.decided += 1
        self.by_reason[delivery.reason] = self.by_reason.get(delivery.reason, 0) + 1
        self.by_level[delivery.level] = self.by_level.get(delivery.level, 0) + 1
        if delivery.spoken:
            self.spoken += 1
        if delivery.spoken or delivery.reason == AWAITING_HOST:
            self.spoken_today += 1  # 「待取」也占名额、也算"同类已说"
            self.spoken_kinds.add(kind_token)


__all__ = ["AccountView", "DeliveryReport", "DeliveryStage", "JudgeMaterials", "OmissionView"]
