"""夜批：把语义树的五步按固定顺序跑一遍，外加开跑前的守门。

顺序是**线性的、不级联的**（派生树按固定先后处理已封口的历史，不做"输入又变了就重建"）：

```
B0  守门   配置对齐 + 孤儿结算 —— 不对就停下来报，不带着错的口径跑一夜
B1  重建树 （不在本模块：它排在 after_rebuild 钩子之前，本模块收到的是那一代树）
B1' 备料   刷旁册 → 建映射器 · 常态两个窗 · 情境材料
B2  映射   map_closed_day(… situation_for, baseline_for …)        ← 唯一的 LLM 触点
B1''桥     概念 → kind → 曲线（**放在映射之后**：今天的命中也算进去）→ 机会口
B3  开承诺 前件命中 → 扫全部假设 → 逐条开账，快照进承诺不再改
B4  结算   来了就结 / 等够看清的机会右删失 / 无节律型按 FIFO
B5  投影   读数 → 两面 → 残差 → profile/人物 → 落盘（**安慰剂不进投影**；「还立着的前提」那份投影 09-30 删了：前因与后果是两个行为，没结的账不是"前提"）
B6  闭环   补齐安慰剂 → 扫出"不一样"（调节 / 常态漂移）→ 触点③ 解释并提结构假设
           → 写进 hypotheses/，**从写入日起攒账、不回填**
```

**为什么桥在映射之后**：桥数的是"这个概念命中过哪些 kind"，今天刚映射出来的命中也该算；而它的消费者
（机会口）到 B3 才用得上。常态相反，它是 B2 的**材料**，而且按裁定**不含当天**（拿今天和自己比，
晚睡那条规则会被自己稀释），所以排在映射之前。

**机会口必须拿这一代树**：B1 重建完才建，否则同一天的两条承诺会拿到两代不同的对照期望。

**还没接进常驻 worker**：`assembly` 里 `_nightly_stages` 的第二个槽位就是留给它的，但接线要先把
十几个待定数值写进 `config.scene`，而那些数正是 54 天重放要定的。所以这一批只把跑法做成**可注入的对象**，
由重放脚本构造；接线排在重放之后（见待改清单 G-7）。

**闭环是可选的一拍**：没有注入触点③（``closure``）就跳过，前五步照跑——它要调模型，而重放与离线读数
不该被模型可用性卡住。安慰剂的补齐是纯算法，只要给了 ``placebos`` 就做。

**失败在哪停**：守门失败直接抛（口径不对时跑出来的数字是错的，宁可不跑）；映射里模型失败不塌整天
（映射器自己记未决）；投影是最后一步，前面都成了才落盘。

**2026-09-30 第三轮评审后改的几处编排**：
- 读数传 ``until``（命中记录可信到哪一刻），与结算同一个时刻——不传的话对称截断和无节律型分母都是关着的（R3-03）；
- ``until`` 按连续盖章的日子算、日界用主体时区（R3-11）；中间漏了一夜报进 signals；
- 重映射之后（命中变了）先撤那一天的账再开（R3-18）；
- 覆盖口每夜开头清缓存（R3-19）；桥、残差、叙事只读 ``≤ day`` 的日子（R3-22）；
- 闭环：问过的不再问、叙事取那条关系的承诺周围、按节律分型、写入时刻用夜批的 ``now``（R3-13 / R3-12）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo

from habitus.behavior.tree import BehaviorTree
from habitus.prediction.model import PredictionTree
from habitus.runtime.scene_closure import (
    NARRATIVE_DAYS,
    ClosureReport,
    day_narratives,
    drift_facts,
    relation_narratives,
    run_closure,
    split_facts,
)
from habitus.runtime.scene_opportunities import TreeOpportunities, generation_of
from habitus.runtime.scene_rhythms import TreeRhythms
from habitus.runtime.scene_situations import SceneDayContext
from habitus.scene.calendar import DayTypeCalendar
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.rhythm import Rhythm
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.hypotheses.closure import ClosureAuthor
from habitus.scene.hypotheses.model import Hypothesis
from habitus.scene.hypotheses.placebo import PLACEBO_PER_CONSEQUENT, placebo_hypotheses, split_by_origin
from habitus.scene.hypotheses.store import HypothesisStore
from habitus.scene.ledger.model import CoverageProvider
from habitus.scene.ledger.opening import LedgerConfig, open_claims_for_day
from habitus.scene.ledger.settlement import mapped_until, missing_days, settle_due_all
from habitus.scene.ledger.store import LedgerStore
from habitus.scene.occurrences.baselines import RECENT_WINDOW_DAYS
from habitus.scene.occurrences.mapper import ConceptMapper, map_closed_day
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.views import (
    ViewsConfig,
    ViewsStore,
    behaviour_views,
    entity_slices,
    materialize_views,
    profile_view,
    read_relations,
    residue_candidates,
)
from habitus.scene.views.kinds import ConceptOverlap, concept_kinds, concept_overlap, kinds_by_concept
from habitus.scene.views.placebo import PlaceboReport, placebo_report
from habitus.scene.views.relations import RelationReading


class SceneNightError(RuntimeError):
    """守门没过：口径不对或账上有孤儿结算。带着它跑出来的数字是错的，所以停。"""


@dataclass(frozen=True)
class SceneNightConfig:
    """一夜要用到的全部数值。都是**待定值**，54 天重放定完再写进 ``config.scene``。"""

    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    views: ViewsConfig = field(default_factory=ViewsConfig)
    #: 残差升级判据：某个 kind 攒够 ``residue_k`` 次且跨 ``residue_days`` 天就够资格交给触点① 写定义。
    residue_k: int = 5
    residue_days: int = 3
    recent_days: int = RECENT_WINDOW_DAYS
    #: 每个后件配几条安慰剂（裁定里的"安慰剂 M"）。0 = 不做安慰剂。
    placebos: int = PLACEBO_PER_CONSEQUENT
    #: 一夜最多问几次触点③。
    closure_rounds: int = 3
    #: 窗口落地两边各展几槽的容差 = 预测树的 ``prediction.pool_half_width``（2026-10-01 用户定复用它：预测层算的
    #: 与预测树算的都带上下槽位容差）。组合根从预测配置抄过来；这里不另起一个数。
    slack_slots: int = 0

    def __post_init__(self) -> None:
        for label in ("placebos", "closure_rounds", "slack_slots"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        for label in ("residue_k", "residue_days", "recent_days"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")


@dataclass(frozen=True)
class NightReport:
    """一夜的账：每一步各报自己的数，出了岔子的都在 ``signals`` 里，不静默。"""

    day: date
    generation: str
    concepts: int
    hypotheses: int
    mapped: int
    resumed: int
    unresolved: int
    opened: int
    without_control: int
    settled: int
    pending: int
    relations: int
    views_written: int
    #: 安慰剂那把尺子的读数：无关前件上"显著"的占比就是误报率。
    placebo: PlaceboReport | None = None
    #: 闭环这一拍：问了几次、写下几条新假设。
    closure: ClosureReport | None = None
    signals: tuple[str, ...] = ()

    def summary(self) -> str:
        return (
            f"{self.day}：映射 {self.mapped}（续跑 {self.resumed}、未决 {self.unresolved}）· "
            f"开承诺 {self.opened}（没对照 {self.without_control}）· 结算 {self.settled}（未结 {self.pending}）· "
            f"读数 {self.relations} 条 · 落盘 {self.views_written} 份"
            + ("" if self.closure is None else f" · {self.closure.summary()}")
        )


class SceneNightlyRun:
    """一夜的跑法。构造时拿到的都是**已经建好的东西**，本模块不读配置、不认识 YAML。

    ``mapper_for`` 是"拿这一版概念集建一个映射器"——**刷概念向量旁册也在它里面**（映射器构造时会拒绝
    旁册不全，而刷旁册要 embedding、是异步的）。这样本模块不必知道嵌入器与旁册存储长什么样。
    """

    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        concepts: ConceptStore,
        hypotheses: HypothesisStore,
        hits: ConceptHitStore,
        ledger: LedgerStore,
        views: ViewsStore,
        coverage: CoverageProvider,
        calendar: DayTypeCalendar,
        timezone: tzinfo,
        mapper_for: Callable[[ConceptSet], Awaitable[ConceptMapper]],
        closure: ClosureAuthor | None = None,
        subject: str | None = None,
        config: SceneNightConfig | None = None,
    ) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        resolved = config or SceneNightConfig()
        if not isinstance(resolved, SceneNightConfig):
            raise TypeError("config must be SceneNightConfig")
        self.behavior_tree = behavior_tree
        self.concepts = concepts
        self.hypotheses = hypotheses
        self.hits = hits
        self.ledger = ledger
        self.views = views
        self.coverage = coverage
        self.calendar = calendar
        self.timezone = timezone
        self.mapper_for = mapper_for
        if closure is not None and not isinstance(closure, ClosureAuthor):
            raise TypeError("closure must be a ClosureAuthor or None")
        self.closure = closure
        self.subject = subject
        self.config = resolved

    async def run(self, day: date, *, tree: PredictionTree, now: datetime) -> NightReport:
        """跑一天。``tree`` 是刚重建好的那一代（B1 的产物），``now`` 是夜批时刻。"""

        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise TypeError("now must be a timezone-aware datetime")
        concepts = self.concepts.read_all()
        known = {item.identity: item for item in self.hypotheses.read_all()}
        signals = list(self._guard(known))
        forget = getattr(self.coverage, "forget", None)
        if callable(forget):
            forget()  # 同一个覆盖口跨夜用：上一夜读进缓存的"次日"那时还没封口（评审 B-9）

        mapper = await self.mapper_for(concepts)
        # 常态按峰各算（R3-26）要的峰：用 ``day`` 之前的命中搭桥取节律——今天的还没映射，也不该让今天影响今天的常态。
        before = TreeOpportunities(tree, kinds_by_concept(concept_kinds(self._records_through(day - timedelta(days=1)), concepts)), generation=generation_of(tree), slack_slots=self.config.slack_slots)
        context = SceneDayContext(
            day,
            concepts=concepts,
            hits=self.hits,
            calendar=self.calendar,
            timezone=self.timezone,
            subject=self.subject,
            recent_days=self.config.recent_days,
            rhythms=TreeRhythms(tree, before.kinds).rhythms(concepts.behaviors()),
            slack_minutes=before.slack_minutes,
        )
        signals.extend(f"baseline: {key} 样本不够，引用它的候选会记未决" for key in context.baselines.missing)
        signals.extend(f"drift: {item.render()}" for item in context.baselines.drifting)

        mapping = await map_closed_day(
            self.behavior_tree,
            self.hits,
            mapper,
            day,
            now=now,
            situation_for=context.situation_for,
            baseline_for=context.baseline_for,
        )

        if mapping.changed:
            # 这一天的命中变了（口径变了重判、或树上少了记录）：靵那天开的承诺、以及别的天开却读了那天记录的结算，
            # 都是按旧命中算的，撤了再开（评审 A-12 / B-5）。承诺 add-only，撤是唯一的改法。
            voided = self._void_day(day, known)
            signals.append(f"remap: {day} 的命中变了（重判 {mapping.rewritten}、删 {mapping.stale_removed}）→ 撤账 {voided} 条后重开")

        opportunities, overlap = self._bridge(tree, concepts, day)
        rhythms = TreeRhythms(tree, opportunities.kinds).rhythms(concepts.behaviors())
        if overlap.diluted:
            # 概念集在互相稀释：一条 occurrence 命中好几个近义概念 → 同一次前件命中开出好几倍的承诺，
            # 本来清楚的因果被切成几条各自更薄的账。最常同时命中的那几对就是该合并或该挂同一上级的。
            signals.append(f"concepts: {overlap.render()}")
        opening = open_claims_for_day(
            day,
            hypotheses=tuple(known.values()),
            concepts=concepts,
            hits=self.hits,
            ledger=self.ledger,
            opportunities=opportunities,
            now=now,
            config=self.config.ledger,
        )
        signals.extend(f"opening: 假设 {identity} 引用了已不存在的概念，这一晚没开账" for identity in opening.unmappable)
        signals.extend(f"opening: 机会口给了账本收不下的快照（{identity}），那条按没有对照开" for identity in opening.unusable_snapshots)

        signals.extend(f"opening: {opening.backfill_refused} 条触发早于假设写入时刻，按不回填没开" for _ in range(1) if opening.backfill_refused)
        done = sorted(self.hits.days_done())
        gaps = missing_days(self.hits, since=done[0], through=day) if done else ()
        signals.extend(f"trust: {gap} 没有盖章，命中记录只信到它之前" for gap in gaps[:3])
        until = mapped_until(self.hits, now, timezone=self.timezone)
        settlements = settle_due_all(
            tuple(known.values()),
            concepts=concepts,
            hits=self.hits,
            ledger=self.ledger,
            coverage=self.coverage,
            now=now,
            until=until,
            config=self.config.ledger,
        )
        known = self._top_up_placebos(concepts, known, now=now, signals=signals)
        real, _placebo = split_by_origin(known)
        written, relations, readings = self._project(concepts, real, known, day=day, now=now, until=until if until is not None else now)
        placebo = placebo_report(readings, known)
        if placebo.placebo_tested:
            signals.append(f"placebo: {placebo.render()}")
        closure = await self._close(concepts, known, readings, context, day, now, rhythms, signals)
        return NightReport(
            day=day,
            generation=opportunities.generation,
            concepts=len(concepts),
            hypotheses=len(known),
            mapped=mapping.mapped,
            resumed=mapping.resumed,
            unresolved=mapping.unresolved,
            opened=opening.opened,
            without_control=opening.without_control,
            settled=sum(item.settled for item in settlements),
            pending=sum(item.pending for item in settlements),
            relations=relations,
            views_written=written,
            placebo=placebo,
            closure=closure,
            signals=tuple(signals),
        )

    def _top_up_placebos(
        self, concepts: ConceptSet, known: Mapping[str, Hypothesis], *, now: datetime, signals: list[str]
    ) -> dict[str, Hypothesis]:
        """补齐安慰剂（七c ⑭ 那把尺子）。已经有的不动——它一旦写盘就开始攒账，每晚换一批就永远量不出东西。"""

        resolved = dict(known)
        if not self.config.placebos:
            return resolved
        for hypothesis in placebo_hypotheses(concepts, tuple(resolved.values()), now=now, per_consequent=self.config.placebos):
            try:
                self.hypotheses.write(hypothesis, concepts)
            except Exception as exc:  # noqa: BLE001 - 写不进去要报出来
                signals.append(f"placebo: 写不进 {hypothesis.identity}：{type(exc).__name__}: {exc}")
                continue
            resolved[hypothesis.identity] = hypothesis
            signals.append(f"placebo: 新配 {hypothesis.label()}（量误报率用，不给人看）")
        return resolved

    async def _close(
        self,
        concepts: ConceptSet,
        known: Mapping[str, Hypothesis],
        readings: Sequence[RelationReading],
        context: SceneDayContext,
        day: date,
        now: datetime,
        rhythms: Mapping[str, Rhythm],
        signals: list[str],
    ) -> ClosureReport | None:
        """B6 闭环：扫出"不一样"→ 交触点③ → 写进 hypotheses/（从写入日起攒账、不回填）。

        没注入触点③ 就整拍跳过：它要调模型，而重放与离线读数不该被模型可用性卡住。
        叙事取**那条关系的承诺周围**；漂移的近期 / 早先两段不重叠，不够两段就先不问。
        """

        if self.closure is None:
            return None
        days = [past for past in sorted(self.hits.days_done()) if past <= day]
        if not days:
            return None

        def narratives(hypothesis: Hypothesis, situation: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
            return relation_narratives(self.ledger, self.hits, hypothesis, situation, concepts)

        facts: list = list(split_facts(readings, known, narratives))
        if len(days) >= 2 * NARRATIVE_DAYS and context.baselines.drifting:
            recent, earlier = days[-NARRATIVE_DAYS:], days[-2 * NARRATIVE_DAYS : -NARRATIVE_DAYS]
            facts += list(
                drift_facts(
                    context.baselines,
                    {
                        drift.concept: (day_narratives(self.hits, recent, concepts), day_narratives(self.hits, earlier, concepts))
                        for drift in context.baselines.drifting
                    },
                )
            )
        elif context.baselines.drifting:
            signals.append(f"closure: 漂移 {len(context.baselines.drifting)} 条，但已映射不满 {2 * NARRATIVE_DAYS} 天、两段叙事分不开，先不问")
        if not facts:
            return ClosureReport()
        report = await run_closure(
            self.closure,
            facts,
            concepts,
            self.hypotheses,
            day=day,
            now=now,
            known=tuple(known.values()),
            rhythms=rhythms,
            max_rounds=self.config.closure_rounds,
        )
        signals.extend(report.signals)
        signals.extend(f"closure: {line}" for line in report.explanations)
        return report

    def _void_day(self, day: date, known: Mapping[str, Hypothesis]) -> int:
        """重映射之后撤账：那天开的承诺与结算（``void_day``）+ 别的天开、却读了那天记录的结算（``void_settlements_touching``）。"""

        voided = 0
        for identity in known:
            voided += len(self.ledger.void_day(identity, day))
            voided += len(self.ledger.void_settlements_touching(identity, day))
        return voided

    def _guard(self, known: Mapping[str, Hypothesis]) -> tuple[str, ...]:
        """B0：守门——账上不能有孤儿结算（结算在、承诺没了）：那说明有人手删过文件或作废撤了一半，
        继续跑会把一条没有承诺的结算算进分母。

        （读侧与账本"等几次机会"的口径对齐那条 2026-10-01 随账改按钟面窗口记而消失：收口点只剩"窗口过完"，
        写侧 ``settle_probability`` 与读侧 ``settled_by`` 读的是承诺上同一个窗口。）
        """

        orphans = {identity: self.ledger.orphan_settlements(identity) for identity in known}
        broken = {identity: refs for identity, refs in orphans.items() if refs}
        if broken:
            raise SceneNightError(
                "the ledger carries settlements whose claims are gone: "
                + "; ".join(f"{identity} × {len(refs)}" for identity, refs in sorted(broken.items()))
            )
        return ()

    def _bridge(self, tree: PredictionTree, concepts: ConceptSet, day: date) -> tuple[TreeOpportunities, ConceptOverlap]:
        """B1''：桥 + 机会口，顺带数概念之间的重叠。

        桥数的是命中记录里写着的 kind，所以排在映射之后（今天的也算）；**只读 ``≤ day`` 的日子**——重跑更早的
        某一天时盘上有它之后的命中，读进桥就是把未来泄给过去（评审 B-14）。重叠用同一批记录数——
        它是概念集质量的体检项，不数就只能看着承诺量莫名偏高而不知道为什么。
        """

        records = self._records_through(day)
        spreads = concept_kinds(records, concepts)
        provider = TreeOpportunities(tree, kinds_by_concept(spreads), generation=generation_of(tree), slack_slots=self.config.slack_slots)
        return provider, concept_overlap(records, concepts)

    def _records_through(self, day: date) -> list[ConceptHits]:
        """盘上 ``≤ day`` 的全部命中记录（桥与节律都只许读到这一天，不把未来泄给过去）。"""

        return [record for past in sorted(self.hits.days_done()) if past <= day for record in self.hits.read_day(past)]

    def _project(
        self,
        concepts: ConceptSet,
        real: Mapping[str, Hypothesis],
        known: Mapping[str, Hypothesis],
        *,
        day: date,
        now: datetime,
        until: datetime,
    ) -> tuple[int, int, tuple[RelationReading, ...]]:
        """B5：读数 → 两面 → 残差 → profile/人物 → 落盘。全是算法，一个模型调用都没有。

        **读数算全部、投影只落真假设**：安慰剂的读数是那把尺子要用的，但它不是给人看的因果——
        印进 ``behaviours/`` 会被当成真关系读。

        **读数传 ``until``**（命中记录可信到哪一刻），与结算判"时间走到哪了"用同一个时刻：读侧靠它判
        "命运已定"（对称截断）与"仍立着多久"（无节律型分母的第三项）；不传就两样都关着（评审 A-3 / B-3 / C-3）。
        """

        readings = read_relations(tuple(known.values()), ledger=self.ledger, concepts=concepts, config=self.config.views, now=until)
        shown = tuple(item for item in readings if item.hypothesis_identity in real)
        residue = residue_candidates(
            self.hits,
            [past for past in sorted(self.hits.days_done()) if past <= day],
            k=self.config.residue_k,
            d=self.config.residue_days,
            claimed=concepts.claimed_kinds(),
        )
        written = materialize_views(
            self.views,
            hypotheses=real,
            readings=shown,
            behaviours=behaviour_views(shown, real),
            residue=residue,
            profile=profile_view(shown),
            entities=entity_slices(shown, concepts),
            concepts=concepts,
            now=now,
            k=self.config.residue_k,
            d=self.config.residue_days,
        )
        return len(written), len(shown), readings


__all__ = ["NightReport", "SceneNightConfig", "SceneNightError", "SceneNightlyRun"]
