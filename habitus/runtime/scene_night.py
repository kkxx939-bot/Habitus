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
B5  投影   读数 → 两面 → 还立着的 → 残差 → profile/人物 → 落盘
```

**为什么桥在映射之后**：桥数的是"这个概念命中过哪些 kind"，今天刚映射出来的命中也该算；而它的消费者
（机会口）到 B3 才用得上。常态相反，它是 B2 的**材料**，而且按裁定**不含当天**（拿今天和自己比，
晚睡那条规则会被自己稀释），所以排在映射之前。

**机会口必须拿这一代树**：B1 重建完才建，否则同一天的两条承诺会拿到两代不同的对照期望。

**还没接进常驻 worker**：`assembly` 里 `_nightly_stages` 的第二个槽位就是留给它的，但接线要先把
十几个待定数值写进 `config.scene`，而那些数正是 54 天重放要定的。所以这一批只把跑法做成**可注入的对象**，
由重放脚本构造；接线排在重放之后（见待改清单 G-7）。

**失败在哪停**：守门失败直接抛（口径不对时跑出来的数字是错的，宁可不跑）；映射里模型失败不塌整天
（映射器自己记未决）；投影是最后一步，前面都成了才落盘。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, tzinfo

from habitus.behavior.tree import BehaviorTree
from habitus.prediction.model import PredictionTree
from habitus.runtime.scene_opportunities import TreeOpportunities, generation_of
from habitus.runtime.scene_situations import SceneDayContext
from habitus.scene.calendar import DayTypeCalendar
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.hypotheses.model import Hypothesis
from habitus.scene.hypotheses.store import HypothesisStore
from habitus.scene.ledger.model import CoverageProvider
from habitus.scene.ledger.opening import LedgerConfig, open_claims_for_day
from habitus.scene.ledger.settlement import mapped_until, settle_due_all
from habitus.scene.ledger.store import LedgerStore
from habitus.scene.occurrences.baselines import RECENT_WINDOW_DAYS
from habitus.scene.occurrences.mapper import ConceptMapper, map_closed_day
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.views import (
    ViewsConfig,
    ViewsStore,
    behaviour_views,
    entity_slices,
    materialize_views,
    profile_view,
    read_relations,
    require_aligned,
    residue_candidates,
    standing_intentions,
)
from habitus.scene.views.kinds import concept_kinds, kinds_by_concept


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

    def __post_init__(self) -> None:
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
    signals: tuple[str, ...] = ()

    def summary(self) -> str:
        return (
            f"{self.day}：映射 {self.mapped}（续跑 {self.resumed}、未决 {self.unresolved}）· "
            f"开承诺 {self.opened}（没对照 {self.without_control}）· 结算 {self.settled}（未结 {self.pending}）· "
            f"读数 {self.relations} 条 · 落盘 {self.views_written} 份"
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
        self.subject = subject
        self.config = resolved

    async def run(self, day: date, *, tree: PredictionTree, now: datetime) -> NightReport:
        """跑一天。``tree`` 是刚重建好的那一代（B1 的产物），``now`` 是夜批时刻。"""

        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise TypeError("now must be a timezone-aware datetime")
        concepts = self.concepts.read_all()
        known = {item.identity: item for item in self.hypotheses.read_all()}
        signals = list(self._guard(known))

        mapper = await self.mapper_for(concepts)
        context = SceneDayContext(
            day,
            concepts=concepts,
            hits=self.hits,
            calendar=self.calendar,
            timezone=self.timezone,
            ledger=self.ledger,
            hypotheses=known,
            subject=self.subject,
            recent_days=self.config.recent_days,
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

        opportunities = self._opportunities(tree, concepts)
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

        settlements = settle_due_all(
            tuple(known.values()),
            concepts=concepts,
            hits=self.hits,
            ledger=self.ledger,
            coverage=self.coverage,
            now=now,
            until=mapped_until(self.hits, now),
            config=self.config.ledger,
        )
        written, relations = self._project(concepts, known, now=now)
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
            signals=tuple(signals),
        )

    def _guard(self, known: Mapping[str, Hypothesis]) -> tuple[str, ...]:
        """B0：两处守门。

        ① 读侧与账本的口径必须对齐（``settlement_horizon`` 与 ``censor_after`` 不一致时，读出来的
           "等够了几次机会"和账本判删失用的不是同一个数）；
        ② 账上不能有孤儿结算（结算在、承诺没了）——那说明有人手删过文件或作废撤了一半，
           继续跑会把一条没有承诺的结算算进分母。
        """

        require_aligned(self.config.views, self.config.ledger)
        orphans = {identity: self.ledger.orphan_settlements(identity) for identity in known}
        broken = {identity: refs for identity, refs in orphans.items() if refs}
        if broken:
            raise SceneNightError(
                "the ledger carries settlements whose claims are gone: "
                + "; ".join(f"{identity} × {len(refs)}" for identity, refs in sorted(broken.items()))
            )
        return ()

    def _opportunities(self, tree: PredictionTree, concepts: ConceptSet) -> TreeOpportunities:
        """B1''：桥 + 机会口。桥数的是命中记录里写着的 kind，所以排在映射之后（今天的也算）。"""

        records = [record for past in sorted(self.hits.days_done()) for record in self.hits.read_day(past)]
        spreads = concept_kinds(records, concepts)
        return TreeOpportunities(tree, kinds_by_concept(spreads), generation=generation_of(tree))

    def _project(self, concepts: ConceptSet, known: Mapping[str, Hypothesis], *, now: datetime) -> tuple[int, int]:
        """B5：读数 → 两面 → 还立着的 → 残差 → profile/人物 → 落盘。全是算法，一个模型调用都没有。"""

        readings = read_relations(tuple(known.values()), ledger=self.ledger, concepts=concepts, config=self.config.views)
        intentions = standing_intentions(self.ledger, known, now=now)
        residue = residue_candidates(
            self.hits,
            sorted(self.hits.days_done()),
            k=self.config.residue_k,
            d=self.config.residue_days,
            claimed=concepts.claimed_kinds(),
        )
        written = materialize_views(
            self.views,
            hypotheses=known,
            readings=readings,
            behaviours=behaviour_views(readings, known),
            intentions=intentions,
            residue=residue,
            profile=profile_view(readings),
            entities=entity_slices(readings, concepts),
            concepts=concepts,
            now=now,
            k=self.config.residue_k,
            d=self.config.residue_days,
        )
        return len(written), len(readings)


__all__ = ["NightReport", "SceneNightConfig", "SceneNightError", "SceneNightlyRun"]
