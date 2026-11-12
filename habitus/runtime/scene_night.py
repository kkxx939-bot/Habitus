"""语义树的几步（语义树新方案 ``13`` 第五节；原来的夜批）与它们在换代、模型队列里的分法（实时化阶段 2）。

顺序是**线性的、不级联的**（派生树按固定先后处理已封口的历史，不做"输入又变了就重建"）：

```
B1  同步词表（裁定 20）：读上次同步之后的变更
      · 每个类一个基础概念：新类生成、改名 / 改判据跟着改、停用的标停用；汇总概念的成员改写成现编号
      · 迁移改了编号的那些已映射日子：重映射                                                ← 映射的 LLM 触点
B2  概念   类清单变了的 lane 交触点①写细分 / 汇总 / 情境概念（``scene_authoring``）  ← 写概念的 LLM 触点
B3  映射   按条：一条记录所在的槽过后留够余量就映射（不等那一天封口），材料只由它自己决定；口径或输入变了的重判；
           没映射过的历史回填，最新的先（续跑按每条记录"挂在它类上的细分概念"认口径：只有真受影响的记录才问模型） ← 映射的 LLM 触点
    封口   已封口的日子里每条记录都作数的：统一重算情境、写回、核条数、盖日完成章（纯算法、零模型）
B4  关系   候选调节条件补问、先验补问（都按输入摘要缓存）                                   ← 两个 LLM 触点
           每条 lane 全部两两检验（截止日 = 序列的截止日）→ 接上这条 lane 盘上之前最近的一晚折叠状态
           （候选 / 成立 / 失效 / 前向未复现）→ 这一晚的关系表与迁移日志落盘                 ← 纯算法、零模型，只读缓存
```

**调模型的归模型队列，换代里没有模型调用**：

- 模型队列（``runtime.model_queue``）一项一项做，今天这一代完成之后：``author``（B2，写成的待生效）→ ``relation_stage.ask``
  （B4 的补问：先验对着下一次换代之后的概念集问，候选条件对着今天这一代的时间线问）→ ``syncing.run(remap=True)``（迁移
  牵到的重映射、记下同步到的版本）；映射（B3：``mapping.plan`` 按条认欠账、``mapping.map_one`` 一条一项）不等这一代。
  每一项要的概念集、节律都从盘上与序列现算（``materials``），不靠上一项在内存里传。
- 换代调 ``generate``：同步词表里不调模型的那一半（概念落盘，幂等；有迁移牵到的重映射还没做时不记同步到的版本，留给队列）
  → 封口（``mapping.seal``，按并入前的概念集）→ 并入待生效的概念（这时定创建时刻）→ 写这一代的节律 → 按缓存折叠关系 → 时间线与材料交给机会表。有记录还不作数的日子不盖章，
  关系检验读不到它（"不知道"）。

同步词表两边都会走到：概念落盘按全部类核对、是幂等的，同步到的版本只在重映射做完之后才记——谁先做结果都一样。
B4 读的是盘上**全部**已映射的命中（不只今天）：关系每晚全量重算，不做增量统计。

**读的是第 N 晚的事件序列**（``series.reader`` 读一次，组合根交进来，机会表用同一份）：映射哪几条、属于哪条 lane、
当天的空白都按它，行为树只用来取那几条的全文。第 N 晚只映射 N 之前（已封口）的日子。

**桥不读命中**：概念记着自己的类，节律按类编号从序列算（``temporal``；基础 / 细分 = 自己那个类，汇总 = 成员类合起来），
所以桥在映射之前就定了。常态是映射的**材料**，按裁定**不含当天**：命中历史取那一天之前已完成的日子，按峰分份用的节律取
那一天的那一代落盘的那一份（``runtime.scene_rhythms``；没有就按截止到那一天的序列现算，``rhythms_for``）。

**失败在哪停**：映射里模型失败不塌整天（映射器自己记未决）。换代里各步是尽力而为（``generate`` 的 ``failures``）：一步没做成
走退路，时间线与材料照样算给机会表。纯计算都下沉到线程（``asyncio.to_thread``），循环上只留调模型与落盘。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, tzinfo

from habitus.behavior.tree import BehaviorTree
from habitus.conditions import DayTypeCalendar
from habitus.runtime import scene_pending as pending
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_authoring import ConceptAuthoring
from habitus.runtime.scene_night_mapping import MappingStage
from habitus.runtime.scene_night_relations import RelationStage
from habitus.runtime.scene_night_seal import SealStage
from habitus.runtime.scene_night_sync import Synced, SyncStage
from habitus.runtime.scene_rhythms import RhythmFiles
from habitus.scene.concepts.calendar_situations import calendar_key, most_specific
from habitus.scene.concepts.catalog import ClassCatalog
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.rhythm import Rhythm, RhythmRulers, rhythms_of
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.occurrences.baselines import RECENT_WINDOW_DAYS, declared_keys
from habitus.scene.occurrences.mapper import ConceptMapper
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.recurrence import Recurrence, coverage_by_lane, recurrences_of
from habitus.scene.relations import (
    LaneTimeline,
    RelationConfig,
    Segment,
)
from habitus.scene.relations.state import Status
from habitus.scene.relations.store import RelationStore, RelationStoreError
from habitus.series import EventSeries
from habitus.temporal import Coverage


@dataclass(frozen=True)
class SceneNightConfig:
    """一夜要用到的数值。都是**待定值**，跑真实数据定完再写进 ``config.scene``。"""

    recent_days: int = RECENT_WINDOW_DAYS
    #: 常态按峰分份时两边各展几槽的容差，组合根从 ``config.scene.baseline_slack_slots`` 抄过来（裁定 59）。
    slack_slots: int = 0

    def __post_init__(self) -> None:
        if self.slack_slots < 0:
            raise ValueError("slack_slots must be a non-negative integer")
        if self.recent_days <= 0:
            raise ValueError("recent_days must be a positive integer")


@dataclass(frozen=True)
class NightMaterials:
    """B4 这一晚用过的材料，交给排在语义树之后的那几拍复用（预测层的机会表要在**同一条**时间线上算）：
    同步词表之后的概念集、各 lane 的时间线。只在内存里传一晚，不落盘。"""

    concepts: ConceptSet
    timelines: Mapping[str, LaneTimeline]
    #: 每个类的复现规律（裁定 59）：预测层「离上次够久」从这里读，由机会表落盘。
    recurrences: Mapping[str, Recurrence] = field(default_factory=dict)
    #: 各 lane 的观测覆盖：机会表抄成每天看着的比例（账的久别用，26 号清单 3.3）。
    coverage: Mapping[str, Coverage] = field(default_factory=dict)


@dataclass(frozen=True)
class NightReport:
    """一夜的账：每一步各报自己的数，出了岔子的都在 ``signals`` 里，不静默。"""

    night: date
    concepts: int
    #: B4：每条 lane 各状态的关系数（样本不够 / 已检 / 候选 / 成立 / 失效 / 前向未复现）与这一晚的迁移条数。
    relations: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    transitions: int = 0
    signals: tuple[str, ...] = ()
    #: B4 这一代没判（模型整个调不通，关系沿用上一代）：为什么；判了是 None。
    relations_held: str | None = None
    #: 封口（换代的语义树一步）：盖了章的日子数、写回了情境的记录数、还不能盖章的日子数与其中没映射好的记录数、跟着重算
    #: 情境的读者日数、这一段用了多久。
    sealed_days: int = 0
    seal_rewritten: int = 0
    unsealed_days: int = 0
    unsealed_records: int = 0
    reader_days: int = 0
    seal_seconds: float = field(default=0.0, compare=False)
    #: 这一代并进概念集的待生效概念数（写概念的结果，上一代之后写的）。
    merged_concepts: int = 0
    #: 这一晚的材料（见 ``NightMaterials``）；不进比较与打印。
    materials: NightMaterials | None = field(default=None, compare=False, repr=False)

    def summary(self) -> str:
        counts = "；".join(
            f"{lane} 候选 {numbers.get('candidate', 0)}、成立 {numbers.get('established', 0)}、失效 {numbers.get('expired', 0)}、前向未复现 {numbers.get('rejected', 0)}"
            for lane, numbers in sorted(self.relations.items())
        )
        return (
            f"第 {self.night} 晚：概念 {self.concepts} · 封口 {self.sealed_days} 天（不能盖章 {self.unsealed_days} 天）· "
            f"并入 {self.merged_concepts} 个待生效概念 · 关系 {counts or '无'}（迁移 {self.transitions} 条）"
        )


def line_days(lines: Mapping[str, LaneTimeline]) -> tuple[date, ...]:
    """各条 lane 人在场的日子（日历备注要给的那些天）。"""

    return tuple(sorted({day for line in lines.values() for day in line.days()}))


class SceneNightlyRun:
    """一夜的跑法。构造时拿到的都是**已经建好的东西**，这个类不读配置、不认识 YAML。``catalog`` 是词表口（组合根用 ``VocabularyReader`` 装的 ``ClassCatalog``）；``mapper_for`` 是"拿这一版
    概念集建一个映射器"（概念集在同步词表之后才定）。
    """

    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        catalog: ClassCatalog,
        concepts: ConceptStore,
        hits: ConceptHitStore,
        calendar: DayTypeCalendar,
        timezone: tzinfo,
        mapper_for: Callable[[ConceptSet], ConceptMapper],
        relations: RelationStore,
        relation_config: RelationConfig,
        authoring: ConceptAuthoring | None = None,
        advice: RelationAdvice | None = None,
        subject: str | None = None,
        rulers: RhythmRulers,
        config: SceneNightConfig | None = None,
        rhythm_files: RhythmFiles | None = None,
        frozen_vocabulary: Callable[[], int | None] | None = None,
    ) -> None:
        resolved = config or SceneNightConfig()
        self.behavior_tree = behavior_tree
        self.catalog = catalog
        self.concepts = concepts
        self.hits = hits
        self.calendar = calendar
        self.timezone = timezone
        self.mapper_for = mapper_for
        self.relations = relations
        self.relation_config = relation_config
        self.authoring = authoring
        self.advice = advice
        self.subject = subject
        self.config = resolved
        self.rulers = rulers
        #: 每一代的节律文件（换代写、映射读）；没接就每次按截止到那一天的序列现算。
        self.rhythm_files = rhythm_files
        #: 最近一代做完时的词表版本（完成标记上记的）；没接、或一代都还没有是 None。那之后才有或成员才变的类，
        #: "近几天同类记录"这份材料按没给（``recent_gate``）。
        self.frozen_vocabulary = frozen_vocabulary
        self._build_stages()

    def _build_stages(self) -> None:
        """B1 同步词表、B3 映射、封口与 B4 关系各是一个 stage；一夜的数值、映射器的建法与模型建议每次从这里现读（构造后可换）。"""

        self.mapping = MappingStage(
            behavior_tree=self.behavior_tree,
            hits=self.hits,
            calendar=self.calendar,
            timezone=self.timezone,
            subject=self.subject,
            config=lambda: self.config,
            rhythms=self.rhythms_for,
            slack=lambda: self.slack_minutes,
            recent_given=self.recent_gate,
        )
        self.sealing = SealStage(self.mapping)
        self.syncing = SyncStage(
            catalog=self.catalog, concepts=self.concepts, hits=self.hits, calendar=self.calendar,
            mapping=self.mapping, rulers=self.rulers, mapper_for=lambda found: self.mapper_for(found),
        )  # fmt: skip
        self.relation_stage = RelationStage(
            catalog=self.catalog,
            concepts=self.concepts,
            hits=self.hits,
            calendar=self.calendar,
            relations=self.relations,
            relation_config=self.relation_config,
            advice=lambda: self.advice,
        )

    @property
    def slack_minutes(self) -> int:
        return self.config.slack_slots * self.relation_config.slot_minutes

    async def generate(
        self,
        *,
        series: EventSeries,
        now: datetime,
        failures: list[tuple[str, BaseException]] | None = None,
        relations_held: str | None = None,
    ) -> NightReport:
        """换代里语义树那一步，没有模型调用：同步词表（概念落盘）→ 封口 → 并入待生效的概念 → 这一代的节律 → 按缓存折叠
        关系 → 材料。``now`` 是这一次的时刻；B1、封口、B4 各一个 span（``scene.b1``、``scene.seal``、``scene.b4``）。

        ``failures`` 给了就是尽力而为：某一步抛了，记进 ``failures``（哪一步, 异常）、走退路接着做——B1 没做成按盘上现有的
        概念集、B4 没做成这一代不写关系表（机会表退到上一代的关系表）——时间线与材料照样交给机会表。不给就照原样往上抛。

        ``relations_held`` 给了（换代那一刻模型队列正在整体退避），B4 这一代不判：不写关系表、时间线照算，机会表退到上一代
        关系表（实时化方案第 2 节"模型调不通 → 关系沿用上一次判定的"），原因记在 ``NightReport.relations_held``。"""

        if now.utcoffset() is None:
            raise TypeError("now must be a timezone-aware datetime")
        from habitus.runtime.scene_night_run import NightRun

        return await NightRun(self, series, now, failures, relations_held).run()

    # ── 各步：模型队列一项一项调，``generate`` 按先后串 ─────────────────────────

    async def materials(
        self, series: EventSeries, coverage: Mapping[str, Coverage] | None = None
    ) -> tuple[ConceptSet, ConceptMapper, Mapping[str, Rhythm]]:
        """盘上现有的概念集，与据它算的映射器、节律（纯计算）。"""

        concepts = await asyncio.to_thread(self.concepts.read_all)
        if coverage is None:
            coverage = await asyncio.to_thread(coverage_by_lane, series, self.rulers)
        rhythms = await asyncio.to_thread(self._rhythms, series, concepts, coverage)
        return concepts, self.mapper_for(concepts), rhythms

    async def prospect(self, series: EventSeries, now: datetime) -> tuple[ConceptSet, frozenset[str]]:
        """下一次换代之后概念集会是什么样、其中现在已经在概念集里的那些身份（``runtime.scene_pending.prospect``）。"""

        return await pending.prospect(self, series, now)

    async def author(
        self, series: EventSeries, signals: list[str], coverage: Mapping[str, Coverage] | None = None
    ) -> int:
        """B2 写概念：写成的存成待生效（下一次换代并入），返回写了几个（``runtime.scene_pending.author``）。"""

        return await pending.author(self, series, signals, coverage)

    def merge_pending(self, now: datetime, signals: list[str]) -> int:
        """把待生效的概念并进概念集（换代的语义树一步，在同步词表之后），返回并入了几个。"""

        return pending.merge_pending(self, now, signals)

    def _materials(
        self,
        series: EventSeries,
        concepts: ConceptSet,
        lines: Mapping[str, LaneTimeline],
        coverage: Mapping[str, Coverage],
    ) -> NightMaterials:
        return NightMaterials(
            concepts=concepts,
            timelines=lines,
            recurrences=recurrences_of(series, self.rulers, coverage),
            coverage=coverage,
        )

    # ── B2–B3 ───────────────────────────────────────────────────────────────

    def rhythms_for(self, day: date, concepts: ConceptSet, series: EventSeries) -> Mapping[str, Rhythm]:
        """映射 ``day`` 那天的记录时常态按峰分份用的节律：那一天的那一代落盘的那一份；没有这份文件（阶段 2 之前的历史、
        那一代的语义树一步没做成）就按截止到那一天的序列现算——都不含当天。没有概念声明常态键时用不着节律，不读也不算。"""

        if not declared_keys(concepts):
            return {}
        if self.rhythm_files is not None:
            frozen = self.rhythm_files.read(day)
            if frozen is not None:
                return frozen
        before = series.until(min(day, series.cutoff))
        return self._rhythms(before, concepts, coverage_by_lane(before, self.rulers))

    def freeze_rhythms(self, series: EventSeries, concepts: ConceptSet, coverage: Mapping[str, Coverage]) -> int:
        """把这一代（第 ``series.cutoff`` 代）的节律落盘一份：之后映射这一天的记录都读它。返回带峰的概念数。"""

        if self.rhythm_files is None:
            return 0
        rhythms = self._rhythms(series, concepts, coverage)
        self.rhythm_files.write(series.cutoff, rhythms)
        return sum(1 for rhythm in rhythms.values() if rhythm.peaks)

    def recent_gate(self) -> Callable[[str], bool] | None:
        """现在哪些类的"近几天同类记录"给得出：最近一代之后才有、或成员才变的类给不出（那份材料按"没给"，下一代补判）。
        没接完成标记、或一代都还没有时都给。"""

        version = self.frozen_vocabulary() if self.frozen_vocabulary is not None else None
        if version is None or version >= self.catalog.version():
            return None
        changes = self.catalog.changes_since(version)
        touched = {*changes.split_from, *changes.split_from.values(), *changes.merged_into}
        touched.update(changes.merged_into.values())
        for moved in changes.moved:
            touched.update((moved.source, moved.target))
        return lambda token: token not in touched

    def _rhythms(
        self, series: EventSeries, concepts: ConceptSet, coverage: Mapping[str, Coverage]
    ) -> Mapping[str, Rhythm]:
        return rhythms_of(
            series, concepts, self.rulers, anchors=self._anchors(concepts, series.cutoff), coverage=coverage
        )

    def _anchors(self, concepts: ConceptSet, night: date) -> dict[str, tuple[str, ...]]:
        """后果 → 上一晚已经成立的日历关系说的那几种日子（只留最具体的那一族；裁定 59）。读不到上一晚就没有。"""

        held: dict[str, list[str]] = {}
        for lane in self.relations.lanes():
            try:
                state = self.relations.previous(lane, night)
            except RelationStoreError:
                # 读不到上一晚就按没有锚日算（见文档串）
                continue
            if state is None:
                continue
            for relation in state.of(Status.ESTABLISHED):
                key = relation.key
                if key.segment is Segment.SITUATION and not key.condition and relation.upward:
                    if calendar_key(concepts, key.antecedent) is not None:
                        held.setdefault(key.consequent, []).append(key.antecedent)
        return {
            consequent: tuple(concepts[item].name for item in most_specific(concepts, antecedents))
            for consequent, antecedents in held.items()
        }


__all__ = [
    "NightMaterials",
    "NightReport",
    "SceneNightConfig",
    "SceneNightlyRun",
    "Synced",
]
