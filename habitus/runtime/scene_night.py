"""语义树的几步（语义树新方案 ``13`` 第五节；原来的夜批）与它们在换代、模型队列里的分法（实时化阶段 2）。

顺序是**线性的、不级联的**（派生树按固定先后处理已封口的历史，不做"输入又变了就重建"）：

```
B1  同步词表（裁定 20）：这一代在用的词（换代数出来、显式交进来，裁定 101）
      · 每个在用的词一个行为概念：新在用的生成、改名 / 改定义跟着改；数据里新出现的日历取值补日历情境
    封口   已封口的日子里还没封好、或记录变了的（补进或延长了记录）：命中记录由代码直接生成
           （词编号、lane、是否已归类、那一天成立的日历情境）、核条数、盖日完成章（纯算法、零模型）
B4  关系   候选调节条件补问、先验补问（都按输入摘要缓存）                                   ← 两个 LLM 触点
           每条 lane 全部两两检验（截止日 = 序列的截止日）→ 接上这条 lane 盘上之前最近的一晚折叠状态
           （候选 / 成立 / 失效 / 前向未复现）→ 这一晚的关系表与迁移日志落盘                 ← 纯算法、零模型，只读缓存
```

**调模型的归模型队列，换代里没有模型调用**：

- 模型队列（``runtime.model_queue``）一项一项做，今天这一代完成之后：``relation_stage.ask``（B4 的补问：先验对着"概念集 +
  同步词表将要写的"问）→ ``syncing.run``（记下同步到的版本，读盘上的在用状态）。
- 换代调 ``generate``：同步词表（概念落盘，幂等）→ 封口 → 按缓存折叠关系 → 时间线与材料交给机会表。

B4 读的是盘上**全部**已封好的命中（不只今天）：关系每晚全量重算，不做增量统计。

**读的是第 N 晚的事件序列**（``series.reader`` 读一次，组合根交进来，机会表用同一份）：封哪几天、属于哪条 lane 都按它。
第 N 晚只封 N 之前（已封口）的日子。

**失败在哪停**：换代里各步是尽力而为（``generate`` 的 ``failures``）：一步没做成走退路，时间线与材料照样算给机会表。
纯计算都下沉到线程（``asyncio.to_thread``），循环上只留落盘。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime

from habitus.behavior.tree import BehaviorTree
from habitus.conditions import DayTypeCalendar
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_night_relations import RelationStage
from habitus.runtime.scene_night_seal import SealStage
from habitus.runtime.scene_night_sync import Synced, SyncStage
from habitus.scene.concepts.catalog import CatalogWords, ClassCatalog
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.recurrence import Recurrence, recurrences_of
from habitus.scene.relations import LaneTimeline, RelationConfig
from habitus.scene.relations.store import RelationStore
from habitus.series import EventSeries
from habitus.temporal import Coverage, Rulers


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
    #: 封口（换代的语义树一步）：盖了章的日子数、写了的命中记录数、封不了的日子数、这一段用了多久。
    sealed_days: int = 0
    seal_rewritten: int = 0
    unsealed_days: int = 0
    seal_seconds: float = field(default=0.0, compare=False)
    #: 这一晚的材料（见 ``NightMaterials``）；不进比较与打印。
    materials: NightMaterials | None = field(default=None, compare=False, repr=False)

    def summary(self) -> str:
        counts = "；".join(
            f"{lane} 候选 {numbers.get('candidate', 0)}、成立 {numbers.get('established', 0)}、失效 {numbers.get('expired', 0)}、前向未复现 {numbers.get('rejected', 0)}、退休 {numbers.get('retired', 0)}"
            for lane, numbers in sorted(self.relations.items())
        )
        return (
            f"第 {self.night} 晚：概念 {self.concepts} · 封口 {self.sealed_days} 天（封不了 {self.unsealed_days} 天）· "
            f"关系 {counts or '无'}（迁移 {self.transitions} 条）"
        )


def line_days(lines: Mapping[str, LaneTimeline]) -> tuple[date, ...]:
    """各条 lane 人在场的日子（日历备注要给的那些天）。"""

    return tuple(sorted({day for line in lines.values() for day in line.days()}))


class SceneNightlyRun:
    """一夜的跑法。构造时拿到的都是**已经建好的东西**，这个类不读配置、不认识 YAML。``catalog`` 是词表口（组合根用
    词表的在用状态装的 ``ClassCatalog``，模型队列记同步到的版本时读它）。"""

    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        catalog: ClassCatalog,
        concepts: ConceptStore,
        hits: ConceptHitStore,
        calendar: DayTypeCalendar,
        relations: RelationStore,
        relation_config: RelationConfig,
        advice: RelationAdvice | None = None,
        rulers: Rulers,
    ) -> None:
        self.behavior_tree = behavior_tree
        self.concepts = concepts
        self.hits = hits
        self.calendar = calendar
        self.relations = relations
        self.relation_config = relation_config
        self.advice = advice
        self.rulers = rulers
        self.sealing = SealStage(hits=hits, calendar=calendar)
        self.syncing = SyncStage(catalog=catalog, concepts=concepts, calendar=calendar, rulers=rulers)
        self.relation_stage = RelationStage(
            concepts=concepts,
            hits=hits,
            calendar=calendar,
            relations=relations,
            relation_config=relation_config,
            advice=lambda: self.advice,
        )

    async def generate(
        self,
        *,
        series: EventSeries,
        words: CatalogWords,
        now: datetime,
        failures: list[tuple[str, BaseException]] | None = None,
        relations_held: str | None = None,
    ) -> NightReport:
        """换代里语义树那一步，没有模型调用：同步词表（概念落盘）→ 封口 → 按缓存折叠关系 → 材料。``words`` 是这一代在用的词
        （换代数出来、显式交进来，一代之内下游看到的词表不变）；``now`` 是这一次的时刻；
        B1、封口、B4 各一个 span（``scene.b1``、``scene.seal``、``scene.b4``）。

        ``failures`` 给了就是尽力而为：某一步抛了，记进 ``failures``（哪一步, 异常）、走退路接着做——B1 没做成按盘上现有的
        概念集、B4 没做成这一代不写关系表（机会表退到上一代的关系表）——时间线与材料照样交给机会表。不给就照原样往上抛。

        ``relations_held`` 给了（换代那一刻模型队列正在整体退避），B4 这一代不判：不写关系表、时间线照算，机会表退到上一代
        关系表（实时化方案第 2 节"模型调不通 → 关系沿用上一次判定的"），原因记在 ``NightReport.relations_held``。"""

        if now.utcoffset() is None:
            raise TypeError("now must be a timezone-aware datetime")
        from habitus.runtime.scene_night_run import NightRun

        return await NightRun(self, series, words, now, failures, relations_held).run()

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
            recurrences=recurrences_of(series, self.rulers, concepts, coverage),
            coverage=coverage,
        )


__all__ = [
    "NightMaterials",
    "NightReport",
    "SceneNightlyRun",
    "Synced",
]
