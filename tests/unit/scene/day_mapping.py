"""测试用：一天按条映射再封口，一次做完（收口删了产品里离线一次做完的 ``map_closed_day``，领域测试用这里由按条的入口拼起来的
同一个做法：``current`` 认续跑 → ``map_record`` 按条映射 → ``plan_seal`` / ``apply_seal`` 封口）。只在测试里用。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

from habitus.behavior.tree import BehaviorTree
from habitus.scene.occurrences.daily import (
    BaselineFor,
    DayRecords,
    DaySeal,
    SituationFor,
    apply_seal,
    current,
    day_documents,
    map_record,
    plan_seal,
)
from habitus.scene.occurrences.mapper import ConceptMapper, ConceptMapperError
from habitus.scene.occurrences.model import UnresolvedReason
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.series import EventSeries
from habitus.series.model import SkipReason


@dataclass(frozen=True)
class DayMappingReport:
    day: date
    mapped: int
    resumed: int
    duplicates_skipped: int
    stale_removed: int
    unresolved: int
    #: 盘上原本就有、这次又重新判了一遍的记录数（口径变了、编号被迁移改了、或输入变了）。它加上 ``stale_removed``
    #: 不为零，就说明这一天的命中**变了**——靶它开的承诺与结算已经是脏的，夜批要撤了重开（评审 A-12 / B-5）。
    rewritten: int = 0
    not_events_skipped: int = 0
    #: 候选没变、只就地重算了情境的记录数（不问模型）。
    refreshed: int = 0
    #: 这一天里有几个（记录, 细分概念）是模型这一次没答成——有就不盖完成章，之后回来重问（E10）。
    model_failed: int = 0
    #: 有几个是两遍判得不一样——映射器一致性在生产上的读数（`10` 判决线要的就是它）。
    inconsistent: int = 0
    #: 这一天盖了完成章没有（有记录还不作数——模型没答成、材料没给到——就不盖）。
    sealed: bool = True

    @property
    def changed(self) -> bool:
        return self.rewritten > 0 or self.stale_removed > 0


async def map_closed_day(
    tree: BehaviorTree,
    store: ConceptHitStore,
    mapper: ConceptMapper,
    day: date,
    *,
    now: datetime,
    series: EventSeries,
    situation_for: SituationFor,
    baseline_for: BaselineFor,
    force: bool = False,
    recent_given: Callable[[str], bool] | None = None,
) -> DayMappingReport:
    """把已封口的一天从行为树映射到 ``occurrences/`` 并封口：按条映射该映射的（``map_record``），再统一重算情境、盖章
    （``plan_seal`` / ``apply_seal``）。离线回放与测试用；线上按条映射归模型队列、封口归换代。

    - 映射哪几条、各是类还是「待定」、属于哪条 lane、当天的空白、近几天的同类记录，全部按事件序列 ``series`` 读；行为树
      只用来取这几条的全文；
    - **续跑**：同口径、编号没变、且输入没变的记录不重问模型；
    - 行为树这一天比盘上少时拒绝清掉多出来的记录，除非 ``force``；
    - ``situation_for`` / ``baseline_for`` 都必填：要空就显式传 ``lambda _d: ()`` / ``lambda _d: {}``。
    """

    if day >= series.cutoff:
        raise ConceptMapperError(f"{day} is not sealed in a series cut at {series.cutoff}")
    documents = day_documents(tree, series, day)
    on_disk = len(store.read_day(day))
    if len(documents) < on_disk and not force:
        # 盘上比树上多：树读不到（根指错 / 封口日算错）或缩水了。静默清掉多出来的再盖章，就是"做完了但内容不对"。
        raise ConceptMapperError(
            f"the behaviour tree shows {len(documents)} occurrences on {day} but the hit store holds {on_disk}; pass force=True to drop the extra records"
        )
    removed = store.retain_only(day, frozenset(document.address.identity_name for document in documents.values()))
    records = DayRecords(series, day, mapper, baseline_for=baseline_for, recent_given=recent_given)
    kept: set[str] = set()
    rewritten = 0
    for uri, document in documents.items():
        entry = records.records[uri]
        material = records.material(entry, document)
        existing = store.read(document.address) if store.exists(document.address) else None
        if current(existing, mapper, entry, material):
            kept.add(uri)
            continue
        rewritten += existing is not None
        await map_record(store, mapper, document, entry, material)
    plan = plan_seal(documents, store, mapper, records, situation_for=situation_for)
    # 原来就作数、这次只就地重算了情境的（这次新映射的那几条写回情境不算在里面）
    refreshed = sum(1 for record in plan.rewrite if record.occurrence_uri in kept)
    return _report(store, mapper, plan, series, now, len(kept) - refreshed, refreshed, rewritten, len(removed))


def _report(
    store: ConceptHitStore,
    mapper: ConceptMapper,
    plan: DaySeal,
    series: EventSeries,
    now: datetime,
    resumed: int,
    refreshed: int,
    rewritten: int,
    removed: int,
) -> DayMappingReport:
    day = plan.day
    reasons = [reason for item in store.read_day(day) for reason in item.unresolved.values()]
    marker = apply_seal(store, mapper, plan, now=now) if plan.ready else None
    return DayMappingReport(
        day=day,
        mapped=plan.expected - resumed - refreshed,
        resumed=resumed,
        refreshed=refreshed,
        duplicates_skipped=series.skipped_on(day, SkipReason.DUPLICATE),
        stale_removed=removed,
        unresolved=len(reasons) if marker is None else marker.unresolved,
        rewritten=rewritten,
        not_events_skipped=series.skipped_on(day, SkipReason.NOT_EVENT),
        # 模型这一次没答成的不盖章：盖了章这一天就不会再被访问，"没答成"就永远是"看不到"（第四轮评审 E10）
        model_failed=0 if marker is not None else reasons.count(UnresolvedReason.MODEL_FAILED),
        inconsistent=reasons.count(UnresolvedReason.INCONSISTENT),
        sealed=marker is not None,
    )
