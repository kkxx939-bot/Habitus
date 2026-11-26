"""一天的封口：命中记录由代码直接生成，不经模型。

一条记录的命中记录只取决于它在序列上的样子（地址、类编号、lane、是否已归类、最后所见）与那一天成立的日历情境，所以
封口时整天一次生成：

- ``day_digest``：这一天的摘要——序列上这一天每条记录的地址、编号、最后所见，与概念集里日历情境的指纹。完成标记记着它，
  补进或延长了记录、日历情境多了，摘要就变了，这一天要重封。
- ``plan_seal``（纯读盘、纯计算）：这一天应有的全部命中记录、其中与盘上不同要写的、盘上多出来要清掉的。
- ``apply_seal``：清掉多出来的、写要写的、核条数与口径、盖日完成章。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from habitus.foundation.integrity import canonical_digest
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.occurrences.situations import SituationOutcome
from habitus.scene.occurrences.store import ConceptHitStore, DayMarker
from habitus.series import SeriesRecord

#: 命中记录的口径：记录的形状或生成规则改了就升，旧口径的那几天会被认出来重封。
HITS_VERSION = "scene_hits_v1"


def day_digest(records: Sequence[SeriesRecord], concepts: ConceptSet) -> str:
    """一天的摘要：序列上这一天每条记录（地址、编号、最后所见）与概念集里日历情境的指纹。"""

    return canonical_digest(
        {
            "version": HITS_VERSION,
            "records": sorted(
                [record.uri, record.kind_token, record.last_observed_at.isoformat()] for record in records
            ),
            "situations": sorted(concepts[name].fingerprint for name in concepts.situations()),
        }
    )[:16]


@dataclass(frozen=True)
class DaySeal:
    """一天封口的计划（纯读盘、纯计算）：这一天应有的全部命中记录、其中要写的、盘上多出来要清掉的。"""

    day: date
    #: 这一天应有的命中记录，按序列的先后。
    records: tuple[ConceptHits, ...]
    #: 与盘上不同（没有、编号或最后所见变了、日历情境变了）要写的。
    write: tuple[ConceptHits, ...]
    #: 盘上有、序列里没有的命中记录（叶名）：盖章前清掉。
    stale: tuple[str, ...]
    keep: frozenset[str]
    digest: str


def plan_seal(
    store: ConceptHitStore,
    day: date,
    records: Sequence[SeriesRecord],
    concepts: ConceptSet,
    situations: SituationOutcome,
    *,
    now: datetime,
) -> DaySeal:
    """一天封口要做什么：序列上这一天的每条记录各生成一份命中记录（编号、lane、是否已归类、最后所见、那一天成立与判过的
    日历情境），与盘上的比，不同的才写。"""

    on_disk = {item.occurrence_uri: item for item in store.read_day(day)}
    fresh: list[ConceptHits] = []
    write: list[ConceptHits] = []
    for record in records:
        built = ConceptHits(
            occurrence_uri=record.uri,
            kind_token=record.kind_token,
            classified=record.classified,
            last_observed_at=record.last_observed_at,
            situation_hits=situations.hits,
            situations_checked=situations.checked,
            lane=record.lane,
            hits_version=HITS_VERSION,
            sealed_at=now,
        )
        existing = on_disk.get(built.occurrence_uri)
        if existing is not None and _same(existing, built):
            fresh.append(existing)
            continue
        fresh.append(built)
        write.append(built)
    keep = frozenset(item.address.identity_name for item in fresh)
    return DaySeal(
        day=day,
        records=tuple(fresh),
        write=tuple(write),
        stale=tuple(sorted({item.address.identity_name for item in on_disk.values()} - keep)),
        keep=keep,
        digest=day_digest(records, concepts),
    )


def apply_seal(store: ConceptHitStore, plan: DaySeal, *, now: datetime) -> DayMarker:
    """把封口计划落盘：清掉序列里没有的命中记录、写要写的、核条数与口径、盖日完成章。"""

    store.retain_only(plan.day, plan.keep)
    for record in plan.write:
        store.write(record)
    return store.complete_day(
        plan.day, records=len(plan.records), completed_at=now, hits_version=HITS_VERSION, digest=plan.digest
    )


def _same(existing: ConceptHits, built: ConceptHits) -> bool:
    """盘上那份与新生成的是不是同一份（封口时刻不算）。"""

    return (
        existing.kind_token,
        existing.classified,
        existing.lane,
        existing.last_observed_at,
        existing.hits_version,
        tuple(hit.identity for hit in existing.situation_hits),
        existing.situations_checked,
    ) == (
        built.kind_token,
        built.classified,
        built.lane,
        built.last_observed_at,
        built.hits_version,
        tuple(hit.identity for hit in built.situation_hits),
        built.situations_checked,
    )


__all__ = ["HITS_VERSION", "DaySeal", "apply_seal", "day_digest", "plan_seal"]
