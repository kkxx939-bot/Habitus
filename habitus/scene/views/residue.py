"""residue/：残差候选——还没映射到任何概念的 kind 攒了多少、跨了几天、离升级判据差多少。

① 的残差升级靠这里触发：某个 kind 累计 ≥ K 次且跨 ≥ D 天 → 交 LLM 写定义。K、D 在重放上定。
"没映射到"＝ 记录的 ``hits`` 空且 ``unresolved`` 空——未决的不算残差，那是材料没给到，不是概念集没覆盖。
已被某个概念认领的 kind（``ConceptSource.kind_token``）不再是候选：新概念不回填历史命中，升级前那些记录永远 ``hits=[]``，
不排除就会每晚再报一次"可升级"。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from habitus.scene.occurrences.store import ConceptHitStore


@dataclass(frozen=True)
class ResidueCandidate:
    kind_token: str
    occurrences: int
    days: int
    first_seen: date
    last_seen: date
    example_uri: str
    short_by_occurrences: int
    short_by_days: int

    @property
    def ready(self) -> bool:
        return self.short_by_occurrences == 0 and self.short_by_days == 0


def residue_candidates(hits: ConceptHitStore, days: Iterable[date], *, k: int, d: int, claimed: frozenset[str] = frozenset()) -> tuple[ResidueCandidate, ...]:
    """扫这些天的命中记录，按 kind 攒残差；``claimed`` 里的 kind 跳过；按次数降序、再按 kind。"""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0 or isinstance(d, bool) or not isinstance(d, int) or d <= 0:
        raise ValueError("k and d are positive integers")
    tally: dict[str, list] = {}
    for day in sorted(set(days)):
        for record in hits.read_day(day):
            if record.hits or record.unresolved or record.kind_token in claimed:
                continue
            entry = tally.setdefault(record.kind_token, [0, set(), day, day, record.occurrence_uri])
            entry[0] += 1
            entry[1].add(record.started_at.date())
            entry[2] = min(entry[2], record.started_at.date())
            entry[3] = max(entry[3], record.started_at.date())
    found = [
        ResidueCandidate(
            kind_token=kind,
            occurrences=count,
            days=len(day_set),
            first_seen=first,
            last_seen=last,
            example_uri=uri,
            short_by_occurrences=max(0, k - count),
            short_by_days=max(0, d - len(day_set)),
        )
        for kind, (count, day_set, first, last, uri) in tally.items()
    ]
    return tuple(sorted(found, key=lambda item: (-item.occurrences, item.kind_token)))


__all__ = ["ResidueCandidate", "residue_candidates"]
