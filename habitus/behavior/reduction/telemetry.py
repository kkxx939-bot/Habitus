"""归约各步骤的可观测属性：从各步自己的产物算计数与时效，不读存储、不碰模型。

归约的复杂度在"什么时候能落、为什么没落"：链要等窗口过去才封口，被链接牵住的链要一起等，
成稿时还会因为名字、链接目标或文档上限被扣下。这里把这些去向分别计数，并算出行为从发生到
上树的时效——预测层能多早用上一件事，取决于这几个数。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from habitus.behavior.model import BehaviorKind
from habitus.behavior.reduction.chains import ChainAssembly
from habitus.behavior.reduction.pending import PendingJudgements
from habitus.behavior.reduction.record import ReducibleJudgement
from habitus.behavior.semantic.model import BehaviorSemanticRefreshResult, BehaviorSemanticRefreshStatus
from habitus.behavior.telemetry import Attributes, count_notes_by_leading_token, seconds_between

_LEDGER_ONLY = "ledger-only"


def assembly_attributes(pending: PendingJudgements) -> Attributes:
    """待归约判断组成的链；丢掉的关系按关系种类计（首词就是关系名或 ``chain``）。"""

    assembly = pending.assembly
    attributes: Attributes = {
        "chains": len(assembly.chains),
        "chain_judgements": sum(len(chain.consumed) for chain in assembly.chains),
        "superseded_judgements": sum(len(chain.superseded) for chain in assembly.chains),
        "gaps": len(assembly.gaps),
        "absorbed_unreadable": len(assembly.absorbed_unreadable),
        "quarantined_records": len(pending.quarantined),
        "quarantined_judgements": len(assembly.quarantined_ids),
        "dropped_edges": len(assembly.dropped_edges),
    }
    attributes.update(count_notes_by_leading_token(assembly.dropped_edges, prefix="dropped_"))
    return attributes


def seal_attributes(
    assembly: ChainAssembly,
    ready_indexes: Sequence[int],
    ready_gaps: Sequence[ReducibleJudgement],
    *,
    now: datetime,
    horizon: datetime,
    frontier_cutoff: datetime | None,
) -> Attributes:
    """封口：视界落后现在多久、谁在拖它，以及到了时间却被链接牵住没放行的链。"""

    ready = set(ready_indexes)
    unsealed = [chain for index, chain in enumerate(assembly.chains) if index not in ready]
    due = sum(1 for chain in assembly.chains if chain.newest_evidence_at < horizon)
    return {
        "ready_chains": len(ready),
        "ready_gaps": len(ready_gaps),
        "unsealed_chains": len(unsealed),
        "unsealed_gaps": len(assembly.gaps) - len(ready_gaps),
        # 证据已过视界、却因并行/因果链接的另一端还没封口而一起等的链。
        "chains_held_by_links": due - len(ready),
        "horizon_lag_seconds": seconds_between(now, horizon),
        # 最早一条还没被融合覆盖的观测把视界往回拉；融合队列卡住时它会一直涨。
        "frontier_pending": frontier_cutoff is not None,
        "frontier_age_seconds": 0.0 if frontier_cutoff is None else seconds_between(now, frontier_cutoff),
        "oldest_unsealed_wait_seconds": (
            seconds_between(now, min(chain.head.evidence_ready_at for chain in unsealed)) if unsealed else 0.0
        ),
    }


def kind_attributes(
    *,
    requests: int,
    classified: int,
    pending: int,
    not_events: int,
    model_calls: int,
    signals: Iterable[str],
) -> Attributes:
    """白天归类：多少条链交去归类、归进在用类 / 进待定 / 判为非事件各多少；信号按首词计。

    ``model_calls`` 含结构不对之后的重问。
    """

    attributes: Attributes = {
        "chains": requests,
        "classified": classified,
        "pending": pending,
        "not_events": not_events,
        "model_calls": model_calls,
    }
    attributes.update(count_notes_by_leading_token(signals, prefix="signal_"))
    return attributes


def stage_attributes(
    documents: Sequence[Mapping[str, Any]],
    notes: Sequence[str],
    *,
    ready_chains: int,
    ready_gaps: int,
    shrink_rounds: int,
    checkpoint_bytes: int,
) -> Attributes:
    """成稿：就绪的链有多少真的成了文档，其余被扣下（名字不可寻址、链接目标没成稿、过不了校验，
    或检查点超界被二分推迟到下一轮）。"""

    kinds = Counter(str(item.get("kind")) for item in documents)
    occurrences = kinds.get(BehaviorKind.OCCURRENCE.value, 0)
    attributes: Attributes = {
        "ready_chains": ready_chains,
        "ready_gaps": ready_gaps,
        "occurrences": occurrences,
        "gaps": kinds.get(BehaviorKind.GAP.value, 0),
        "gaps_by_reference": kinds.get(_LEDGER_ONLY, 0),
        "chains_held_back": ready_chains - occurrences,
        "disambiguated": sum(
            1
            for item in documents
            if item.get("kind") == BehaviorKind.OCCURRENCE.value and item["payload"].get("original_name") is not None
        ),
        "links": sum(len(item.get("links", ())) for item in documents),
        "shrink_rounds": shrink_rounds,
        "checkpoint_bytes": checkpoint_bytes,
        "notes": len(notes),
    }
    attributes.update(count_notes_by_leading_token(notes, prefix="note_"))
    return attributes


def publish_attributes(
    documents: Sequence[Mapping[str, Any]], *, published_at: datetime, kind_pending: int
) -> Attributes:
    """落盘：文档数与上树时效。

    ``end_to_publish`` 从行为最后一次被看到算起，``onset_to_publish`` 从系统第一次能知道它开始了
    算起——后者是"一件正在发生的事要多久才能被下游用上"。
    """

    kinds = Counter(str(item.get("kind")) for item in documents)
    attributes: Attributes = {
        "ledger_entries": len(documents),
        "occurrences": kinds.get(BehaviorKind.OCCURRENCE.value, 0),
        "gaps": kinds.get(BehaviorKind.GAP.value, 0),
        "gaps_by_reference": kinds.get(_LEDGER_ONLY, 0),
        "kind_pending": kind_pending,
    }
    occurrences = [item["payload"] for item in documents if item.get("kind") == BehaviorKind.OCCURRENCE.value]
    if occurrences:
        ended = [seconds_between(published_at, _instant(payload["last_observed_at"])) for payload in occurrences]
        onset = [seconds_between(published_at, _instant(payload["onset_available_at"])) for payload in occurrences]
        attributes["end_to_publish_seconds_max"] = max(ended)
        attributes["end_to_publish_seconds_min"] = min(ended)
        attributes["onset_to_publish_seconds_max"] = max(onset)
    return attributes


def refresh_attributes(results: Sequence[BehaviorSemanticRefreshResult], *, days: int, failed_days: int) -> Attributes:
    """摘要刷新：每个目录的去向；``written`` 才真的调了模型（日目录）或重写了文件。"""

    statuses = Counter(result.status for result in results)
    attributes: Attributes = {"days": days, "failed_days": failed_days}
    for status in BehaviorSemanticRefreshStatus:
        attributes[f"directories_{status.value}"] = statuses.get(status, 0)
    return attributes


def _instant(value: object) -> datetime:
    return datetime.fromisoformat(str(value))


__all__ = [
    "assembly_attributes",
    "kind_attributes",
    "publish_attributes",
    "refresh_attributes",
    "seal_attributes",
    "stage_attributes",
]
