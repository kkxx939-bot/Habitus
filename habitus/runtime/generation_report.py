"""每代报告文件 ``<scene_root>/generation/reports/<日子>.json``（数据逻辑看这里，不开 OTLP 也能看）：

- 每一步：结局、原因、耗时、起止、调模型次数（按触点）、这一步原有的报告；
- 各个退路的计数（关系判定退到上一代、结算留到下次、单元历史跳过、换代时还没映射的记录数……）；
- 换代时模型队列各类还欠着几项（``queue_owed``：出错重试、词表、同步词表、写概念、映射、候选条件与先验；没接队列是 null）；
- 这一代读到的输入是哪一代（序列截止日、词表版本、各 lane 关系表那一晚、已盖章天数），截止日之前还没归约的空判断条数；
- 昨天没在看的时段（「未观测」空白，从这一代读到的序列里取）；
- **关系表双向变化**（用户 11-09：「关系的变化，无论是增加还是减少，这些都需要看」）：每条 lane 这一代的关系表比上一晚多出 /
  少了 / 状态变了的每一条，带键、前后状态、当晚读数（p、最小可达 p、方向、靠先验）、族、进族路径、迁移日志里的原因。两晚的表都从
  盘上读（``scene.relations.store``），差由 ``scene.relations.diff`` 算；另报带着要读数的行数、退休行总数与这一代退了几条。

一代一份，没写成的那一代也写（看得见停在哪一步）；与完成标记同一保留期。写报告失败只记观测，不挡换代。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path

from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes
from habitus.runtime.generation_result import FALLBACKS, GenerationResult
from habitus.runtime.generation_trace import prune_dated
from habitus.scene.relations.diff import RelationChange, diff_states, entry_paths, family_of
from habitus.scene.relations.state import LaneState, Status, Transition
from habitus.scene.relations.store import RelationStore, RelationStoreError

_SCHEMA = "generation_report_v1"
_MAX_BYTES = 64 * 1024 * 1024


class GenerationReports:
    def __init__(self, root: str | Path, *, artifact_root: str | Path, relations: RelationStore | None) -> None:
        self.root = Path(root)
        self.artifact_root = Path(artifact_root)
        self.relations = relations

    def path(self, day: date) -> Path:
        return self.root / f"{day.isoformat()}.json"

    def read(self, day: date) -> dict[str, object] | None:
        try:
            raw = read_regular_bytes(self.path(day), artifact_root=self.artifact_root, max_bytes=_MAX_BYTES)
        except FileNotFoundError:
            return None
        loaded = json.loads(raw.decode("utf-8"))
        return loaded if isinstance(loaded, dict) else None

    def write(self, result: GenerationResult, *, now: datetime, relations: Mapping[str, object] | None = None) -> None:
        """写这一代的报告；``relations`` 不给就沿用文件里已有的关系变化。"""

        if relations is None:
            previous = self.read(result.day)
            kept = None if previous is None else previous.get("relations")
            relations = kept if isinstance(kept, dict) else {}
        payload = {
            "schema": _SCHEMA,
            "day": result.day.isoformat(),
            "generated_at": now.isoformat(),
            "complete": result.complete,
            "failed_step": result.failed,
            "reason": result.reason,
            "nothing": result.nothing,
            "started_at": result.started_at.isoformat(),
            "finished_at": None if result.finished_at is None else result.finished_at.isoformat(),
            "lag_seconds": result.lag_seconds,
            "inputs": {
                "series_cutoff": None if result.series_cutoff is None else result.series_cutoff.isoformat(),
                "vocabulary_version": result.vocabulary_version,
                "relation_nights": {
                    lane: None if night is None else night.isoformat()
                    for lane, night in sorted(result.relation_nights.items())
                },
                "stamped_days": result.stamped_days,
                "unreduced_gaps": result.unreduced_gaps,
            },
            "fallbacks": {kind: result.fallbacks.get(kind, 0) for kind in FALLBACKS},
            "queue_owed": None if result.queue_owed is None else dict(sorted(result.queue_owed.items())),
            "unwatched": [
                {"started_at": start.isoformat(), "ended_at": end.isoformat()} for start, end in result.unwatched
            ],
            "problems": list(result.problems),
            "steps": {
                item.step: {
                    "outcome": item.outcome,
                    "reason": item.reason,
                    "duration_seconds": round(item.duration_seconds, 3),
                    "started_at": item.started_at.isoformat(),
                    "finished_at": item.finished_at.isoformat(),
                    "model_calls": item.report.get("model_calls", {}),
                    "report": {key: value for key, value in item.report.items() if key != "model_calls"},
                }
                for item in result.steps
            },
            "relations": dict(relations),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        atomic_replace_bytes(self.path(result.day), encoded, artifact_root=self.artifact_root)

    def relation_changes(self, night: date) -> tuple[dict[str, object], tuple[tuple[str, RelationChange], ...]]:
        """各 lane 第 ``night`` 晚的关系表比上一晚的变化：（报告里那一节, 逐条（lane, 变化）给指标用）。这一晚没有表的 lane 不报。"""

        if self.relations is None:
            return {}, ()
        section: dict[str, object] = {}
        flat: list[tuple[str, RelationChange]] = []
        for lane in self.relations.lanes():
            try:
                if night not in self.relations.nights(lane):
                    continue  # 这一代没写这条 lane 的关系表（关系判定没做成、沿用上一代）：没有变化可报
                after = self.relations.read(lane, night)
                before = self.relations.previous(lane, night)
                transitions = self.relations.transitions(lane)
            except RelationStoreError as exc:
                section[lane] = {"error": str(exc)[:500]}
                continue
            changes = diff_states(before, after)
            section[lane] = _lane_section(before, after, changes, transitions)
            flat.extend((lane, change) for change in changes)
        return section, tuple(flat)

    def prune(self, newest: date, kept_days: int) -> int:
        return prune_dated(self.root, self.artifact_root, newest - timedelta(days=kept_days), suffix=".json")


def _lane_section(
    before: LaneState | None, after: LaneState, changes: tuple[RelationChange, ...], transitions: tuple[Transition, ...]
) -> dict[str, object]:
    """一条 lane 一代的那一节：各类变化的条数、带着要读数的行数（``carried``）、退休行总数（``retired``）、这一代退了几条
    （``retired_tonight``，这一代迁移日志里变成退休的条数）与逐条变化（原因取这一代每条关系最后一条迁移的）。"""

    moved = tuple(item for item in transitions if item.night == after.night)
    reasons = {item.key.text: item.reason for item in moved}
    paths = entry_paths((change.key for change in changes), before, after)
    rows = []
    for change in changes:
        key = change.key
        tonight = None if change.relation is None else change.relation.tonight
        rows.append(
            {
                "key": key.text,
                "antecedent": key.antecedent,
                "consequent": key.consequent,
                "segment": key.segment.value,
                "condition": key.condition or None,
                "kind": change.kind,
                "before": None if change.before is None else change.before.value,
                "after": None if change.after is None else change.after.value,
                "p_value": None if tonight is None else tonight.p_value,
                "minimum_p": None if tonight is None else tonight.minimum_p,
                "verdict": None if tonight is None else tonight.verdict.value,
                "upward": None if change.relation is None else change.relation.upward,
                "prior_only": None if change.relation is None else change.relation.prior_only,
                "family": family_of(key),
                "entry_path": paths.get(key),
                "reason": reasons.get(key.text),
            }
        )
    counts = {kind: sum(1 for change in changes if change.kind == kind) for kind in ("added", "removed", "status")}
    return {
        "night": after.night.isoformat(),
        "previous_night": None if before is None else before.night.isoformat(),
        **counts,
        "carried": len(after.carried()),
        "retired": after.count(Status.RETIRED),
        "retired_tonight": sum(1 for item in moved if item.after is Status.RETIRED),
        "changes": rows,
    }


__all__ = ["GenerationReports"]
