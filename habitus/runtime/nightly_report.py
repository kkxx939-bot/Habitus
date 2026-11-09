"""每晚报告文件 ``<scene_root>/nightly/reports/<这一晚>.json``（数据逻辑看这里，不开 OTLP 也能看）：

- 每段：状态、结局原因、尝试次数、耗时、调模型次数（按触点）、这一段原有的报告（词表段的报告不再丢）——都从 Job 账抄；
- **关系表双向变化**（用户 11-09：「关系的变化，无论是增加还是减少，这些都需要看」）：每条 lane 这一晚的关系表比上一晚多出 /
  少了 / 状态变了的每一条，带键、前后状态、当晚读数（p、最小可达 p、方向、靠先验）、族、进族路径、迁移日志里的原因。两晚的表都从
  盘上读（``scene.relations.store``），差由 ``scene.relations.diff`` 算；结构那一段成了之后算一次，之后各段重写报告时沿用。

报告与 Job 账同一保留期。写报告失败只记观测，不挡线。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path

from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes
from habitus.runtime.nightly_jobs import NightLedger
from habitus.runtime.nightly_trace import prune_dated
from habitus.scene.relations.diff import RelationChange, diff_states, entry_paths, family_of
from habitus.scene.relations.state import LaneState, Transition
from habitus.scene.relations.store import RelationStore, RelationStoreError

_SCHEMA = "nightly_report_v1"
_MAX_BYTES = 64 * 1024 * 1024


class NightReports:
    def __init__(self, root: str | Path, *, artifact_root: str | Path, relations: RelationStore | None) -> None:
        self.root = Path(root)
        self.artifact_root = Path(artifact_root)
        self.relations = relations

    def path(self, night: date) -> Path:
        return self.root / f"{night.isoformat()}.json"

    def read(self, night: date) -> dict[str, object] | None:
        try:
            raw = read_regular_bytes(self.path(night), artifact_root=self.artifact_root, max_bytes=_MAX_BYTES)
        except FileNotFoundError:
            return None
        loaded = json.loads(raw.decode("utf-8"))
        return loaded if isinstance(loaded, dict) else None

    def write(self, ledger: NightLedger, *, now: datetime, relations: Mapping[str, object] | None = None) -> None:
        """按 Job 账重写这一晚的报告；``relations`` 不给就沿用文件里已有的关系变化。"""

        if relations is None:
            previous = self.read(ledger.night)
            kept = None if previous is None else previous.get("relations")
            relations = kept if isinstance(kept, dict) else {}
        payload = {
            "schema": _SCHEMA,
            "night": ledger.night.isoformat(),
            "generated_at": now.isoformat(),
            "complete": ledger.complete,
            "steps": {
                job.step: {
                    "state": job.state,
                    "reason": job.reason,
                    "attempts": job.attempts,
                    "duration_seconds": round(job.duration_seconds, 3),
                    "started_at": None if job.started_at is None else job.started_at.isoformat(),
                    "finished_at": None if job.finished_at is None else job.finished_at.isoformat(),
                    "next_attempt_at": None if job.next_attempt_at is None else job.next_attempt_at.isoformat(),
                    "model_calls": job.report.get("model_calls", {}),
                    "report": {key: value for key, value in job.report.items() if key != "model_calls"},
                }
                for job in ledger.jobs
            },
            "relations": dict(relations),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        atomic_replace_bytes(self.path(ledger.night), encoded, artifact_root=self.artifact_root)

    def relation_changes(self, night: date) -> tuple[dict[str, object], tuple[tuple[str, RelationChange], ...]]:
        """各 lane 第 ``night`` 晚的关系表比上一晚的变化：（报告里那一节, 逐条（lane, 变化）给指标用）。这一晚没有表的 lane 不报。"""

        if self.relations is None:
            return {}, ()
        section: dict[str, object] = {}
        flat: list[tuple[str, RelationChange]] = []
        for lane in self.relations.lanes():
            try:
                after = self.relations.read(lane, night)
                before = self.relations.previous(lane, night)
                reasons = _reasons(self.relations.transitions(lane), night)
            except RelationStoreError as exc:
                section[lane] = {"error": str(exc)[:500]}
                continue
            changes = diff_states(before, after)
            section[lane] = _lane_section(before, after, changes, reasons)
            flat.extend((lane, change) for change in changes)
        return section, tuple(flat)

    def prune(self, newest: date, kept_days: int) -> int:
        return prune_dated(self.root, self.artifact_root, newest - timedelta(days=kept_days), suffix=".json")


def _reasons(transitions: tuple[Transition, ...], night: date) -> dict[str, str]:
    """这一晚每条关系最后一条迁移的原因（键的文本 → 原因）。"""

    return {item.key.text: item.reason for item in transitions if item.night == night}


def _lane_section(
    before: LaneState | None, after: LaneState, changes: tuple[RelationChange, ...], reasons: Mapping[str, str]
) -> dict[str, object]:
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
        "changes": rows,
    }


__all__ = ["NightReports"]
