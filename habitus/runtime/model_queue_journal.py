"""模型队列留下的痕迹（``runtime.model_queue``）：本地流水（一天一份，一行一项），与每一项、每一轮的指标和日志。"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from datetime import date, timedelta, tzinfo
from pathlib import Path

from habitus.foundation.observability import ObservationStatus, Observer, record_observation
from habitus.infrastructure.store.filesystem import ensure_real_directory, require_safe_artifact_path
from habitus.runtime.generation_marks import GenerationMarkError, GenerationMarks
from habitus.runtime.generation_summary import reason_code
from habitus.runtime.generation_trace import prune_dated
from habitus.runtime.model_queue_state import CATEGORIES, STATUS, Item, ItemResult, ModelQueueState, Round

logger = logging.getLogger("habitus.model_queue")

_MAX_JOURNAL_BYTES = 64 * 1024 * 1024
_SIGNALS_KEPT = 200


class QueueJournal:
    """模型队列的本地流水 ``<scene_root>/model_queue/<日期>.jsonl``：一行一项（类别、结局、原因、用时、调模型次数、读数与
    信号文字）。信号带用户派生文字，只留在本地；保留天数同完成标记。"""

    def __init__(self, root: str | Path, *, artifact_root: str | Path, zone: tzinfo) -> None:
        self.root = Path(root)
        self.artifact_root = Path(artifact_root)
        self.zone = zone

    def path(self, day: date) -> Path:
        return self.root / f"{day.isoformat()}.jsonl"

    def read(self, day: date) -> tuple[dict[str, object], ...]:
        """读回一天的流水（测试与运维用）；没有文件是空。"""

        try:
            text = self.path(day).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ()
        return tuple(json.loads(line) for line in text.splitlines() if line.strip())

    def append(self, result: ItemResult, *, key: str | None, day: date | None, signals: Sequence[str]) -> None:
        payload = {
            "at": result.started_at.isoformat(),
            "category": result.category,
            "key": key,
            "day": None if day is None else day.isoformat(),
            "outcome": result.outcome,
            "reason": result.reason,
            "duration_seconds": round(result.duration_seconds, 3),
            "model_calls": result.model_calls,
            "detail": dict(result.detail),
            "signals": list(signals[:_SIGNALS_KEPT]),
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        ensure_real_directory(self.root, artifact_root=self.artifact_root)
        target = self.path(result.started_at.astimezone(self.zone).date())
        path = require_safe_artifact_path(self.artifact_root, target, label="model queue journal")
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            if os.fstat(descriptor).st_size + len(line) > _MAX_JOURNAL_BYTES:
                return
            os.write(descriptor, line.encode("utf-8"))
        finally:
            os.close(descriptor)

    def prune(self, newest: date, kept_days: int) -> int:
        if not self.root.is_dir():
            return 0
        return prune_dated(self.root, self.artifact_root, newest - timedelta(days=kept_days), suffix=".jsonl")


class QueueRecorder:
    """模型队列的指标、日志、本地流水与留存清理。都是旁路：出错只记观测，不挡队列。"""

    def __init__(
        self, observer: Observer, journal: QueueJournal | None, marks: GenerationMarks | None, kept_days: int
    ) -> None:
        self.observer = observer
        self.journal = journal
        self.marks = marks
        self.kept_days = kept_days

    def item(self, result: ItemResult, item: Item, signals: list[str]) -> None:
        """一项的指标、日志与本地流水。都是旁路：出错只记观测。"""

        attributes: dict[str, str | int | float | bool] = {
            "category": result.category,
            "outcome": result.outcome,
            "duration_seconds": result.duration_seconds,
            "model_calls": result.model_calls,
        }
        lock = result.detail.get("lock_seconds")
        if isinstance(lock, int | float):
            attributes["sweep_lock_seconds"] = float(lock)  # 词表整理活整段持 sweep 锁多久
        record_observation(
            self.observer,
            "model_queue",
            "item",
            STATUS.get(result.outcome, ObservationStatus.FAILURE),
            attributes=attributes,
        )
        logger.info(
            "model queue item category=%s outcome=%s reason=%s duration=%.1fs model_calls=%d",
            result.category,
            result.outcome,
            reason_code(result.reason) or "-",
            result.duration_seconds,
            result.model_calls,
        )
        if self.journal is None:
            return
        try:
            self.journal.append(result, key=item.key, day=item.day, signals=signals)
        except Exception as exc:  # noqa: BLE001 - 流水是旁路：写不出不挡队列
            self.observe("journal", ObservationStatus.FAILURE, {"error_type": type(exc).__name__})

    def round(self, state: ModelQueueState, current: Round) -> None:
        """一轮看完：各类还欠多少、最老一项等了多久记指标；欠着的类跨过了几代从完成标记上数；留存清理。"""

        for category in CATEGORIES:
            count = state.owed.get(category, 0)
            since = state.since.get(category)
            waited = 0.0 if not count or since is None else max(0.0, (current.now - since).total_seconds())
            self.observe(
                "owed", ObservationStatus.SUCCESS, {"category": category, "count": count, "oldest_wait_seconds": waited}
            )
        if self.marks is not None and any(state.owed.values()):
            try:
                finished = [mark.finished_at for day in self.marks.days() if (mark := self.marks.read(day)) is not None]
            except GenerationMarkError:
                finished = []  # 标记读不了健康面另报
            for category, since in state.since.items():
                if state.owed.get(category):
                    state.crossed[category] = sum(1 for moment in finished if moment > since)
        if self.journal is not None:
            try:
                self.journal.prune(current.today, self.kept_days)
            except Exception as exc:  # noqa: BLE001 - 留存清理是旁路
                self.observe("prune", ObservationStatus.FAILURE, {"error_type": type(exc).__name__})

    def backoff(self, failures: int, delay: float) -> None:
        logger.warning("model queue backing off failures=%d delay=%.0fs", failures, delay)
        self.observe("backoff", ObservationStatus.DEGRADED, {"failures": failures, "delay_seconds": delay})

    def recovered(self, waited: float) -> None:
        logger.info("model queue recovered after %.0fs", waited)
        self.observe("recovered", ObservationStatus.SUCCESS, {"unreachable_seconds": waited})

    def scan_failed(self, category: str, code: str) -> None:
        logger.warning("model queue cannot read what is owed category=%s error=%s", category, code)
        self.observe("scan", ObservationStatus.FAILURE, {"category": category, "error_type": code})

    def observe(
        self, operation: str, status: ObservationStatus, attributes: dict[str, str | int | float | bool]
    ) -> None:
        record_observation(self.observer, "model_queue", operation, status, attributes=attributes)


__all__ = ["QueueJournal", "QueueRecorder"]
