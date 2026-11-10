"""换代与预测拍的 trace：span 落本地 ``<scene_root>/generation/traces/<日期>.jsonl``（一行一个 span：名字、父子、起止、属性、
结局），开了 OTLP 时同时进 OTel（``ManagedObservability``）——不开 collector 也能看全。

- ``TraceFiles.sink(day)`` 给 ``foundation.tracing.open_trace`` 用：活的 span 开始时进 OTel 的当前 span（子 span、模型调用都
  挂在它下面），结束时把属性一并挂上、写一行本地文件。
- ``ModelCallTracer`` 是一个观测后端：调用链上有 span 时，每一次模型调用（``model`` 类事件）补一个 ``model.call`` 子 span，挂在
  发起它的段或子步下面，并把调用次数按（触点, 后端, 模型）累计到各级父 span 上；同时按（段, 触点）记一条指标事件。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, MutableMapping
from contextlib import AbstractContextManager
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

from habitus.foundation.observability import ObservationEvent, ObservationStatus, Observer, record_observation
from habitus.foundation.tracing import SpanRecord, TraceSink, ancestors, record_child, tracing_active
from habitus.infrastructure.store.filesystem import (
    durable_unlink,
    ensure_real_directory,
    list_real_directory,
    require_safe_artifact_path,
)

#: 一天的 trace 文件最多这么大；再多的 span 丢掉（观测是旁路，不让它把盘写满）。
_MAX_TRACE_BYTES = 64 * 1024 * 1024
_MAX_FILES = 4_096
#: 换代每一步的 span 名前缀：模型调用按"属于哪一步"记指标时认它。
STEP_SPAN_PREFIX = "generation."
#: 换代那一个父 span（各步的子 span 挂在它下面）。
GENERATION_SPAN = "generation.run"


class SpanBackend(Protocol):
    """OTel 那一侧（``ManagedObservability``）：开一个当前 span、事后补一个已结束的 span。"""

    def start_span(
        self,
        category: str,
        operation: str,
        *,
        attributes: Mapping[str, str | int | float | bool] | None = None,
        traceparent: str | None = None,
    ) -> AbstractContextManager[MutableMapping[str, str | int | float | bool] | None]: ...

    def record_span(
        self,
        name: str,
        *,
        start_ns: int,
        end_ns: int,
        attributes: Mapping[str, str | int | float | bool],
        error_type: str | None = None,
    ) -> None: ...


class TraceFiles:
    """一天一份 trace 文件；保留天数同完成标记。"""

    def __init__(self, root: str | Path, *, artifact_root: str | Path, backend: SpanBackend | None = None) -> None:
        self.root = Path(root)
        self.artifact_root = Path(artifact_root)
        self.backend = backend

    def path(self, day: date) -> Path:
        return self.root / f"{day.isoformat()}.jsonl"

    def sink(self, day: date) -> TraceSink:
        return _DaySink(self, day)

    def read(self, day: date) -> tuple[dict[str, object], ...]:
        """读回一天的 span（测试与运维用）；没有文件是空。"""

        try:
            text = self.path(day).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ()
        return tuple(json.loads(line) for line in text.splitlines() if line.strip())

    def append(self, day: date, record: SpanRecord) -> None:
        line = json.dumps(_encode(record), ensure_ascii=False, sort_keys=True) + "\n"
        ensure_real_directory(self.root, artifact_root=self.artifact_root)
        path = require_safe_artifact_path(self.artifact_root, self.path(day), label="generation trace file")
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            if os.fstat(descriptor).st_size + len(line) > _MAX_TRACE_BYTES:
                return
            os.write(descriptor, line.encode("utf-8"))
        finally:
            os.close(descriptor)

    def prune(self, newest: date, kept_days: int) -> int:
        return prune_dated(self.root, self.artifact_root, newest - timedelta(days=kept_days), suffix=".jsonl")


class _DaySink:
    def __init__(self, files: TraceFiles, day: date) -> None:
        self.files = files
        self.day = day

    def started(self, record: SpanRecord) -> object:
        backend = self.files.backend
        if backend is None or record.ended_at is not None:
            return None
        category, _, operation = record.name.partition(".")
        manager = backend.start_span(category, operation or category, attributes=dict(record.attributes))
        late = manager.__enter__()
        return (manager, late)

    def finished(self, record: SpanRecord, handle: object) -> None:
        _fold_model_calls(record)
        backend = self.files.backend
        if isinstance(handle, tuple):
            manager, late = handle
            if isinstance(late, dict):
                late.update(record.attributes)
            manager.__exit__(None, None, None)
        elif backend is not None and record.ended_at is not None and record.name == "model.call":
            backend.record_span(
                record.name,
                start_ns=int(record.started_at.timestamp() * 1_000_000_000),
                end_ns=int(record.ended_at.timestamp() * 1_000_000_000),
                attributes=dict(record.attributes),
                error_type=record.error_type,
            )
        self.files.append(self.day, record)


def _fold_model_calls(record: SpanRecord) -> None:
    """子 span 里的模型调用摊进这个 span 的属性：总数、按触点、按后端 / 模型。"""

    if record.name == "model.call":
        return
    by_touch: dict[str, int] = {}
    by_model: dict[str, int] = {}
    for (touch, provider, model), count in record.model_calls.items():
        by_touch[touch] = by_touch.get(touch, 0) + count
        by_model[f"{provider}/{model}"] = by_model.get(f"{provider}/{model}", 0) + count
    record.attributes["model_calls"] = sum(by_touch.values())
    for touch, count in sorted(by_touch.items()):
        record.attributes[f"model_calls.touch.{touch}"[:128]] = count
    for model, count in sorted(by_model.items()):
        record.attributes[f"model_calls.model.{model}"[:128]] = count


def _encode(record: SpanRecord) -> dict[str, object]:
    return {
        "name": record.name,
        "trace_id": record.trace_id,
        "span_id": record.span_id,
        "parent_id": record.parent_id,
        "started_at": record.started_at.isoformat(),
        "ended_at": None if record.ended_at is None else record.ended_at.isoformat(),
        "duration_seconds": round(record.duration_seconds, 6),
        "status": record.status,
        "error_type": record.error_type,
        "attributes": dict(record.attributes),
    }


class ModelCallTracer:
    """观测后端：把模型调用变成 ``model.call`` 子 span（只在调用链上有 span 时）。``metrics`` 收按（段, 触点）的计数事件。"""

    def __init__(self, metrics: Observer | None = None) -> None:
        self.metrics = metrics

    def record(self, event: ObservationEvent) -> None:
        if event.category != "model" or event.operation not in ("complete", "complete_async") or not tracing_active():
            return
        attributes = event.attributes
        touch = str(attributes.get("touch") or "chat")
        # 后端与模型记实际作答的那一对（答复上带的）；没有答复（失败）时只有路由配置的那一对
        provider = str(attributes.get("response_provider") or attributes.get("provider") or "unknown")
        model = str(attributes.get("response_model") or attributes.get("model") or "unknown")
        retries = attributes.get("retry_count")
        failed = event.status is ObservationStatus.FAILURE
        record_child(
            "model.call",
            started_at=event.occurred_at - timedelta(seconds=event.duration_seconds),
            ended_at=event.occurred_at,
            attributes={
                **{
                    key: value
                    for key, value in attributes.items()
                    if key not in ("retry_count", "provider", "model", "response_provider", "response_model")
                },
                "touch": touch,
                "provider": provider,
                "model": model,
                "route_provider": str(attributes.get("provider") or "unknown"),
                "route_model": str(attributes.get("model") or "unknown"),
                "attempts": (retries if isinstance(retries, int) else 0) + 1,
                "duration_seconds": event.duration_seconds,
                "succeeded": not failed,
                **({"error_type": str(attributes.get("error_code") or "model_error")} if failed else {}),
            },
            error_type=str(attributes.get("error_code") or "model_error") if failed else None,
        )
        chain = ancestors()
        for item in chain:
            item.count_model_call(touch, provider, model)
        step = next(
            (
                item.name.removeprefix(STEP_SPAN_PREFIX)
                for item in reversed(chain)
                if item.name.startswith(STEP_SPAN_PREFIX) and item.name != GENERATION_SPAN
            ),
            None,
        )
        if self.metrics is not None and step is not None:
            record_observation(
                self.metrics, "generation", "model_call", event.status, attributes={"step": step, "touch": touch}
            )


def prune_dated(directory: Path, artifact_root: Path, floor: date, *, suffix: str) -> int:
    """按日子命名的文件里比 ``floor`` 早的删掉。"""

    removed = 0
    for entry in list_real_directory(directory, artifact_root=artifact_root, max_entries=_MAX_FILES):
        if not entry.is_file() or not entry.name.endswith(suffix):
            continue
        try:
            day = date.fromisoformat(entry.name.removesuffix(suffix))
        except ValueError:
            continue
        if day < floor and durable_unlink(directory / entry.name, artifact_root=artifact_root):
            removed += 1
    return removed


__all__ = ["GENERATION_SPAN", "STEP_SPAN_PREFIX", "ModelCallTracer", "SpanBackend", "TraceFiles", "prune_dated"]
