"""把观察事件投影成小而固定的低基数指标，并以 Prometheus 文本暴露的进程内指标后端。"""

from __future__ import annotations

import math
import re
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from habitus.foundation.observability import ObservationEvent, ObservationStatus

_DEFAULT_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
_LABEL_ATTRIBUTES = frozenset(
    {
        "provider",
        "model",
        "http_method",
        "http_route",
        "http_status_class",
        "stage",
        "worker_state",
        "job_status",
        # Conversation Source Consumer 的取值由枚举封闭，基数恒定。
        "consumer",
    }
)


@dataclass(frozen=True)
class ObservabilitySnapshot:
    counters: Mapping[str, int]
    duration_seconds: Mapping[str, float]
    gauges: Mapping[str, float]
    recent_events: tuple[ObservationEvent, ...]
    histogram_counts: Mapping[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricUpdate:
    """由同一投影规则交给进程内指标和可选 OTel 指标。"""

    name: str
    kind: str
    value: float
    labels: tuple[tuple[str, str], ...] = ()


def project_metric_updates(event: ObservationEvent) -> tuple[MetricUpdate, ...]:
    """把事件投影为小而固定的低基数指标集合。"""

    base_labels = (
        ("category", _label_value(event.category)),
        ("operation", _label_value(event.operation)),
        ("status", event.status.value),
    )
    updates = [
        MetricUpdate("operations_total", "counter", 1.0, base_labels),
        MetricUpdate("operation_duration_seconds", "histogram", event.duration_seconds, base_labels),
    ]
    attributes = event.attributes
    if event.category == "http" and event.operation == "request":
        labels = _selected_labels(attributes, "http_method", "http_route", "http_status_class")
        updates.extend(
            (
                MetricUpdate("http_requests_total", "counter", 1.0, labels),
                MetricUpdate("http_request_duration_seconds", "histogram", event.duration_seconds, labels),
            )
        )
    if event.category == "model":
        labels = _selected_labels(attributes, "provider", "model") + (("operation", _label_value(event.operation)),)
        updates.append(MetricUpdate("model_requests_total", "counter", 1.0, labels + (("status", event.status.value),)))
        retries = _non_negative_number(attributes.get("retry_count"))
        if retries:
            updates.append(MetricUpdate("model_retries_total", "counter", retries, labels))
        for attribute, metric_name in (
            ("input_tokens", "model_input_tokens_total"),
            ("output_tokens", "model_output_tokens_total"),
        ):
            value = _non_negative_number(attributes.get(attribute))
            if value:
                updates.append(MetricUpdate(metric_name, "counter", value, labels))
    if event.category == "retrieval":
        labels = (("operation", _label_value(event.operation)), ("status", event.status.value))
        updates.append(MetricUpdate("retrieval_requests_total", "counter", 1.0, labels))
        degradations = _non_negative_number(attributes.get("degradation_count"))
        if degradations:
            updates.append(MetricUpdate("retrieval_degradations_total", "counter", degradations, labels[:1]))
    if event.category == "lock" and event.status is not ObservationStatus.SUCCESS:
        updates.append(
            MetricUpdate(
                "lock_contention_total",
                "counter",
                1.0,
                (("operation", _label_value(event.operation)), ("status", event.status.value)),
            )
        )
    updates.extend(_domain_metric_updates(event))
    if event.category == "observability" and event.operation == "snapshot":
        for attribute, metric_name in (
            ("queue_staged", "memory_jobs_staged"),
            ("queue_queued", "memory_jobs_queued"),
            ("queue_running", "memory_jobs_running"),
            ("queue_failed", "memory_jobs_failed"),
            ("queue_committed", "memory_jobs_committed_retained"),
            ("queue_oldest_age_seconds", "memory_job_oldest_age_seconds"),
            ("queue_high_watermark", "memory_job_sequence_high_watermark"),
            ("active_locks", "active_locks"),
            ("hanging_locks", "hanging_locks"),
            ("max_active_lock_age_seconds", "active_lock_max_age_seconds"),
        ):
            value = _non_negative_number(attributes.get(attribute))
            if value is not None:
                updates.append(MetricUpdate(metric_name, "gauge", value, labels))
    return tuple(updates)


# 行为管线的吞吐与损耗：(operation, attribute, metric)。属性是该步骤自己算出的计数，
# 布尔属性按 0/1 计。
_BEHAVIOR_COUNTERS: tuple[tuple[str, str, str], ...] = (
    ("physical_delivery", "segments", "behavior_physical_segments_received_total"),
    ("physical_delivery", "gaps", "behavior_physical_gaps_received_total"),
    ("physical_delivery", "units_recorded", "behavior_physical_units_recorded_total"),
    ("physical_delivery", "units_revision_rejected", "behavior_physical_revisions_rejected_total"),
    ("physical_delivery", "units_unusable_name", "behavior_physical_unusable_names_total"),
    ("reduction_publish", "occurrences", "behavior_occurrences_published_total"),
    ("reduction_publish", "gaps", "behavior_gaps_published_total"),
    ("kind_classify", "classified", "behavior_kind_classified_total"),
    ("kind_classify", "pending", "behavior_kind_pending_total"),
    ("kind_classify", "not_events", "behavior_kind_not_events_total"),
    ("kind_classify", "model_calls", "behavior_kind_model_calls_total"),
    ("reduction_vocabulary", "model_calls", "behavior_kind_model_calls_total"),
    ("reduction_vocabulary", "changed_days", "behavior_kind_changed_days_total"),
    ("reduction_publish", "kind_pending", "behavior_kind_pending_admitted_total"),
    ("semantic_refresh", "failed_days", "behavior_semantic_refresh_failures_total"),
)
# 行为管线的时效：小时级的等待放不进操作时长直方图的桶，按最近一次的值记 gauge。
_BEHAVIOR_GAUGES: tuple[tuple[str, str, str], ...] = (
    ("reduction_seal", "horizon_lag_seconds", "behavior_seal_horizon_lag_seconds"),
    ("reduction_seal", "oldest_unsealed_wait_seconds", "behavior_oldest_unsealed_chain_wait_seconds"),
    ("reduction_seal", "unsealed_chains", "behavior_unsealed_chains"),
    ("reduction_publish", "end_to_publish_seconds_max", "behavior_end_to_publish_seconds"),
    ("reduction_publish", "onset_to_publish_seconds_max", "behavior_onset_to_publish_seconds"),
)


#: 换代一步一跑可以是几个小时：它的耗时直方图另用一组到数小时的桶（通用直方图最高 30 秒，一整代都落进 +Inf）。
LONG_DURATION_BUCKETS = (1.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0)
_LONG_DURATION_METRICS = frozenset({"generation_step_duration_seconds", "generation_duration_seconds"})
#: 关系变化计数的标签（取值都是封闭枚举：lane、变化种类、状态、跨度段、带不带条件）。
_RELATION_CHANGE_LABELS = ("lane", "kind", "status_from", "status_to", "segment", "conditioned")
#: 预测拍按触发种类计的建了几条（种类是封闭枚举：日子、事件、情境）。
_FORESIGHT_SOURCE_PREFIX = "recorded_"


def _domain_metric_updates(event: ObservationEvent) -> tuple[MetricUpdate, ...]:
    if event.category == "behavior":
        return _behavior_metric_updates(event)
    if event.category in ("generation", "scene"):
        return _generation_metric_updates(event)
    if event.category == "foresight" and event.operation == "tick":
        return _tick_metric_updates(event)
    return ()


def _generation_metric_updates(event: ObservationEvent) -> tuple[MetricUpdate, ...]:
    """换代：每步耗时与结局、每代写成没有与多久、从昨天定稿到写出隔多久、各个退路的计数、按（步, 触点）的调模型次数、
    关系表的双向变化。标签只放低基数的。"""

    attributes = event.attributes
    operation = (event.category, event.operation)
    if operation == ("generation", "step"):
        labels: tuple[tuple[str, str], ...] = (("step", _label_value(attributes.get("step"))),)
        duration = _non_negative_number(attributes.get("duration_seconds")) or 0.0
        outcome = labels + (("outcome", _label_value(attributes.get("outcome"))),)
        return (
            MetricUpdate("generation_step_duration_seconds", "histogram", duration, labels),
            MetricUpdate("generation_step_outcomes_total", "counter", 1.0, outcome),
        )
    if operation == ("generation", "run"):
        outcome = (("outcome", _label_value(attributes.get("outcome"))),)
        updates = [
            MetricUpdate("generation_runs_total", "counter", 1.0, outcome),
            MetricUpdate(
                "generation_duration_seconds",
                "histogram",
                _non_negative_number(attributes.get("duration_seconds")) or 0.0,
                outcome,
            ),
        ]
        lag = _non_negative_number(attributes.get("lag_seconds"))
        if lag is not None:
            updates.append(MetricUpdate("generation_lag_seconds", "gauge", lag))
        gaps = _non_negative_number(attributes.get("unreduced_gaps"))
        if gaps is not None:
            updates.append(MetricUpdate("generation_unreduced_gaps", "gauge", gaps))
        return tuple(updates)
    if operation == ("generation", "fallback"):
        count = _non_negative_number(attributes.get("count")) or 0.0
        kind = (("kind", _label_value(attributes.get("kind"))),)
        return (
            MetricUpdate("generation_fallbacks_total", "counter", count, kind),
            MetricUpdate("generation_fallbacks_last", "gauge", count, kind),
        )
    if operation == ("generation", "model_call"):
        touch = (("step", _label_value(attributes.get("step"))), ("touch", _label_value(attributes.get("touch"))))
        return (MetricUpdate("generation_model_calls_total", "counter", 1.0, touch),)
    if operation == ("scene", "relation_change"):
        count = _non_negative_number(attributes.get("count")) or 0.0
        change = tuple((name, _label_value(attributes.get(name))) for name in _RELATION_CHANGE_LABELS)
        return (MetricUpdate("scene_relations_changed_total", "counter", count, change),) if count else ()
    return ()


def _tick_metric_updates(event: ObservationEvent) -> tuple[MetricUpdate, ...]:
    """预测拍：离槽边界晚了多少、一拍用时（通用的操作直方图）、建了几条（按种类，补建单列）、说了几条（合盖后补说单列）、
    没在看记了几段、共多少分钟。"""

    attributes = event.attributes
    updates: list[MetricUpdate] = []
    lateness = _non_negative_number(attributes.get("lateness_seconds"))
    if lateness is not None:
        updates.append(MetricUpdate("foresight_tick_lateness_seconds", "histogram", lateness))
    catch_up = "true" if attributes.get("catch_up") is True else "false"
    for key, value in attributes.items():
        if not key.startswith(_FORESIGHT_SOURCE_PREFIX):
            continue
        number = _non_negative_number(value)
        if number:
            labels = (("source", _label_value(key.removeprefix(_FORESIGHT_SOURCE_PREFIX))), ("catch_up", catch_up))
            updates.append(MetricUpdate("foresight_expectations_recorded_total", "counter", number, labels))
    for attribute, metric_name in (
        ("spoken", "foresight_reminders_spoken_total"),
        ("spoken_after_gap", "foresight_reminders_spoken_after_gap_total"),
        ("unwatched_spans", "foresight_unwatched_spans_total"),
        ("unwatched_minutes", "foresight_unwatched_minutes_total"),
    ):
        number = _non_negative_number(attributes.get(attribute))
        if number:
            updates.append(MetricUpdate(metric_name, "counter", number))
    return tuple(updates)


def _behavior_metric_updates(event: ObservationEvent) -> tuple[MetricUpdate, ...]:
    updates: list[MetricUpdate] = []
    attributes = event.attributes
    # 行为侧两条 lane 各是一整套、发的事件同名：带了 lane 的按 lane 分开记，否则一支的读数会被另一支盖掉。
    lane = attributes.get("lane")
    labels = (("lane", lane),) if isinstance(lane, str) and lane else ()
    for operation, attribute, metric_name in _BEHAVIOR_COUNTERS:
        if event.operation != operation or attribute not in attributes:
            continue
        raw = attributes[attribute]
        value = float(raw) if isinstance(raw, bool) else _non_negative_number(raw)
        if value:
            updates.append(MetricUpdate(metric_name, "counter", value, labels))
    for operation, attribute, metric_name in _BEHAVIOR_GAUGES:
        if event.operation != operation:
            continue
        value = _non_negative_number(attributes.get(attribute))
        if value is not None:
            updates.append(MetricUpdate(metric_name, "gauge", value))
    return tuple(updates)


class MetricRegistry:
    """线程安全的轻量指标注册表，输出 Prometheus 文本且不依赖 SDK。"""

    def __init__(
        self,
        *,
        max_recent_events: int = 256,
        namespace: str = "habitus",
        duration_buckets: tuple[float, ...] = _DEFAULT_DURATION_BUCKETS,
        enabled: bool = True,
    ) -> None:
        if not 1 <= max_recent_events <= 10_000:
            raise ValueError("max_recent_events must be between 1 and 10000")
        if not namespace or len(namespace) > 64:
            raise ValueError("namespace must be non-empty bounded text")
        normalized_buckets = tuple(float(value) for value in duration_buckets)
        if (
            not normalized_buckets
            or any(not math.isfinite(value) or value <= 0 for value in normalized_buckets)
            or tuple(sorted(set(normalized_buckets))) != normalized_buckets
        ):
            raise ValueError("duration_buckets must be unique positive ascending numbers")
        self.namespace = _metric_name(namespace)
        self.duration_buckets = normalized_buckets
        self.enabled = enabled
        self._lock = threading.RLock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[list[int], float, int]] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._recent: deque[ObservationEvent] = deque(maxlen=max_recent_events)

    def record(self, event: ObservationEvent) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._recent.append(event)
            for update in project_metric_updates(event):
                key = (update.name, update.labels)
                if update.kind == "counter":
                    self._counters[key] = self._counters.get(key, 0.0) + update.value
                elif update.kind == "gauge":
                    self._gauges[key] = update.value
                elif update.kind == "histogram":
                    buckets = self._buckets(update.name)
                    counts, total, count = self._histograms.get(
                        key,
                        ([0] * len(buckets), 0.0, 0),
                    )
                    for index, bucket in enumerate(buckets):
                        if update.value <= bucket:
                            counts[index] += 1
                    self._histograms[key] = (counts, total + update.value, count + 1)
                else:  # pragma: no cover - 投影函数只创建三种固定类型
                    raise ValueError("unsupported metric update kind")

    def _buckets(self, name: str) -> tuple[float, ...]:
        return LONG_DURATION_BUCKETS if name in _LONG_DURATION_METRICS else self.duration_buckets

    def set_gauge(
        self,
        name: str,
        value: int | float,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        if not name:
            raise ValueError("gauge name must be non-empty text")
        if not self.enabled:
            return
        if not math.isfinite(float(value)):
            raise TypeError("gauge value must be a finite number")
        resolved_labels = _mapping_labels(labels)
        with self._lock:
            self._gauges[(_metric_name(name), resolved_labels)] = float(value)

    def snapshot(self) -> ObservabilitySnapshot:
        with self._lock:
            counters = {
                _snapshot_key(name, labels): int(round(value)) for (name, labels), value in self._counters.items()
            }
            durations = {
                _snapshot_key(name, labels): total
                for (name, labels), (_counts, total, _count) in self._histograms.items()
            }
            gauges = {_snapshot_key(name, labels): value for (name, labels), value in self._gauges.items()}
            histograms = {
                _snapshot_key(name, labels): tuple(counts)
                for (name, labels), (counts, _total, _count) in self._histograms.items()
            }
            return ObservabilitySnapshot(
                counters=counters,
                duration_seconds=durations,
                gauges=gauges,
                recent_events=tuple(self._recent),
                histogram_counts=histograms,
            )

    def prometheus_text(self, *, namespace: str | None = None) -> str:
        """输出包含 HELP、TYPE 与直方图的 Prometheus exposition 文本。"""

        prefix = self.namespace if namespace is None else _metric_name(namespace)
        lines: list[str] = []
        with self._lock:
            counter_names = sorted({name for name, _labels in self._counters})
            for name in counter_names:
                full_name = f"{prefix}_{_metric_name(name)}"
                lines.extend(
                    (f"# HELP {full_name} Habitus bounded observation counter.", f"# TYPE {full_name} counter")
                )
                for (metric_name, labels), value in sorted(self._counters.items()):
                    if metric_name == name:
                        lines.append(f"{full_name}{_render_labels(labels)} {_format_number(value)}")

            histogram_names = sorted({name for name, _labels in self._histograms})
            for name in histogram_names:
                full_name = f"{prefix}_{_metric_name(name)}"
                lines.extend(
                    (f"# HELP {full_name} Habitus bounded operation duration.", f"# TYPE {full_name} histogram")
                )
                for (metric_name, labels), (counts, total, count) in sorted(self._histograms.items()):
                    if metric_name != name:
                        continue
                    for bucket, bucket_count in zip(self._buckets(metric_name), counts, strict=True):
                        bucket_labels = labels + (("le", _format_number(bucket)),)
                        lines.append(f"{full_name}_bucket{_render_labels(bucket_labels)} {bucket_count}")
                    lines.append(f"{full_name}_bucket{_render_labels(labels + (('le', '+Inf'),))} {count}")
                    lines.append(f"{full_name}_sum{_render_labels(labels)} {_format_number(total)}")
                    lines.append(f"{full_name}_count{_render_labels(labels)} {count}")

            gauge_names = sorted({name for name, _labels in self._gauges})
            for name in gauge_names:
                full_name = f"{prefix}_{_metric_name(name)}"
                lines.extend((f"# HELP {full_name} Habitus current bounded state.", f"# TYPE {full_name} gauge"))
                for (metric_name, labels), value in sorted(self._gauges.items()):
                    if metric_name == name:
                        lines.append(f"{full_name}{_render_labels(labels)} {_format_number(value)}")
        return "\n".join(lines) + ("\n" if lines else "")


def _selected_labels(
    attributes: Mapping[str, str | int | float | bool],
    *names: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, _label_value(attributes[name])) for name in names if name in _LABEL_ATTRIBUTES and name in attributes
    )


def _mapping_labels(labels: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if labels is None:
        return ()
    if len(labels) > 8:
        raise ValueError("metric labels must be a bounded mapping")
    result: list[tuple[str, str]] = []
    for name, value in labels.items():
        if name not in _LABEL_ATTRIBUTES:
            raise ValueError("metric label is not in the low-cardinality allowlist")
        result.append((_metric_name(name), _label_value(value)))
    return tuple(sorted(result))


def _non_negative_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _label_value(value: object) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        return "unknown"
    return normalized[:128]


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    rendered = ",".join(f'{_metric_name(name)}="{_escape_label(value)}"' for name, value in labels)
    return "{" + rendered + "}"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _snapshot_key(name: str, labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return name
    return f"{name}[{','.join(f'{key}={value}' for key, value in labels)}]"


def _format_number(value: float) -> str:
    return f"{value:.9g}"


def _metric_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_:]", "_", value)
    if not normalized or normalized[0].isdigit():
        normalized = f"metric_{normalized}"
    return normalized.lower()


__all__ = [
    "MetricRegistry",
    "MetricUpdate",
    "ObservabilitySnapshot",
    "project_metric_updates",
]
