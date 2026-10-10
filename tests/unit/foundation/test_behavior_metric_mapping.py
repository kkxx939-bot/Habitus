"""行为管线的指标映射不许挂在已不存在的操作名上（词表重写时 ``kind_resolve`` 被删，指标静默归零过一次）。"""

from __future__ import annotations

from pathlib import Path

from habitus.foundation.observability import (
    _BEHAVIOR_COUNTERS,
    _BEHAVIOR_GAUGES,
    ObservationEvent,
    ObservationStatus,
    _behavior_metric_updates,
)

SRC = Path(__file__).resolve().parents[3] / "habitus"


def test_every_mapped_operation_is_emitted_somewhere_in_production_code() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in SRC.rglob("*.py") if path.name != "observability.py")
    missing = sorted({operation for operation, _, _ in (*_BEHAVIOR_COUNTERS, *_BEHAVIOR_GAUGES) if f'"{operation}"' not in source})
    assert missing == []


def test_vocabulary_classification_feeds_its_counters() -> None:
    event = ObservationEvent(
        category="behavior",
        operation="kind_classify",
        status=ObservationStatus.SUCCESS,
        duration_seconds=1.0,
        attributes={"classified": 7, "pending": 2, "not_events": 1, "model_calls": 3},
    )
    names = {update.name: update.value for update in _behavior_metric_updates(event)}
    assert names == {
        "behavior_kind_classified_total": 7.0,
        "behavior_kind_pending_total": 2.0,
        "behavior_kind_not_events_total": 1.0,
        "behavior_kind_model_calls_total": 3.0,
    }
