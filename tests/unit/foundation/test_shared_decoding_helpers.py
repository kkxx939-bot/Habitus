"""S1 并到 foundation 的公共小工具：报错文字由调用方给、UTC 文字格式、读回字段解码与旁路观测记录。"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from habitus.foundation.decoding import int_field, optional_text_field, text_field
from habitus.foundation.observability import ObservationEvent, ObservationStatus, record_observation
from habitus.foundation.timestamps import format_utc_z, parse_utc_z, require_aware_utc

EAST_8 = timezone(timedelta(hours=8))


def test_require_aware_utc_converts_offsets_and_raises_the_callers_text_for_naive_values() -> None:
    assert require_aware_utc(datetime(2026, 7, 1, 8, tzinfo=EAST_8), "unused") == datetime(2026, 7, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match=r"^journal timestamp must be timezone-aware$"):
        require_aware_utc(datetime(2026, 7, 1), "journal timestamp must be timezone-aware")


def test_format_utc_z_writes_utc_with_z_and_fixed_microseconds_only_when_asked() -> None:
    value = datetime(2026, 7, 1, 8, tzinfo=EAST_8)
    assert format_utc_z(value) == "2026-07-01T00:00:00Z"
    assert format_utc_z(value, microseconds=True) == "2026-07-01T00:00:00.000000Z"
    assert format_utc_z(value.replace(microsecond=5)) == "2026-07-01T00:00:00.000005Z"


def test_parse_utc_z_keeps_each_failure_text_and_the_empty_string_choice() -> None:
    assert parse_utc_z("2026-07-01T08:00:00+08:00", invalid="bad", naive="naive") == datetime(2026, 7, 1, tzinfo=UTC)
    assert parse_utc_z("2026-07-01T00:00:00Z", invalid="bad", naive="naive") == datetime(2026, 7, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match=r"^bad$"):
        parse_utc_z(1, invalid="bad", naive="naive")
    with pytest.raises(ValueError, match=r"^bad$"):
        parse_utc_z("", invalid="bad", naive="naive")
    with pytest.raises(ValueError, match=r"^naive$"):
        parse_utc_z("2026-07-01T00:00:00", invalid="bad", naive="naive")
    # 只查类型的调用方：空串交给 fromisoformat，透出它自己的报错
    with pytest.raises(ValueError) as caught:
        parse_utc_z("", invalid="bad", naive="naive", empty_is_invalid=False)
    assert str(caught.value) != "bad"
    with pytest.raises(ValueError) as unparsable:
        parse_utc_z("time", invalid="bad", naive="naive")
    assert str(unparsable.value) not in {"bad", "naive"}


def test_field_decoders_reject_the_wrong_type_with_the_callers_text() -> None:
    assert int_field(3, "n") == 3
    for bad in (True, "3", 3.0, None):
        with pytest.raises(ValueError, match=r"^n must be an integer$"):
            int_field(bad, "n must be an integer")
    assert text_field("", "t") == ""
    with pytest.raises(ValueError, match=r"^t$"):
        text_field(None, "t")
    assert optional_text_field(None, "o") is None
    assert optional_text_field("x", "o") == "x"
    with pytest.raises(ValueError, match=r"^o$"):
        optional_text_field(1, "o")


class _Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


class _Broken:
    def record(self, event: ObservationEvent) -> None:
        raise RuntimeError("observer backend is down")


def test_record_observation_records_point_events_and_swallows_observer_failures() -> None:
    recorder = _Recorder()
    record_observation(recorder, "lock", "acquire", ObservationStatus.DEGRADED, attributes={"k": 1})
    event = recorder.events[0]
    assert (event.category, event.operation, event.status) == ("lock", "acquire", ObservationStatus.DEGRADED)
    assert event.duration_seconds == 0.0
    assert dict(event.attributes) == {"k": 1}

    record_observation(recorder, "lock", "acquire", ObservationStatus.SUCCESS, started=0.0)
    assert recorder.events[1].duration_seconds > 0.0

    # 后端失败与事件构造失败（属性越界）都只丢这条观察，不向业务抛
    record_observation(_Broken(), "lock", "acquire", ObservationStatus.FAILURE)
    record_observation(recorder, "lock", "acquire", ObservationStatus.SUCCESS, attributes={"k": "x" * 300})
    assert len(recorder.events) == 2
