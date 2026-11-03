"""带时区时刻的统一解析：``datetime`` 或 ISO-8601 文字（``Z`` 记作 UTC）规范成 UTC。

各层的时刻校验都走这一份，只各自把错误翻成本层的错误类型。
"""

from __future__ import annotations

from datetime import UTC, datetime


class TimestampError(ValueError):
    """时刻不能规范成带时区的 UTC。"""


class TimestampFormatError(TimestampError):
    """文字不是 ISO-8601。"""


class TimestampZoneError(TimestampError):
    """时刻没带时区。"""


def utc_timestamp(value: datetime | str) -> datetime:
    """``datetime`` 或 ISO-8601 文字 → UTC；文字解析不了抛 ``TimestampFormatError``，没带时区抛 ``TimestampZoneError``。"""

    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise TimestampFormatError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TimestampZoneError("timestamp must include timezone")
    return parsed.astimezone(UTC)


__all__ = ["TimestampError", "TimestampFormatError", "TimestampZoneError", "utc_timestamp"]
