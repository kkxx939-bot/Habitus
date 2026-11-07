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


def require_aware_utc(value: datetime, message: str) -> datetime:
    """带时区的 ``datetime`` → UTC；没带时区抛 ``ValueError(message)``，报错文字由调用方给。"""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(message)
    return value.astimezone(UTC)


def format_utc_z(value: datetime, *, microseconds: bool = False) -> str:
    """转成 UTC 后写 ISO-8601，``+00:00`` 记作 ``Z``；``microseconds`` 为真时固定写满六位微秒。"""

    timespec = "microseconds" if microseconds else "auto"
    return value.astimezone(UTC).isoformat(timespec=timespec).replace("+00:00", "Z")


def parse_utc_z(value: object, *, invalid: str, naive: str, empty_is_invalid: bool = True) -> datetime:
    """落盘读回的 ISO-8601 文字 → UTC。

    不是文字（``empty_is_invalid`` 为真时空串也算）抛 ``ValueError(invalid)``；``fromisoformat`` 解析不了的原样抛
    它自己的 ``ValueError``；没带时区抛 ``ValueError(naive)``。
    """

    if not isinstance(value, str) or (empty_is_invalid and not value):
        raise ValueError(invalid)
    return require_aware_utc(datetime.fromisoformat(value.replace("Z", "+00:00")), naive)


__all__ = [
    "TimestampError",
    "TimestampFormatError",
    "TimestampZoneError",
    "format_utc_z",
    "parse_utc_z",
    "require_aware_utc",
    "utc_timestamp",
]
