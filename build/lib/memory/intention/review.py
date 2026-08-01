"""只产生提醒、不改变 Intention 业务状态的时间复核模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from memory.document import MemoryDocument
from memory.model import MemoryKind


class MemoryIntentionReviewLevel(str, Enum):
    """距离最近明确确认时间形成的逐级提醒。"""

    CURRENT = "current"
    FIRST_REVIEW = "first_review"
    SECOND_REVIEW = "second_review"
    STRONG_REVIEW = "strong_review"


@dataclass(frozen=True)
class MemoryIntentionReviewConfig:
    """Intention 三级复核提醒阈值；任何阈值都不表示自动失效。"""

    first_review_after_days: int = 30
    second_review_after_days: int = 60
    strong_review_after_days: int = 180

    def __post_init__(self) -> None:
        values = (
            self.first_review_after_days,
            self.second_review_after_days,
            self.strong_review_after_days,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("intention review day thresholds must be integers")
        if not 0 < values[0] < values[1] < values[2] <= 36_500:
            raise ValueError("intention review day thresholds must be positive and strictly increasing")


@dataclass(frozen=True)
class MemoryIntentionReview:
    """一个活动 Intention 在指定时刻的只读复核提示。"""

    level: MemoryIntentionReviewLevel
    last_confirmed_at: datetime
    unconfirmed_days: int

    def __post_init__(self) -> None:
        level = MemoryIntentionReviewLevel(self.level)
        object.__setattr__(self, "level", level)
        timestamp = _utc_timestamp(self.last_confirmed_at, "last_confirmed_at")
        object.__setattr__(self, "last_confirmed_at", timestamp)
        if (
            isinstance(self.unconfirmed_days, bool)
            or not isinstance(self.unconfirmed_days, int)
            or self.unconfirmed_days < 0
        ):
            raise ValueError("unconfirmed_days must be a non-negative integer")

    @property
    def requires_review(self) -> bool:
        return self.level is not MemoryIntentionReviewLevel.CURRENT


class MemoryIntentionReviewer:
    """按系统时间确定性计算提醒，绝不修改、隐藏或删除 Intention。"""

    def __init__(self, config: MemoryIntentionReviewConfig | None = None) -> None:
        if config is not None and not isinstance(config, MemoryIntentionReviewConfig):
            raise TypeError("config must be MemoryIntentionReviewConfig")
        self.config = config or MemoryIntentionReviewConfig()

    def review(
        self,
        document: MemoryDocument,
        *,
        now: datetime,
    ) -> MemoryIntentionReview | None:
        """完成事项不需要时间复核；其余事项始终保持可召回。"""

        if not isinstance(document, MemoryDocument):
            raise TypeError("document must be MemoryDocument")
        if document.kind is not MemoryKind.INTENTION:
            raise ValueError("intention review requires an Intention document")
        if document.fields.get("status") == "completed":
            return None
        current = _utc_timestamp(now, "now")
        confirmed = document.metadata.last_confirmed_at
        if confirmed is None:
            raise ValueError("active Intention is missing last_confirmed_at")
        if current < confirmed:
            raise ValueError("intention review time precedes last_confirmed_at")
        elapsed = current - confirmed
        level = self._level(elapsed)
        return MemoryIntentionReview(
            level=level,
            last_confirmed_at=confirmed,
            unconfirmed_days=elapsed.days,
        )

    def _level(self, elapsed: timedelta) -> MemoryIntentionReviewLevel:
        if elapsed >= timedelta(days=self.config.strong_review_after_days):
            return MemoryIntentionReviewLevel.STRONG_REVIEW
        if elapsed >= timedelta(days=self.config.second_review_after_days):
            return MemoryIntentionReviewLevel.SECOND_REVIEW
        if elapsed >= timedelta(days=self.config.first_review_after_days):
            return MemoryIntentionReviewLevel.FIRST_REVIEW
        return MemoryIntentionReviewLevel.CURRENT


def _utc_timestamp(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{label} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return value.astimezone(timezone.utc)


__all__ = [
    "MemoryIntentionReview",
    "MemoryIntentionReviewConfig",
    "MemoryIntentionReviewer",
    "MemoryIntentionReviewLevel",
]
