"""scene 读侧测试共用的现场：一棵行为树 + 发布行为的助手。

旧语义树（规律级、关联记录）随 2026-09-26 的重构整块删掉，这里只剩行为树那一半；读侧的上下文全部
来自行为树。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from tests.unit.behavior.tree_payloads import gap_payload, occurrence_payload

CST = timezone(timedelta(hours=8))
DAY1 = date(2026, 8, 15)
DAY2 = date(2026, 8, 16)
DAY3 = date(2026, 8, 17)
SUBJECT = "家庭成员A"


def at(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CST)


def publish(
    tree: BehaviorTree,
    day: date,
    name: str,
    hour: int,
    minute: int,
    *,
    kind: str | None = None,
    links: tuple[tuple[str, str], ...] = (),
    lasts_minutes: int = 10,
    **overrides: Any,
) -> str:
    """往行为树发布一条 occurrence（可带前向 links），返回它的 URI。``lasts_minutes`` 定最后所见。"""

    started = at(day, hour, minute)
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: started + timedelta(hours=3))
    overrides = dict(overrides)
    payload = occurrence_payload(
        occurred_on=day,
        name=name,
        kind_token=kind or name,
        started_at=started,
        last_observed_at=started + timedelta(minutes=lasts_minutes),
        onset_available_at=started + timedelta(seconds=2),
        basis=(),
        goal=overrides.pop("goal", None),
        **overrides,
    )
    document = writer.publish(BehaviorKind.OCCURRENCE, payload, links=links)
    return str(BehaviorURI.from_address(document.address))


def publish_gap(
    tree: BehaviorTree,
    day: date,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    kind: str = "未观测",
    end_day: date | None = None,
) -> str:
    """往行为树发布一段观测空白（``未观测`` / ``没读懂``），返回它的 URI。映射器的时间线要列出这些段。

    ``end_day`` 给了就是跨午夜的空白（02:00 睡到 09:00 那一类）：它挂在**开始**那天的目录下，
    所以读次日的覆盖时必须往前多读一天。
    """

    ended = at(end_day or day, end[0], end[1])
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: ended + timedelta(hours=1))
    document = writer.publish(
        BehaviorKind.GAP,
        gap_payload(occurred_on=day, gap_kind=kind, started_at=at(day, start[0], start[1]), ended_at=ended),
    )
    return str(BehaviorURI.from_address(document.address))


class Site:
    """一棵真实的行为树 + 一个可拨的时钟。"""

    def __init__(self, tmp_path: Path, *, now: datetime) -> None:
        self.behavior_tree = BehaviorTree(tmp_path / "behavior" / "tree")
        self.now = now
        self.lock_store = ProcessLocalLockStore()

    def seed(self) -> dict[str, str]:
        uris = {
            "buy": publish(self.behavior_tree, DAY1, "去超市买菜", 15, 20, kind="买菜"),
            "phone1": publish(self.behavior_tree, DAY1, "看手机", 21, 0),
            "discuss": publish(self.behavior_tree, DAY2, "商量晚餐", 19, 12, kind="交谈"),
            "wash": publish(self.behavior_tree, DAY2, "洗菜", 19, 43),
            "phone2": publish(self.behavior_tree, DAY2, "看手机", 20, 15),
        }
        writer = BehaviorDocumentWriter(self.behavior_tree, ProcessLocalLockStore(), clock=lambda: self.now)
        writer.publish(
            BehaviorKind.GAP,
            gap_payload(occurred_on=DAY2, started_at=at(DAY2, 20, 30), ended_at=at(DAY2, 20, 50)),
        )
        return uris


__all__ = ["CST", "DAY1", "DAY2", "DAY3", "SUBJECT", "Site", "at", "publish", "publish_gap"]
