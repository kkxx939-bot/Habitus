"""scene 读侧测试共用的现场：一棵行为树 + 发布行为的助手。

按天情景树已经删掉（2026-09-14），所以这里不再有情景日。读侧现在要的上下文全部来自行为树；
规律级那一半（按候选取记录）是读侧改写要接的，还没有接上。
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
    **overrides: Any,
) -> str:
    """往行为树发布一条 occurrence（可带前向 links），返回它的 URI。"""

    started = at(day, hour, minute)
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: started + timedelta(hours=3))
    payload = occurrence_payload(
        occurred_on=day,
        name=name,
        kind_token=kind or name,
        started_at=started,
        last_observed_at=started + timedelta(minutes=10),
        onset_available_at=started + timedelta(seconds=2),
        basis=(),
        goal=None,
        **overrides,
    )
    document = writer.publish(BehaviorKind.OCCURRENCE, payload, links=links)
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


__all__ = ["CST", "DAY1", "DAY2", "DAY3", "SUBJECT", "Site", "at", "publish"]
