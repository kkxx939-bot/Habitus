"""把 habitus 树里两支的每一轮记录交给记忆检索：记忆检索只认纯值接口，读行为树的实现放在 runtime 这座桥里。

只读 ``occurrence``（他做的事）；「未观测」「没读懂」是 ``gap``，不算记录。「续话」「非事件」「表外」也是
occurrence，照样读出。读盘在调用方的线程里做；某一天读失败就记下这一天、跳过，不让整次查询失败。
记录的地址就是树上的 ``habitus://occurrences/<支>/…``（支在地址里，裁定 112）。
"""

from __future__ import annotations

from datetime import date, datetime

from habitus.behavior.document.model import BehaviorDocument
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.memory.retrieval import BehaviorRecord, BehaviorRecordBatch


class BehaviorTreeRecords:
    """两支（会话、物理）的视图合起来读。"""

    def __init__(self, trees: tuple[BehaviorTree, ...]) -> None:
        self.trees = trees

    def days(self) -> tuple[date, ...]:
        return tuple(sorted({day for tree in self.trees for day in tree.days(BehaviorKind.OCCURRENCE)}))

    def read_days(self, days: tuple[date, ...]) -> BehaviorRecordBatch:
        records: list[BehaviorRecord] = []
        failed: set[date] = set()
        for tree in self.trees:
            available = set(tree.days(BehaviorKind.OCCURRENCE))
            for day in days:
                if day not in available:
                    continue
                try:
                    documents = tree.read_day(BehaviorKind.OCCURRENCE, day)
                    day_records = [_record(tree, document) for document in documents]  # 整天读成了才收，读一半不算
                except Exception:
                    failed.add(day)
                    continue
                records.extend(day_records)
        return BehaviorRecordBatch(tuple(records), tuple(sorted(failed)))


def _record(tree: BehaviorTree, document: BehaviorDocument) -> BehaviorRecord:
    """文档字段是落盘的 JSON 快照：日期与时刻是 ISO 文字。"""

    fields = document.fields
    return BehaviorRecord(
        address=tree.uri_for(document.address).uri,
        lane=tree.lane,
        day=date.fromisoformat(str(fields["occurred_on"])),
        started_at=datetime.fromisoformat(str(fields["started_at"])),
        name=fields["name"],
        goal=fields.get("goal"),
        summary=fields["summary"],
        steps=tuple(step["semantics"] for step in fields["basis"]),
        original_name=fields.get("original_name"),
    )


__all__ = ["BehaviorTreeRecords"]
