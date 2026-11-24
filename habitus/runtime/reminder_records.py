"""提醒记录（裁定 113）：``habitus/reminders/<支>/YYYY/MM/DD/<预期的编号>.md``，系统每决定要说一次提醒（账里记下「待取」）
就有一条，之后补上宿主取没取、说了什么、他怎么回应、结算成什么。只做记录，预测的任何计算都不读它。

按预期所在的那一天分目录（和预测的账一致：账的每个写入口都带着这一天，路径直接拼出来）。

**怎么更新**：账每写一条（``runtime.foresight_reminder_records``），就从账把"来自账的那几段"整份重算、整份替换；"宿主取没取"
那一段账里没有（只在出口的传输状态里，过了最晚时限就清掉），由出口写取走记录时把整段值传进来，从账重算时保留记录里已有的这一段。
账按天删掉以后重算是空的：重算得到"没有这条"时不写，记录留着。没有"扫一遍全部重建"的入口。

**锁**：一条记录一把进程内的锁，是最里面的一把：读旧记录、读账、整份替换都在锁里（两个线程——预测 worker 与会话消费者——
同时更新同一条时，后读的一方一定看得到先写的那一份，不会拿旧内容盖掉新的）。持着它不拿别的锁，尤其不读出口：出口是持着
自己的锁来写账的，反过来拿会和它先后相反。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from habitus.infrastructure.tree.layout import REMINDERS_DIRECTORY
from habitus.runtime.record_files import RecordFiles, StoredRecord

REMINDER_TYPE = "reminder"
#: 来自账的那几段（每次从账整份重算）。
LEDGER_KEYS = frozenset(
    {
        "expectation_id",
        "lane",
        "kind_token",
        "kind_name",
        "day",
        "source",
        "window_start",
        "window_end",
        "deadline",
        "decided_at",
        "text",
        "spoken",
        "unspoken",
        "responses",
        "settlement",
        "rulings",
    }
)
#: 宿主取没取（出口传进来，重算时保留）。
CLAIMS = "claims"
_LANE_NAMES = {"session": "会话", "physical": "物理"}
_ENDED = {
    None: "还在手里",
    "expired": "租约过期",
    "released": "没问出口，放回",
    "spoken": "说出口了",
    "removed": "出口清掉了（过了时限、时限后才问或问之前已经做了，见后面的结局）",
}


def render_reminder(fields: Mapping[str, Any]) -> str:
    """记录正文：给人看的几段，全部由字段拼出。"""

    lines = [
        f"# 提醒：{fields['kind_name']}",
        "",
        f"- 预期：{fields['expectation_id']}（{_LANE_NAMES.get(fields['lane'], fields['lane'])}，{fields['day']}，"
        f"{fields['source']}触发）",
        f"- 词：{fields['kind_name']}（{fields['kind_token']}）",
        f"- 时窗：{fields['window_start']} 至 {fields['window_end']}；最晚：{fields['deadline']}",
        f"- 决定要说：{fields['decided_at']}",
        f"- 写好的话：{fields['text']}",
        "",
        "## 宿主取没取",
        "",
    ]
    claims = fields[CLAIMS]
    if claims is None:
        lines.append("- 没有取走的记录")
    else:
        lines.append(f"- 放回 {claims['attempts']} 次")
        lines.extend(
            f"- {item['claimed_at']} {item['host']} 取走（{_ENDED.get(item['ended'], item['ended'])}）"
            for item in claims["handoffs"]
        )
    lines.extend(["", "## 说了没有", ""])
    spoken, unspoken = fields["spoken"], fields["unspoken"]
    if spoken is not None:
        lines.append(f"- {spoken['at']} 经 {spoken['via']} 说出口（轮 {spoken['turn']}）：{spoken['text']}")
    if unspoken is not None:
        lines.append(f"- 没说成：{unspoken['reason']}（{unspoken['at']}）")
    if spoken is None and unspoken is None:
        lines.append("- 还没说")
    lines.extend(["", "## 他怎么回应", ""])
    lines.extend(f"- {item['at']} {item['polarity']}（轮 {item['turn']}）" for item in fields["responses"])
    if not fields["responses"]:
        lines.append("- 还没有回应")
    lines.extend(["", "## 结算", ""])
    settlement = fields["settlement"]
    if settlement is None:
        lines.append("- 还没结算")
    else:
        offset = "" if settlement["offset_minutes"] is None else f"，偏差 {settlement['offset_minutes']} 分钟"
        found = "" if settlement["occurrence_uri"] is None else f"，对上 {settlement['occurrence_uri']}"
        lines.append(f"- {settlement['outcome']}（{settlement['at']}）{offset}{found}")
    lines.extend(
        f"- {item['ruling']}（{item['at']}），晚了 {item['delay_minutes']} 分钟，对上 {item['occurrence_uri']}"
        for item in fields["rulings"]
    )
    return "\n".join(lines) + "\n"


class ReminderRecords:
    """一支的提醒记录。``tree_root`` 是整棵 habitus 树的根；``clock`` 给版本的更新时间。"""

    def __init__(
        self,
        tree_root: str | Path,
        *,
        lane: str,
        max_children: int,
        max_bytes: int,
        clock: Callable[[], datetime],
    ) -> None:
        self.files = RecordFiles(
            tree_root,
            kind=REMINDER_TYPE,
            keys=LEDGER_KEYS | {CLAIMS},
            render=render_reminder,
            max_children=max_children,
            max_bytes=max_bytes,
        )
        self.lane = lane
        self.clock = clock
        #: 写记录失败的次数（账照常写成；失败报出到日志、计数在这里）。
        self.failures = 0
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def path(self, day: date, expectation_id: str) -> Path:
        return self.files.path(
            REMINDERS_DIRECTORY,
            self.lane,
            f"{day.year:04d}",
            f"{day.month:02d}",
            f"{day.day:02d}",
            f"{expectation_id}.md",
        )

    def exists(self, day: date, expectation_id: str) -> bool:
        """这条预期有没有记录（只看文件在不在，不读账）。"""

        return self.files.exists(self.path(day, expectation_id))

    def read(self, day: date, expectation_id: str) -> StoredRecord | None:
        return self.files.read(self.path(day, expectation_id))

    def update(
        self,
        day: date,
        expectation_id: str,
        from_ledger: Callable[[], Mapping[str, Any] | None],
        *,
        claims: Mapping[str, Any] | None = None,
    ) -> bool:
        """在这条记录的锁里：读旧记录、``from_ledger()`` 从账重算来自账的那几段（没有「待取」或账已删是 None：不写），
        取走那一段用传进来的 ``claims``（没传就保留旧的），整份替换；字段没变不写。返回写没写。"""

        with self._lock(expectation_id):
            path = self.path(day, expectation_id)
            old = self.files.read(path)
            computed = from_ledger()
            if computed is None:
                return False
            fields = {**computed, CLAIMS: dict(claims) if claims is not None else (old.fields[CLAIMS] if old else None)}
            return self.files.write(path, fields, old=old, now=self.clock())

    def _lock(self, expectation_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(expectation_id, threading.Lock())


__all__ = ["CLAIMS", "LEDGER_KEYS", "REMINDER_TYPE", "ReminderRecords", "render_reminder"]
