"""习惯卡片（裁定 113）：``habitus/habits/<支>/<词的编号>.md``，每个在用过的词一张，只做记录，预测的任何计算都不读它。

每一代在语义树之后（``runtime.generation_steps.habits_step``）把这一支目录里的卡片和词表此刻的状态对一遍：

- 词在用：有这一代的数（``scene.habits``）就按数改写，没有（序列里还没有它的记录、或这一代语义树没交出材料）就保持旧数；
  没卡片又没有数的不建。
- 词已停用（还活着）：卡片留着，状态写成已停用和停用的日子，数保持最后一次的值；转回在用就改回在用、接着更新。
- 词不在词表里了（删了，含两次换代之间选词时挤掉的）：删卡片。自己长的叶子词到期是直接删，不经过停用，卡片跟着删。

文件名用词的编号（永不改、不重用），名字写在卡片里；初始词换版本改了名就改写一次。卡片里只有日期、整数与状态，不放
这一代的日子与写入时刻：字段没变就不写（版本不动），这一步被重跑时什么都不改。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from habitus.behavior.kinds.ids import ClassId, KindIdError
from habitus.behavior.kinds.model import Vocabulary
from habitus.infrastructure.tree.layout import HABITS_DIRECTORY
from habitus.runtime.record_files import RecordFileError, RecordFiles, StoredRecord
from habitus.scene.habits import HabitNumbers

HABIT_TYPE = "habit"
ACTIVE, EXITED = "在用", "已停用"
_LANE_NAMES = {"session": "会话", "physical": "物理"}
_NUMBERS = (
    "first_day",
    "last_day",
    "days",
    "below_only_days",
    "intervals",
    "usable_intervals",
    "censored_intervals",
    "median_days",
)
_KEYS = frozenset({"id", "name", "lane", "parent", "status", "exited_on", *_NUMBERS})


@dataclass(frozen=True)
class HabitSync:
    """对一遍的结果：改写（含新建）几张、删几张、在用却还没有数所以没建的几个。"""

    written: int
    deleted: int
    waiting: int


def render_habit(fields: Mapping[str, Any]) -> str:
    """卡片正文：给人看的几行，全部由字段拼出。"""

    status = fields["status"] if fields["exited_on"] is None else f"{fields['status']}（{fields['exited_on']} 起）"
    median = "还没有" if fields["median_days"] is None else f"{fields['median_days']} 天"
    lines = [
        f"# {fields['name']}",
        "",
        f"- 编号：{fields['id']}（{_LANE_NAMES.get(fields['lane'], fields['lane'])}）",
        f"- 上级：{fields['parent'] or '无'}",
        f"- 状态：{status}",
        f"- 第一次：{fields['first_day']}；最近一次：{fields['last_day']}",
        f"- 一共做过 {fields['days']} 天，其中 {fields['below_only_days']} 天只有下级的记录",
        f"- 相邻间隔 {fields['intervals']} 段，能用的 {fields['usable_intervals']} 段、截尾的 {fields['censored_intervals']} 段；"
        f"常见间隔 {median}",
    ]
    return "\n".join(lines) + "\n"


class HabitCards:
    """一支的习惯卡片目录。``tree_root`` 是整棵 habitus 树的根。"""

    def __init__(self, tree_root: str | Path, *, lane: str, max_children: int, max_bytes: int) -> None:
        self.files = RecordFiles(
            tree_root,
            kind=HABIT_TYPE,
            keys=_KEYS,
            render=render_habit,
            max_children=max_children,
            max_bytes=max_bytes,
        )
        self.lane = lane

    @property
    def directory(self) -> Path:
        return self.files.root / HABITS_DIRECTORY / self.lane

    def read(self, word_id: str) -> StoredRecord | None:
        return self.files.read(self.files.path(HABITS_DIRECTORY, self.lane, f"{word_id}.md"))

    def sync(self, vocabulary: Vocabulary, numbers: Mapping[str, HabitNumbers] | None, *, now: datetime) -> HabitSync:
        """按词表此刻的状态对一遍（见模块说明）。``numbers`` 是这一代算出的数；这一代没有（序列里没有记录、语义树没交出材料）
        是 None：只对停用与删除，不清任何数。"""

        if vocabulary.lane.value != self.lane:
            raise ValueError("the habit cards and the vocabulary belong to different lanes")
        written = deleted = waiting = 0
        alive = {str(word.id) for word in vocabulary.words}
        if self.files.files.directory_exists(self.directory):
            for name in self.files.files.markdown_names(self.directory):
                try:
                    ClassId.parse(name)
                except KindIdError as exc:
                    raise RecordFileError(f"habit card {name}.md is not named after a word of this lane") from exc
                if name not in alive:
                    deleted += self.files.files.unlink(self.files.path(HABITS_DIRECTORY, self.lane, f"{name}.md"))
        for word in vocabulary.words:
            active = word.id in vocabulary.active
            if not active and word.id not in vocabulary.exited:
                continue  # 候选：从没在用过，或卡片的状态由上一次在用 / 停用定了
            path = self.files.path(HABITS_DIRECTORY, self.lane, f"{word.id}.md")
            old = self.files.read(path)
            found = None if numbers is None else numbers.get(str(word.id))
            if found is not None and active:
                counts = _numbers(found)
            elif old is not None:
                counts = {key: old.fields[key] for key in _NUMBERS}  # 停用了、或这一代没有它的数：保持最后一次的值
            else:
                waiting += active
                continue
            fields: dict[str, Any] = {
                "id": str(word.id),
                "name": word.name,
                "lane": self.lane,
                "parent": None if word.parent is None else str(word.parent),
                "status": ACTIVE if active else EXITED,
                "exited_on": None if active else vocabulary.exited[word.id].isoformat(),
            }
            fields.update(counts)
            written += self.files.write(path, fields, old=old, now=now)
        return HabitSync(written, deleted, waiting)


def _numbers(found: HabitNumbers) -> dict[str, Any]:
    return {
        "first_day": found.first_day.isoformat(),
        "last_day": found.last_day.isoformat(),
        "days": found.days,
        "below_only_days": found.below_only_days,
        "intervals": found.intervals,
        "usable_intervals": found.usable_intervals,
        "censored_intervals": found.censored_intervals,
        "median_days": found.median_days,
    }


__all__ = ["ACTIVE", "EXITED", "HABIT_TYPE", "HabitCards", "HabitSync", "render_habit"]
