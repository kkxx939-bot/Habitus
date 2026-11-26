"""关系卡片（裁定 115）：``habitus/relations/<支>/<前因>/<后果>--<段>[--<条件摘要>].md``，关系表里跨代带着的每一行与退休的
每一行一张，只做记录，预测、语义树、归约的任何计算都不读它。树根下的这个 ``relations`` 和语义树工作目录里存关系表的
``relations`` 同名不同根。

每一代在习惯卡片之后（``runtime.generation_steps.relation_cards_step``）拿盘上最晚那张关系表把这一支的卡片对一遍：表里的行
改写（字段没变不写），表里没有的按 ``scene.relations.cards.gone_fields`` 留着标结束或删掉。字段与文件名的规则在
``scene.relations.cards``；这里只拼地址、读写文件。卡片上的指向是普通字符串（后果那个词的关系目录——顺着走就是链——与前因、
后果的习惯卡片），不进记忆卡片的 ``links`` / ``backlinks``；指向的东西不存在不算错。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from habitus.behavior.kinds.ids import ClassId, KindIdError
from habitus.infrastructure.tree.address import format_uri
from habitus.infrastructure.tree.layout import HABITS_DIRECTORY, RELATION_CARDS_DIRECTORY
from habitus.runtime.record_files import RecordFileError, RecordFiles, StoredRecord
from habitus.scene.relations.cards import CARD_KEYS, card_location, gone_fields, render_card, row_fields
from habitus.scene.relations.engine import RelationKey
from habitus.scene.relations.spans import Segment
from habitus.scene.relations.state import LaneState, Status

RELATION_TYPE = "relation"


@dataclass(frozen=True)
class RelationSync:
    """对一遍的结果：改写（含新建）几张、删几张、留着标了结束的几张（含以前就标过的）。"""

    written: int
    deleted: int
    ended: int


class RelationCards:
    """一支的关系卡片目录。``tree_root`` 是整棵 habitus 树的根。"""

    def __init__(self, tree_root: str | Path, *, lane: str, max_children: int, max_bytes: int) -> None:
        self.files = RecordFiles(
            tree_root,
            kind=RELATION_TYPE,
            keys=CARD_KEYS,
            render=render_card,
            max_children=max_children,
            max_bytes=max_bytes,
        )
        self.lane = lane

    @property
    def directory(self) -> Path:
        return self.files.root / RELATION_CARDS_DIRECTORY / self.lane

    def read(self, antecedent: str, name: str) -> StoredRecord | None:
        return self.files.read(self.files.path(RELATION_CARDS_DIRECTORY, self.lane, antecedent, f"{name}.md"))

    def sync(self, table: LaneState, names: Mapping[str, str], *, now: datetime) -> RelationSync:
        """拿第 ``table.night`` 代的关系表把这一支的卡片对一遍。``names`` 是词的编号 → 名字（词表此刻的样子）。"""

        if table.lane != self.lane:
            raise ValueError("the relation cards and the relation table belong to different lanes")
        stored = self._stored()
        written = deleted = ended = 0
        seen: set[tuple[str, str]] = set()
        for row in table.relations:
            if not (row.status.carried or row.status is Status.RETIRED):
                continue  # "检验过没过"：每代重算、不跨代，不写卡片
            location = card_location(row.key)
            if location in seen:
                # 条件摘要撞了（截断的 sha256）：两条关系会写进同一个文件，明确失败，不让后一条盖掉前一条
                raise RecordFileError(f"two relations map to the card {location[0]}/{location[1]}.md")
            seen.add(location)
            old = stored.get(location)
            fields = row_fields(row, None if old is None else old.fields, names=names, pointers=self._pointers(row.key))
            written += self.files.write(self._path(*location), fields, old=old, now=now)
        for location, old in sorted(stored.items()):
            if location in seen:
                continue
            kept = gone_fields(old.fields, table.night)
            if kept is None:
                deleted += self.files.files.unlink(self._path(*location))
            else:
                ended += 1
                written += self.files.write(self._path(*location), kept, old=old, now=now)
        for directory in self.files.files.directories(self.directory) if self._exists() else ():
            if not self.files.files.content_children(directory):
                self.files.files.rmdir(directory)  # 前因的卡片都删了：目录跟着删
        return RelationSync(written, deleted, ended)

    def _stored(self) -> dict[tuple[str, str], StoredRecord]:
        """盘上全部卡片（按（前因目录, 文件名））；文件名必须是卡片上那条关系算出来的名字，手放的、改过名的报错。"""

        found: dict[tuple[str, str], StoredRecord] = {}
        if not self._exists():
            return found
        for directory in self.files.files.directories(self.directory):
            for name in self.files.files.markdown_names(directory):
                record = self.files.read(self._path(directory.name, name))
                assert record is not None
                if _location_of(record.fields) != (directory.name, name):
                    raise RecordFileError(f"relation card {directory.name}/{name}.md is not named after its relation")
                found[(directory.name, name)] = record
        return found

    def _exists(self) -> bool:
        return self.files.files.directory_exists(self.directory)

    def _path(self, antecedent: str, name: str) -> Path:
        return self.files.path(RELATION_CARDS_DIRECTORY, self.lane, antecedent, f"{name}.md")

    def _pointers(self, key: RelationKey) -> dict[str, str | None]:
        return {
            "next": format_uri((RELATION_CARDS_DIRECTORY, self.lane, key.consequent)),
            "antecedent_habit": self._habit(key.antecedent),
            "consequent_habit": self._habit(key.consequent),
        }

    def _habit(self, identity: str) -> str | None:
        """这个概念的习惯卡片地址；日历情境（不是词）没有习惯卡片。"""

        try:
            ClassId.parse(identity)
        except KindIdError:
            return None
        return format_uri((HABITS_DIRECTORY, self.lane, f"{identity}.md"))


def _location_of(fields: Mapping[str, Any]) -> tuple[str, str]:
    key = RelationKey(
        fields["lane"], fields["antecedent"], fields["consequent"], Segment(fields["segment"]), fields["condition"]
    )
    return card_location(key)


__all__ = ["RELATION_TYPE", "RelationCards", "RelationSync"]
