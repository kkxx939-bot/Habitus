"""规律级的 L1（概览）与 L0（一句话）。

L1 是"这个候选反复出现时有哪几种情形、各覆盖哪些日期"。它**不另调模型**：那几句话是模型在逐条
记录里说的，L1 只是按候选汇总；L0 又从 L1 机械派生。所以这两层零模型调用、零判断。

**为什么 L1 自己带一份规范结构。** 每次更新都从全部 L2 重扫的话，一个跑了一年的候选每夜要读
三百多份文档。所以 L1 增量追加，落盘形状与记录一致（正文 + 尾部规范 JSON，解码时回读比对）；
"从 L2 全量重建"是一个显式的修复操作，不在夜批路径上。

**由来住在这里，但只写一次。** 它与 L0 的更新节奏完全不同：L0 随情境分布变，由来定了就定了。
放在同一份结构里而不是另起一个侧车，是因为只要"已经有了就不覆盖"这条由代码保证，重算 L1 就不会
把它冲掉；另起一个文件只是把同一条约束搬个地方。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime

from habitus.foundation.ids import canonical_path_identity, canonical_text_identity
from habitus.scene.model import RegularityLevel
from habitus.scene.regularity.store import RegularityTree, RegularityTreeError

_MARKER = "\n<!-- HABITUS_REGULARITY_OVERVIEW\n"
_FOOTER = "\n-->\n"
_RECORD_TYPE = "regularity_overview"
_METADATA_KEYS = frozenset({"record_type", "kind", "origin", "situations"})


class OverviewError(ValueError):
    """概览的结构或序列化与我们自己的产物矛盾。"""


@dataclass(frozen=True)
class Situation:
    """一种反复出现的情形：它是什么，覆盖了哪些日期。"""

    text: str
    days: tuple[date, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip() or self.text != self.text.strip():
            raise OverviewError("a situation must be non-empty text without surrounding whitespace")
        if "\n" in self.text or "\r" in self.text:
            raise OverviewError("a situation must be a single line")
        days = tuple(self.days)
        if any(isinstance(day, datetime) or not isinstance(day, date) for day in days):
            raise OverviewError("situation days must be plain dates")
        if list(days) != sorted(set(days)):
            raise OverviewError("situation days must be strictly ascending without repeats")
        object.__setattr__(self, "days", days)

    @property
    def identity(self) -> str:
        """情形是一句人话，身份走文本归一——拿路径段校验器去卡它会让"周五 19:00 去"这种句子炸掉。"""

        return canonical_text_identity(self.text, "situation text")

    def covering(self, day: date) -> Situation:
        """把这一天记进来；已经有了就原样返回。"""

        return self if day in self.days else Situation(self.text, tuple(sorted({*self.days, day})))


@dataclass(frozen=True)
class Overview:
    """一个候选的 L1。``situations`` 按首次覆盖的日期排序——那就是它们出现的先后。"""

    kind_token: str
    situations: tuple[Situation, ...] = ()
    origin: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind_token, str) or not self.kind_token.strip():
            raise OverviewError("overview kind_token must be non-empty text")
        situations = tuple(self.situations)
        if any(not isinstance(item, Situation) for item in situations):
            raise TypeError("overview situations must be Situation values")
        identities = [item.identity for item in situations]
        if len(set(identities)) != len(identities):
            raise OverviewError("overview repeats a situation")
        if self.origin is not None and (not self.origin.strip() or "\n" in self.origin):
            raise OverviewError("overview origin must be a single non-empty line")
        object.__setattr__(self, "situations", situations)

    def index_of(self, text: str) -> int | None:
        """按规范身份找这句话对应的那一种；找不到返回 None。"""

        try:
            wanted = canonical_text_identity(text, "situation text")
        except (TypeError, ValueError):
            return None
        return next((index for index, item in enumerate(self.situations) if item.identity == wanted), None)

    def with_occurrence(self, *, day: date, text: str) -> Overview:
        """把这一天归进某一种情形：已有的就多记一个日期，没有的就新开一种（追加在末尾）。"""

        index = self.index_of(text)
        if index is None:
            return Overview(self.kind_token, (*self.situations, Situation(text, (day,))), self.origin)
        updated = list(self.situations)
        updated[index] = updated[index].covering(day)
        return Overview(self.kind_token, tuple(updated), self.origin)

    def with_origin(self, origin: str) -> Overview:
        """由来只写一次：已经有了就原样返回。"""

        return self if self.origin is not None else Overview(self.kind_token, self.situations, origin)

    @property
    def abstract(self) -> str:
        """L0：一句话。取覆盖天数最多的那一种情形——机械派生，不调模型、不加判断。"""

        if not self.situations:
            return "还没有归纳出情形。"
        ranked = sorted(self.situations, key=lambda item: (-len(item.days), item.text))
        return ranked[0].text

    @property
    def markdown_body(self) -> str:
        lines = [f"# {canonical_path_identity(self.kind_token, 'overview kind token')}", ""]
        if self.origin is not None:
            lines += [f"- 由来：{self.origin}", ""]
        if not self.situations:
            lines.append("（还没有归纳出情形）")
        lines += [
            f"- S{index}：{item.text}（{'、'.join(day.isoformat() for day in item.days) or '—'}）"
            for index, item in enumerate(self.situations, start=1)
        ]
        return "\n".join(lines) + "\n"

    def encode(self) -> str:
        body = self.markdown_body
        if _MARKER in body:
            raise OverviewError("overview body contains the reserved metadata marker")
        metadata = {
            "record_type": _RECORD_TYPE,
            "kind": canonical_path_identity(self.kind_token, "overview kind token"),
            "origin": self.origin,
            "situations": [
                {"text": item.text, "days": [day.isoformat() for day in item.days]} for item in self.situations
            ],
        }
        rendered = json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).replace(
            "--", "\\u002d\\u002d"
        )
        return f"{body}{_MARKER}{rendered}{_FOOTER}"

    @classmethod
    def decode(cls, raw: str, *, kind_token: str) -> Overview:
        if not isinstance(raw, str) or raw.count(_MARKER) != 1 or not raw.endswith(_FOOTER):
            raise OverviewError("overview must contain one terminal metadata comment")
        body, _separator, tail = raw.partition(_MARKER)
        try:
            metadata = json.loads(tail[: -len(_FOOTER)])
        except json.JSONDecodeError as exc:
            raise OverviewError("overview metadata is not strict JSON") from exc
        if not isinstance(metadata, Mapping) or set(metadata) != _METADATA_KEYS:
            raise OverviewError("overview metadata has an invalid shape")
        if metadata["record_type"] != _RECORD_TYPE:
            raise OverviewError("overview document type is not an overview")
        if metadata["kind"] != canonical_path_identity(kind_token, "overview kind token"):
            raise OverviewError("overview belongs to a different candidate")
        overview = cls(
            kind_token=kind_token,
            situations=tuple(_situation(item) for item in metadata["situations"]),
            origin=metadata["origin"],
        )
        if overview.encode() != raw:
            raise OverviewError("overview is not canonically serialized")
        return overview

    @classmethod
    def read(cls, tree: RegularityTree, kind_token: str) -> Overview:
        """读这个候选的 L1；还没有就是空的——新候选本来如此，不是错误。"""

        if not tree.layer_exists(kind_token, RegularityLevel.OVERVIEW):
            return cls(kind_token)
        return cls.decode(tree.read_layer(kind_token, RegularityLevel.OVERVIEW), kind_token=kind_token)

    def write(self, tree: RegularityTree) -> None:
        """L1 与 L0 一起落；存储层先写 L1 再写 L0，所以 L0 存在就意味着 L1 也在。"""

        tree.write_layers(self.kind_token, overview=self.encode(), abstract=self.abstract)


def _situation(item: object) -> Situation:
    if not isinstance(item, Mapping) or set(item) != {"text", "days"}:
        raise OverviewError("a situation entry has an invalid shape")
    days = item["days"]
    if not isinstance(days, list):
        raise OverviewError("situation days must be an array")
    try:
        return Situation(text=item["text"], days=tuple(date.fromisoformat(day) for day in days))
    except (TypeError, ValueError) as exc:
        raise OverviewError("a situation entry is not decodable") from exc


def rebuild(tree: RegularityTree, kind_token: str) -> Overview:
    """修复用：从全部 L2 重建 L1 与由来。夜批不走这条路——它要读这个候选的全部历史。

    取材按**日目录**而不是完成标记：一天可能记录已落盘、标记还没写（下轮重做），那几条的情境
    仍然是真实产出，重建时把它们算进来才和增量追加的结果一致。
    """

    if not isinstance(tree, RegularityTree):
        raise TypeError("tree must be a RegularityTree")
    overview = Overview(kind_token)
    for day in tree.day_directories(kind_token):
        for document in tree.read_day(kind_token, day):
            # 由来的真源就在最早那条记录的上下文里。不还原的话，修一次 L1 就把它永久抹掉——
            # 而 ``_publish`` 的"之前没写过"判据之后恒为假，再也不会补回来。
            overview = overview.with_origin(document.context)
            if document.situation is not None:
                overview = overview.with_occurrence(day=day, text=document.situation)
    return overview


def read_abstract(tree: RegularityTree, kind_token: str) -> str | None:
    try:
        return tree.read_layer(kind_token, RegularityLevel.ABSTRACT)
    except RegularityTreeError:
        return None


__all__ = ["Overview", "OverviewError", "Situation", "read_abstract", "rebuild"]
