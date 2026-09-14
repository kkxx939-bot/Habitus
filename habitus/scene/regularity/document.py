"""一次关联记录：这次发生的上下文与它的前因。

形状是**固定的三个字段加若干条边**，不走 schema 注册表——注册表是为归组那种可变形状
（成员、待用前提、角色）准备的，而这里每份文档长得一模一样，声明式的那套只会多一层间接。

落盘格式与按天情景文档一致（正文 + 终端 HTML 注释里的规范 JSON），两片区域在磁盘上看起来
是同一种东西；编码是规范化的，解码时回读比对，不是"尽力解析"。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from habitus.behavior.uri import BehaviorURI, BehaviorURINodeType
from habitus.foundation.ids import canonical_text_identity
from habitus.foundation.integrity import canonicalize
from habitus.scene.model import AssociationAddress
from habitus.scene.regularity.link import SceneStoredLink, normalize_stored_links, parse_stored_links
from habitus.scene.uri import SceneURI

_MARKER = "\n<!-- HABITUS_ASSOCIATION_FIELDS\n"
_FOOTER = "\n-->\n"
_RECORD_TYPE = "association"
_METADATA_KEYS = frozenset({"record_type", "created_at", "fields", "links"})
_FIELD_NAMES = ("occurrence_uri", "context", "situation", "left", "consumed")


def premise_identity(producer_uri: str, text: str) -> tuple[str, str]:
    """一条前提的身份：**产生它的那条行为 + 那句话的规范形式**。

    粒度必须细到这一层。旧实现把兑现记在"整条行为被指过一次"上，于是去超市一趟留下的"家里有菜"
    和"买到了灯泡"，只要其中一条被用掉，另一条跟着消失。前提一次写下、之后只被引用，文本不再变，
    所以拿它作身份的一半是稳的。

    文本走 ``canonical_text_identity`` 而**不是** ``canonical_path_identity``：那一个是给会变成
    目录名的东西用的，会拒掉 ``/``、拒掉以句点结尾、拒掉 Windows 保留名——而"冰箱里有菜/水果"
    "明天 9:00 要复诊"都是模型会写出来的正常句子，用它校验等于让一句人话把整条流程卡死。
    """

    return str(BehaviorURI.parse(producer_uri)), canonical_text_identity(text, "premise text")


class AssociationDocumentError(ValueError):
    """关联记录的结构、身份或序列化与我们自己的产物矛盾。"""


@dataclass(frozen=True)
class AssociationDocument:
    """一次发生的关联：指回那条 occurrence、一句"为什么"、归属的情境、以及前因边。

    ``occurrence_uri`` 不是冗余——地址只到"哪个候选的哪个时刻"，而那一刻在行为树上叫什么名字
    （原话）只有 URI 说得出；而且"关联与 occurrence 一一对应"这条要能被机械核对。

    ``situation`` 可以为空：装配层把"哪一种情境都不像"当成降级而不是丢弃（上下文才是这条记录
    的主产物），所以文档必须接得住它。为空时正文里那一行整行不出现，正文仍然完整承载全部字段。

    ``left`` 与 ``consumed`` 是待用前提的**两件事实**，都按前提本身记，不按行为、不按情景：
    ``left`` 是这次铺下了什么（一句话 + 它在等哪一类行为），``consumed`` 是这次用掉了哪几条
    （产生方的行为 URI + 那句话）。**没有状态、没有过期**——"还立不立得住"是读的时候的问题，
    "到期没到期"更是预测层的判断，不该在这一层。兑现也照旧写一条 ``needs`` 边给读侧走图用，
    但权威是 ``consumed`` 这个字段：边只带得上目标 URI，带不上是哪一条前提。
    """

    address: AssociationAddress
    created_at: datetime
    occurrence_uri: str
    context: str
    situation: str | None = None
    left: tuple[tuple[str, str], ...] = ()
    consumed: tuple[tuple[str, str], ...] = ()
    links: tuple[SceneStoredLink, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.address, AssociationAddress):
            raise TypeError("association address must be an AssociationAddress")
        if not isinstance(self.created_at, datetime) or self.created_at.utcoffset() is None:
            raise ValueError("association created_at must be a timezone-aware datetime")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        for name in ("context", "situation"):
            value = getattr(self, name)
            if value is None and name == "situation":
                # 归不进任何情境是合法结果：这一次确实哪一种都不像，不是缺数据。硬要一句话
                # 才能落盘，就等于逼装配层去编一个占位串——那才是真的在凭空写内容。
                continue
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"association {name} must be non-empty text without surrounding whitespace")
            if "\n" in value or "\r" in value:
                raise ValueError(f"association {name} must be a single line")
        object.__setattr__(self, "left", _pairs(self.left, "association left", ("text", "consumed_by")))
        object.__setattr__(self, "consumed", _pairs(self.consumed, "association consumed", ("producer_uri", "text")))
        occurrence = BehaviorURI.parse(self.occurrence_uri)
        if occurrence.node_type is not BehaviorURINodeType.DOCUMENT:
            raise ValueError("association occurrence_uri must identify a behaviour document")
        # 关联与 occurrence 一一对应：地址里的时刻必须就是那条 occurrence 的开始时刻，
        # 否则文档会指着一条与自己身份不符的行为。
        if occurrence.to_address().started_at != self.address.started_at:
            raise ValueError("association address must carry the occurrence's own start time")
        # 时刻单独不构成身份：行为树允许同一微秒上两条不同名的 occurrence（撞车消歧正是按名字分）。
        if occurrence.to_address().identity_name != self.address.identity_name:
            raise ValueError("association address must carry the occurrence's own name")
        object.__setattr__(self, "occurrence_uri", str(occurrence))
        here = occurrence.to_address().started_at
        for producer_uri, _text in self.consumed:
            producer = BehaviorURI.parse(producer_uri)
            if producer.node_type is not BehaviorURINodeType.DOCUMENT:
                raise ValueError("a consumed premise must name a behaviour document")
            # 用掉的前提必须先被建立：指向一条比这次还晚的行为，是我们自己产物里的矛盾。
            if producer.to_address().started_at > here:
                raise ValueError("a consumed premise cannot have been produced later than this occurrence")
        seen = {premise_identity(uri, text) for uri, text in self.consumed}
        if len(seen) != len(self.consumed):
            raise ValueError("association consumed repeats a premise")
        # 身份算得出来，是"这条记录将来读得回来"的一部分：算不出来的记录不许落盘，否则它会在
        # 下一轮的投影里炸，而那时已经没有任何地方能把它拦下来。
        left_identities = {canonical_text_identity(text, "association left text") for text, _waits in self.left}
        if len(left_identities) != len(self.left):
            raise ValueError("association left repeats a premise")
        if self.situation is not None:
            canonical_text_identity(self.situation, "association situation")
        links = normalize_stored_links(self.links, label="association links")
        uri = self.uri
        if any(link.from_uri != uri for link in links):
            raise ValueError("association forward link has the wrong source URI")
        object.__setattr__(self, "links", links)

    @property
    def uri(self) -> SceneURI:
        return SceneURI.from_association(self.address)

    @property
    def fields(self) -> Mapping[str, Any]:
        return {name: getattr(self, name) for name in _FIELD_NAMES}

    @property
    def markdown_body(self) -> str:
        """给人读的正文；结构字段是权威，这里是它的完整呈现。

        两条纪律：

        **每个字段都要出现在正文里。**解码时正文与字段互相比对，正文没承载的字段就盖不住——
        改一个只在 JSON 里的值，文档照样"自洽"。审查用变异实测过三处漏网：情境编号（L1 的日期
        归属靠它）、``created_at``、以及每条边的 ``link_type``（``needs`` 被改成 ``results_from``
        照样通过）。

        **标题只用路径推得出来的身份**（``identity_kind`` / ``identity_name``），不用人写法。
        从磁盘还原时地址是按路径重建的，人写法（``Gym`` 还是 ``gym``）根本不在路径里；把它写进
        正文，同一份文档换个写法的地址去读就会报"正文与字段不符"。人写法的真源在行为树上那条
        occurrence，这里由 ``occurrence_uri`` 指过去。
        """

        lines = [
            f"# {self.address.identity_kind} · {self.address.identity_name}",
            "",
            self.context,
            "",
        ]
        if self.situation is not None:
            lines.append(f"- 情境：{self.situation}")
        # 一对的两半各占一行。拼在同一行用分隔符隔开的话，文本里出现同一个分隔符时正文就定不出
        # 分割点——只改 JSON 一侧把分割点挪走能骗过正文比对，而且两条语义不同的前提会渲染成
        # 一模一样的一行。
        for text, waits_for in self.left:
            lines += [f"- 留下：{text}", f"  等待：{waits_for}"]
        for producer_uri, text in self.consumed:
            lines += [f"- 用掉：{text}", f"  产生：{producer_uri}"]
        lines += [
            f"- 这次：{self.occurrence_uri}",
            f"- 关联于：{self.created_at.isoformat(timespec='microseconds').replace('+00:00', 'Z')}",
        ]
        # 边的**类型**也要写出来：两条指向同一目标、类型不同的边，正文里不能长得一模一样。
        lines.extend(f"- {link.link_type.value}：{link.to_uri}" for link in self.links)
        return "\n".join(lines) + "\n"


def encode(document: AssociationDocument) -> str:
    if not isinstance(document, AssociationDocument):
        raise TypeError("document must be an AssociationDocument")
    body = document.markdown_body
    if _MARKER in body:
        raise AssociationDocumentError("association body contains the reserved metadata marker")
    metadata = {
        "record_type": _RECORD_TYPE,
        "created_at": document.created_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "fields": canonicalize(document.fields),
        "links": [link.to_dict() for link in document.links],
    }
    # 结构字段住在 HTML 注释里，正文中任意 ``--`` 都会提前闭合该注释；统一转义成 JSON 的 -。
    rendered = json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).replace(
        "--", "\\u002d\\u002d"
    )
    return f"{body}{_MARKER}{rendered}{_FOOTER}"


def decode(raw: str, *, expected_address: AssociationAddress) -> AssociationDocument:
    """还原一份关联记录；**回读比对**，不做"尽力解析"。"""

    if not isinstance(raw, str):
        raise TypeError("raw association document must be a string")
    if not isinstance(expected_address, AssociationAddress):
        raise TypeError("expected_address must be an AssociationAddress")
    if raw.count(_MARKER) != 1 or not raw.endswith(_FOOTER):
        raise AssociationDocumentError("association document must contain one terminal metadata comment")
    body, _separator, tail = raw.partition(_MARKER)
    try:
        metadata = json.loads(tail[: -len(_FOOTER)], object_pairs_hook=_unique_object, parse_constant=_reject)
    except (json.JSONDecodeError, RecursionError, AssociationDocumentError) as exc:
        raise AssociationDocumentError("association metadata is not strict JSON") from exc
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
        raise AssociationDocumentError("association metadata has an invalid shape")
    if metadata["record_type"] != _RECORD_TYPE:
        raise AssociationDocumentError("association document type is not an association")
    fields = metadata["fields"]
    if not isinstance(fields, dict) or set(fields) != set(_FIELD_NAMES):
        raise AssociationDocumentError("association fields have an invalid shape")
    try:
        document = AssociationDocument(
            address=expected_address,
            created_at=_timestamp(metadata["created_at"]),
            links=parse_stored_links(metadata["links"], label="association links"),
            **{name: fields[name] for name in _FIELD_NAMES},
        )
    except (TypeError, ValueError) as exc:
        raise AssociationDocumentError("association fields do not satisfy the record contract") from exc
    if document.markdown_body != body:
        raise AssociationDocumentError("association body does not match its structured fields")
    if encode(document) != raw:
        raise AssociationDocumentError("association document is not canonically serialized")
    return document


def _pairs(value: object, label: str, names: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    """一组 (甲, 乙) 文本对；规范化成有序元组，两端都必须是单行非空文本。"""

    if not isinstance(value, list | tuple):
        raise TypeError(f"{label} must be a sequence of pairs")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError(f"{label} must contain ({names[0]}, {names[1]}) pairs")
        first, second = item
        for part, name in ((first, names[0]), (second, names[1])):
            if not isinstance(part, str) or not part.strip() or part != part.strip() or "\n" in part or "\r" in part:
                raise ValueError(f"{label} {name} must be a single non-empty line without surrounding whitespace")
        pairs.append((first, second))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{label} contains a duplicate entry")
    return tuple(sorted(pairs))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise AssociationDocumentError("association metadata contains a duplicate key")
        seen[key] = value
    return seen


def _reject(value: str) -> Any:
    raise AssociationDocumentError(f"association metadata contains the non-JSON constant {value}")


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise AssociationDocumentError("association created_at must be text")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AssociationDocumentError("association created_at is not an ISO-8601 instant") from exc


__all__ = ["AssociationDocument", "AssociationDocumentError", "decode", "encode"]
