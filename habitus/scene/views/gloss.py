"""规律树的读口：把某个候选在某些日子的关联记录按 ``occurrence_uri`` 取出来，贴到历史卡上。

只读、零判断。记录是夜批关联写下的（``scene/association``），这里不解释它，只把五样东西原样带出：
当时是什么情况（context）、属于哪种情形（situation）、前因（``results_from`` 链接的目标）、
用掉的前提（consumed）、留下的前提（left）。

只读**有完成标记**的日子：一天的记录是逐条写、最后打标记的，读到写了一半的一天等于读到半句话。
``version`` 与刷新器判"已关联"用的是同一把尺子，传同一个值两边才对得上。

情形列表按周几分成两组：算法按（周几，槽）算，语义侧沿同样的维度取——周三判断打球时，判断者
先看周三出现过的情形，其他周几的另放。``Situation.days`` 已经带日期，周几机械得出，存储不动。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType

from habitus.behavior.uri import BehaviorURI
from habitus.scene.model import SceneLinkType
from habitus.scene.regularity.overview import Overview, Situation
from habitus.scene.regularity.store import RegularityTree

_WEEKDAYS = 7


@dataclass(frozen=True)
class AssociationGloss:
    """一次发生的关联记录里给判断者看的那部分。

    ``causes`` 是（行为 URI，行为名）对：名字从 URI 的地址里解出来（地址保留原始行为名，不是折叠后的
    身份），读它的人不用认识 URI 的形状；两组前提是原文对。
    """

    occurrence_uri: str
    context: str
    situation: str | None
    causes: tuple[tuple[str, str], ...]
    consumed: tuple[tuple[str, str], ...]
    left: tuple[tuple[str, str], ...]


def association_glosses(
    tree: RegularityTree, kind_token: str, days: Iterable[date], *, version: str | None = None
) -> Mapping[str, AssociationGloss]:
    """候选在这些日子里的关联记录，键是 ``occurrence_uri``；没有完成标记的日子一条都不取。"""

    if not isinstance(tree, RegularityTree):
        raise TypeError("tree must be a RegularityTree")
    if not isinstance(kind_token, str) or not kind_token:
        raise ValueError("kind_token must be non-empty text")
    done = tree.days_for(kind_token, version=version)
    glosses: dict[str, AssociationGloss] = {}
    for day in sorted(set(days)):
        if day not in done:
            continue
        for document in tree.read_day(kind_token, day):
            glosses[document.occurrence_uri] = AssociationGloss(
                occurrence_uri=document.occurrence_uri,
                context=document.context,
                situation=document.situation,
                causes=tuple(
                    _cause(str(link.to_uri)) for link in document.links if link.link_type is SceneLinkType.RESULTS_FROM
                ),
                consumed=document.consumed,
                left=document.left,
            )
    return MappingProxyType(glosses)


def _cause(uri: str) -> tuple[str, str]:
    """前因链接的目标：行为 URI 就解出它的名字；别的（情景 URI）原样带出。"""

    try:
        return uri, BehaviorURI.parse(uri).to_address().name
    except (TypeError, ValueError):
        return uri, uri


def situations_of(
    tree: RegularityTree, kind_token: str, *, weekday: int
) -> tuple[tuple[Situation, ...], tuple[Situation, ...]]:
    """候选的情形列表，分成（这个周几出现过的，只在其他周几出现过的）两组。"""

    if not isinstance(tree, RegularityTree):
        raise TypeError("tree must be a RegularityTree")
    if isinstance(weekday, bool) or not isinstance(weekday, int) or not 0 <= weekday < _WEEKDAYS:
        raise ValueError("weekday must be an integer between 0 and 6")
    overview = Overview.read(tree, kind_token)
    here = tuple(item for item in overview.situations if any(day.weekday() == weekday for day in item.days))
    elsewhere = tuple(item for item in overview.situations if item not in here)
    return here, elsewhere


__all__ = ["AssociationGloss", "Situation", "association_glosses", "situations_of"]
