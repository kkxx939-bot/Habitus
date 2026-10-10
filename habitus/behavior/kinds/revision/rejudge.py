"""封闭集合重判：拆改之后，受影响的条目只在那几类之间重新选，不放回全部类（设计 四-③）。

模型在封闭集合里答「都不是」或「不是一件事」时，条目留在原类——封闭集合的意思就是不许溢出。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from habitus.behavior.kinds.changes import Move
from habitus.behavior.kinds.classify.chunked import classify_in_chunks
from habitus.behavior.kinds.classify.classifier import ClassifyResult, DaytimeClassifier, Outcome
from habitus.behavior.kinds.classify.request import ClassifyRequest
from habitus.behavior.kinds.ids import ClassId
from habitus.behavior.kinds.model import BehaviorClass, Exclusion, Vocabulary
from habitus.behavior.kinds.revision.facts import Member


def closed_vocabulary(classes: Sequence[BehaviorClass]) -> Vocabulary:
    """只含这几类的词表；指向集合外的「不含」保留文字、去掉去向。"""

    inside = {item.id for item in classes}
    return Vocabulary(version=0, classes={item.id: _trim(item, inside) for item in classes})


def _trim(item: BehaviorClass, inside: set[ClassId]) -> BehaviorClass:
    excludes = tuple(
        exclusion if exclusion.goes_to in inside else Exclusion(exclusion.text) for exclusion in item.excludes
    )
    return replace(item, excludes=excludes)


async def rejudge(
    members: Sequence[Member],
    closed: Sequence[BehaviorClass],
    *,
    keep: ClassId,
    classifier: DaytimeClassifier,
    checkpoint: Callable[[], object],
) -> tuple[tuple[Move, ...], ClassifyResult]:
    """在 ``closed`` 里重判 ``members``；返回需要改 token 的迁移（答到集合外的留在 ``keep``）。"""

    if not members:
        return (), ClassifyResult({}, 0, ())
    requests = [ClassifyRequest(member.occurrence, member.lane, member.content) for member in members]
    result = await classify_in_chunks(classifier, requests, closed_vocabulary(closed), checkpoint=checkpoint)
    moves: list[Move] = []
    for member in members:
        verdict = result.verdicts[member.occurrence]
        target = verdict.token if verdict.outcome is Outcome.CLASS else str(keep)
        if target != member.token:
            moves.append(Move(member.occurrence, member.token, target))
    return tuple(moves), result


__all__ = ["closed_vocabulary", "rejudge"]
