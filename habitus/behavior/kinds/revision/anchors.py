"""锚点自检（裁定 17 ②）：拆改改过的判据会不会把别的类以后的新条目错拉进来——只用用户自己的历史。

1. 挑锚点：每个没动过的在用类，取最近的 ``anchors_per_class`` 条成员；
2. 只留站得稳的：用旧清单归两遍，两遍都归回原类的才算锚点（排除模型自己的随机抖动）；
3. 用新清单重归：被拉走的锚点超过 ``max_anchor_pull`` 这一比例，这次改动作废。

封闭集合重判已经保证别的类里归好的条目不会被移动；锚点防的是"以后"。站得稳的锚点按（词表版本，lane）在一次拆改里只算一次，
各个提议共用。一个锚点都没有时闸过不了——量不了就不改。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from habitus.behavior.kinds.classify.chunked import classify_in_chunks
from habitus.behavior.kinds.classify.classifier import DaytimeClassifier, KindVerdict, Outcome
from habitus.behavior.kinds.classify.request import ClassifyRequest
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import Vocabulary
from habitus.behavior.kinds.revision.facts import Member, members_of


@dataclass(frozen=True)
class AnchorCheck:
    passed: bool
    anchors: int
    pulled: int
    model_calls: int

    def describe(self) -> str:
        return f"anchors {self.anchors}, pulled {self.pulled}"


class AnchorGate:
    def __init__(self, classifier: DaytimeClassifier) -> None:
        if not isinstance(classifier, DaytimeClassifier):
            raise TypeError("classifier must be DaytimeClassifier")
        self.classifier = classifier
        self.config = classifier.config
        #: 键是（词表版本，lane，这次取到的那批最近成员）：同一次拆改里多次自检复用；下一周成员换了就重挑，
        #: 不沿用上一周"站得稳的锚点"（裁定 17②"最近几条成员"；第二轮评审 P2-5）。
        self._stable: dict[tuple[int, Lane, tuple[str, ...]], tuple[Member, ...]] = {}

    async def check(
        self,
        lane: Lane,
        before: Vocabulary,
        after: Vocabulary,
        touched: frozenset[ClassId],
        members: Sequence[Member],
        *,
        checkpoint: Callable[[], object],
    ) -> AnchorCheck:
        calls = 0
        candidates = self._recent_members(lane, before, members)
        key = (before.version, lane, tuple(member.occurrence for member in candidates))
        if key not in self._stable:
            self._stable[key], calls = await self._stable_anchors(candidates, before, checkpoint)
        anchors = [member for member in self._stable[key] if ClassId.parse(member.token) not in touched]
        if not anchors:
            return AnchorCheck(False, 0, 0, calls)
        result = await classify_in_chunks(self.classifier, _requests(anchors), after, checkpoint=checkpoint)
        pulled = sum(1 for member in anchors if _token(result.verdicts[member.occurrence]) != member.token)
        allowed = int(self.config.max_anchor_pull * len(anchors))
        return AnchorCheck(pulled <= allowed, len(anchors), pulled, calls + result.model_calls)

    def _recent_members(self, lane: Lane, vocabulary: Vocabulary, members: Sequence[Member]) -> tuple[Member, ...]:
        """每个在用类最近的几条成员：锚点候选。"""

        candidates: list[Member] = []
        for item in vocabulary.active(lane):
            own = sorted(members_of(members, str(item.id)), key=lambda m: (m.day, m.occurrence), reverse=True)
            candidates += own[: self.config.anchors_per_class]
        return tuple(candidates)

    async def _stable_anchors(
        self, candidates: Sequence[Member], vocabulary: Vocabulary, checkpoint: Callable[[], object]
    ) -> tuple[tuple[Member, ...], int]:
        if not candidates:
            return (), 0
        first = await classify_in_chunks(self.classifier, _requests(candidates), vocabulary, checkpoint=checkpoint)
        second = await classify_in_chunks(self.classifier, _requests(candidates), vocabulary, checkpoint=checkpoint)
        stable = tuple(
            member
            for member in candidates
            if _token(first.verdicts[member.occurrence]) == member.token == _token(second.verdicts[member.occurrence])
        )
        return stable, first.model_calls + second.model_calls


def _requests(members: Sequence[Member]) -> list[ClassifyRequest]:
    return [ClassifyRequest(member.occurrence, member.lane, member.content) for member in members]


def _token(verdict: KindVerdict) -> str | None:
    return verdict.token if verdict.outcome is Outcome.CLASS else None


__all__ = ["AnchorCheck", "AnchorGate"]
