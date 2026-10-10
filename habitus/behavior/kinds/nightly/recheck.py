"""待定重判：每晚新增认作"已有某类"的组，交白天归类按当晚词表重判一次，由白天归类定。

为什么要有这一步：每晚新增与白天归类各说一套（晚上说"就是已有的 X"，白天说"都不是"）时，原来的做法是只留信号、
跳过——这组既不归进 X，也长不成自己的类，在待定池里一直卡着（冻结方案 54 天结束时卡着 7 条）。

为什么交白天归类定、而不信每晚新增：白天归类看得到类的全部信息（判据、提醒句、不含、例子），每晚新增只看类名与判据；
白天归类本来就是"记录归哪一类"的唯一入口，这里不多出一条判定规则。

为什么只问一次：词表几乎每晚都因长新类而升版本，"版本变了就再问"等于每晚都问，问多了总有一次被塞进最像的类
（2026-10-09 54 天对照：体彩那条前五次都判「都不是」，第六次被归进"阅读并理解技术资料"）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from habitus.behavior.kinds.classify import (
    ClassifyRequest,
    DaytimeClassifier,
    OccurrenceContent,
    Outcome,
    classify_in_chunks,
)
from habitus.behavior.kinds.model import Vocabulary
from habitus.behavior.kinds.pending import PendingEntry

#: 按 occurrence 取它在树上的内容（原话 + 概要 + 目标 + 步骤）；取不到（树上已没有）给 None。
ContentOf = Callable[[str], OccurrenceContent | None]


@dataclass(frozen=True)
class RecheckResult:
    """``assigned``：重判归进了类的（occurrence → 类编号）；``released``：重判过、仍不归已有类的，从此照常长新类。"""

    assigned: Mapping[str, str]
    released: frozenset[str]
    model_calls: int
    signals: tuple[str, ...]


class PendingRecheck:
    def __init__(self, classifier: DaytimeClassifier) -> None:
        if not isinstance(classifier, DaytimeClassifier):
            raise TypeError("classifier must be DaytimeClassifier")
        self.classifier = classifier

    async def recheck(
        self,
        entries: Sequence[PendingEntry],
        vocabulary: Vocabulary,
        *,
        content_of: ContentOf,
        checkpoint: Callable[[], object],
    ) -> RecheckResult:
        signals: list[str] = []
        requests: list[ClassifyRequest] = []
        for entry in entries:
            content = content_of(entry.occurrence)
            if content is None:
                signals.append(f"kind_pending_recheck_missing {entry.occurrence}: not on the tree; left as is")
                continue
            requests.append(ClassifyRequest(entry.occurrence, entry.lane, content))
        if not requests:
            return RecheckResult({}, frozenset(), 0, tuple(signals))
        result = await classify_in_chunks(self.classifier, requests, vocabulary, checkpoint=checkpoint)
        assigned = {
            key: verdict.token for key, verdict in result.verdicts.items() if verdict.outcome is Outcome.CLASS
        }
        # 「都不是」与「不是一件事」都不归已有类：这条本来就在待定池里，白天归类没归进去，就放行去长新类
        released = frozenset(result.verdicts) - frozenset(assigned)
        return RecheckResult(assigned, released, result.model_calls, (*signals, *result.signals))


__all__ = ["ContentOf", "PendingRecheck", "RecheckResult"]
