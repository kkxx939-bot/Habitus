"""分块归类：按批大小切块，块与块之间调一次 ``checkpoint``（调用方给的续租约钩子；词表这一层不认识锁）。

白天归类、拆改的封闭集合重判与锚点自检都可能一次几十上百条；整段不续约会把 sweep 租约耗死。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from habitus.behavior.kinds.classify.classifier import ClassifyResult, DaytimeClassifier, KindVerdict
from habitus.behavior.kinds.classify.request import ClassifyRequest
from habitus.behavior.kinds.model import Vocabulary


async def classify_in_chunks(
    classifier: DaytimeClassifier,
    requests: Sequence[ClassifyRequest],
    vocabulary: Vocabulary,
    *,
    checkpoint: Callable[[], object],
) -> ClassifyResult:
    verdicts: dict[str, KindVerdict] = {}
    signals: list[str] = []
    calls = 0
    size = classifier.config.batch_size
    for start in range(0, len(requests), size):
        result = await classifier.classify(requests[start : start + size], vocabulary)
        verdicts.update(result.verdicts)
        signals.extend(result.signals)
        calls += result.model_calls
        checkpoint()
    return ClassifyResult(verdicts, calls, tuple(signals))


__all__ = ["classify_in_chunks"]
