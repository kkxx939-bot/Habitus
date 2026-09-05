"""Benchmark 共用的确定性质量与性能指标。"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"[\u3400-\u9fff]|[a-z0-9]+", re.IGNORECASE)


def evidence_recall(context: str, evidence_texts: Sequence[str]) -> float | None:
    """计算回答模型实际可见上下文覆盖了多少条标注证据。"""

    if not isinstance(context, str):
        raise TypeError("context must be text")
    evidence = tuple(evidence_texts)
    if any(not isinstance(item, str) or not item.strip() for item in evidence):
        raise ValueError("evidence_texts must contain non-empty text")
    if not evidence:
        return None
    return sum(_evidence_hit(context, item) for item in evidence) / len(evidence)


def latency_distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    """返回跨 suite 统一的耗时分布。"""

    normalized = tuple(_finite_non_negative(value, "latency") for value in values)
    if not normalized:
        return {
            "count": 0,
            "average": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    ordered = sorted(normalized)
    return {
        "count": len(ordered),
        "average": sum(ordered) / len(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "maximum": ordered[-1],
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    """在线性插值规则下计算百分位。"""

    if not values:
        raise ValueError("percentile values must not be empty")
    if isinstance(fraction, bool) or not isinstance(fraction, int | float) or not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be between zero and one")
    ordered = sorted(_finite_non_negative(value, "percentile value") for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(fraction)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _evidence_hit(context: str, evidence: str) -> bool:
    normalized_context = _normalized(context)
    normalized_evidence = _normalized(evidence)
    if normalized_evidence in normalized_context:
        return True
    evidence_tokens = _TOKEN.findall(normalized_evidence)
    if len(evidence_tokens) < 4:
        return False
    context_counts = Counter(_TOKEN.findall(normalized_context))
    evidence_counts = Counter(evidence_tokens)
    overlap = sum(min(count, context_counts[token]) for token, count in evidence_counts.items())
    return overlap / sum(evidence_counts.values()) >= 0.8


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _finite_non_negative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return normalized


__all__ = ["evidence_recall", "latency_distribution", "percentile"]
