"""门：读时从结算里数出来的量。

``verified_counts`` 是每个 (行为, 情形) 被验证过的承诺数——纯计数，没有时间项、没有阈值。阈值（要几次才
够格开口）是产品档位，住在配置里；比较在调用方。情形为空串的那本账是"引用的卡还没有关联记录"。

一条承诺引用了几种情形就各记一次：它在那几种情形下都被验证过。合成一个"情形集合"的账另说，
等有数据了再定（见 ``TODO(FORESIGHT-LOSS-001)``）。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from habitus.foresight.ledger.model import Settlement


def verified_counts(settlements: Iterable[Settlement]) -> Mapping[tuple[str, str], int]:
    counts: Counter[tuple[str, str]] = Counter()
    for item in settlements:
        if not item.verified:
            continue
        for situation in item.situations or ("",):
            counts[(item.kind_token, situation)] += 1
    return dict(counts)


def verified_count(settlements: Iterable[Settlement], kind_token: str, situation: str = "") -> int:
    return verified_counts(settlements).get((kind_token, situation), 0)


__all__ = ["verified_count", "verified_counts"]
