"""对答案：一条承诺 vs 那天定稿之后行为树上的行。

判据只有一条：那天该行为**在说这话之后**的第一次发生——落在时窗里是「验证」，在窗外是「偏离」（记差几槽，
早于窗是负数），那天没再发生是「落空」。槽号按**承诺自己那一代的槽宽**算：槽号配上槽宽才有意义，
换参数重建的新一代不能拿来核对旧承诺。

"那天定稿了"由调用方判，用的是归约自己的事实（``BehaviorReductionRunner.closed_days``：那天的链都已落树、
且封口视界已过那天的本地结束）。**不能用预测树的出处日**——树每轮从行为树全量重建，今天上午的行中午就在树上，
拿它当定稿会在今天还没过完时就把今天的承诺结成「落空」，而结算是一次性的。

承诺上记的是**说话那一刻的类编号**；到结算时词表可能已经拆改（每周的定期拆改会把那天的行重打成新编号）。所以
"该行为"按 ``current``（编号 → 现在对应哪些在用编号，顺着合并、拆分往下找；组合根注入词表的 ``current_ids``）认：
合并后认合并进去的那一类，拆分后认拆出来的任何一类。

本模块不认识存储、不读时钟。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from habitus.foresight.ledger.model import Claim, Settlement
from habitus.prediction.model import SlotKey
from habitus.scene.views import FlowRow


def settle(
    claim: Claim,
    rows: Sequence[FlowRow],
    *,
    settled_at: datetime,
    current: Callable[[str], Sequence[str]],
) -> Settlement:
    """``rows`` 是那天的全部行（``DayIndex.rows``）；只看这一类（按现在的编号认）、开始晚于说话那一刻的。"""

    spoken = claim.judged_at.astimezone(UTC)
    # 原编号总带上：迁移做到一半时，那天的行可能还是原编号（第四轮评审 E3）
    kinds = {claim.kind_token, *current(claim.kind_token)}
    first = next(
        (row for row in rows if row.kind_token in kinds and row.at.astimezone(UTC) > spoken),
        None,
    )

    def record(outcome: str, uri: str | None, offset: int | None) -> Settlement:
        return Settlement(
            claim_id=claim.claim_id,
            kind_token=claim.kind_token,
            day=claim.day,
            situations=claim.situations,
            outcome=outcome,
            occurrence_uri=uri,
            slot_offset=offset,
            settled_at=settled_at,
        )

    if first is None:
        return record("落空", None, None)
    started = SlotKey.of(first.at, slot_minutes=claim.slot_minutes).slot
    start, end = claim.window
    if start <= started <= end:
        return record("验证", first.uri, None)
    return record("偏离", first.uri, started - end if started > end else started - start)


__all__ = ["settle"]
