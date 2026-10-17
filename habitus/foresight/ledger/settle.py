"""对答案（``19-预测层方案.md`` 第五节）：一条预期 vs 行为树上那几天的行。纯函数，不认识存储、不读时钟。

判据：``observed_from`` 之后、最晚时限之前这一类**第一次**发生（形成时已经做了的也算：每个机会都进账，裁定 49）——

- 开口之后才发生：「提醒后发生」，不进自然命中，记提醒到发生的延迟；
- 时窗开始之前：「早到」（记早了几分钟，负数）；时窗之内：「准时」；时窗之后、最晚时限之前：「晚了但做了」（记晚了几分钟）；
- 到最晚时限没发生：「过期待决」，不是终态——之后在这本账的下一次机会之前补做了就是「后补」，回话另记。

"该行为"按 ``current``（编号 → 现在对应哪些在用编号）认：预期记的是形成时的编号，结算时词表可能已拆改。
只认真做成的：occurrence 的 status 是 abandoned（放弃了）的那一行不算这件事发生了（裁定 54：假命中）；interrupted 算——他做了，
只是没做完。

事件触发可能在前因还没封口时就建了（实例是前因的开始时刻）；封口之后前因归成了别的类、或成了非事件（不进行），这次机会就不存在
——``antecedent_holds`` 判它，结算与递送共用（裁定 54：前因不成立 → 作废）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from habitus.foresight.ledger.model import Expectation, Omission, Ruling, Settlement
from habitus.scene.views import FlowRow

#: 不算"做了"的结束方式。
NOT_DONE = frozenset({"abandoned"})


def settle(
    expectation: Expectation,
    rows: Sequence[FlowRow],
    *,
    spoken_at: datetime | None,
    settled_at: datetime,
    current: Callable[[str], Sequence[str]],
) -> Settlement:
    """``rows`` 是从形成那天到最晚时限那天的全部行（按时刻排）；``spoken_at`` 是这条预期第一次被说出的时刻（没说过是 None）。"""

    def record(outcome: str, uri: str | None = None, offset: int | None = None) -> Settlement:
        return Settlement(
            expectation_id=expectation.expectation_id,
            kind_token=expectation.kind_token,
            day=expectation.day,
            settled_at=settled_at,
            outcome=outcome,
            occurrence_uri=uri,
            offset_minutes=offset,
        )

    first = first_occurrence(expectation, rows, current=current)
    if first is None:
        return record("过期待决")
    at = _utc(first.at)
    if spoken_at is not None and _utc(spoken_at) < at:
        return record("提醒后发生", first.uri, _minutes(at - _utc(spoken_at)))
    start, end = expectation.window
    if at < _utc(start):
        return record("早到", first.uri, -_minutes(_utc(start) - at))
    if at <= _utc(end):
        return record("准时", first.uri)
    return record("晚了但做了", first.uri, _minutes(at - _utc(end)))


def first_occurrence(
    expectation: Expectation, rows: Sequence[FlowRow], *, current: Callable[[str], Sequence[str]]
) -> FlowRow | None:
    """``observed_from`` 之后、最晚时限之前这一类第一次发生的那一行。"""

    kinds = _kinds(expectation, current)
    for row in sorted(rows, key=lambda item: _utc(item.at)):
        if row.kind_token in kinds and expectation.covers(_utc(row.at)) and row.status not in NOT_DONE:
            return row
    return None


def antecedent_holds(
    expectation: Expectation,
    antecedent_kind: str | None,
    classes: Sequence[str] | None,
    *,
    current: Callable[[str], Sequence[str]],
) -> bool:
    """这条预期的前因此刻还成不成立。只管事件触发；``antecedent_kind`` 是前因那条记录现在的类（按开始时刻找；找不到——成了
    非事件或被并掉——是 None）；``classes`` 是这条关系的前因类（机会表；关系已不在表里时为 None：说不清，不判作废）。"""

    if expectation.source != "事件" or classes is None:
        return True
    if antecedent_kind is None:
        return False
    wanted = {token for item in classes for token in (item, *current(item))}
    return antecedent_kind in wanted


def rule_overdue(
    expectation: Expectation,
    rows: Sequence[FlowRow],
    *,
    before: datetime | None,
    ruled_at: datetime,
    current: Callable[[str], Sequence[str]],
) -> Ruling | None:
    """过期待决之后：最晚时限之后、``before``（这本账下一条预期形成的时刻；没有就是看到的尽头）之前补做了 → 「后补」。
    下一次机会按账找、不按类找（裁定 48、49）：周四漏跑、周二照常跑，周二是周二那本账的机会，对周四这本账是"下一次之前"
    的补做——是不是真补，交给强度（补做延迟在这类历史间隔里的位置，裁定 51），这里只认方向。
    没补做不裁（等回话或下一次机会）；回话的裁决由调用方按回话记，不经这里。"""

    kinds = _kinds(expectation, current)
    deadline = _utc(expectation.deadline)
    horizon = _utc(before) if before is not None else None
    for row in sorted(rows, key=lambda item: _utc(item.at)):
        at = _utc(row.at)
        if row.kind_token not in kinds or at <= deadline or row.status in NOT_DONE:
            continue
        if horizon is not None and at >= horizon:
            return None
        return Ruling(
            expectation_id=expectation.expectation_id,
            ruled_at=ruled_at,
            ruling="后补",
            occurrence_uri=row.uri,
            delay_minutes=max(1, _minutes(at - deadline)),
        )
    return None


def omissions(
    rows: Sequence[FlowRow],
    expectations: Sequence[Expectation],
    *,
    noticed_at: datetime,
    current: Callable[[str], Sequence[str]],
) -> tuple[Omission, ...]:
    """这一天的行里，发生时没有任何一条在时的预期的那些（只看成了类的行：占位标记不是预测对象）。"""

    found: list[Omission] = []
    by_kind: dict[str, list[Expectation]] = {}
    for item in expectations:
        for token in _kinds(item, current):
            by_kind.setdefault(token, []).append(item)
    for row in rows:
        if row.status in NOT_DONE:
            continue
        candidates = by_kind.get(row.kind_token, [])
        if not candidates and not _is_class(row.kind_token):
            continue
        if any(item.covers(_utc(row.at)) for item in candidates):
            continue
        found.append(
            Omission(
                occurrence_uri=row.uri,
                kind_token=row.kind_token,
                day=row.at.date(),
                noticed_at=noticed_at,
                had_expectation=bool(candidates),
            )
        )
    return tuple(found)


def _kinds(expectation: Expectation, current: Callable[[str], Sequence[str]]) -> set[str]:
    # 原编号总带上：迁移做到一半时，那天的行可能还是原编号
    return {expectation.kind_token, *current(expectation.kind_token)}


def _is_class(token: str) -> bool:
    """类编号形如 ``s-k0013``；「待定」「非事件」这些占位标记不是预测对象。编号规则归词表包，这里只认"有 ``-k`` 这一段"。"""

    _lane, separator, rest = token.partition("-k")
    return bool(separator) and rest.isdigit()


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _minutes(delta: timedelta) -> int:
    return int(round(delta.total_seconds() / 60.0))


__all__ = ["first_occurrence", "omissions", "rule_overdue", "settle"]
