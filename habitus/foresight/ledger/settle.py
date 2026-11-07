"""对答案（``19-预测层方案.md`` 第五节）：一条预期 vs 行为树上那几天的行。纯函数，不认识存储、不读时钟。

判据：``observed_from`` 之后、最晚时限之前这一类**第一次**发生（形成时已经做了的也算：每个机会都进账，裁定 49）——

- 开口之后才发生：「提醒后发生」，不进自然命中，记提醒到发生的延迟；
- 时窗开始之前：「早到」（记早了几分钟，负数）；时窗之内：「准时」；时窗之后、最晚时限之前：「晚了但做了」（记晚了几分钟）；
- 到最晚时限没发生：「过期待决」，不是终态——之后在这本账的下一次机会之前补做了就是「后补」，回话另记。

"该行为"按 ``current``（编号 → 现在对应哪些在用编号）认：预期记的是形成时的编号，结算时词表可能已拆改。
只认真做成的：occurrence 的 status 是 abandoned（放弃了）的那一行不算这件事发生了（裁定 54：假命中）；interrupted 算——他做了，
只是没做完。

事件触发可能在前因还没封口时就建了（实例是前因的开始时刻）；封口之后前因归成了别的类、或成了非事件（不进行），这次机会就不存在
——``antecedent_holds`` 判它（裁定 54：前因不成立 → 作废）。结算与递送开口前对"前因还成立吗"读同一份：前因那一行按实例找、
类按 ``current`` 认词表拆改（``event_antecedent_holds``）；对"做了没"也是同一个口径（``counts_as_done``：说中的范围里、
这一类按 ``current`` 认、没放弃）。只有一处两边不同：情境要不要按最新的日历重核——结算重核（``antecedent_holds_when_settling``），
递送开口前不重核（``antecedent_holds_when_delivering``）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import NamedTuple

from habitus.foresight.ledger.model import Expectation, Omission, Ruling, Settlement
from habitus.foresight.opportunities.model import RelationOpportunities
from habitus.scene.views import FlowRow

#: 不算"做了"的结束方式。
NOT_DONE = frozenset({"abandoned"})


class AntecedentRow(NamedTuple):
    """找前因时一行要的三样：开始、最后所见、现在的类（结算从树上的行取，递送从这一拍读到的行取）。"""

    started_at: datetime
    last_seen_at: datetime
    kind_token: str


def settle(
    expectation: Expectation,
    rows: Sequence[FlowRow],
    *,
    spoken_at: datetime | None,
    settled_at: datetime,
    current: Callable[[str], Sequence[str]],
) -> Settlement:
    """``rows`` 是从起算那天（``observed_from``；晚形成的预期，起算可能早于形成那天）到最晚时限那天的全部行（按时刻排）；
    ``spoken_at`` 是这条预期第一次被说出的时刻（没说过是 None）。"""

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

    for row in sorted(rows, key=_row_at):
        if counts_as_done(expectation, row.kind_token, row.at, abandoned=row.status in NOT_DONE, current=current):
            return row
    return None


def counts_as_done(
    expectation: Expectation,
    kind_token: str,
    started_at: datetime,
    *,
    abandoned: bool,
    current: Callable[[str], Sequence[str]],
) -> bool:
    """一条 ``started_at`` 开始、类为 ``kind_token`` 的记录算不算这条预期说的那件事做了：落在说中的范围里（起算之后、最晚时限之前）、
    这一类按 ``current`` 认（原编号也算：迁移做到一半时那天的行可能还是原编号）、没放弃。结算（``first_occurrence``）与递送判
    "这回已经做了"（``foresight.delivery.done_since``）同一个口径。"""

    return kind_token in _kinds(expectation, current) and expectation.covers(_utc(started_at)) and not abandoned


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


def event_antecedent_holds(
    expectation: Expectation,
    relation: RelationOpportunities | None,
    *,
    rows_on: Callable[[date], Iterable[AntecedentRow]],
    current: Callable[[str], Sequence[str]],
) -> bool:
    """事件触发：前因那条记录（按实例——前因的开始时刻——在那天与前后各一天里找，封口后的样子）现在还属于这条关系的前因类吗。
    先找开始时刻对得上的；封口时前因的开始可能被更正挪了一点，就找跨过那一刻的那一行（行为有先后、不会两行同时）。类按 ``current``
    认词表拆改。不是事件触发、关系已不在表里（或没有机会表）、实例读不出时刻的：说不清，不判作废。``rows_on`` 给某一天的行，
    只在用得着时才问。结算与递送开口前共用这一个。"""

    if expectation.source != "事件" or relation is None:
        return True
    try:
        started = datetime.fromisoformat(expectation.trigger_instance).astimezone(expectation.formed_at.tzinfo)
    except ValueError:
        return True
    moment = started.astimezone(UTC)
    rows = [row for offset in (0, -1, 1) for row in rows_on(started.date() + timedelta(days=offset))]
    found = next((row.kind_token for row in rows if _utc(row.started_at) == moment), None)
    if found is None:
        found = next(
            (row.kind_token for row in rows if _utc(row.started_at) <= moment <= _utc(row.last_seen_at)),
            None,
        )
    return antecedent_holds(expectation, found, relation.antecedent_classes, current=current)


def antecedent_holds_when_settling(
    expectation: Expectation,
    relation: RelationOpportunities | None,
    *,
    official: bool,
    calendar_on: Callable[[date], Iterable[tuple[str, str]]],
    rows_on: Callable[[date], Iterable[AntecedentRow]],
    current: Callable[[str], Sequence[str]],
) -> bool:
    """结算时这次机会的前因还成不成立。``relation`` 是机会表里这条预期的关系（关系已不在表里、或没有机会表：说不清，不判作废）；
    ``official`` 是机会表对没见过的日型按不按官方含义归组；``calendar_on`` 按最新的日历给某一天的日历情境，``rows_on`` 给某一天
    树上的行，两者只在用得着时才问。

    - 情境触发：那一天按最新的日历还处在这个日历情境里吗（放假数据后来补上了，原来按名义当成工作日的那天就作废）；
    - 事件触发：``event_antecedent_holds``（与递送开口前同一个）。"""

    if expectation.source == "情境":
        if relation is None or relation.calendar_key is None:
            return True
        return relation.calendar_holds(dict(calendar_on(expectation.day)), official=official)
    return event_antecedent_holds(expectation, relation, rows_on=rows_on, current=current)


def antecedent_holds_when_delivering(
    expectation: Expectation,
    relation: RelationOpportunities | None,
    *,
    rows_on: Callable[[date], Iterable[AntecedentRow]],
    current: Callable[[str], Sequence[str]],
) -> bool:
    """开口之前再核一遍事件触发的前因：它可能是没封口时建的，封口后归成了别的类、或成了非事件（找不到了）；``rows_on`` 给某一天
    这一拍读到的行。找法与类的映射和结算同一个（``event_antecedent_holds``）。情境不按日历重核（结算重核；递送要不要也重核
    待定）。"""

    return event_antecedent_holds(expectation, relation, rows_on=rows_on, current=current)


def rule_overdue(
    expectation: Expectation,
    rows: Sequence[FlowRow],
    *,
    before: datetime | None,
    ruled_at: datetime,
    current: Callable[[str], Sequence[str]],
) -> Ruling | None:
    """过期待决之后：最晚时限之后、``before``（这本账下一条预期开始看的时刻；没有就是看到的尽头）之前补做了 → 「后补」。
    下一次机会按账找、不按类找（裁定 48、49）：周四漏跑、周二照常跑，周二是周二那本账的机会，对周四这本账是"下一次之前"
    的补做——是不是真补，交给强度（补做延迟在这类历史间隔里的位置，裁定 51），这里只认方向。
    没补做不裁（等回话或下一次机会）；回话的裁决由调用方按回话记，不经这里。"""

    kinds = _kinds(expectation, current)
    deadline = _utc(expectation.deadline)
    horizon = _utc(before) if before is not None else None
    for row in sorted(rows, key=_row_at):
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


def _row_at(row: FlowRow) -> datetime:
    return _utc(row.at)


def _minutes(delta: timedelta) -> int:
    return int(round(delta.total_seconds() / 60.0))


__all__ = [
    "NOT_DONE",
    "AntecedentRow",
    "antecedent_holds",
    "antecedent_holds_when_delivering",
    "antecedent_holds_when_settling",
    "counts_as_done",
    "event_antecedent_holds",
    "first_occurrence",
    "omissions",
    "rule_overdue",
    "settle",
]
