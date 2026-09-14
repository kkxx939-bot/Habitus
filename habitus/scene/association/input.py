"""一次关联调用的输入装配：把行为树、规律级与前因事实凑成六份材料。

**范围不是日历窗口。** 前因只可能落在"上一次这个候选发生"到"这次"之间——更早的话，它应该已经
导致了上一次。这个范围**自适应**：日频就是一天，周频就是一周，月频就是一个月，长度由行为自己的
复发节奏决定。"上一次"取的是规律级**已完成**的日期里最大的那个小于今天的，不回头扫行为树。

第一次发生没有"上一次"，范围退化成当天；那时产出的是"由来"，由编排层决定落到哪一格。

**那一段里的行为怎么压。** 两条确定性筛法都用树自己的数（``CauseFacts``）：树上有转移边指向这个
候选的排在前，其余按全天基线升序（越罕见越前），再截一个上限。数字本身不进提示词。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.scene.association.model import (
    AssociationInput,
    CauseRow,
    DayFacts,
    OccurrenceRow,
    PendingRow,
    SituationRow,
)
from habitus.scene.association.premises import PremiseTable
from habitus.scene.backlog import CauseFacts
from habitus.scene.calendar import DayTypeCalendar
from habitus.scene.regularity.overview import Overview
from habitus.scene.regularity.store import RegularityTree

#: 一次调用最多摊开多少条前因候选、多少条未兑现前提。两道都是渲染预算，不是失效判断。
MAX_CAUSE_ROWS = 12
#: 同一个动作最多占几格。没有它，一个高频动作会把整份前因候选吃光。
MAX_CAUSE_ROWS_PER_ACTION = 2
MAX_PENDING_ROWS = 8


def build_association_input(
    kind_token: str,
    day: date,
    *,
    behavior_tree: BehaviorTree,
    regularity_tree: RegularityTree,
    premises: PremiseTable,
    causes: CauseFacts,
    calendar: DayTypeCalendar,
    max_cause_rows: int = MAX_CAUSE_ROWS,
    max_pending_rows: int = MAX_PENDING_ROWS,
) -> AssociationInput | None:
    """装配这个候选在这一天的一次调用；那天它一次都没发生就返回 None。"""

    documents = _day_stream(behavior_tree, day)
    if not documents:
        return None
    rows = tuple(_row(no, document) for no, document in enumerate(documents, start=1))
    targets = tuple(row.no for row in rows if row.kind_token == kind_token)
    if not targets:
        return None
    overview = Overview.read(regularity_tree, kind_token)
    since = _previous_day(regularity_tree, kind_token, day)
    # 按整批**最晚**的目标筛：同一天早一次、晚一次时，晚的那次本该看得见当天早些建立的前提。
    # 早的那次会不会误用晚建立的前提，由记录层的"前提不能比这次晚"挡着，不靠这里收紧。
    latest = rows[targets[-1] - 1].started_at
    return AssociationInput(
        kind_token=kind_token,
        day=day,
        targets=targets,
        occurrences=rows,
        facts=_facts(day, behavior_tree, calendar),
        situations=tuple(
            SituationRow(no=no, text=situation.text, days=situation.days)
            for no, situation in enumerate(overview.situations, start=1)
        ),
        causes=_cause_rows(behavior_tree, kind_token, since, day, causes, limit=max_cause_rows),
        pending=_pending_rows(premises, kind_token, before=latest, limit=max_pending_rows),
        origin=overview.origin,
    )


def _pending_rows(
    premises: PremiseTable, kind_token: str, *, before: datetime, limit: int
) -> tuple[PendingRow, ...]:
    if limit <= 0:
        return ()
    standing, _signals = premises.waiting_for(kind_token, before=before, limit=limit)
    return tuple(
        PendingRow(
            no=no,
            text=premise.text,
            producer_uri=premise.producer_uri,
            created_on=premise.created_on,
            consumed_by=premise.waits_for,
        )
        for no, premise in enumerate(standing, start=1)
    )


def _day_stream(behavior_tree: BehaviorTree, day: date) -> list[BehaviorDocument]:
    """当天的流，与旧归组同一口径：跳过改名留痕的那些，按时刻升序、同刻按叶名定序。"""

    documents = [
        document
        for document in behavior_tree.read_day(BehaviorKind.OCCURRENCE, day)
        if document.fields.get("original_name") is None
    ]
    documents.sort(key=lambda document: (document.address.started_at.astimezone(UTC), document.address.identity_name))
    return documents


def _row(no: int, document: BehaviorDocument) -> OccurrenceRow:
    fields = document.fields
    return OccurrenceRow(
        no=no,
        uri=str(BehaviorURI.from_address(document.address)),
        name=str(fields["name"]),
        kind_token=str(fields["kind_token"]),
        started_at=document.address.started_at,
        summary=str(fields["summary"]),
        goal=None if fields.get("goal") is None else str(fields["goal"]),
    )


def _facts(day: date, behavior_tree: BehaviorTree, calendar: DayTypeCalendar) -> DayFacts:
    """情境事实全部确定性算出，不经模型：周几、几月、日型、观测空白。"""

    gaps = tuple(
        (gap.address.started_at, datetime.fromisoformat(str(gap.fields["ended_at"])))
        for gap in sorted(behavior_tree.read_day(BehaviorKind.GAP, day), key=lambda item: item.address.started_at)
    )
    return DayFacts(weekday=day.weekday(), month=day.month, day_note=calendar.describe(day), observed_gaps=gaps)


def _previous_day(regularity_tree: RegularityTree, kind_token: str, day: date) -> date | None:
    """这个候选上一次被**关联完成**的那一天。

    它不完全等于"上一次发生"：被封锁的、被预算推迟的、答不全的那些天不在里面，于是 ``since``
    会回退到更早，前因窗口跟着变宽。方向是安全的（只会多看几天，不会漏），代价是真正相关的
    几条可能被更早的挤掉——而每动作配额正是为此设的。
    """

    earlier = [covered for covered in regularity_tree.days_for(kind_token) if covered < day]
    return max(earlier) if earlier else None


def _cause_rows(
    behavior_tree: BehaviorTree,
    kind_token: str,
    since: date | None,
    day: date,
    causes: CauseFacts,
    *,
    limit: int,
) -> tuple[CauseRow, ...]:
    """上一次到这次之间的行为，按前因事实排序后截断。首次发生没有上一次，这一段就是空的。"""

    if since is None or limit <= 0:
        return ()
    documents = [
        document
        for probe in _days_between(since, day)
        for document in _day_stream(behavior_tree, probe)
        if str(document.fields["kind_token"]) != kind_token
    ]
    if not documents:
        return ()
    ordered = causes.ranked_for(kind_token, (str(item.fields["kind_token"]) for item in documents))
    ranked = {fact.action: index for index, fact in enumerate(ordered)}
    # 同一个动作在这一段里可能发生过很多次。不给每个动作配额的话，"每次打球前都在看手机"这种
    # 又高频又高转移的动作会把 12 格全吃掉，体检、约球这类真正值得看的一条都进不来。
    documents.sort(
        key=lambda document: (
            ranked.get(str(document.fields["kind_token"]), len(ranked)),
            -document.address.started_at.timestamp(),
        )
    )
    kept = _with_action_quota(documents, limit, per_action=MAX_CAUSE_ROWS_PER_ACTION)
    kept.sort(key=lambda document: document.address.started_at.astimezone(UTC))
    return tuple(
        CauseRow(
            no=no,
            uri=str(BehaviorURI.from_address(document.address)),
            name=str(document.fields["name"]),
            started_at=document.address.started_at,
            summary=str(document.fields["summary"]),
        )
        for no, document in enumerate(kept, start=1)
    )


def _with_action_quota(documents: list[BehaviorDocument], limit: int, *, per_action: int) -> list[BehaviorDocument]:
    """每个动作最多占 ``per_action`` 格，取到 ``limit`` 为止。

    **余量不回填。**只有两三个动作时，回填会让那 12 行变成十来条几乎一样的"看手机"。上限是预算
    不是指标：少而不同的几行，比多而重复的一堆更能让模型看出前因。
    """

    used: dict[str, int] = {}
    kept: list[BehaviorDocument] = []
    for document in documents:
        action = str(document.fields["kind_token"])
        if used.get(action, 0) >= per_action:
            continue
        used[action] = used.get(action, 0) + 1
        kept.append(document)
        if len(kept) == limit:
            break
    return kept


def _days_between(since: date, day: date) -> Iterable[date]:
    probe = since
    while probe < day:
        yield probe
        probe += timedelta(days=1)


__all__ = ["MAX_CAUSE_ROWS", "MAX_PENDING_ROWS", "build_association_input"]
