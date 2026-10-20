"""一刻的装配（方案第四节前半）：最近几天的行 → 时间线 → 三种触发（事件、情境、日子）出草稿 → 每条附反向前因清单 → 账本的预期。

读口：行为树经 ``scene.views`` 的 ``DayIndexCache``（已封口的行），还没封口的判断经 ``UnsealedReader`` 读好传进来。
回看 7 天（第 2–7 天那一段最远看到 7 天前）。预期身份 = 类 + 触发种类 + 触发身份 + 触发实例，**不含机会表那一晚、不含日子**：
同一个触发实例（同一条前因、同一个上次的同一簇、同一天）全账本只建一条，哪一晚的表只记在 ``night`` 字段上；跨天的触发
（次日、第 2–7 天、离上次够久）第二天重算出同一个身份，账本按身份查重、不再写一份。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta

from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger.model import Expectation, expectation_identity
from habitus.foresight.ledger.settle import NOT_DONE
from habitus.foresight.model import Moment, Seen, UnsealedRow
from habitus.foresight.opportunities.model import OpportunityTable
from habitus.foresight.triggers.days import day_drafts
from habitus.foresight.triggers.draft import Draft
from habitus.foresight.triggers.events import antecedents_of, event_drafts
from habitus.foresight.triggers.marks import timeline_of
from habitus.foresight.triggers.situations import situation_drafts
from habitus.scene.views import DayIndexCache
from habitus.temporal.slots import absolute_slot, slot_index

LOOKBACK_DAYS = 7


def seen_rows(moment: Moment, cache: DayIndexCache, unsealed: Sequence[UnsealedRow]) -> tuple[Seen, ...]:
    """回看窗内到此刻为止开始过的事：已封口的行 + 还没封口、归上了类的判断，按开始时刻排。"""

    now = moment.at
    rows: list[Seen] = []
    for offset in range(LOOKBACK_DAYS, -1, -1):
        day = moment.day - timedelta(days=offset)
        rows.extend(
            Seen(
                row.at,
                min(row.last_observed_at, now),
                row.kind_token,
                row.uri,
                row.name,
                row.summary or "",
                abandoned=row.status in NOT_DONE,
                place=row.place,
                subjects=row.subjects,
            )
            for row in cache.day(day).rows
            if row.at <= now
        )
    rows.extend(
        Seen(
            row.started_at,
            min(row.last_observed_at, now),
            row.kind_token,
            f"unsealed:{row.started_at.isoformat()}",
            row.name or "",
            row.summary or "",
        )
        for row in unsealed
        if row.kind_token is not None and row.started_at <= now
    )
    return tuple(sorted(rows, key=lambda item: (item.started_at, item.source)))


def form_expectations(
    *,
    moment: Moment,
    table: OpportunityTable | None,
    rows: Sequence[Seen],
    shadow_unestablished: bool = False,
    generation: str = "",
    version: str = "",
    implementation: str = "",
    recent: int,
    calendar_of: Callable[[date], Mapping[str, str]] | None = None,
) -> tuple[Expectation, ...]:
    """此刻对得上历史的全部预期。``table`` 为 None 就一条也不建（三种触发都从机会表读）。``recent`` 是证据的窗口 M：
    「离上次够久」的时窗只看最近 M 次命中，每条预期的依据只列最近 M 次（窗外的分量已低于 α；账本单条也不会越写越大）。
    ``calendar_of`` 给某一天的日历条件（带条件的关系里日历情境条件白天判它要；不给就只有今天的，取 ``moment.calendar``）。"""

    if isinstance(recent, bool) or not isinstance(recent, int) or recent < 1:
        raise ForesightError("recent must be a positive integer")
    if table is None:
        return ()
    if moment.slot != slot_index(moment.at, slot_minutes=table.slot_minutes):
        raise ForesightError("moment.slot must be mapped with the opportunity table's slot width")
    done_today = frozenset(row.kind_token for row in rows if row.day == moment.day and not row.abandoned)
    statuses = relation_statuses(shadow_unestablished)
    # TODO(FORESIGHT-EVENT-FIRST-001)：触发以行为事件与语义树的关系为主、时间与天气等是外部条件（裁定 53）。本 TODO 是
    # 代码侧唯一规格来源，另两处只引用它。
    # 为什么：零模型全年盘上挂在钟点上的账准度低（周几 18%、几号 13%），从前因出发的事件触发最准（days_2_7 41%）；
    # 钟面账在节假日照样给"周二跑步"开窗。
    # 已做（裁定 59）：钟面、几号两种触发随预测树删了；周几 / 几号 / 日型成了语义树的日历情境关系，走「情境」触发
    # （triggers/situations.py）——锚在工作日的规律节假日自然不建；「离上次够久」读语义树登记的复现。哪本账可靠由账自己
    # 分出来（裁定 49），不另判"外部约束类"。
    # 已做（裁定 61、62）：成立的带条件关系进机会表（opportunities/build.py），账键就是带条件的关系键（"到家 → 跑步｜不下雨"
    #    各记一本），触发时判此刻条件成不成立（triggers/conditions.py）；条件合的那本账自己攒得起来就开口，不另写公式。
    # 还没做：
    # ② 天气接进外部条件包（habitus/conditions，同一个 FactProvider 协议）：必须答得出"那天那时是多少"（有历史），只有
    #    预报的源不能用；语义树的调节条件按字段读（scene/relations/conditions.py）。
    # 影响：conditions.Field、组合根注入事实源。
    # 风险：每多一个条件切一刀，跑步全年 68 次，调节检验开不开由每条关系自己的最小可达 p 定（每边至少 2 次，``conditions.ready_to_test``），带条件的账要攒
    # 几个月；条件源越多候选越多，交并检验不管总共试了多少条，接源时一起定是否做多重比较校正；现有造的数据没有天气，
    # 测不了这一步，要先造天气调节的习惯。
    line = timeline_of(
        rows,
        lane=table.lane,
        slot_minutes=table.slot_minutes,
        horizon=moment.day + timedelta(days=1),
        conditions=tuple(spec for relation in table.relations for spec in relation.conditions),
        calendar_of=calendar_of
        if calendar_of is not None
        else lambda day: moment.calendar if day == moment.day else {},
    )
    now_slot = absolute_slot(moment.at, slot_minutes=table.slot_minutes)
    drafts: list[Draft] = [
        *event_drafts(table, line, moment, now_slot=now_slot, statuses=statuses),
        *situation_drafts(table, moment, done_today=done_today, statuses=statuses),
        *day_drafts(table, moment, done_today=done_today, recent=recent),
    ]
    found: dict[str, Expectation] = {}
    for draft in drafts:
        antecedents = antecedents_of(table, draft.kind_token, line, moment, now_slot=now_slot, statuses=statuses)
        identity = expectation_identity(draft.kind_token, draft.source, draft.trigger, draft.trigger_instance)
        found[identity] = Expectation(
            expectation_id=identity,
            kind_token=draft.kind_token,
            day=moment.day,
            formed_at=moment.at,
            source=draft.source,
            trigger=draft.trigger,
            trigger_instance=draft.trigger_instance,
            window=draft.window,
            deadline=draft.deadline,
            basis=draft.basis[-recent:],
            opportunities=draft.opportunities,
            hits=draft.hits,
            observed_from=draft.observed_from,
            antecedents=antecedents,
            generation=generation,
            night=table.night,
            version=version,
            implementation=implementation,
        )
    return tuple(found[key] for key in sorted(found))


def relation_statuses(shadow_unestablished: bool) -> tuple[str, ...]:
    """建预期与重标单元历史认哪几种关系状态（同一份，裁定 56 第 6 条）：成立的；``shadow_uses_unestablished`` 开着时加候选。"""

    return ("established", "candidate") if shadow_unestablished else ("established",)


__all__ = ["LOOKBACK_DAYS", "form_expectations", "relation_statuses", "seen_rows"]
