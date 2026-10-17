"""一刻的装配（方案第四节前半）：最近几天的行 → 时间线 → 三种触发出草稿 → 每条附反向前因清单 → 账本的预期。

读口：行为树经 ``scene.views`` 的 ``DayIndexCache``（已封口的行），还没封口的判断经 ``UnsealedReader`` 读好传进来。
回看 7 天（第 2–7 天那一段最远看到 7 天前）。预期身份 = 类 + 触发种类 + 触发身份 + 触发实例，**不含机会表那一晚、不含日子**：
同一个触发实例（同一条前因、同一个上次的同一簇、同一天）全账本只建一条，哪一晚的表只记在 ``night`` 字段上；跨天的触发
（次日、第 2–7 天、离上次够久）第二天重算出同一个身份，账本按身份查重、不再写一份。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger.model import Expectation, expectation_identity
from habitus.foresight.model import Moment, Seen, UnsealedRow
from habitus.foresight.opportunities.model import OpportunityTable
from habitus.foresight.triggers.clock import clock_drafts
from habitus.foresight.triggers.days import day_drafts
from habitus.foresight.triggers.draft import Draft
from habitus.foresight.triggers.events import antecedents_of, event_drafts
from habitus.foresight.triggers.marks import absolute_slot, timeline_of
from habitus.prediction.model import PredictionTree
from habitus.scene.views import DayIndexCache

LOOKBACK_DAYS = 7


def seen_rows(moment: Moment, cache: DayIndexCache, unsealed: Sequence[UnsealedRow]) -> tuple[Seen, ...]:
    """回看窗内到此刻为止开始过的事：已封口的行 + 还没封口、归上了类的判断，按开始时刻排。"""

    now = moment.at
    rows: list[Seen] = []
    for offset in range(LOOKBACK_DAYS, -1, -1):
        day = moment.day - timedelta(days=offset)
        rows.extend(
            Seen(row.at, min(row.last_observed_at, now), row.kind_token, row.uri, row.name, row.summary or "")
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
    tree: PredictionTree | None,
    table: OpportunityTable | None,
    rows: Sequence[Seen],
    shadow_unestablished: bool = False,
    generation: str = "",
    version: str = "",
    implementation: str = "",
) -> tuple[Expectation, ...]:
    """此刻对得上历史的全部预期。``tree`` 为 None 就没有钟面触发，``table`` 为 None 就没有事件与日子触发（也没有前因清单）。"""

    if tree is not None and table is not None and tree.slot_minutes != table.slot_minutes:
        raise ForesightError("the prediction tree and the opportunity table must share one slot width")
    if tree is not None and moment.slot != (moment.at.hour * 60 + moment.at.minute) // tree.slot_minutes:
        raise ForesightError("moment.slot must be mapped with the prediction tree's slot width")
    done_today = frozenset(row.kind_token for row in rows if row.day == moment.day)
    statuses = ("established", "candidate") if shadow_unestablished else ("established",)
    # TODO(FORESIGHT-EVENT-FIRST-001)：触发改成以行为事件为主、时间与天气等降为外部条件（裁定 53，用户 2026-10-17：
    # 只记、不在这一版改）。本 TODO 是代码侧唯一规格来源，另两处只引用它。
    # 为什么：零模型全年盘 7,237 条预期，挂在时间上的占 95%（离上次够久 2,998 自然命中 35%、钟面周几 2,793 中 18%、
    # 几号 1,058 中 13%），从前因出发的事件触发只有 388 条却最准（days_2_7 41%）；钟面账在节假日照样给"周二跑步"开窗。
    # 改法：
    # ① 事件为主：这个后果在语义树上有成立的前因关系时，不再为它建钟面草稿；时窗从前因起算（events.py 已是这样）。
    #    节假日没有"下班到家"，预期自然不建，裁定 52 的"日型进小包"随之降级。
    # ② 钟面降为兜底：只在后果没有任何成立前因时建（前因观测不到，如起床要等物理 lane）；几号 / 离上次够久照旧，
    #    哪本账可靠由账自己分出来（裁定 49），不另判"外部约束类"。
    # ③ 外部条件进关系：日型、天气、地点、同在的人都当语义树关系的调节条件（scene/relations/conditions.py），
    #    机会表接带条件的关系（opportunities/build.py 现在整条跳过），账键带上条件（"到家 → 跑步｜不下雨"各记一本），
    #    触发时判此刻条件成不成立；条件合的那本账自己攒得起来就开口，就是 2026-09-17"外部条件调门控"的落法，不另写公式。
    # ④ 天气要新加外部事实来源接缝：必须答得出"那天那时是多少"（有历史），只有预报的源不能用；日历接 scene 的日型接缝。
    # 影响：triggers/assemble.py、clock.py、opportunities/build.py、ledger 的 account_key、conditions.Field、组合根注入事实源。
    # 风险：每多一个条件切一刀，跑步全年 68 次，调节检验要前因 ≥ 40（moderation_min_antecedents）才开，带条件的账要攒
    # 几个月；条件源越多候选越多，交并检验不管总共试了多少条，接源时一起定是否做多重比较校正；造的数据按钟面埋习惯、
    # 没有天气，测不了这个转向，要先造事件驱动与天气调节的习惯。
    drafts: list[Draft] = []
    line = None
    now_slot = 0
    if table is not None:
        line = timeline_of(
            rows, lane=table.lane, slot_minutes=table.slot_minutes, horizon=moment.day + timedelta(days=1)
        )
        now_slot = absolute_slot(moment.at, slot_minutes=table.slot_minutes)
        drafts.extend(event_drafts(table, line, moment, now_slot=now_slot, statuses=statuses))
    if tree is not None:
        drafts.extend(clock_drafts(tree, table, moment, done_today=done_today))
    if table is not None:
        drafts.extend(day_drafts(table, moment, done_today=done_today))
    found: dict[str, Expectation] = {}
    for draft in drafts:
        antecedents = (
            antecedents_of(table, draft.kind_token, line, moment, now_slot=now_slot, statuses=statuses)
            if table is not None and line is not None
            else ()
        )
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
            basis=draft.basis,
            opportunities=draft.opportunities,
            hits=draft.hits,
            observed_from=draft.observed_from,
            antecedents=antecedents,
            generation=generation,
            night=table.night if table is not None else None,
            version=version,
            implementation=implementation,
        )
    return tuple(found[key] for key in sorted(found))


__all__ = ["LOOKBACK_DAYS", "form_expectations", "seen_rows"]
