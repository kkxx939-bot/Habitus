"""把证据渲染成 Markdown：给人看，也给模型看。

展示与逻辑分开：本模块**一个数都不算**，只把 ``assemble`` 装配好的东西摆出来。所以改渲染
永远不会改判断依据，而判断依据变了这里会立刻少一块——这正是要拆开的理由。

三条呈现纪律：

- **数字连着它的出处**。每一层写成"3/4（3 天）"而不是"0.75"：判断者要能看出这个数薄不薄。
- **没说的要说出来**。还没关联完成的日子、被保护闸截掉的日子与卡都单独写一行；
  不写的话，"给你看的这几条"会被读成"一共就这几条"。
- **卡上之前、这次、之后三段分开**。判断者比的是"之前"那段像不像此刻，"之后"那段是接下来会是什么。
- **卡有编号**。每个候选的卡各自从 #1 编，判断者的 ``basis`` 引用的就是这个编号；装配层按同一序换回 URI。

**不砍**（2026-09-16 定）：方案还在实施阶段、效果没确认，先不做任何预算与截断；包里有什么就渲什么。
"""

from __future__ import annotations

from datetime import UTC, datetime

from habitus.foresight.assemble import CandidateEvidence, EvidencePack
from habitus.foresight.cards import HistoryCard, NowScene
from habitus.foresight.model import LAYER_LABELS, CandidateNumbers, Moment
from habitus.scene.views import ActionRef, ContextView, FlowRow, Neighbour, ObservationGap, Situation

_WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def render_candidate(evidence: CandidateEvidence) -> str:
    """一个候选的完整证据。"""

    return "\n".join(_candidate_lines(evidence))


def render_pack(pack: EvidencePack) -> str:
    """整包：此刻场景在前，摊开的候选各一节（每节数字在前、卡在后），只列名的候选收在最后一节。"""

    lines = [f"# 证据包 · 一代 {pack.generation}", "", render_moment(pack.moment), _clock_face(pack), "", *_now(pack.now)]
    for item in pack.expanded:
        lines.extend(["", *_candidate_lines(item)])
    named = pack.named_only
    if named:
        lines.extend(["", "## 只列名的候选（这个时段附近没发生过）"])
        lines.extend(
            f"- {item.kind_token}：边际 {item.numbers.marginal:.3f} · 全天 {item.provenance.all_day.value:.4f}"
            f" · 出处 {len(item.provenance.all_day.days)} 天"
            for item in named
        )
    return "\n".join(lines)


def render_moment(moment: Moment) -> str:
    """"此刻"那一行：日期、周几、时分、槽号，有当地日历就带上。"""

    at = moment.at
    head = f"此刻：{at:%Y-%m-%d} {_WEEKDAYS[moment.weekday]} {at:%H:%M}（第 {moment.slot} 槽）"
    return head if moment.day_note is None else f"{head}；当地日历：{moment.day_note}"


def _clock_face(pack: EvidencePack) -> str:
    """钟面那一行：判断者说的"第几槽"都按这个数。"""

    return f"钟面：槽宽 {pack.slot_minutes} 分钟，一天 {pack.slots_per_day} 槽，第 0 槽从 00:00 起"


def _now(scene: NowScene) -> list[str]:
    """此刻场景：已封口与未封口的行合成一条时间线，按时刻排；空白两类都列。"""

    lines = [f"## 此刻场景（{scene.since:%H:%M}–{scene.moment.at:%H:%M}）"]
    timeline: list[tuple[datetime, str]] = [(row.at, _row(row)) for row in scene.flow]
    for row in scene.unsealed:
        tag = "（未封口）" if row.kind_token is not None else "（未封口、未归类）"
        timeline.append((row.started_at, f"{row.started_at:%H:%M} {row.name}{tag}"))
    timeline.sort(key=lambda item: item[0].astimezone(UTC))
    lines.append("到此刻已发生：" + (" ｜ ".join(text for _at, text in timeline) if timeline else "（无）"))
    counts = scene.done_today
    lines.append(
        "今天做过：" + (" · ".join(f"{kind} {count}" for kind, count in counts.items()) if counts else "（无）")
    )
    gaps = [(gap, False) for gap in scene.gaps] + [(gap, True) for gap in scene.unsealed_gaps]
    if gaps:
        lines.append("观测空白：" + " ｜ ".join(_gap(gap, unsealed=unsealed) for gap, unsealed in gaps))
    return lines


def _gap(gap: ObservationGap, *, unsealed: bool) -> str:
    text = f"{gap.started_at:%H:%M}–{gap.ended_at:%H:%M}（{gap.kind}）"
    return f"{text}（未封口）" if unsealed else text


def _candidate_lines(evidence: CandidateEvidence) -> list[str]:
    """一个候选一节：先数字（四层表、发布的率），再卡，再情形。先候选、再上下文（2026-09-16 定）。"""

    lines = [f"## {evidence.kind_token}", "", *_table(evidence), *_numbers(evidence.numbers)]
    if not evidence.expanded:
        lines.extend(["", "（这个时段附近没发生过，只列名字与数字）"])
        return lines
    lines.extend(["", *_cards(evidence)])
    situations = _situations(evidence)
    if situations:
        lines.extend(["", *situations])
    return lines


def _table(evidence: CandidateEvidence) -> list[str]:
    rows = ["| 层 | 数 | 出处 | 没背景 |", "| --- | --- | --- | --- |"]
    for layer in evidence.provenance:
        if layer.hits is not None and layer.exposure is not None:
            if layer.exposure <= 0.0:
                # 分母为零是"这一格我们根本没看过"（那个周几还没进过观测跨度），不是"看了没发生"。
                # 渲染成 0.00/0.00 = 0.000 会让判断者把它读成一个有把握的实测零。
                value = "从没看过这一格"
            else:
                # 裸比值：分子分母摆出来，判断者自己掂量这个数薄不薄。**这两个是衰减加权量**
                # （权重 = 时间衰减 × 覆盖比例），不是次数，所以写成小数而不是 2/4。
                value = f"{layer.hits:.2f}/{layer.exposure:.2f} = {layer.value:.3f}"
        else:
            value = f"{layer.value:.4f}"
        missing = f"{len(layer.unassociated)} 天" if layer.unassociated else "—"
        if not layer.days and layer.value > 0.0:
            # 真实数据上常见：这一层一天都没发生过，率却不是 0——那是收缩链的 Laplace 先验
            # 在说话。不标出来的话，一个 0.04 会被当成"别的周几这个点会做"的实测结论。
            value = f"{value}（只有先验，没有证据）"
        rows.append(f"| {layer.label} | {value} | {len(layer.days)} 天 | {missing} |")
    return rows


def _numbers(numbers: CandidateNumbers) -> list[str]:
    """树发布的结论那一行，摆在四层推导之后、卡之前。"""

    trend = "—" if numbers.trend is None else f"{numbers.trend:.2f}（证据 {numbers.trend_n_eff:.1f}）"
    published = (
        f"发布的率：边际 {numbers.marginal:.3f} · 危险 {numbers.hazard:.3f} · 累积 {numbers.cumulative:.3f}"
        f" · lift 全天 {numbers.lift_all_day:.1f} / 周几 {numbers.lift_weekday:.1f}"
        f" · 计数 {numbers.count:.2f} · 机会 {numbers.n_eff:.2f} · 趋势 {trend}"
    )
    done = f"今天已做 {numbers.done_today} 次" if numbers.done_today else "今天还没做"
    recurrence = numbers.recurrence
    if recurrence is None:
        return [published, f"复发：没有间隔样本 · {done}"]
    since = "" if recurrence.overdue is None else f" · 距上次占中位数 {recurrence.overdue:.2f}"
    return [
        published,
        f"复发：p10 {_days(recurrence.p10)} · p50 {_days(recurrence.p50)} · p90 {_days(recurrence.p90)} 天"
        f"（样本 {recurrence.sample_count:.1f}）{since} · {done}",
    ]


def _days(seconds: float) -> str:
    return f"{seconds / 86_400.0:.1f}"


def _cards(evidence: CandidateEvidence) -> list[str]:
    background = evidence.background
    total = len(background.cards)
    lines = [f"### 历史 · {total} 次发生，每次一张卡（按 # 编号引用）"]
    notes = []
    unassociated = evidence.unassociated
    if unassociated:
        notes.append(f"{len(unassociated)} 天有数、语义层还没关联，那几天的卡没有关联记录")
    if any(background.dropped_days.values()):
        detail = "、".join(
            f"{LAYER_LABELS[name]} {count}" for name, count in background.dropped_days.items() if count
        )
        notes.append(f"更早的日子没有展开（{detail} 天）")
    if notes:
        lines.append(f"（{'；'.join(notes)}）")
    if total == 0:
        lines.append("（这几天里没有一次落在这个时段附近）")
        return lines
    for number, card in enumerate(background.cards, start=1):
        lines.extend(_card(card, number))
    return lines


def _card(card: HistoryCard, number: int) -> list[str]:
    own = card.own
    head = f"- #{number} {_stamp(card.at)}–{own.last_observed_at:%H:%M} {own.name} 〔{LAYER_LABELS[card.layer]}〕"
    if card.gloss is not None and card.gloss.situation is not None:
        head += f"（{card.gloss.situation}）"
    bits = _view_bits(card.view)
    if bits:
        head += " ｜ " + " ｜ ".join(bits)
    return [
        head,
        "  之前：" + (" ｜ ".join(_row(row) for row in card.before) if card.before else "（无）"),
        "  之后：" + (" ｜ ".join(_row(row) for row in card.after) if card.after else "（无）"),
        "  关联：" + _gloss(card),
    ]


def _gloss(card: HistoryCard) -> str:
    if card.gloss is None:
        return "那天还没关联" if not card.day_associated else "那天关联了，这一次没有记录"
    gloss = card.gloss
    parts = [gloss.context]
    if gloss.causes:
        parts.append("前因：" + "、".join(name for _uri, name in gloss.causes))
    if gloss.consumed:
        parts.append("用掉：" + "、".join(text for _producer, text in gloss.consumed))
    if gloss.left:
        parts.append("留下：" + "、".join(f"{text}（在等：{waits}）" for text, waits in gloss.left))
    return " ｜ ".join(parts)


def _situations(evidence: CandidateEvidence) -> list[str]:
    here, elsewhere = evidence.situations_here, evidence.situations_elsewhere
    if not here and not elsewhere:
        return []
    lines = ["### 情形"]
    if here:
        lines.append("本周几出现过：" + "；".join(_situation(item) for item in here))
    if elsewhere:
        lines.append("只在其他周几出现过：" + "；".join(_situation(item) for item in elsewhere))
    return lines


def _situation(item: Situation) -> str:
    return f"{item.text}（{len(item.days)} 天）"


def _row(row: FlowRow) -> str:
    """一行一条：时分、名字、这个 token 在那一行自己那天的全天次数——"与某人交谈(100)"一眼看出是底噪。"""

    return f"{row.at:%H:%M} {row.name}({row.day_count})"


def _stamp(at: datetime) -> str:
    return f"{at:%Y-%m-%d} {_WEEKDAYS[at.weekday()]} {at:%H:%M}"


def _view_bits(view: ContextView) -> list[str]:
    parts: list[str] = []
    if view.first_of_day:
        parts.append("当天首次")
    if view.preceding is not None:
        parts.append(f"紧邻上一条：{_neighbour(view.preceding)}")
    if view.following is not None:
        parts.append(f"紧邻下一条：{_neighbour(view.following)}")
    if view.causes:
        # 起因是语义边（results_from），与"此前"不是一回事：一个说因果，一个只说先后。
        parts.append("起因：" + "、".join(_label(item) for item in view.causes))
    if view.last_time is not None:
        parts.append(f"距上次 {view.last_time.days_ago} 天")
    if view.concurrent:
        parts.append("同时：" + "、".join(item.name for item in view.concurrent))
    if view.subjects:
        parts.append("和谁：" + "、".join(view.subjects))
    if view.day_note is not None:
        parts.append(f"那天：{view.day_note}")
    return parts


def _label(item: ActionRef) -> str:
    return item.name


def _neighbour(neighbour: Neighbour) -> str:
    """三态照实说：找到了是名字，``∅`` 是"窗口内确实什么都没有"，删失是"那段没看清"。

    把删失说成 ∅ 就是在观测最差的地方下最确凿的结论——树那边为此专门有一条删失规则，
    渲染这边不能又给抹平。
    """

    if neighbour.action is not None:
        return neighbour.action.name
    return "那段没看清（删失）" if neighbour.censored else "没有（∅）"


__all__ = ["render_candidate", "render_moment", "render_pack"]
