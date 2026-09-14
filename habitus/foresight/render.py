"""把一个候选的证据渲染成 Markdown：给人看，也给模型看。

展示与逻辑分开：本模块**一个数都不算**，只把 ``assemble`` 装配好的东西摆出来。所以改渲染
永远不会改判断依据，而判断依据变了这里会立刻少一块——这正是要拆开的理由。

两条呈现纪律：

- **数字连着它的出处**。每一层写成"3/4（3 天）"而不是"0.75"：判断者要能看出这个数薄不薄。
- **没说的要说出来**。还没关联完成的日子与被保护闸截掉的日子都单独写一行；
  不写的话，"给你看的这几条"会被读成"一共就这几条"。

保护闸按**从远到近**砍：先砍全天层的逐条实例，再跨周几、邻域，最后才是本槽——离此刻越远的
层，少看几条损失越小。四层的**数字与出处一个都不砍**，砍掉多少如实写在正文里。
"""

from __future__ import annotations

from habitus.foresight.assemble import CandidateEvidence
from habitus.foresight.context import LayerBackground
from habitus.foresight.errors import ForesightError
from habitus.scene.views import ActionRef, ContextView, Neighbour

# 砍的次序：离此刻最远的先砍。
_TRIM_ORDER = ("all_day", "cross_weekday", "pool", "slot")
_WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def render_candidate(evidence: CandidateEvidence, *, max_chars: int | None = None) -> str:
    """一个候选的完整证据。``max_chars`` 是保护闸，给了就按上面的次序砍到放得下。"""

    if max_chars is not None and (isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0):
        raise ForesightError("max_chars must be a positive integer or None")
    budget = {layer.layer.name: len(layer.views) for layer in evidence.layers}
    text = _render(evidence, budget)
    if max_chars is None:
        return text
    for name in _TRIM_ORDER:
        while len(text) > max_chars and budget[name] > 0:
            budget[name] -= 1
            text = _render(evidence, budget)
        if len(text) <= max_chars:
            break
    return text


def _render(evidence: CandidateEvidence, budget: dict[str, int]) -> str:
    lines = [f"## {evidence.kind_token}", "", _moment(evidence), "", *_table(evidence)]
    for layer in evidence.layers:
        lines.extend(["", *_layer(layer, budget[layer.layer.name], evidence.moment.at.year)])
    return "\n".join(lines)


def _moment(evidence: CandidateEvidence) -> str:
    moment = evidence.moment
    head = f"此刻：{moment.at:%Y-%m-%d} {_WEEKDAYS[moment.weekday]} {moment.at:%H:%M}（第 {moment.slot} 槽）"
    return head if moment.day_note is None else f"{head}；当地日历：{moment.day_note}"


def _table(evidence: CandidateEvidence) -> list[str]:
    rows = ["| 层 | 数 | 出处 | 没背景 |", "| --- | --- | --- | --- |"]
    for item in evidence.layers:
        layer = item.layer
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


def _layer(item: LayerBackground, shown: int, year: int) -> list[str]:
    layer = item.layer
    if not layer.days:
        return [f"### {layer.label} · 没有出处日", "（这一层一天都没发生过，没有历史可看）"]
    lines = [f"### {layer.label} · 这 {len(layer.days)} 天当时的情形"]
    notes = []
    if layer.unassociated:
        notes.append(f"{len(layer.unassociated)} 天有数、语义层还没关联，没有那句上下文")
    if item.dropped_days:
        notes.append(f"更早的 {item.dropped_days} 天没有展开")
    trimmed = len(item.views) - shown
    if trimmed > 0:
        notes.append(f"另有 {trimmed} 条因篇幅未展开")
    if notes:
        lines.append(f"（{'；'.join(notes)}）")
    if shown <= 0:
        return lines
    lines.extend(f"- {_view(view, year)}" for view in item.views[-shown:])
    return lines


def _view(view: ContextView, year: int) -> str:
    # 名字与 kind 不是一回事（"打球"这个 kind 那天叫"去球场打球"），历史条目要按名字给。
    # 天数上限管的是**天数**不是跨度：一个月频行为的最近 40 天能横跨三年，只写月日的话
    # 两条 "06-20" 会来自不同年份，而"此刻"那行是带年份的，对比之下更容易读错。
    stamp = f"{view.at:%m-%d %H:%M}" if view.at.year == year else f"{view.at:%Y-%m-%d %H:%M}"
    parts = [f"{stamp} {view.name}" if view.name else stamp]
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
    return " ｜ ".join(parts)


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


__all__ = ["render_candidate"]
