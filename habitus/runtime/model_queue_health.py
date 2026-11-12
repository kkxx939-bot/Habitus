"""模型队列在健康面上要人看的（``runtime.health`` 读；都是 non-critical，清完、恢复后各项自己消掉）：

- ``model_queue_backlog``：欠账到了阈值（项数，或最老一项等了多久；``scene.model_queue_backlog_warning_*``），带各类条数；
- ``model_queue_backoff``：正在整体退避（模型整个调不通），带从什么时候起、连着几次、下一次什么时候再做；
- ``model_queue_crossed``：有活欠着跨过了换代（哪几类、跨了几代）——那几代是按"不知道"算出来的，队列补上之后下一代用上；
- ``model_queue_scan``：有的类认欠账时读盘出错（哪几类、原因码）——那一类的活这一趟没看成，后面的类照做。

另有一项读数（不是故障）``model_queue_pending``：待生效的拆改与概念各几份、先验两种问法各要问几题（``queue_pending``）。
"""

from __future__ import annotations

from datetime import datetime

from habitus.runtime.model_queue_state import ModelQueueState, QueueSettings


def queue_problems(state: ModelQueueState, settings: QueueSettings, now: datetime) -> list[tuple[str, str]]:
    """（检查项的名字, 说明）；没有要报的是空。"""

    owed = {category: count for category, count in state.owed_now().items() if count}
    found: list[tuple[str, str]] = []
    waited = state.oldest_wait_seconds(now)
    if sum(owed.values()) >= settings.warning_items or (owed and waited >= settings.warning_wait_seconds):
        counts = ", ".join(f"{category}={count}" for category, count in owed.items())
        found.append(("model_queue_backlog", f"owed {sum(owed.values())}: {counts}; oldest waited {waited:.0f}s"))
    if state.down() and state.unreachable_since is not None:
        later = "" if state.retry_at is None else f"; next attempt at {state.retry_at.isoformat()}"
        found.append(
            (
                "model_queue_backoff",
                f"model unreachable since {state.unreachable_since.isoformat()}; failures={state.failures}{later}",
            )
        )
    crossed = {category: count for category, count in state.crossed.items() if count and owed.get(category)}
    if crossed:
        detail = ", ".join(f"{category}×{count}" for category, count in sorted(crossed.items()))
        found.append(("model_queue_crossed", f"work owed across generations: {detail}"))
    if state.errors:
        detail = ", ".join(f"{category}: {code}" for category, code in sorted(state.errors.items()))
        found.append(("model_queue_scan", f"cannot read what is owed: {detail}"))
    return found


def queue_pending(state: ModelQueueState) -> str:
    """待生效的东西与先验要问的题各有几份（实时化阶段 2「生效时点」「先验混合问法」；不是故障，健康面上记一项读数）：
    待生效的拆改计划、待生效的概念（都等下一次换代）、先验两种问法各要问几题。一样都没有是空串。"""

    labels = (
        ("revisions", "pending revisions"),
        ("concepts", "pending concepts"),
        ("priors_by_consequent", "prior questions by consequent"),
        ("priors_by_antecedent", "prior questions by antecedent"),
    )
    return ", ".join(f"{label}={state.pending[key]}" for key, label in labels if state.pending.get(key))


__all__ = ["queue_pending", "queue_problems"]
