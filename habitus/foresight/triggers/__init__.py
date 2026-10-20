"""触发与候选（``19-预测层方案.md`` 第三、四节，第九节第 3 步）：此刻对得上历史就建一条预期，不设门槛。

- **事件**（``events``）：有件事刚发生——语义树成立（或候选）关系的前因此刻在它那一段跨度里，后果就是候选；时窗从机会表里
  历史那几次后果的实际时刻取。反向也在这里：每个候选 → 以它为后果的全部关系，逐条标前因在场 / 不在场、抬高 / 压低。
- **情境**（``situations``）：今天处在语义树上成立的日历情境里（周三、20 号、工作日……）且到了时窗——历史上条件成立那几天
  后果自然发生的时刻（p10–p90），从机会表的情境段读（裁定 59）。
- **日子**（``days``）：离上次够久——距上次落在历史间隔成的簇里，从机会表登记的复现读（``temporal.cluster_at``）。
- **装配**（``assemble``）：把最近几天的行排成一条时间线（链合并与语义树同一把尺），跑三种触发，给每条候选附反向前因清单，
  装成账本的预期。

「没做完」「约定的日子」两种触发这一步没接（真实 54 天盘上被打断 / 放弃的 17 条，同类再来的 9 条全是另一件事，
类这一级分不出"接着做完"；原话抽日期、长跨度关系都在待办）。本包零模型、零统计；落盘与读树在组合根（``runtime/foresight_tick.py``）。
"""

from habitus.foresight.triggers.assemble import form_expectations
from habitus.foresight.triggers.days import day_drafts
from habitus.foresight.triggers.draft import Draft
from habitus.foresight.triggers.events import antecedents_of, event_drafts
from habitus.foresight.triggers.marks import timeline_of
from habitus.foresight.triggers.situations import situation_drafts

__all__ = [
    "Draft",
    "antecedents_of",
    "day_drafts",
    "event_drafts",
    "form_expectations",
    "situation_drafts",
    "timeline_of",
]
