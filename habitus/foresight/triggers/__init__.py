"""触发与候选（``19-预测层方案.md`` 第三、四节，第九节第 3 步）：此刻对得上历史就建一条预期，不设门槛。

- **事件**（``events``）：有件事刚发生——语义树成立（或候选）关系的前因此刻在它那一段跨度里，后果就是候选；时窗从机会表里
  历史那几次后果的实际时刻取。反向也在这里：每个候选 → 以它为后果的全部关系，逐条标前因在场 / 不在场、抬高 / 压低。
- **钟面**（``clock``）：到了钟点——预测树这个周几这个行为的时刻分布（p10–p90）此刻开着、今天还没做。
- **日子**（``days``）：到了日子（往月同一个几号做过）/ 离上次够久（距上次落在历史间隔之内），从机会表的日历读。
- **装配**（``assemble``）：把最近几天的行排成一条时间线（链合并与语义树同一把尺），跑三种触发，给每条候选附反向前因清单，
  装成账本的预期。

「没做完」「约定的日子」「处在情境里」三种触发这一步没接（真实 54 天盘上被打断 / 放弃的 17 条，同类再来的 9 条全是另一件事，
类这一级分不出"接着做完"；原话抽日期、长跨度关系都在待办）。本包零模型、零统计；落盘与读树在组合根（``runtime/foresight_tick.py``）。
"""

from habitus.foresight.triggers.assemble import form_expectations
from habitus.foresight.triggers.clock import clock_drafts
from habitus.foresight.triggers.days import day_drafts
from habitus.foresight.triggers.draft import Draft
from habitus.foresight.triggers.events import antecedents_of, event_drafts
from habitus.foresight.triggers.marks import timeline_of

__all__ = ["Draft", "antecedents_of", "clock_drafts", "day_drafts", "event_drafts", "form_expectations", "timeline_of"]
