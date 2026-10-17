"""预测层：在背后对这个人每个规律记账，攒够把握之后在"该来了还没来"的那一刻提一句，人说不要就退回去。只到提醒。

方案定稿见桌面 ``事件融合与词表重做/19-预测层方案.md``（裁定 45–51）。三样东西：**预期**（每个机会都建，不设门槛）、
**账**（每一对 触发 → 事 一本，只增不改；递法只读账上的事实）、**递法**（影子 / 提醒两档，级从两条轴的带遗忘 Beta 下界 / 上界
对着可接受误提醒率定的线算出来）。预测器（模型 + 三棵树）可以换，回路不变。

本包只读派生树、不写任何树；不开文件、不拿锁（落盘在组合根 ``runtime/foresight_ledger.py`` 与
``runtime/foresight_opportunities.py``）。模型触点只在 ``judge`` 子包。账本的形状与编解码（``ledger``）、结算规则
（``ledger.settle``）、夜里的机会表（``opportunities``）、此刻的触发与候选（``triggers``）、可靠度（``reliability``）、
反馈强度（``strength``）、递法（``ladder``）。
"""

from habitus.foresight.errors import ForesightError
from habitus.foresight.ladder import Evidence, Standing, Tolerance, account_key, climb, current_facts, decide
from habitus.foresight.model import Moment, NoUnsealed, Seen, UnsealedReader, UnsealedRow
from habitus.foresight.opportunities import OpportunityTable, build_table
from habitus.foresight.triggers import form_expectations

__all__ = [
    "Evidence",
    "ForesightError",
    "Moment",
    "NoUnsealed",
    "OpportunityTable",
    "Seen",
    "Standing",
    "UnsealedReader",
    "Tolerance",
    "UnsealedRow",
    "account_key",
    "build_table",
    "climb",
    "decide",
    "current_facts",
    "form_expectations",
]
