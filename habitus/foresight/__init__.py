"""预测层：把一刻的证据装成一个包——候选、四层数字、每次发生一张历史卡、此刻场景——给判断用。

本包只读派生树、不写任何树（行为树永远只有观测→融合→归约那一个写入口）。包根这一层零模型：候选与
四层拆解、发布的率、历史卡（邻域序列 + 视图）、此刻场景、整包的 Markdown 渲染都是纯函数。
旧语义树的关联记录随重构摘掉（2026-09-26）；新语义树的产物接回来时卡的形状再变。
判断（LLM）在子包 ``habitus.foresight.judge``，是本层唯一的模型触点；节奏与存储按实施方案第六步接上。
"""

# TODO(FORESIGHT-PERF-001): 按体量会变成问题的开销（2026-09-12 三方审查实测；方案确认阶段不谈成本，
# 等真实数据跑起来再说——用户裁定 2026-09-15）。
# 1. **每一拍都重新解码整棵树**（``runtime/foresight.py`` 的 ``EvidenceAssembler.assemble`` →
#    ``store.load_generation``）：读整个 tree.json、``json.loads``、``codec.decode``，再 ``CellIndex.of``
#    全量扫一遍。第六步接上 ``ForesightWorker`` 之后这是每个槽一次（15 分钟槽宽一天 96 次），不再是
#    脚本跑一次。按 ``prediction/model.py`` 自己的估算，一年后这棵树约 33 MiB。改造：代是不可变的，按
#    ``(generation, digest)`` 记住 ``(tree, CellIndex)`` 即可；"一次查询一份"那条纪律说的是
#    ``DayIndexCache``（要读今天），不是树。
# 2. **四层各自调一次 ``history_contexts``**（``context.py``）。卡已按 URI 去重（同一次发生只投影
#    一次），但四层的筛选仍各跑一遍，``_project → neighbours`` 也会为每条视图重排三天的时间线。
#

# TODO(FORESIGHT-LOSS-001): 下半环的"loss 门控"方案（《Loss 与门控方案》，2026-09-19）**已被《语义树重构》
# 取代**（2026-09-25 定）：不再做条件相合度的 loss，也不做关联长出 relevant_conditions 那两刀。留下的两样在
# 新方案里各有位置——
#   - 账本（``foresight/ledger``：带时窗的「会」落承诺、封口后结算为 验证/偏离/落空）是新方案里"负样本自产"的那本
#     账，将并入 ``scene/ledger``（假设触发的承诺与判断者的承诺形状相同，来源不同；合不合待跑起来再定）。
#   - 事实门（``scene/facts.py``）保留，给概念命中里的情境概念当一个来源。
# 刻意还没动的：``Claim.situations`` 暂恒为空（旧语义树的情形已删，新树的概念命中接回来时按概念分家）；
# ``Settlement.reminded / response`` 按新方案改成另一份提醒记录（承诺写完不改），那一刀再动。
# 待用户定的数：θ（提醒 / 代劳两档）、偏离多少还算验证（按 slot_offset 的真实分布定）。
# 影响大小：大——账本是负样本与将来真因果（提醒 = 干预）的唯一来源；提醒上线前干预记录必须先就位。

from habitus.foresight.assemble import CandidateEvidence, EvidencePack, assemble, moment_at
from habitus.foresight.cards import HistoryCard, NowScene, history_card, now_scene
from habitus.foresight.context import CandidateBackground, candidate_background
from habitus.foresight.errors import ForesightError
from habitus.foresight.model import (
    LAYER_LABELS,
    LAYER_NAMES,
    CandidateNumbers,
    Layer,
    Moment,
    NoUnsealed,
    Provenance,
    RecurrenceNumbers,
    UnsealedReader,
    UnsealedRow,
)
from habitus.foresight.numbers import CellIndex, candidate_numbers, provenance
from habitus.foresight.render import render_candidate, render_moment, render_pack

__all__ = [
    "LAYER_LABELS",
    "LAYER_NAMES",
    "CandidateBackground",
    "CandidateEvidence",
    "CandidateNumbers",
    "CellIndex",
    "EvidencePack",
    "ForesightError",
    "HistoryCard",
    "Layer",
    "Moment",
    "NoUnsealed",
    "NowScene",
    "Provenance",
    "RecurrenceNumbers",
    "UnsealedReader",
    "UnsealedRow",
    "assemble",
    "candidate_background",
    "candidate_numbers",
    "history_card",
    "moment_at",
    "now_scene",
    "provenance",
    "render_candidate",
    "render_moment",
    "render_pack",
]
