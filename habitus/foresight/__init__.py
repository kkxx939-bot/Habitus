"""预测层：把一刻的证据装成一个包——候选、四层数字、每次发生一张历史卡、此刻场景——给判断用。

本包只读派生树、不写任何树（行为树永远只有观测→融合→归约那一个写入口）。包根这一层零模型：候选与
四层拆解、发布的率、历史卡（邻域序列 + 视图 + 关联记录）、此刻场景、整包的 Markdown 渲染都是纯函数。
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

from habitus.foresight.assemble import (
    AssociatedDays,
    CandidateEvidence,
    EvidencePack,
    GlossesFor,
    SituationsFor,
    assemble,
    moment_at,
)
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
    "AssociatedDays",
    "CandidateBackground",
    "CandidateEvidence",
    "CandidateNumbers",
    "CellIndex",
    "EvidencePack",
    "ForesightError",
    "GlossesFor",
    "HistoryCard",
    "Layer",
    "Moment",
    "NoUnsealed",
    "NowScene",
    "Provenance",
    "RecurrenceNumbers",
    "SituationsFor",
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
