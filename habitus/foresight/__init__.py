"""预测层：把树的数字与语义树的背景装配成给判断用的证据，四层数字各配各自那批日子的背景。

本包只读派生树、不写任何树（行为树永远只有观测→融合→归约那一个写入口）。现状是第一块的
前半段：四层拆解、每层的背景装配、Markdown 渲染，零模型。
"""

# TODO(FORESIGHT-PERF-001): 三处按体量会变成问题的开销（2026-09-12 三方审查实测，现在的数据量
# 下都不痛，先记下来）。
# 1. **每次 ``assemble`` 重新解码整棵树**（``runtime/foresight.py``）。``store.load()`` 读整个
#    tree.json、``json.loads``、``codec.decode``，再 ``CellIndex.of`` 全量扫一遍。按
#    ``prediction/model.py`` 自己的估算，一年后这棵树约 33 MiB——问一个候选的证据要付一次
#    33 MiB 的 JSON 解码。改造：代是不可变的，按 ``(generation, digest)`` 记住 ``(tree, CellIndex)``
#    即可；"一次查询一份"那条纪律说的是 ``DayIndexCache``（要读今天），不是树。
# 2. **渲染的截断是二次的**（``render.py``）。每砍一条就整篇重渲一次。实测：1,160 条视图的
#    候选截到 4,000 字要 0.94s。现在有 ``max_days_per_layer`` 兜着（每层至多 40 天），到不了那个
#    量级；真要修就改成先按条目长度记账、或对预算做二分。
# 3. **四层嵌套，同一条 occurrence 最多被投影四次**（``context.py``）。``slot ⊆ pool ⊆ cross``、
#    ``all_day`` 包住全部，但四层各自调一次 ``history_contexts``，而 ``_project`` 没有按 URI 记忆；
#    ``_project → neighbours`` 还会为每条视图重排三天的时间线。改造：在缓存上加一个
#    ``dict[str, ContextView]`` 记忆即可同时消掉两个倍数。
#
# TODO(FORESIGHT-PACK-001): 两个配置项还没有消费者，一个事实还没进包。
# - **背景侧目前恒空**：``layer_background`` 经 ``scene.views`` 读的是按天情景树，而它的写入方
#   （按天归组）已经删掉（2026-09-13），新的规律级还没有读侧。四层数字照常，语义背景一条都取不到，
#   而且不会报错——只会在 ``Provenance.unassociated`` 里如实摆着"全部出处日都没有背景"。读侧改写到
#   规律级是下一步。
# - ``foresight.max_pack_chars`` 与 ``max_similar_scenes`` 已声明、已进 example.yaml，但生产代码
#   一处都没读：``render_candidate`` 目前只有测试在调。等证据包（多候选 + 相似情景）成形时接上。
# - ``render_candidate`` 的 ``max_chars`` 是**尽力而为**：四层的数字与出处一个都不砍，所以四个
#   预算都归零之后仍可能超出（实测一个候选的下限是 360 字），而调用方拿不到任何信号。接包的
#   时候要么返回"是否放得下"，要么在文本里留一行。
# - **包里没有树真正发布的那个率**。四层拆解给的是链上各环节的证据，而 ``curves[...].marginal[slot]``
#   （链的产物）、``hazard``、``cumulative``、两个 lift、``trend`` 一个都没进去。判断者看得到推导、
#   看不到结论。等 ``CandidateNumbers`` 那一块时一并补上。

from habitus.foresight.assemble import (
    AssociatedDays,
    CandidateEvidence,
    candidate_evidence,
    candidates_evidence,
    moment_at,
)
from habitus.foresight.context import LayerBackground, layer_background
from habitus.foresight.errors import ForesightError
from habitus.foresight.model import LAYER_LABELS, LAYER_NAMES, Layer, Moment, Provenance
from habitus.foresight.numbers import CellIndex, provenance
from habitus.foresight.render import render_candidate

__all__ = [
    "LAYER_LABELS",
    "LAYER_NAMES",
    "AssociatedDays",
    "CandidateEvidence",
    "CellIndex",
    "ForesightError",
    "Layer",
    "LayerBackground",
    "Moment",
    "Provenance",
    "candidate_evidence",
    "candidates_evidence",
    "layer_background",
    "moment_at",
    "provenance",
    "render_candidate",
]
