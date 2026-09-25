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

# TODO(FORESIGHT-LOSS-001): 下半环"门控 + loss"分四刀落地（方案见《Loss 与门控方案》artifact，2026-09-19），
# 每刀刻意少做的部分记在这里，做完前一刀不许忘掉后一刀。
#
# 第一刀（账本与资格，无外部依赖）**已做（2026-09-19，分支 foresight-loss，经三方审查收缩）**：
#   foresight/ledger/{model,claims,settle,gate,codec}.py + runtime/foresight_ledger.py（落盘）
#   + runtime/foresight_settlement.py（夜批：树重建 → 结算 → 关联）。
#   只记**带时窗的「会」**：「不会」没有时点、「说不准」是弃权、说不出时窗的「会」核对不了，三者都不进账。
#   结算的定稿日用归约自己的事实（``BehaviorReductionRunner.closed_days``，组合根注入）——**不能用预测树的
#   出处日**：树每轮全量重建，今天上午的行中午就在树上，拿它当定稿会在今天还没过完时把承诺结成「落空」，
#   而结算是一次性的。
#   刻意留空 / 少做的：
#   - ``Claim.conditions``（第二刀填）；``Settlement.reminded / response``（干预账本未建，提醒上线前必须先有
#     ——时间不可逆）；没有 ReminderPolicy、没有 ``suggest_after_verified`` 配置（提醒通道那一刀再加，
#     现在加等于把不消费的数写进 example.yaml）。
#   - 弃权读数（说不准之后多久发生了）没做：它不进门，等真要调判断层时再从判断记录里算。
#   - 门只到 ``verified_counts(ledger.settlements())``：读时全量扫结算，没有阈值比较；成本按裁定不优化。
#   - 「偏离」只记事实（``slot_offset``，早于窗是负数）：偏多少还算数是读时的事，用户还没定。
#   - 情形键按 (行为, 情形) 分家，一条承诺引用几种情形就各记一次；"混引多种情形该不该合成一个集合的账"
#     待用户裁定。
#   - 判断本身**不落盘**（只有账本里的承诺），所以承诺不可从别处重放；真要重放判断得先持久化 Judgement。
# 第二刀（事实门）**已做（2026-09-19，经两方审查收缩）**：scene/facts.py——``FactProvider`` 契约
#   （``version`` + ``keys() -> FactKey(name, kind ∈ 类别|数值, unit)`` + ``at(本地时刻)``）、``NoFacts``、
#   ``CompositeFacts``；承诺上记三样：答了什么（``conditions``）、问了哪些键（``condition_keys``）、
#   谁按什么口径答的（``facts_version``）。三样缺一都事后补不回来：只看答案分不清"那天这个源离线"与
#   "从来没有这个键"，也分不清"日型"是名义日历答的还是真日历答的。账本仍不 import scene（自带 ``Conditions``
#   形状）；账本 schema 随之升到 v2（v1 的文件按 v1 解只会读出半条，所以硬拒而不是补默认值）。
#   刻意少做的：
#   - **没有任何真实提供者**：缺省是显式的 ``NoFacts``（与 ``_calendar`` 退到名义日历同一条纪律）——不拿一个
#     没有数据源的合成键（比如按周几算的"日型"、按钟点分的"时段"）去充数：那种键是 ``Claim`` 已有字段
#     （day / slot / slot_minutes）的纯函数，会让第一版 loss 的数字看起来稳定，实际什么都没量到。
#     **所以第三刀在接上真源之前只能跑通链路、算不出有意义的 loss。**
#   - 日型归 ``scene/calendar.py`` 那个接缝（读时算、不冻结）；将来要它进条件就写一个 ``CalendarFacts(calendar)``
#     适配器，由那一个源同时答两处，不要在事实门里另算一份。
#   - **没动关联**：``DayFacts.facts`` 与记录里的 conditions / relevant 留到第四刀一起做——改关联输入要升
#     提示词版本、全量重关联一次，两刀各升一次是白付两遍代价。
#   - 事实门**没有配置旋钮**：提供者由组合根注入（``build_foresight_components(facts=...)``），example.yaml 里
#     没有它——等有了真源、需要填路径或凭据时再进配置。
#   - 条件**不进证据包、不进判断提示词**：判断看行为流，条件只喂门控。要不要给判断者看，等 loss 有数据再说。
#   - 事实门自己不吞源的异常；预测层那一侧吞（记成"没问到、口径 unavailable"）：模型已经答过了，
#     条件是旁证，不许因为它丢掉整条承诺。
# 第三刀（loss 影子模式）：foresight/loss.py。证据 = 已验证承诺那些次的 Claim.conditions；类别按份额、数值按分位；
#   键权重**过渡**用"各键在证据里取值的集中度"——正式权重要等第四刀的 relevant；只记不拦，对着结算算误提醒率与覆盖率。
# 第四刀（关联长出条件，还没排期）：association/schema.py 加 relevant_conditions（只能引用输入里有的键）、
#   regularity/document.py 记录加 conditions / relevant、views/gloss.py 带上、refresher._source_digest 含 facts；
#   升 ASSOCIATION_PROMPT_VERSION → 全量重关联一次（三包量级约 175 次调用）；然后 loss 的权重换成 relevant 比例。
# 之后：第二段认知 foresight/cognition/（记忆只在这一段进，桥住组合根，MemoryTree 直读零副作用）；执行档位由晋升计数定。
# 待用户定：T₁（建议档要几次验证）、θ、承诺落空 / 被拒绝对资格做什么、偏离多少还算数。
# 影响大小：大——不做第一刀，判断层没有真值可调、提醒上线后干预混淆无法回溯；第二至四刀缺任何一刀，loss 都只是"未知、不拦"。

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
