"""Sample → Pattern 的确定性学习聚合器。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from foundation.integrity import canonical_digest, canonical_json, canonicalize
from prediction.document import PredictionDocument
from prediction.learning.config import PredictionLearningConfig, PredictionLearningError
from prediction.learning.keys import (
    label_branch_identity,
    logical_state_key,
    previous_step_key,
    sequence_state_identity,
    temporal_state_identity,
    treatment_branch_identity,
)
from prediction.learning.prior import PredictionBehaviorPrior
from prediction.learning.vocabulary import PredictionBehaviorVocabulary
from prediction.model import PredictionKind
from prediction.pattern.document import PredictionPatternDocument
from prediction.pattern.factory import PredictionPatternFactory
from prediction.uri import PredictionURI

_LEVELS = frozenset({"action", "event"})
_OBSERVED_LABEL_STATUSES = frozenset({"observed", "terminal"})
_ROOT_CONTEXT_KEY = ""
_LEVEL_NAMES = {"action": "动作", "event": "事件"}

# TODO(PRED-REGULARITY-001): 当前聚合用写死的两族状态处理所有行为，未区分行为之间可预测性的差异；
# 已与用户多轮讨论定稿一套「时间预测树 + 规律性度量」的完整方案取代它，尚未实现。
# 按"范围 → 产物（时间预测树）→ 度量 → 估计纪律 → 日循环 → 已验证 → 待定"组织；历轮被推翻的旧形态
#（按行为组织输出、前向选择链作为最终产物、LR 替换估计）只在文末"已关闭的支线"里留一句结论。
#
# ── 范围 ─────────────────────────────────────────────────────────────────────────────
# - 预测目标只覆盖**被外部约束**和**被习惯化**的行为。自发随机的行为照常记录、计入曝光量、不为任何
#   分支作证，但**不得从数据中剔除**——剔除会抽掉分母，抬高所有具体行为的条件概率，并让留白塌向零。
# - 任务框架是**时间槽**（这个 15 分钟槽里行为 B 有没有开始），不是活动序列。合成数据对照过：活动
#   序列框架下纯随机行为靠自相关能刷到 0.28 的解释比例；换成时间槽后掉到 0.07 排末位。
#
# ── 产物：时间预测树（本层唯一产物；定案版）───────────────────────────────────────────
# - 最终裁定（用户）：本层只算概率、零语义零 LLM；情景/关联/状态/日历/地点全部后置预测层——在
#   本层做情景语义关联会让设计混乱。与既有纪律"预测树只是预测算法的前置数据准备，推演类逻辑一律
#   不进树"完全同构。早先的"两张表"不是删除而是**合并进树**：表一成为节点、表二成为边。
# - 主干是**钟面叠加**：一天的 96 个槽位（15 分钟为工作值、可调），全部历史天数叠到同一个钟面——
#   不是日历时间线（那是行为树本身）。
# - 记录形状（计数与概率都存：计数+曝光是原始账本供预测层再加工，概率是发布时派生的成品供在线
#   直读）：
#     节点 (槽位 t, 动作 B)：{ 计数, 曝光, P(B│t), n_eff, lift }
#     边   (B → B′)      ：{ 计数, P(B′│B), lift, n_eff, 间隔分位数(开始到开始) }
# - **每个数字必带两个伴随值，因为概率自己会骗人**（同为 P=0.10：看电视@20:00 是 lift≈1.2 的
#   底噪、吃药@07:00 是 lift≈15 的真峰、只见过 3 次的是不可信的巧合）：
#     n_eff  这一格背后的衰减加权有效样本数——是每格自己的，不是该动作的总样本。回答"这个概率
#            是看了多少次得出的"。
#     lift   这一格的率 ÷ 该动作**自己的全天平均**（边为该边份额 ÷ 目标动作整体份额）。回答
#            "这一刻对这件事特殊吗，还是他本来就常做"——分母是他自己的平均，天然排除高频干扰。
#   预测层拿这两个伴随值自行划线；本层不做任何"可不可预测"的判定。
# - 一阶是刻意的：本层只给一阶原料，高阶组合（多步前缀、情景识别、关联）归预测层。早先否定
#   bigram，否定的是拿一阶转移当**唯一状态键**；此处它是原料之一，时间主干才是骨架。
# - 留白两条曲线照旧：h_不规律（已知但无规律）与 h_未见（真正 escape）。
# - 动态性内建：夜间整棵重建、原子发布、时间衰减、旧规律自然淡出——树每天更新，无新增机制。
#
# ── 度量的下落（原独立度量机器退出核心路径）────────────────────────────────────────────
# - 原方案的"度量"拆开是三件事，各有去处：算条件概率——就是树的计算本身；判可预测性（Δ/显著性）
#   ——归预测层，由 n_eff+lift 两个伴随值承载证据、预测层划线；选条件（前向选择/候选池）——随
#   条件定死为"时刻+上一步"而消失。
# - 置换检验降级为**离线验证工具**：合成数据与 CASAS 回归测试用（验证实现正确性：分层跨度不塌、
#   机制识别不乱、纯时钟共现的假转移不显著、随机转移源提升≈1），不进每夜生产路径。已有实现与
#   六户验证结论继续有效（见"已验证"节）。
# - 历史结论仍然成立的实现陷阱：per-type 统计必须在全部检验槽上算（只数发生槽会让高频行为靠
#   自相关刷虚高）；lifelog 类数据的适配参数需做敏感性测试。
#
# ── 估计纪律（一趟扫描五步；三条卫生是底线）─────────────────────────────────────────────
# - 五步：1 逐条 occurrence 按槽计数 +w（w=时间衰减 exp(−天龄/τ)）→ 2 逐天逐槽记曝光（只数被
#   观测到的槽）→ 3 节点概率 = 收缩后的计数/曝光（槽位 ← 池化 ← 全天，池化为纯统计平滑，可选
#   von Mises 圆周核）→ 4 按时间配对转移：每条 occurrence 找窗口内下一个开始，中间有观测空洞
#   则该对**删失**、不计为"无转移"；自转移允许 → 5 边概率 = 收缩后的份额，附 lift 与间隔分位数。
#   单人一年万条 occurrence 毫秒级，**每夜整棵重建，不做增量**。
# - 三条不可省的卫生（省了数字就是错的）：**收缩**（稀疏槽会出 0/1 极端值）；**覆盖分母**（观测
#   空洞会系统性压低概率——看见就记、没看见不记，本层不解释缺失）；**转移删失**（观测中断不是
#   "之后什么都没做"）。转移**开始锚定**（ended_at 常不可知，不锚结束）；边 lift 的分母 = 目标
#   动作在全部"下一个"事件中的份额（量纲须一致）。
# - **给预测层的组合契约**：节点概率与边概率共享时钟信息，**禁止朴素相乘**；树上是无条件率，要推
#   "大概几点做"的时刻分布，须先按当日已发生状态构造风险集（"还没发生"）——这一步在预测层做，
#   本层不持有当日状态。
# - 校准检查不能省：收缩使输出偏离真实频率，留出段画可靠性曲线，偏了套现有等渗校准；低频动作按
#   预测概率跨动作合并做。
# - **干预混淆在提醒上线前堵住**：occurrence（或 PredictionRun 结算）带"此前有没有提醒"标记，被
#   提醒后的发生不进自然率——事后无法分离两种数据。
# - 实现注意：槽索引按本地时间构造（夏令时 92/100 槽、偏移随 occurrence 各自还原），要有测试。
#
# ── 日循环（在线/离线分界）─────────────────────────────────────────────────────────────
# - 白天在线：观测 → 融合 → occurrence 落树（现有链路不变）；预测层查**昨夜的树** + 自己维护的
#   当日实况（今天已发生什么、距上次多久——这些状态属于预测层，不属于本层）。本层白天零动作、
#   全程零 LLM。
# - 夜间批处理：1 重算槽位概率 → 2 重算转移边 → 3 置换检验刷新显著性标记 → 4 整代原子发布。
#   次日预测查新树。首次出现的新动作当天预测不了：留白接住，当晚进树。
#
# ── 已验证 ────────────────────────────────────────────────────────────────────────────
# - 合成数据（答案已知）：机制识别全部正确，阴性对照（未植入星期效应的 dow）全列为零——度量不会
#   凭空造信号。
# - CASAS HH 六户真实住宅：分层跨度 0.39–0.60，高端稳定是做饭/吃药/起床/进门，低端稳定是招待客人/
#   在桌前工作，**分层假设成立**。按情境组织在 hh115 上直接给出可读的一天结构（00-06 睡觉起夜、
#   06-08 吃药、08-10 早饭三件套+看书、12-14 午饭三件套、20-22 晚药），组内顺序与常识一致；未归属
#   约四成。同轮教训：LSC-ADL lifelog 跑出假阴性（0.15）是测错对象——其标签是瞬时状态不是行为
#   事件；验证数据集的标签必须是目标行为事件。适配器参数（同标签中断多久算新一次）会实质改变结论，
#   必须做敏感性测试。
# - **公开数据集只能验证方案是否有效，不能定我们的候选池与参数**（CASAS 住户多为退休老人，星期
#   无用是这群人的属性不是人类行为的属性）。其长期用途：回归测试（分层跨度不塌、机制识别不乱、
#   对照条件 Δ≈0）与万条量级成本检查。
#
# ── 已关闭的支线 ─────────────────────────────────────────────────────────────────────
# - 以 L1-正则 logistic 回归替换计数估计：实测只在正例 ≥300 的高频行为上稳定 +0.011 bits，中低频
#   平手；且 L1 按单系数惩罚系统性偏向低基数条件（12 桶的时刻被摊薄、几乎从不入选），解释被编码
#   方式扭曲，要修得上 group lasso 并重做校准与置换——收益配不上代价，**不换**。同轮顺带证实：
#   朴素地把全部条件相加会灾难性过度自信（log-loss 达基线 13 倍），逐个入选的选择机制是承重的。
# - 按行为组织输出（每行为一条链）：不是最终产物形态，但其统计内核（集中度、显著性、条件分布）
#   与情境组织共用；"链"这一读法仅保留为调试视图。
# - LLM 先验伪计数的冷启动通道（把 prior.py 在线化、接进收缩链顶层）：**用户决定直接放弃**。
#   冷启动期系统只靠留白与不说话，是可接受的行为；prior.py 保持现状，不接入本方案。
# - 情境格子/情景谓词形态（L1 预切 2 小时格子、成员组挂格子、LLM 生成情景谓词、格子划分进
#   generation）：被"时间主干 + 两张概率表"取代。动机：情景宽度参数消失（情景边界由行为簇自己
#   决定，不再切）、顺序不被格子边界劈断、预测的两种模式（时间驱动的启动 / 顺序驱动的推进）自然
#   分开。**语义（情景命名、簇对齐、关联方式）整体上移预测层**——度量层曾经承担的语义角色全部
#   退出。该形态在 CASAS 上验证过的结论（分层、成员、顺序与常识一致）仍然成立，因为统计内核未变，
#   只是组织方式换了。情景实例 ≈ 搁置已久的 Episode，这一定位保留给行为树/预测层设计。
# - 门控机制（进度门三态/间隔门/未知态/行为日边界）与一切"当日状态"条件：随情景语义**整体后置
#   预测层**——树上只有无条件率，风险集与时刻分布推导由预测层结合当日实况完成。相应参数（近因
#   窗口、间隔桶边界、行为日边界、未知态覆盖阈值、门控层 τ）随之移出待定清单。
# - 三层"时间→情景→行为"在线打标结构（LLM 在行为落树时判断情景归属）：提议后用户裁定不采——
#   在本层做情景语义关联会让设计混乱，情景连同其在线打标整体后置预测层。
#
# ── 方案边界（构造上不解决的问题及其分派；逐条经用户裁定）──────────────────────────────
# - **日状态/换挡**：本层是逐行为的边际模型，没有"今天是哪种日子"的变量，反常日会整体错且无法
#   中途换挡。日期/日型/节假日语义**不在本层转化**（太重），归预测层的情景与语义处理；留白占比
#   突升是本层能给的唯一换挡信号。
# - **出门≠未发生**：**不区分**。系统的职责是看见了就记、没看见就不记；缺失只记录、不解释——
#   让行为更容易被看见是上游的设计职责，不是本层的建模对象。覆盖时间线只表示"系统在不在观测"，
#   不试图判断"人在不在场"。两张表估计的是**被观测到的行为**，这是系统的本体论，如实即可。
# - **外部协变量**（天气/日历事件/访客/他人安排）：不做。语法留扩展点，但现阶段这些偏离全部
#   如实落进留白。
# - **因果/规范/规划**（为什么没做、该不该做、几点前必须出门）：不属于统计层，由预测层的因果
#   信息与语义关联承担；禁止往两张表上硬挂这类语义。
# - **月频以下行为**（理发/体检/交租）：样本复杂度的数学下界，置换检验永远过不了——本层明确
#   不服务，交给预测层用 LLM 主观判断或其他机制（显式日历规则等）实现；这也是放弃 LLM 先验
#   通道的代价兑现处。本层服务的行为带：日频到周频。
# - **干预饥饿**（提醒常态化后自然数据流枯竭）：归入预测后的**数据回写设计**（PRED-ALGO-003
#   反馈闭环）——回写要支持按实际发生情况判断预测准确性与提醒影响；排除法防污染但制造饥饿，
#   完整解在回写层面，不在本层。
# - **悬而未决（待议）**：类型身份的自愈（上游分裂/误合并时本层只见"不显著"，分不出真不规律
#   还是身份碎了——同义分裂鲁棒性实验是它重要的原因）；效用验证（离线统计指标与提醒价值不同构，
#   只有上线反馈能量）。
#
# ── 待定（只能在**我们自己的**真实数据上定档，公开数据集不算）──────────────────────────────
# - 九个核心参数：衰减 τ、收缩 τ（槽位层/池化层两层）、池化宽度、槽位宽度、转移窗口、留白平滑、
#   夜间重算周期、发布保留代数。另有离线验证专用的置换参数（R、块长、三段切分比例）不进生产
#   路径。判据已定数值待量：槽位宽度 = min(提醒精度目标, 链内相邻间隔低分位)；转移窗口 = 链内/
#   链间间隔双峰分布的谷底。全部留显式配置、不给默认值。
# - loss 形式（PRED-LOSS-002）只用于离线验证与校准检查，不再影响生产路径的树本身。
# - 合成数据加同义分裂与随机漏报两种扰动的鲁棒性实验尚未做，现在就能做。
#
# - 影响大小：大。它替换本模块的状态定义与条件选择；估计式（折扣、强度、封顶、衰减、删失）与
#   generation 发布机制原样保留。行为树侧的硬依赖：本地时刻+日历日、跨次类型身份、顺序与前序、
#   地点、首次可知时刻、结束方式（含"没看到结束"与"没做完"之分）、**观测覆盖时间线**（危险率分母
#   要排除观测空洞，覆盖信息在观测层且会被生命周期清理，清理前必须沉淀覆盖摘要）、**提醒标记**
#  （干预混淆，见概率算法一节）。
# - 时机：方案已定稿、结构已在真实数据上验证；进主链要等行为树重构落地、用我们自己的数据把参数
#   定档。在此之前不要写进主链。


@dataclass(frozen=True)
class _Observation:
    """一条 TransitionSample 中参与聚合的最小事实。"""

    uri: str
    level: str
    domain: str
    context: dict[str, Any] | None
    branch: dict[str, Any] | None
    raw_semantics: str | None
    raw_actor: str | None
    weight: float
    cutoff: datetime
    group_id: str


class _BranchAccumulator:
    """一个 (State, 行为) 组合的加权计数器。"""

    def __init__(self, identity: Mapping[str, Any]) -> None:
        self.identity: dict[str, Any] = dict(identity)
        self.group_weights: dict[str, float] = {}
        self.raw_count = 0
        self.references: list[tuple[datetime, str]] = []
        self.semantics_weights: dict[str, float] = {}
        self.actor_weights: dict[str, float] = {}

    def add(self, observation: _Observation, *, scaled_weight: float, record: bool) -> None:
        if record:
            self.raw_count += 1
            self.references.append((observation.cutoff, observation.uri))
        self.group_weights[observation.group_id] = (
            self.group_weights.get(observation.group_id, 0.0) + scaled_weight
        )
        if observation.raw_semantics is not None:
            self.semantics_weights[observation.raw_semantics] = (
                self.semantics_weights.get(observation.raw_semantics, 0.0) + scaled_weight
            )
        if observation.raw_actor is not None:
            self.actor_weights[observation.raw_actor] = (
                self.actor_weights.get(observation.raw_actor, 0.0) + scaled_weight
            )


class _StateAccumulator:
    """一个抽象格状态的曝光、分支与删失计数器。

    时间族状态用核平滑把一条样本按分数权重摊到相邻时段；分数贡献只进
    加权计数，原始支持度、证据引用与统计窗口只在主时段记账一次。
    """

    def __init__(
        self,
        level: str,
        domain: str,
        context: dict[str, Any] | None,
        bucket: str | None = None,
    ) -> None:
        self.level = level
        self.domain = domain
        self.context = context
        self.bucket = bucket
        self.branches: dict[str, _BranchAccumulator] = {}
        self.censored_group_weights: dict[str, float] = {}
        self.raw_count = 0
        self.references: list[tuple[datetime, str]] = []
        self.groups: set[str] = set()
        self.first_observed_at: datetime | None = None
        self.last_observed_at: datetime | None = None

    def add(self, observation: _Observation, *, weight_scale: float = 1.0, record: bool = True) -> None:
        scaled_weight = observation.weight * weight_scale
        if record:
            self.raw_count += 1
            self.references.append((observation.cutoff, observation.uri))
            self.groups.add(observation.group_id)
            if self.first_observed_at is None or observation.cutoff < self.first_observed_at:
                self.first_observed_at = observation.cutoff
            if self.last_observed_at is None or observation.cutoff > self.last_observed_at:
                self.last_observed_at = observation.cutoff
        if observation.branch is None:
            self.censored_group_weights[observation.group_id] = (
                self.censored_group_weights.get(observation.group_id, 0.0) + scaled_weight
            )
            return
        key = canonical_json(observation.branch)
        accumulator = self.branches.get(key)
        if accumulator is None:
            accumulator = _BranchAccumulator(observation.branch)
            self.branches[key] = accumulator
        accumulator.add(observation, scaled_weight=scaled_weight, record=record)


class PredictionPatternLearner:
    """把 TransitionSample 集合聚合为一代 State/Branch Pattern 的纯函数学习器。

    估计采用两层 Pitman–Yor 插值回退：上下文状态（上一步行为）的分布向同层
    根状态收缩，折扣项产生的未分配概率质量显式表示"下一步是未见行为"的留白，
    因此每个 State 的出边概率和严格小于一。计数经过质量加权（样本
    ``source_confidence``）、同 occurrence_group 封顶和时间指数衰减修正；
    删失样本只进入状态曝光量，不为任何具体分支作证。全部超参、词表版本、
    ``learned_at`` 与样本身份共同构成 generation 身份，同输入必同输出；
    本层不落盘、不加锁，物化与激活交给 Pattern 发布器。
    """

    def __init__(
        self,
        *,
        config: PredictionLearningConfig | None = None,
        vocabulary: PredictionBehaviorVocabulary | None = None,
        factory: PredictionPatternFactory | None = None,
        prior: PredictionBehaviorPrior | None = None,
    ) -> None:
        if config is not None and not isinstance(config, PredictionLearningConfig):
            raise TypeError("config must be PredictionLearningConfig")
        if vocabulary is not None and not isinstance(vocabulary, PredictionBehaviorVocabulary):
            raise TypeError("vocabulary must be PredictionBehaviorVocabulary")
        if factory is not None and not isinstance(factory, PredictionPatternFactory):
            raise TypeError("factory must be PredictionPatternFactory")
        if prior is not None and not isinstance(prior, PredictionBehaviorPrior):
            raise TypeError("prior must be PredictionBehaviorPrior")
        self.config = config or PredictionLearningConfig()
        self.vocabulary = vocabulary or PredictionBehaviorVocabulary()
        self.factory = factory or PredictionPatternFactory()
        self.prior = prior

    def match_identity(self) -> dict[str, Any]:
        """预测端构造期握手所需的匹配关键参数快照。

        发布时经 ``learning_identity`` 随 manifest 落盘；预测器配置或词表
        与这份快照不一致时必须显式失败——失配的时段参数会让预测器精确
        命中语义错误的时段状态，失配的词表会让全部键静默 miss 到根。
        """

        return {
            "temporal_bucket_hours": self.config.temporal_bucket_hours,
            "temporal_utc_offset_minutes": self.config.temporal_utc_offset_minutes,
            "vocabulary_version": self.vocabulary.version,
        }

    def learn(
        self,
        samples: Sequence[PredictionDocument],
        *,
        learned_at: datetime,
        consequences: Sequence[PredictionDocument] = (),
    ) -> tuple[PredictionPatternDocument, ...]:
        """从不可变 TransitionSample 聚合出可直接发布的一批 Pattern 文档。

        ``consequences`` 是可选的 ConsequenceSample 集合：按 (域, 行为键)
        聚合出结果类别分布，挂到对应分支的 ``outcomes``，供执行判决的
        风险门消费。同一 Outcome 的多次修订先收敛到最新有效物化再计数。
        """

        if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
            raise PredictionLearningError("samples must be a sequence of PredictionDocument values")
        if (
            not isinstance(learned_at, datetime)
            or learned_at.tzinfo is None
            or learned_at.utcoffset() is None
        ):
            raise PredictionLearningError("learned_at must be a timezone-aware datetime")
        observations: list[_Observation] = []
        versions: set[str] = set()
        materialization_ids: set[str] = set()
        seen: dict[str, PredictionDocument] = {}
        for document in samples:
            if not isinstance(document, PredictionDocument):
                raise PredictionLearningError("samples must contain PredictionDocument values")
            if document.kind is not PredictionKind.TRANSITION:
                raise PredictionLearningError("the baseline learner only aggregates TransitionSamples")
            uri = str(PredictionURI.from_address(document.address))
            existing = seen.get(uri)
            if existing is not None:
                if existing != document:
                    raise PredictionLearningError("one learning round binds one sample URI to different content")
                continue
            seen[uri] = document
            versions.add(str(document.fields["provenance"]["projection_version"]))
            materialization_ids.add(str(document.fields["materialization_id"]))
            observations.append(self._observation(uri, document, learned_at))
        if not observations:
            raise PredictionLearningError("pattern learning requires at least one TransitionSample")
        effective_consequences = self._effective_consequences(consequences, versions)
        if len(versions) != 1:
            raise PredictionLearningError("one learning round must use one projection version")
        projection_version = next(iter(versions))
        outcome_index = self._consequence_outcomes(effective_consequences, learned_at)
        generation_id = canonical_digest(
            {
                "contract": "prediction-pattern-learning-v1",
                "projection_version": projection_version,
                "config": self.config.identity_material(),
                "vocabulary_version": self.vocabulary.version,
                "learned_at": canonicalize(learned_at),
                "sample_materialization_ids": sorted(materialization_ids),
                "consequence_materialization_ids": sorted(
                    str(document.fields["materialization_id"]) for document in effective_consequences
                ),
                "prior_version": None if self.prior is None else self.prior.version,
            }
        )
        states: dict[tuple[str, str, str], _StateAccumulator] = {}
        for observation in observations:
            self._accumulate(
                states,
                (observation.level, observation.domain, _ROOT_CONTEXT_KEY),
                observation,
                context=None,
            )
            if observation.context is not None:
                self._accumulate(
                    states,
                    (observation.level, observation.domain, canonical_json(observation.context)),
                    observation,
                    context=observation.context,
                )
            for bucket, fraction, primary in self._temporal_fractions(observation.cutoff):
                self._accumulate(
                    states,
                    (observation.level, observation.domain, f"hour:{bucket}"),
                    observation,
                    context=None,
                    bucket=bucket,
                    weight_scale=fraction,
                    record=primary,
                )
        documents: list[PredictionPatternDocument] = []
        base_by_group: dict[tuple[str, str], dict[str, float]] = {}
        for group in sorted({key[:2] for key in states}):
            root_accumulator = states[(*group, _ROOT_CONTEXT_KEY)]
            base_by_group[group] = self._distribution(
                root_accumulator,
                parent=None,
                pseudo=self._pseudo_counts(root_accumulator),
            )
        for key in sorted(states):
            level, domain, context_key = key
            accumulator = states[key]
            if context_key == _ROOT_CONTEXT_KEY:
                probabilities = base_by_group[(level, domain)]
            else:
                if accumulator.raw_count < self.config.min_context_support:
                    continue
                probabilities = self._distribution(
                    accumulator,
                    parent=base_by_group[(level, domain)],
                    pseudo=self._pseudo_counts(accumulator),
                )
            documents.extend(
                self._materialize_state(
                    accumulator, probabilities, generation_id, projection_version, outcome_index
                )
            )
        return tuple(documents)

    def _effective_consequences(
        self,
        consequences: Sequence[PredictionDocument],
        versions: set[str],
    ) -> tuple[PredictionDocument, ...]:
        """按逻辑样本收敛 Outcome 修订：只保留最新有效物化参与计数。"""

        if isinstance(consequences, (str, bytes)) or not isinstance(consequences, Sequence):
            raise PredictionLearningError("consequences must be a sequence of PredictionDocument values")
        latest: dict[str, PredictionDocument] = {}
        for document in consequences:
            if not isinstance(document, PredictionDocument):
                raise PredictionLearningError("consequences must contain PredictionDocument values")
            if document.kind is not PredictionKind.CONSEQUENCE:
                raise PredictionLearningError("consequences must contain ConsequenceSamples")
            versions.add(str(document.fields["provenance"]["projection_version"]))
            logical_id = str(document.fields["logical_sample_id"])
            current = latest.get(logical_id)
            if current is None or self._revision_order(document) > self._revision_order(current):
                latest[logical_id] = document
        return tuple(latest[key] for key in sorted(latest))

    @staticmethod
    def _revision_order(document: PredictionDocument) -> tuple[int, str]:
        return (
            int(document.fields["materialization_context"]["outcome_revision"]),
            str(document.fields["materialization_id"]),
        )

    def _consequence_outcomes(
        self,
        consequences: Sequence[PredictionDocument],
        learned_at: datetime,
    ) -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
        """按 (域, 行为键) 聚合结果类别分布，供分支 ``outcomes`` 消费。

        类别 = outcome_type × valence；计数经质量加权、consequence_group
        封顶与时间衰减，折扣产生的留白表示"还有未见过的结果"。延迟取
        中位数——结果延迟长尾重，均值会被单个迟到结果拖爆。
        """

        buckets: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = {}
        for document in consequences:
            fields = document.fields
            supervision = fields["supervision"]
            if bool(supervision["censored"]):
                continue
            domain = self.vocabulary.canonical_token(
                fields["prediction_scope"]["target_domain"], "prediction_scope.target_domain"
            )
            branch_key = canonical_json(
                treatment_branch_identity(fields["treatment"], self.vocabulary)
            )
            cutoff = _parse_datetime(fields["anchor"]["cutoff_at"], "anchor.cutoff_at")
            age_seconds = max(0.0, (learned_at - cutoff).total_seconds())
            weight = float(fields["quality"]["source_confidence"]) * math.exp(
                -age_seconds / (self.config.decay_tau_days * 86400.0)
            )
            outcome = fields["label"]["outcome"]
            category = (str(outcome["outcome_type"]), str(outcome["valence"]))
            group_id = str(fields["lineage"]["consequence_group_id"])
            bucket = buckets.setdefault((domain, branch_key), {})
            entry = bucket.setdefault(
                category,
                {"group_weights": {}, "delays": [], "semantics_weights": {}},
            )
            entry["group_weights"][group_id] = entry["group_weights"].get(group_id, 0.0) + weight
            delay = outcome["delay_seconds"]
            if delay is not None:
                entry["delays"].append(float(delay))
            semantics = str(outcome["semantics"])
            entry["semantics_weights"][semantics] = (
                entry["semantics_weights"].get(semantics, 0.0) + weight
            )
        index: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
        for key, categories in buckets.items():
            counts = {
                category: self._capped_weight(entry["group_weights"])
                for category, entry in categories.items()
            }
            denominator = self.config.strength + sum(counts.values())
            entries: list[dict[str, Any]] = []
            for category in sorted(categories):
                entry = categories[category]
                probability = max(0.0, counts[category] - self.config.discount) / denominator
                if probability <= 0:
                    continue
                outcome_type, valence = category
                delays = sorted(entry["delays"])
                entries.append(
                    {
                        "semantics": _weighted_mode(
                            entry["semantics_weights"], f"{outcome_type}({valence})"
                        ),
                        "probability": probability,
                        "average_delay_seconds": _median(delays),
                        "valence": valence,
                    }
                )
            entries.sort(key=lambda item: (-item["probability"], item["semantics"]))
            index[key] = tuple(entries)
        return index

    @staticmethod
    def _accumulate(
        states: dict[tuple[str, str, str], _StateAccumulator],
        key: tuple[str, str, str],
        observation: _Observation,
        *,
        context: dict[str, Any] | None,
        bucket: str | None = None,
        weight_scale: float = 1.0,
        record: bool = True,
    ) -> None:
        accumulator = states.get(key)
        if accumulator is None:
            accumulator = _StateAccumulator(observation.level, observation.domain, context, bucket)
            states[key] = accumulator
        accumulator.add(observation, weight_scale=weight_scale, record=record)

    def _temporal_fractions(self, cutoff: datetime) -> tuple[tuple[str, float, bool], ...]:
        """按 von Mises 核把一个时刻的权重摊到各时段桶；圆周上无边界断裂。

        23:50 与 00:10 是邻居，直方分桶会人为切断；核平滑让每条样本按
        与桶中心的圆周距离分数计入相邻桶，主桶（时刻所在桶）负责记账。
        """

        hours = self.config.temporal_bucket_hours
        offset = timezone(timedelta(minutes=self.config.temporal_utc_offset_minutes))
        local = cutoff.astimezone(offset)
        theta = 2 * math.pi * (local.hour + local.minute / 60 + local.second / 3600) / 24
        concentration = self.config.temporal_kernel_concentration
        bucket_count = 24 // hours
        raw = [
            math.exp(
                concentration
                * (math.cos(theta - 2 * math.pi * ((index + 0.5) * hours) / 24) - 1.0)
            )
            for index in range(bucket_count)
        ]
        total = sum(raw)
        primary_index = local.hour // hours
        return tuple(
            (
                f"{index * hours:02d}-{(index + 1) * hours:02d}",
                raw[index] / total,
                index == primary_index,
            )
            for index in range(bucket_count)
        )

    def _observation(
        self,
        uri: str,
        document: PredictionDocument,
        learned_at: datetime,
    ) -> _Observation:
        fields = document.fields
        anchor = fields["anchor"]
        level = str(anchor["anchor_type"])
        if level not in _LEVELS:
            raise PredictionLearningError("transition anchors must be at action or event level")
        cutoff = _parse_datetime(anchor["cutoff_at"], "anchor.cutoff_at")
        age_seconds = max(0.0, (learned_at - cutoff).total_seconds())
        weight = float(fields["quality"]["source_confidence"]) * math.exp(
            -age_seconds / (self.config.decay_tau_days * 86400.0)
        )
        supervision = fields["supervision"]
        labeled = (
            not bool(supervision["censored"])
            and str(supervision["label_status"]) in _OBSERVED_LABEL_STATUSES
        )
        branch: dict[str, Any] | None = None
        raw_semantics: str | None = None
        raw_actor: str | None = None
        if labeled:
            label = fields["label"]
            raw_semantics = label["semantics"]
            raw_actor = label["actor"]
            branch = label_branch_identity(label, self.vocabulary)
        history = fields["input"]["behavior_history"]
        steps = history["completed_actions"] if level == "action" else history["completed_events"]
        context = previous_step_key(steps, self.vocabulary)
        return _Observation(
            uri=uri,
            level=level,
            domain=self.vocabulary.canonical_token(
                fields["prediction_scope"]["target_domain"], "prediction_scope.target_domain"
            ),
            context=context,
            branch=branch,
            raw_semantics=raw_semantics,
            raw_actor=raw_actor,
            weight=weight,
            cutoff=cutoff,
            group_id=str(fields["lineage"]["occurrence_group_id"]),
        )

    def _state_identity(self, accumulator: _StateAccumulator) -> dict[str, Any]:
        if accumulator.bucket is not None:
            return temporal_state_identity(accumulator.level, accumulator.domain, accumulator.bucket)
        return sequence_state_identity(accumulator.level, accumulator.domain, accumulator.context)

    def _pseudo_counts(self, accumulator: _StateAccumulator) -> Mapping[str, float]:
        if self.prior is None:
            return {}
        return self.prior.pseudo_counts(logical_state_key(self._state_identity(accumulator)))

    def _distribution(
        self,
        accumulator: _StateAccumulator,
        *,
        parent: Mapping[str, float] | None,
        pseudo: Mapping[str, float],
    ) -> dict[str, float]:
        """Pitman–Yor 插值：局部折扣计数加上按 escape 系数继承的父层概率。

        escape 携带的质量 = 强度 + 各分支实际被折扣走的量（min(count, d)，
        而不是理想 CRP 的 d×T）+ 删失与未见伪计数曝光；这保证在质量加权、
        时间衰减、组封顶产生的分数计数下（单分支计数可小于折扣 d），
        Σ local + escape ≡ 1 恒成立，于是每状态出边概率和严格小于一，
        任何单分支概率不越界。``pseudo`` 是语义先验的伪计数：观测到的
        分支直接加进计数，未观测分支的伪计数只进曝光量——它们没有证据
        不能物化，其先验质量以留白形式存在，等真实观测出现后立即转化。
        """

        discount = self.config.discount
        strength = self.config.strength
        counts = {
            key: self._capped_weight(branch.group_weights) + pseudo.get(key, 0.0)
            for key, branch in accumulator.branches.items()
        }
        unseen_pseudo = sum(
            value for key, value in pseudo.items() if key not in accumulator.branches
        )
        censored = self._capped_weight(accumulator.censored_group_weights)
        exposure = sum(counts.values()) + censored + unseen_pseudo
        denominator = strength + exposure
        discounted = sum(min(count, discount) for count in counts.values())
        escape = (strength + discounted + censored + unseen_pseudo) / denominator
        result: dict[str, float] = {}
        for key, count in counts.items():
            local = max(0.0, count - discount) / denominator
            inherited = escape * parent.get(key, 0.0) if parent is not None else 0.0
            result[key] = local + inherited
        return result

    def _capped_weight(self, group_weights: Mapping[str, float]) -> float:
        cap = self.config.group_weight_cap
        return sum(min(cap, weight) for weight in group_weights.values())

    def _materialize_state(
        self,
        accumulator: _StateAccumulator,
        probabilities: Mapping[str, float],
        generation_id: str,
        projection_version: str,
        outcome_index: Mapping[tuple[str, str], tuple[dict[str, Any], ...]],
    ) -> list[PredictionPatternDocument]:
        assert accumulator.first_observed_at is not None
        assert accumulator.last_observed_at is not None
        predicates: list[dict[str, Any]] = [
            {"field": "anchor_type", "operator": "eq", "value": accumulator.level},
            {"field": "target_domain", "operator": "eq", "value": accumulator.domain},
        ]
        identity = self._state_identity(accumulator)
        if accumulator.bucket is not None:
            predicates.append(
                {"field": "hour_bucket", "operator": "eq", "value": accumulator.bucket}
            )
        elif accumulator.context is not None:
            predicates.append(
                {"field": "previous_step_key", "operator": "eq", "value": accumulator.context}
            )
        state = self.factory.build_state(
            pattern_generation=generation_id,
            identity=identity,
            fields={
                "semantic_summary": self._state_summary(accumulator),
                "state_level": accumulator.level,
                "predicates": predicates,
                "active_goals": (),
                "constraints": (),
                "support_count": accumulator.raw_count,
                "sample_refs": self._bounded_refs(accumulator.references),
                "statistics": {
                    "first_observed_at": accumulator.first_observed_at,
                    "last_observed_at": accumulator.last_observed_at,
                    "source_count": len(accumulator.groups),
                    "confidence": self._shrunk_confidence(accumulator.raw_count),
                },
                "projection_version": projection_version,
            },
        )
        documents = [state]
        for key in sorted(accumulator.branches):
            branch = accumulator.branches[key]
            if branch.raw_count == 0:
                continue
            identity = branch.identity
            if identity["target_kind"] == "termination":
                behavior_type = "termination"
                target_refs: list[str] = []
                fallback_semantics = f"终止（{identity['status']}）"
            else:
                behavior_type = str(identity["behavior_type"])
                target_refs = list(identity["target_refs"])
                fallback_semantics = str(identity["semantics"])
            documents.append(
                self.factory.build_branch(
                    state=state,
                    identity={"target": identity},
                    fields={
                        "target_kind": str(identity["target_kind"]),
                        "behavior": {
                            "actor_role": _weighted_mode(branch.actor_weights, "unspecified"),
                            "behavior_type": behavior_type,
                            "semantics": _weighted_mode(branch.semantics_weights, fallback_semantics),
                            "target_refs": target_refs,
                            "parameters": {},
                        },
                        "conditions": predicates,
                        "support_count": branch.raw_count,
                        "conditional_probability": probabilities[key],
                        "confidence": self._shrunk_confidence(branch.raw_count),
                        "sample_refs": self._bounded_refs(branch.references),
                        "outcomes": outcome_index.get((accumulator.domain, key), ()),
                        "projection_version": projection_version,
                    },
                )
            )
        return documents

    def _state_summary(self, accumulator: _StateAccumulator) -> str:
        level_name = _LEVEL_NAMES[accumulator.level]
        if accumulator.bucket is not None:
            return f"{accumulator.domain} 域{level_name}层 {accumulator.bucket} 时段的下一步分布"
        if accumulator.context is None:
            return f"{accumulator.domain} 域{level_name}层任意上下文的下一步基线分布"
        return (
            f"{accumulator.domain} 域{level_name}层上一步"
            f"「{accumulator.context['semantics']}」之后的下一步分布"
        )

    def _bounded_refs(self, references: Sequence[tuple[datetime, str]]) -> tuple[str, ...]:
        ordered = sorted(references, key=lambda item: item[1])
        ordered.sort(key=lambda item: item[0], reverse=True)
        return tuple(item[1] for item in ordered[: self.config.max_sample_refs])

    def _shrunk_confidence(self, raw_count: int) -> float:
        return raw_count / (raw_count + self.config.confidence_prior)


def _weighted_mode(weights: Mapping[str, float], fallback: str) -> str:
    if not weights:
        return fallback
    return min(weights.items(), key=lambda item: (-item[1], item[0]))[0]


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _parse_datetime(value: object, label: str) -> datetime:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PredictionLearningError(f"{label} must be an ISO timestamp") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PredictionLearningError(f"{label} must be timezone-aware")
    return parsed


__all__ = ["PredictionPatternLearner"]
