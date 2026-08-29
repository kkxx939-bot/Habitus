"""时间预测树的键值与记录对象。

# TODO(PRED-TREE-001): 本层的唯一规格来源（取代已退役的 TODO(PRED-REGULARITY-001)，
# 继承其全部结论并补入本轮讨论的扩展）。按"范围 → 产物 → 估计纪律 → 组合契约 → 日循环 →
# 已验证 → 已关闭的支线 → 待定"组织。实现完成后本 TODO 转为规格记录，不再是待办。
#
# ── 范围 ─────────────────────────────────────────────────────────────────────────────
# - 预测目标只覆盖**被外部约束**与**被习惯化**的行为。自发随机的行为照常记录、**计入曝光分母**、
#   不为任何分支作证，但**绝不从数据中剔除**——剔除会抽掉分母、抬高所有具体行为的条件概率、
#   并让留白塌向零。
# - 任务框架是**时间槽**（这个 15 分钟槽里行为 B 有没有开始），不是活动序列。合成数据对照过：
#   活动序列框架下纯随机行为靠自相关能刷到 0.28 的解释比例，换成时间槽后掉到 0.07 排末位。
# - 被约束与被习惯化的行为在树上**同形**（都是高 lift 的强规律），树区分不了"为什么规律"——
#   那需要人物层语义，属于上层。
#
# ── 产物：一棵树，不是若干张表 ─────────────────────────────────────────────────────────
# - 主干是**钟面叠加**：(周几 × 每天 96 个 15 分钟槽)，全部历史天叠到同一钟面，不是日历时间线
#   （那是行为树本身）。动作节点挂在主干上，动作之间有边，动作自身带复发间隔。
# - **周几维度**（本轮新增）：周几是从日期机械算出的键，不是语义判断——树只当它是字符串，
#   不知道"工作日/周末"的含义（那个由上层从周几自行合并）。它把"每周二打球"这类
#   **看似低频、实为条件高频**的规律从噪声里捞出来：日钟面上 P≈1/7，周维度上 P≈0.9。
#   代价是样本除以 7，周规律需要 3–6 个月数据；学不到时收缩自动退回日钟面行为，不会更差。
# - **月/季不做**：几号（31 桶）是样本下界；季节是**衰减结构下界**——周期长于约 2τ 的规律，
#   上一次的数据已衰减殆尽，不是学得慢而是结构上学不到。月频行为改由**复发间隔**覆盖
#   （不分桶，一年 12 次就是 11 个间隔样本），季节由衰减机制"缓慢跟上"而非"预先知道"。
#   旬（3 桶）统计上可行但收益未知，留作纯增量，等真实数据算出 lift_旬 显著再加。
#
# - 记录形状（计数与概率都存：计数+曝光是原始账本供上层再加工，概率是发布时派生的成品供直读）：
#   另有**全天基线**（每个动作不看时刻的整体发生率）随树发布：树是稀疏存储，一个动作在某格
#   没有记录只说明"没见过"而不是"不会发生"，读侧对空格子的兜底就取这条基线（``query.marginal_at``）。
#     节点 (周几 d, 槽位 t, 动作 B)：
#       原始账本 { 计数, 原始次数(不封顶), 近期计数, 该槽之前的累积计数 }
#       派生成品 { 边际率 P(B│d,t), 危险率 h(B│d,t), 累积率 CumP, lift_全天, lift_周几,
#                  n_eff, 趋势(近期/长期) }
#     边 (B → B′)：{ 计数, P(B′│B), lift, n_eff, 间隔分位数(开始到开始), 槽位直方图 }
#       另有并行边 (B ∥ B′) 与"无后继" (B → ∅) 两类。
#     复发 (B)：{ 间隔分位数, 样本数 }
# - **每个数字必带伴随值，因为概率自己会骗人**（同为 P=0.10：看电视@20:00 是 lift≈1.2 的底噪、
#   吃药@07:00 是 lift≈78 的真峰、只见过 1 次的是不可信的巧合）：
#     n_eff  这一格自己的衰减加权有效**机会**数（不是该动作总样本）。注意它只回答"分母可不可信"，
#            **单次巧合要靠计数 C 本身识别**（C 在树上摆着，划线由上层做）。
#     lift   节点为"这一格的率 ÷ 该动作自己的全天平均"，边为"该边份额 ÷ 目标动作整体份额"
#            （两个分母口径必须一致，都含 ∅，否则 lift 被系统性拉偏）。分母是他自己的平均，
#            天然排除高频干扰。
#     趋势   近期率(短 τ) ÷ 长期率(长 τ)。衰减是对称的——它同样慢地学新习惯、忘旧习惯；
#            没有这个数，换药停服后会连续误报两个月。→0 = 习惯正在消失（抑制提醒），
#            ≫1 = 新习惯正在建立（可以开始提醒）。
# - PRED-RATES-001（已裁定 2026-08-29：三种率**全部保留**）。裁定依据：一个"每周五上午
#   去医院"的场景同时用光三个数——边际率生成候选、危险率给"几点开口"（时刻分布 h·Π(1−h)
#   的唯一来源）、累积率给缺失检测的"该完成线"；三者共用同一本账，砍任何一个不省存储、
#   只砍对应能力。附带条件两条：① 评估仪补齐危险率与累积率的校准——已落地为
#   evaluation.first_occurrence_timing（首次时刻：中位误差 + p10–p90 覆盖率）与
#   evaluation.cumulative_calibration（该完成线：ECE + 分箱）；② 在真实数据的校准数字
#   出来之前，判决层对危险率/累积率保守加权。
# - **三种率的分工**（本轮据 PRED-ALGO-003 修正）：
#     边际率 P = 计数/曝光            —— "这个槽会不会开始做"，对可重复行为正确；
#     危险率 h = 计数/(曝光−之前累积) —— "如果到现在还没做，这个槽做的概率"，
#                                        它使**时刻分布免费得到**：P(第 t 槽才发生)=h(t)·Π(1−h(s))，
#                                        中位数即预计时刻。只描述"当天第一次"，
#                                        第二次及以后由复发间隔承担。
#     累积率 CumP = (之前累积+计数)/曝光 —— "到这个点为止今天通常做了没有"，缺失检测的落点。
#   三者共用同一批原始计数 {C, E, CumC}，派生时一次算出，不各存一套。
# - 一阶是刻意的：本层只给一阶原料，高阶组合（多步前缀、情景识别、语义关联）归上层。
# - **伴随值必须与概率同源**：给了槽位并命中联合格子时，``count``/``n_eff`` 说的就是那一格的
#   命中数与机会数。把槽内概率配上全时段的伴随值，会让"这一格只见过一次"读成"n_eff≈19 的
#   实测结论"，下游的支持度门直接放行。
# - 留白是**两条曲线**不是一个常数：h_不规律（认识这些动作但此刻没有赢家，= 分布的熵）与
#   h_未见（可能发生没见过的事，Good-Turing 式 escape ≈ 只出现一次的动作数/总数）。
#   两者由查询层**从格子的计数分布现算**，不额外存储。escape 高 = 这个格子太杂，别说话。
# - 动态性内建：夜间整棵重建、原子发布、时间衰减、旧规律自然淡出——树每天更新，无新增机制。
#
# ── 估计纪律（一趟扫描；三条卫生是底线）─────────────────────────────────────────────────
# - 计数：每天每槽每动作**封顶 1**（否则 P 不是概率、lift 失去"倍数"含义），同时另存不封顶的
#   原始次数以保住强度信息。权重 = 时间衰减（长 τ 与短 τ 各算一遍，后者供趋势）。
# - 曝光：只数**被观测到的**槽。两类 gap 都扣减覆盖——"未观测"显然扣，"没读懂"同样扣，
#   因为那段时间他真做了我们也记不下来，与没在看等价。上游覆盖信号未接入时按既定退化假设
#   `covered ≡ 1`（代价：曝光分母偏大、概率偏低，已知并接受）。
# - **收缩顺序（本轮修正的技术错误）**：必须从**假设最弱的借用**开始。
#     (周几,槽) ← (周几,池化邻域) ← (跨周几,池化邻域) ← 该动作全天平均
#   时间邻域借用假设"相邻时刻率相近"（弱，通常成立）；跨周几借用假设"各周几率相近"（强，
#   对周规律行为**直接错误**）。顺序颠倒会系统性惩罚周规律——而那正是加周维度要捕捉的东西。
#   实测：12 周数据下先跨周几把真值 0.92 压到 0.69，先时间池化压到 0.75。
#   池化在钟面上是**环形**的（23:50 的邻域含 00:05），可选 von Mises 圆周核。
# - 转移配对：每条 occurrence 找窗口内**下一个开始**（**开始锚定**——ended_at 常不可知）；
#   自转移允许。三条规矩：① 行为树标了 concurrent_with 的**分到并行边**，不算转移——否则
#   "吃饭→看手机"这种同时发生会被记成假因果，且把真正的下一步（吃完饭洗碗）挤掉；
#   ② 窗口内有观测空洞则该对**删失**，不计为"无转移"；③ 分母**含 ∅**，
#   使 Σ_b P(b│a) + P(∅│a) = 1，那个 P(∅) 正是提醒逻辑最依赖的数。
# - 复发间隔：同一动作相邻两次的加权分位数。它**不分桶**，所以不受"频率越低越没救"的诅咒，
#   是月频行为的唯一解法。
# - 消歧重复不计数：original_name 非空的 occurrence 是已知重复（撞车消歧的保命阀留痕），
#   夜批机械跳过——标记在写入时打好，本层只认标记不做判断。
# - 三条不可省的卫生（省了数字就是错的）：**收缩**（稀疏槽出 0/1 极端值）、**覆盖分母**
#   （空洞系统性压低概率）、**转移删失**（观测中断不是"之后什么都没做"）。
# - 槽索引按**本地时间**构造；夏令时当天不是 96 槽（92/100），偏移随 occurrence 各自还原。
#   实现落点（已测）：槽位取本地时分，跳表后的 03:05 落 03:00 那个槽而不是"距午夜两小时零五分"
#   那个槽；回拨那天重复的一小时两次落同一个槽、被当日封顶折成一次。曝光那边仍按 96 槽记，
#   于是春季那天有 4 个不存在的槽被记成"在看但没发生"——一年一次、只影响 4 个槽的分母，
#   已知并接受；真按 92/100 记账要给曝光引入日历时区依赖，代价远大于收益。
# - 单人一年万条 occurrence 毫秒级；**每夜整棵重建，不做增量**——重建成本低，而增量会让口径漂移
#   （kinds 词表变了要用新口径重数历史）。副作用：**树的数值不保证跨夜连续**，上层不得假设
#   "昨天 0.8 今天还是 0.8"。
#
# ── 记账口径（评审实测修正，改动这几条之前先重跑对应测试）─────────────────────────────
# - **计数与曝光必须同口径**。曝光按覆盖比例记，所以：① ``earlier_days`` 也按覆盖比例、且
#   只记在有覆盖的槽上——记在无覆盖的槽上会造出"有计数无曝光"的格子（夜批硬失败），记成整份
#   权重会把危险率风险集 ``E − earlier`` 压成近零的正数（实测危险率炸到 613）；
#   ② 真的发生过的槽一律按"在看"记满曝光——occurrence 与 gap 出自不同判断链，重叠是现实而不是
#   矛盾，硬失败会让一条落在空白里的行为打掉整夜重建。补满曝光同时是保守方向（分母变大）。
# - **危险率的先验必须用风险集分母**（``Σfirst / (总曝光 − Σearlier)``）。用总曝光会把先验
#   系统性压低 3–4 倍，行为在一天里越早发生偏差越大。
# - **边的归一化**：``∅`` 与其余目标走**完全相同**的收缩（同一个先验构造、同一个伪计数）。
#   准确说法是"对全部已知目标（含 ∅）求和为 1"；已发布的边是稀疏的，列出来的加起来 ≤ 1，
#   差额正是先验分给"该源从没去过的目标"的质量，也就是这条边上的逃逸。旧写法转移边收缩、
#   ∅ 是裸比值，生产档下总和只有 0.93。
# - **并行边不走后继搜索**：直接从行为树声明的对里数。它既不受转移窗口约束（长时段重叠很常见），
#   也不该因为中间插进一个真后继而消失；存储只存一个方向，**对称闭包在查询层取**。
# - ``∅`` 是本层的哨兵，真实 kind_token 不得叫这个名字（否则无后继的账会静默覆盖真转移边）。
# - ``reminded`` 为真的 occurrence **硬拒**：干预账本未建，数进去就再也分不开。
#
# ── 给上层的组合契约 ─────────────────────────────────────────────────────────────────
# - **禁止朴素相乘**。节点概率与边概率共享时钟信息，相乘等于把基线乘两遍：周二19:00 且刚吃完饭
#   打球，两个边缘 0.85×0.08=0.068 与真值 0.83 差一个数量级。实测佐证：朴素把全部条件相加会
#   灾难性过度自信，log-loss 达基线 13 倍。
# - **正确顺序**：① 优先查**联合格子**——边带槽位直方图，`P(b│a, d, t) = 该边在该槽的计数 /
#   该槽内 a 的未删失次数`，无需任何独立假设。那个分母**含 ∅**，且不额外存一份：∅ 边同样带
#   槽位直方图，分母由该源全部去向的直方图现加（多存一份就多一份漂移的机会）。
#   少了 ∅ 那一半，"这个点做完 A 通常就收工"会被算成"必然接着做 B"。
#   联合格子**刻意不收缩**（裸比值，一次观测就是 P=1.0）：它的伴随值（该格命中数与机会数）
#   如实暴露证据量，支持度门槛归判决层。给它套边缘那套收缩，先验会把稀疏格子拉向全局份额，
#   恰好抹掉联合查询存在的意义（无独立假设的实测）。同一函数里两套估计纪律并存是设计不是
#   疏漏，不要"修"它。② 联合样本不足时退回 `P ≈ P_基线 × lift₁ × lift₂`
#   （相乘的是 lift 不是 P，只叠加提升不重复叠加基线），超过 1 截断，并**标注为近似值**。
# - **查询必须全量返回**，不得 top-N by P：重要但低频的行为（每周二打球）在按 P 的排行榜上永远
#   排在洗手、看手机后面，筛掉它们等于在树这一层就废掉了预测范围里的一半。
# - **一次查询钉住一代**：跨代混读会让节点与边出自不同批次的统计，组合出来的数字互不一致
#   （旧 PRED-STORE-001 记录过这种静默失真的真实案例）。落点：查询层每个函数都接一棵完整的
#   ``PredictionTree``，本层没有"按需取一个格子"的接口，也就没有中途换代的机会。
# - **``config_digest`` 只覆盖会改变数字的参数**（``estimation_parameters()`` 的 11 项）：
#   重建间隔与保留代数是运维节奏，算进指纹会让"改一次保留代数"把全部已发布的树伪装成
#   "出自另一套统计"而被读侧拒绝。
# - 当日实况**不来自本层也不来自行为树**（行为树要等封口）：来自判断存储，且需要用
#   `behavior://kinds.md` 的 `token_for` 纯查表把原始名换成 kind_token 才能与树对齐；
#   查不到即新名字，树上本来也没有它的统计，跳过。**这条链路零 LLM。**
# - 风险集（"今天还没发生的"）由上层拿累积率对比当日实况自行构造，本层不持有当日状态。
#
# ── 日循环 ───────────────────────────────────────────────────────────────────────────
# - 白天在线：观测 → 融合 → occurrence 落树（现有链路不变）；上层查**昨夜的树** + 自己维护的
#   当日实况。本层白天零动作、全程零 LLM。
# - 夜间批处理：整棵重建 → 校验 → 原子发布（两阶段：先物化校验，再统一翻转指针，任一失败
#   整批不激活、旧代继续服务）→ 清理超出保留代数的老代。首次出现的新动作当天预测不了，
#   留白接住，当晚进树。
#
# ── 已验证（继承自 PRED-REGULARITY-001，结论仍然有效）─────────────────────────────────
# - 合成数据（答案已知）：机制识别全部正确；阴性对照（未植入星期效应的 dow）全列为零——
#   度量不会凭空造信号。
# - CASAS HH 六户真实住宅：可预测性分层真实存在（跨度 0.39–0.60），高端稳定是做饭/吃药/起床/
#   进门，低端稳定是招待客人/在桌前工作。
# - 方法论教训：per-type 统计必须在**全部检验槽**上算（只数发生槽会让高频行为靠自相关刷虚高，
#   且真实数据上看不出来）；验证数据集的标签必须是**行为事件**而非瞬时状态（LSC-ADL 的假阴性
#   是测错了对象）；lifelog 适配参数（同标签中断多久算新一次）会实质改变结论，必须做敏感性测试；
#   **公开数据集只能验证方法有效性、不能定我们的参数**（CASAS 住户多为退休老人，星期无用是
#   这群人的属性，不是人类行为的属性）。
# - 置换检验降级为**离线回归工具**（验证分层跨度不塌、纯时钟共现的假转移不显著、随机转移源
#   提升≈1），不进每夜生产路径。
#
# ── 已关闭的支线 ─────────────────────────────────────────────────────────────────────
# - 以 L1-正则 logistic 回归替换计数估计：实测只在正例 ≥300 的高频行为上稳定 +0.011 bits，
#   中低频平手；且 L1 按单系数惩罚系统性偏向低基数条件，解释被编码方式扭曲，要修得上 group
#   lasso 并重做校准与置换——收益配不上代价，**不换**。
# - 情境格子、情景谓词、前向选择链作为最终产物、按行为组织输出、两张独立概率表：全部退役，
#   其中"两张表"不是删除而是**合并进树**（表一成为节点、表二成为边）。
# - 独立的"规律性度量机器"退出核心路径：算条件概率就是树的计算本身；判可预测性归上层
#   （由 n_eff+lift 承载证据、上层划线）；选条件随条件定死为"时刻+周几+上一步"而消失。
#
# ── 待定（不在本次实现范围）─────────────────────────────────────────────────────────
# - 参数定档：13 个参数全部无数据依据（详见 config.py）。定档方法是 evaluation 的留出回测。
# - 旬维度（月初/中/末三桶）：纯增量，等真实数据算出 lift_旬 显著再加。
# - 干预混淆：occurrence 的 reminded 标记已在行为树就位但恒为 False（干预账本未建）。
#   被提醒后的发生不得进入自然率——此标记必须在提醒功能上线**之前**生效，事后无法分离两种数据。
# - 行为侧规律沉淀进 memory profile 的通道（上层需要"对人的认知"来判断"为什么规律"，
#   而行为侧学到的东西目前没有通道进入 memory）。涉及"memory 是否开第二个写入口"的纪律问题。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime

from prediction.errors import PredictionTreeError

MINUTES_PER_DAY = 1440
WEEKDAYS = 7


def slot_count(slot_minutes: int) -> int:
    """一天被切成多少个槽；槽宽必须整除一天。"""

    if isinstance(slot_minutes, bool) or not isinstance(slot_minutes, int):
        raise PredictionTreeError("slot_minutes must be an integer")
    if slot_minutes <= 0 or MINUTES_PER_DAY % slot_minutes:
        raise PredictionTreeError("slot_minutes must divide 1440 evenly")
    return MINUTES_PER_DAY // slot_minutes


@dataclass(frozen=True, slots=True)
class SlotKey:
    """钟面主干上的一个位置：(周几, 槽位)。

    周几由日期机械算出（0=周一），不是"工作日/周末"这类语义判断——树只当它是个键。
    """

    weekday: int
    slot: int

    def __post_init__(self) -> None:
        for name, value, upper in (("weekday", self.weekday, WEEKDAYS), ("slot", self.slot, None)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PredictionTreeError(f"{name} must be a non-negative integer")
            if upper is not None and value >= upper:
                raise PredictionTreeError(f"{name} must be below {upper}")

    @classmethod
    def of(cls, moment: datetime, *, slot_minutes: int) -> SlotKey:
        """把一个**本地**时刻映射到钟面位置；调用方保证 moment 已是本地时间。"""

        if not isinstance(moment, datetime) or moment.utcoffset() is None:
            raise PredictionTreeError("moment must be a timezone-aware datetime")
        total = slot_count(slot_minutes)
        index = (moment.hour * 60 + moment.minute) // slot_minutes
        # 夏令时当天本地时钟可能跳变，但"时分 → 槽位"的映射本身仍然良定义；
        # 那一天缺失/重复的槽由曝光分母如实反映（见 PRED-TREE-001 估计纪律）。
        return cls(weekday=moment.weekday(), slot=min(index, total - 1))


@dataclass(frozen=True, slots=True)
class NodeCounts:
    """一个 (周几, 槽位, 动作) 格子的原始账本；三种率全部由它派生。

    四个计数分工（都是衰减加权的"天数"，不是次数）：

    - ``occurred_days``：该槽发生过的天数（同日同槽封顶 1，第几次都算）→ **边际率**的分子；
    - ``first_days``：当天**首次**发生落在该槽的天数 → **危险率**的分子；
    - ``earlier_days``：当天在该槽**之前**已经发生过的天数 → 从曝光里扣掉，得到危险率的风险集；
    - ``raw_occurrences``：不封顶的原始次数，保住"今天洗了三次手"这类强度信息。

    ``earlier_days + first_days`` 恰好是"当天在该槽或更早发生过"的天数（``first`` 只在首次那个
    槽计一次，不会与 ``earlier`` 重复），所以**累积率**也由它们直接得到，不需要跨槽累乘。
    """

    occurred_days: float = 0.0
    first_days: float = 0.0
    earlier_days: float = 0.0
    raw_occurrences: float = 0.0
    recent_days: float = 0.0

    def plus(
        self,
        *,
        occurred_days: float = 0.0,
        first_days: float = 0.0,
        earlier_days: float = 0.0,
        raw_occurrences: float = 0.0,
        recent_days: float = 0.0,
    ) -> NodeCounts:
        return NodeCounts(
            occurred_days=self.occurred_days + occurred_days,
            first_days=self.first_days + first_days,
            earlier_days=self.earlier_days + earlier_days,
            raw_occurrences=self.raw_occurrences + raw_occurrences,
            recent_days=self.recent_days + recent_days,
        )


@dataclass(frozen=True, slots=True)
class SlotExposure:
    """一个 (周几, 槽位) 的曝光；与动作无关，是全部率的公共分母。"""

    observed_days: float = 0.0
    recent_days: float = 0.0

    def plus(self, *, observed_days: float = 0.0, recent_days: float = 0.0) -> SlotExposure:
        return SlotExposure(
            observed_days=self.observed_days + observed_days,
            recent_days=self.recent_days + recent_days,
        )


@dataclass(frozen=True, slots=True)
class NodeStatistics:
    """一个格子的发布成品：三种率 + 两个 lift + 可信度 + 趋势。"""

    marginal: float
    hazard: float
    cumulative: float
    lift_all_day: float
    lift_weekday: float
    n_eff: float
    trend: float | None
    counts: NodeCounts


@dataclass(frozen=True, slots=True)
class IntervalQuantiles:
    """间隔分布的三个分位（秒）；回答"该在多久之后说"。"""

    p10: float
    p50: float
    p90: float
    sample_count: float

    def __post_init__(self) -> None:
        if not (self.p10 <= self.p50 <= self.p90):
            raise PredictionTreeError("interval quantiles must be non-decreasing")


@dataclass(frozen=True, slots=True)
class EdgeStatistics:
    """一条边的发布成品。``slot_histogram`` 支持无独立假设的联合查询。"""

    count: float
    probability: float
    lift: float
    n_eff: float
    intervals: IntervalQuantiles | None
    slot_histogram: dict[SlotKey, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecurrenceStatistics:
    """一个动作自身的复发间隔；月频行为靠它，不靠月桶。"""

    intervals: IntervalQuantiles


@dataclass(frozen=True)
class PredictionTree:
    """一代已发布的完整树；``config_digest`` 防止跨参数混读。

    一次查询必须**钉住一代**——跨代混读会让节点与边出自不同批次的统计，组合出来的数字
    互不一致（旧实现真实发生过这种静默失真，见 TODO(PRED-DOWNSTREAM-001) 的发布协议一节）。
    """

    built_at: datetime
    reference_day: date
    config_digest: str
    slot_minutes: int
    nodes: Mapping[tuple[SlotKey, str], NodeStatistics]
    edges: Mapping[tuple[str, str], EdgeStatistics]
    parallels: Mapping[tuple[str, str], EdgeStatistics]
    recurrences: Mapping[str, RecurrenceStatistics]
    exposure: Mapping[SlotKey, SlotExposure]
    # 每个动作不看时刻的整体发生率：稀疏存储下"这一格没有记录"的兜底答案。
    baselines: Mapping[str, float]
    actions: tuple[str, ...]
    observed_days: int
    censored_transitions: float

    def __post_init__(self) -> None:
        if not isinstance(self.built_at, datetime) or self.built_at.utcoffset() is None:
            raise PredictionTreeError("built_at must be a timezone-aware datetime")
        if not isinstance(self.reference_day, date):
            raise PredictionTreeError("reference_day must be a date")
        if not isinstance(self.config_digest, str) or not self.config_digest:
            raise PredictionTreeError("config_digest must be non-empty text")
        # 槽宽随树一起发布：读侧要把"此刻"映射成槽位，而 config_digest 是不可逆的。
        slot_count(self.slot_minutes)


@dataclass(frozen=True, slots=True)
class ObservedAction:
    """喂给估计器的一条已归一化的行为记录。"""

    action: str
    started_at: datetime
    day: date

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or not self.action:
            raise PredictionTreeError("action must be non-empty text")
        if not isinstance(self.started_at, datetime) or self.started_at.utcoffset() is None:
            raise PredictionTreeError("started_at must be a timezone-aware datetime")


@dataclass(frozen=True, slots=True)
class ObservedGap:
    """一段观测空白；两类 gap 在本层同等对待，都扣减曝光。"""

    started_at: datetime
    ended_at: datetime

    def __post_init__(self) -> None:
        for name, value in (("started_at", self.started_at), ("ended_at", self.ended_at)):
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise PredictionTreeError(f"{name} must be a timezone-aware datetime")
        if self.ended_at < self.started_at:
            raise PredictionTreeError("a gap must not end before it starts")


@dataclass(frozen=True)
class BehaviorSnapshot:
    """一次重建的输入快照。

    ``actions`` 按 ``started_at`` 升序（估计器的前置条件），``concurrent`` 里的下标指向
    排序**之后**的位置。``skipped_duplicates`` 只作可观测量，不参与任何计算。
    """

    actions: tuple[ObservedAction, ...]
    gaps: tuple[ObservedGap, ...]
    concurrent: tuple[tuple[int, int], ...]
    skipped_duplicates: int

    @property
    def latest_day(self) -> date | None:
        """快照里最晚的一天；调用方据此选夜批的基准日。"""

        days = [item.day for item in self.actions]
        # 跨日的空白算到它**结束**那天：一段停到今天早上的空白说明今天也在观测范围里。
        days.extend(gap.ended_at.date() for gap in self.gaps)
        return max(days) if days else None


__all__ = [
    "MINUTES_PER_DAY",
    "WEEKDAYS",
    "BehaviorSnapshot",
    "EdgeStatistics",
    "IntervalQuantiles",
    "NodeCounts",
    "NodeStatistics",
    "ObservedAction",
    "ObservedGap",
    "PredictionTree",
    "RecurrenceStatistics",
    "SlotExposure",
    "SlotKey",
    "slot_count",
]
