"""观测片段到行为判断的语义融合层。

本层是行为链上唯一调用模型的一环：观测清洗是无模型的确定性归一，判断的落盘是确定性的身份与
事务，中间这一步负责"看完这段观测，我判断这里发生了什么"。

## 产物是判断，不是事实

融合不产出"一件已经发生完的事"，而是**我们在某个时刻对一段观测做出的判断**。同一段行为在不同
时刻可以有不同判断，它们都为真，只是知道的时间不同；后来的判断不修改先前的判断，只新增一条指向
它。三层递减由两个字段决定——``behavior`` 与 ``goal`` 的空与非空——而不是三种并列的结构。

## 本层不碰行为树

判断到行为树文档（occurrence 与观测空白 gap）的归约属于**另一层**：链要等老出引用窗口才
封口（后续的延续、修正、结果都可能在窗口内到达），节奏与融合相反。把它塞进融合，融合就永远
完不了。

TODO(BHV-REALDATA-001): 首次用真实日/周尺度数据（EgoLife A1_JAKE 七天：第一人称字幕每 2s
一条抽样到 ≥5s、转写带说话人、每天 11–23 点、单天约 4700 条观测）走完 观测→融合→归约 后暴露的
一批缺口。全部是实测撞出来的，不是推理；在 benchmark 那 70 条秒级用例上一个都不触发。按
"处置方式 → 容量 → 校验 → 队列 → 语义层"分组，每条带方案、影响与实例。实验期用驱动脚本里的
猴子补丁绕过（不进仓库），正式修法待用户裁定。

── 处置方式：记账疏漏一律整批拒，实测应降级 ──【已落地 WP1，2026-08-30】────────────────
以下四条校验本身都对（我们自己产物的自洽），错在处置：整批拒绝→让模型重来 3 次→退避 60/120/240s
再来。真实数据上它们高频触发，一个段反复失败就把**严格串行**的队列封死一整天。prompt.py 模块头
自己写的纪律是"记账疏漏在装配层降级、不整批拒"，这四条没照做。
**现状**：四条均已在 ``assembly`` 降级并以 ``BehaviorJudgementBatch.degradations`` 留痕
（去重取首条；剔掉不在场主体、全不在场则整条降为没读懂并剪掉指向它的关系；``continues`` 指向
已完成目标的边剪掉；goal 的 basis 全无帧归属则 goal 置空），``validation`` 保留为后置断言，
worker 把计数记进 ``fusion_degradations`` 可观测事件；回执字段与版本升级留到容量一组一起做。
验证：单元测试按四类失败形状各造一份；真实对照（EgoLife DAY1、真实装配代码、无补丁）：
368 段 **0 次硬拒**，1421 条判断（与补丁版 1422 一致），信号 subject_absent 12 /
continues_completed 57 / goal_dropped 1 / duplicate_assignment 1。对照中撞出并修掉一条连锁：
同批内的 continues 边不能剪——判重规则靠它把"同一件事被看成两条"认作一条，剪掉反而撞成
硬拒；现只剪指向先前上下文（C 行）的那种，同批内保留并留 "kept" 信号。
另记：同一份 DAY1 观测这次切出 368 段、上周是 198 段（切段器在尾部找最大空白下刀，结果对
入队时机敏感）——段边界不确定意味着融合产物不可逐字重放，属于第 6 条容量/切段一组。
- 同一帧内重复指向同一条判断（assembly._assignments "assigns … twice"）：一天里 2 个段连续
  3 次触发。方案：装配前按 judgement_no 去重、保留首条并计数。
- subjects 不在覆盖片段的 participants 里（validation._require_subject_present）：模型会从
  "她 / 大家 / 众人"推出在场者；一天 10 次。方案：剔掉不在场的名字（至少留一个）并计数；
  同时上游契约要求视觉片段的 participants 列出画面里看到的所有人、转写片段包含佩戴者/听者——
  只列说话人时"Jake 与 X 交谈"必然过不了。
- continues 指向已 completed 的先前判断（assembly._require_continuable）：七天里四天各自反复
  触发，退避后重试仍失败。方案：把这条边改写为无关系（或按提示词建议改成 supersedes）并计数，
  不整批拒；prompt.py 已记录把约束贴在 C 行上仍压不住。
- goal 非空但 basis 为空（judgement.BehaviorClaim 不变量）：这是一次**连锁**——装配层
  _reduce_coverage 先把没有帧归属的 basis 条目丢掉（合理降级），丢空后 goal 还在，claim 不变量
  随即硬拒。方案：_reduce_coverage 丢空 basis 时同步把 goal 置空（"少说"降级）并计数。

── 容量：段容量与模型输出预算不匹配 ──【已落地 WP3，2026-08-30】───────────────────────
**现状**：段容量进 ``Config.behavior.max_fragments_per_segment``（默认 60），切段与融合共用同一份
BehaviorFusionConfig；输出截断不再原样重试——整段降为一条没读懂判断（时间轴上留下"观测到了但
没读懂"的空白、覆盖索引照记、队列照走、留 ``segment_truncated`` 信号）；example.yaml 的 chat
路由 timeout 300s、max_output_tokens 8192。溯源只留 ``chain_digest``（原料发布即释放，逐条 id
只会是死引用，实测占文档 70%），basis 步骤不再内联 observation_ids，Markdown 正文只渲染首尾
步骤；单文档仍超限时截断 basis 中段留信号而不是整轮失败。kinds 归一每 25 个名字 CAS 落盘一次并
续 sweep 租约、瞬态错误有界重试、词表撞顶时超限名字暂以原始名作 token 留信号。树新增
``read_day``（目录只解析一次），预测树夜批按天整块读。未做：kinds 批量归一（改提示词，随词表方案）。
- BehaviorFusionConfig.max_fragments_per_segment=512 / max_segment_span_seconds=1800 下，
  逐帧归属表按片段数线性增长、判断本体按判断条数增长，deepseek-chat 8192 输出上限（也是它的
  硬上限）下 **512、160、100 条的段都实测截断**（对话密集段判断多），60 条才稳。而截断的段重试
  还是同样大小、必然再截断，最终把队列封死。方案：段容量按输出预算反推（当前档位 ≈60），且
  截断失败时应**重切成更小的段**而不是原样重试；容量类参数并入 Config（BHV-FUSION-003 余项）。
- example.yaml 的 chat timeout_seconds=30 对日尺度段不够（第一次就 ReadTimeout）；
  max_output_tokens=null 走厂商默认 4096，更早截断。方案：行为侧路由单独给 timeout≥300s、
  max_output_tokens 取模型上限。

── 队列：周尺度下吞吐塌掉 ──【生命周期部分已落地 WP2，2026-08-30】──────────────────────
**现状**：原料"消费即释放"（用户裁定：真正的数据只在行为树上，不做基于保留期的堆积）——作业
COMMITTED 即 discard；判断与交付在链发布、账本写完之后由归约同轮删除（``reduction/runner._release``）；
"处理过没有"改由覆盖索引回答（``behavior/fusion/coverage.py``，按 judged_at 日分区、窗口 =
上游最大补发跨度、整目录过期，``Config.behavior.coverage_window_days``），回执与消费账本按同一窗口
过期；投递口对同身份观测去重不再整批拒收。热区因此只剩未封口的最近一个窗口，第 8 条的全量扫描
不再需要分区。"当日实况"契约改为已封口读树、未封口读判断存储（prediction/model.py、behavior/model.py
已改）。未做：按自然断点并行（先量释放后的单作业耗时再定）。
- 融合循环每拍 enqueue_ready 全量扫描观测存储（8.5 万条）+ 作业存储 claim/stage/commit 各全量
  解析一次（2690 个作业文件、每个带 60 个观测 id）：单作业从 10s 涨到 **62–120s**，一周要
  45–90 小时。方案：BHV-LIFECYCLE-001——观测/作业按时间分区；COMMITTED 作业即时
  discard_committed（方法已有、运行时无人调用）；入队扫描按水位增量。实验期用"120 个作业的
  滑动窗口 + 即时清理"压回 8–13s。
- 串行队列的语义必要性只有 1 小时回看窗口：跨夜没有观测，按天并行融合再合并归约与串行结果
  逐字相同（三个存储都是内容哈希平面文件）。方案：队列按"回看窗口内无观测"的自然断点分片
  并行，或至少按日分片。
- 失败作业退避 60→120→240→… 后仍失败会 FAILED 封锁整条队列；worker 硬崩留下的 300s 租约
  让重启后先空等 5 分钟。方案：截断/校验类失败按上文降级或重切后不再计入 attempts。

── 语义层：真实密度下断链 ─────────────────────────────────────────────────────────
- 一天 947 条 occurrence（微动作粒度，见 BHV-FUSION 折叠问题）超过 BehaviorSemanticConfig
  .max_direct_entries，日概览生成直接失败（"behavior snapshot exceeds its direct entry bound"）。
  方案：先解决折叠粒度；日概览对超限目录应分片生成或按小时目录再分一层，而不是整天放弃。
- 覆盖信号未接入时一天 4 小时真实无观测记为 0 条 gap，"没读懂"也是 0（模型对微动作从不说
  读不懂）——已知退化，但意味着曝光分母在真实数据上系统性偏大。

── 折叠粒度（最大的语义问题）──【已落地 WP4，2026-08-30】────────────────────────────
**裁定**（用户，2026-08-30）："我们需要的是能够在规律的或者被约束下的行为，可以做到一个提醒或者
代劳，那么这么一些无意识的小动作是不应该进入事件的。"由此：一条判断 = 一个**可提醒或可代劳的
行为单位**；步骤（拿起、放下、走到桌前）进那件事的 basis；无意识小动作与过渡帧**归属为空**
（frames 填 []：看到了、看懂了、不构成任何事——不是 gap、不是没读懂，树上不写，曝光分母照算，
占比作为信号 ``fusion_unowned`` 报出）；goal 退回可读字段，"goal 空则 basis 空"的不变量取消；
"实际规律不规律"留给预测层。完整裁定与文献旁证见 ``judgement.py`` 模块头。
**现状**：提示词 v15→v18、schema 的 frames.assignments 去掉 minItems、``BehaviorJudgementBatch``
带 ``unowned_fragment_nos``、回执带 ``unowned_observation_ids``/``unowned_ratio``。benchmark 加
``unowned_fragments`` / ``behaviors_present`` / ``forbidden_behaviors`` / ``subject_free_fragments``
期望与"可提醒单位保留率"，补 6 条用例（EgoLife 真实粒度 ×2、喝水/锁门无目标单位、压制守卫、
过渡帧）；两条"旁人帧"用例改为 subject_free 口径（v17 下模型在"分流成旁人判断"与"无归属"之间
摇摆，两者都没吸收进主体）。
**验证**（EgoLife DAY1 全天 12,394 条观测、无补丁、按纪律逐版真实对照）：
  版本   可读判断  不同名字  中位时长  0秒判断  basis≥3  无归属   姿态类*  走路类**
  v15    1,422     513       9s        553      116      —        —        —
  v16    1,051     273       23s       237      153      3.4%     49       46
  v17    945       238       26s       172      134      5.2%     2        42
  *坐着/看着大家/观看/俯身/坐在原地；**行走/走路/向前走/移动/走动。
  归约后的 occurrence（同日）：v15 1,049 → v17 348 + 21 gap（零调用重放，词表命中账 348 一一对应）。
  **并行实现的对照**（主树另一条 WP4 线，同源 v16 快照、独立演化，2026-08-31 跨会话核对口径）：
  可读判断 1,422 → 989、occurrence 1,049 → 415；本线 v17/v18 多收的一档是"姿态判断经 C 行
  continues 链传染"（见下）。两条线的提示词谱系与 benchmark 协议命名待用户裁定归一。
  转头/点头/扶眼镜/转身/笑在 v16 起全部消失；v16 实测"坐着/看着大家"经 C 行 continues 链传染
  （孤立重跑同一窗口 3/3 干净，起点是抖动、传染是结构性的），v17 补一句"讨论中的姿态填 []、不延续
  姿态判断"后清零。v17 第一版措辞（"归给交谈那条"）把通话吞进做饭——并行用例 6 次只过 1 次——
  改窄后 9/9；这是"提示词每句都要对照实测"的又一例。benchmark 76 条：回归 68/69→改口径后全过、
  探查 7/7、无归属 4.9%、可提醒单位保留率 100%。
**未完 / 边界**：① 走路类（≈42/天）未收：在店里/路上的走动多半是"逛超市/去某处"的步骤，提示词
  没专门写，按纪律留待下一轮对照；② 长交谈仍按 60 条一段各出一条（参与讨论 ×80 等，跨段靠
  continues 并链）——B′"先合并再判断"按裁定先不做，让数据说；③ 装配层"全段无归属则报错"保留：
  DAY1 两轮 0 次触发，benchmark 里 4 帧全是站窗边的用例被迫立一条"站在窗边"，是已知边界不是 bug；
  ④ 折叠后判断数 −34%，远小于"每天 50–85 条真实行为"的目标——剩余主要是交谈按段拆分与走路类，
  不再是微动作。词表膨胀（BHV-KINDS-002）在此基础上继续。
  ⑤ v17 残留的 8 条姿态判断全在 11:37–11:42 连续三段，机制是**上下文先例**而非内容：这三段与
  种子段（11:37:09，12 条）孤立重跑 3/3、5/5 全干净（姿态帧无归属）；DAY1 里是 11:35–11:36 两条
  边缘判断（"观看大家收拾东西""走动"）进了 C 行，下一段照先例立"转身"并 continues，"姿态模式"
  延续到 v17 那句"不延续"生效为止（v16 下同类链延续 25 分钟）。C 行是参照不是模板——**v18**
  按模块头"贴在模型正在看的对象上"把这句贴到【先前的判断】块头（``render_context_judgements``），
  不再加正文。对照（用户裁定只跑对应数据、不跑全量）：11:35:20–11:42:05 原样切 5 段带上下文链
  3/3 干净（种子"观看大家收拾东西"在 C 行里，后段零姿态条）；相关 4 条用例（两条并行、真实讨论、
  压制守卫）×3 全过。提示词到此封版（用户裁定）：残留躲不掉的部分不再靠提示词追，后续若要
  "不进树"，闸门放在 kinds 归一（kind 标"非单位"、归约不发布），与 BHV-KINDS-002 同批。

── 端到端审计修复（2026-08-31，三方审计各两轮，全部有复现脚本；落点各归各处）───────────────
根因 1 **覆盖窗口过载**：7 天覆盖窗口曾同时充当去重判据、封口前沿、释放门槛、撞车命名空间；
  交付只在"其判断发布后"才释放，无判断/重复投递/被隔离判断引用的交付永不释放，窗口一过被当成
  未融合重入队重跑模型、封口视界钉死、旧回执/同址硬冲突封死队列与检查点。修：覆盖记录在其观测
  仍在存储里时**不过期**（``coverage.expire(retain=…)``）；观测按"已覆盖且无判断引用"每轮全量释放
  （``_release_unreferenced``，不再靠判断反查）；撞车占用集合并入 ``tree.exists``；gap 同址按
  ledger-only。不按送达时间丢弃任何观测——下游停机再久也不丢数据。
根因 2 **检查点/merge/刷新缺共同耐久状态**：刷新失败曾扣住检查点，merge 重打后重放撞永久冲突；
  单日超限让整批刷新连坐且集合只增不减。修：待刷新日集合独立耐久（``refresh_pending.json``，
  逐日隔离、成功一天清一天）、``_finish_sweep`` 统一两条返回路径的顺序（写集合→清检查点→刷新→
  释放）、merge/rebuild 遇悬挂检查点拒绝（损坏则指引手删）。
根因 3 **命中账幂等键**：曾按账本条目判"首次"，账本落、账未落的崩溃窗口丢命中；改为检查点内容
  摘要（``hits_applied_checkpoint``，词表 v3，与账同一次 CAS），冻结时钟下也不撞；owner 按
  ``token_for`` 找（merge 后可能是别名）；``-2`` 消歧记录不记账（与 rebuild、预测树同口径）；
  数据时钟钳墙钟当日。
根因 4 **撞车消歧顺序**：曾先占位后物化，占位链物化失败被跳过，幸存者却带 -2 落盘被预测源当重复
  丢掉。修：先干跑物化、通过者才占位；缺失交付按链跳过不整轮失败；树完整性错误不当坏名字吞。
根因 5 **跨段上下文截断**：曾按送达时刻严格小于截断，真实送达抖动下系统性丢掉前半截；改为按
  **事件时间**（判断最后观测时刻 ≤ 本段最早观测发生时刻）+ 已落盘，并列排序加次键。
根因 6 **WP4 降级收尾**：整段无归属曾硬拒逼模型发明→允许空判断批；主体全不在场曾降"没读懂"
  （主体时间轴凭空多 gap、两条同帧集互拒）→ 按 out_of_scope（回执记、不落盘），同帧集没读懂合并。
  对照：相关 7 条 benchmark 用例 ×3 全过（空批、旁人帧、姿态帧）。
周尺度实测（另一条线的冻结周零调用重放，2026-08-31）：9,270 条链的**链组装→命名/物化→发布**是
单阶段长循环，只在阶段边界续约会踩满 sweep 锁 600s TTL。两处根因与修法：
  ① **逐链 ``tree.exists`` 是 O(日目录)**（树自己的 ``read_day`` docstring 早已记载：逐篇读会重新
     枚举整个日目录、13.6 ms/次）。撞车消歧与落盘干跑改为 ``tree.list_day_addresses``——本批涉及的
     每天每类只枚举一次目录。同一现场实测：9,270 条链 72.7s → 0.46s（约 160×）。
  ② **stage 前段整段不写任何东西**（判断枚举、链组装、封口前沿、账本全量加载、命名/物化、干跑），
     实测静默 16 分钟即把租约耗死，事后再多续约点也救不了已过期的租约。现在每段边界都续，且
     判断枚举/前沿扫描/观测枚举/命名/落文档/干跑的循环内部按条数续（发布每 10 条，其余 50–500 条）。
  ③ **环境放大（已证实的根因，不是猜测）**：万级小文件写入触发 Spotlight/fseventsd 索引放大
     （实测 fseventsd 240% CPU），把发布拖出十几分钟的整段静默。逐篇计时定性：9,273 次 publish
     合计 **207.8 秒、最大单篇 0.09 秒**——代码侧没有任何单点卡死；重放根改 ``.noindex`` 后现象
     彻底消失。**部署要求**：行为根若落在被索引的卷上（macOS Spotlight、各类同步盘），万级文件的
     归约会被系统级放大——部署时应把它排除出索引。sweep 锁 TTL 同时收进 Config
     （``behavior.reduction_sweep_lock_ttl_seconds``，默认 600s）：周尺度回填调大它是运维动作，
     不该靠改代码或猴子补丁。
  【NEW-2/3/4 的 DAY1 真实对照】（2026-08-31，用户点头后跑，368 段 ≈ 370 次调用；基线 = 同一份
  DAY1 观测的 v17 产物）。融合层：判断 966→956（可读 945→936）、**作业失败 0**、空判断批 3 段
  （合法产出，不再逼模型编）、subject_absent 降级 20→11 且全部按新口径保留、没读懂观测 83→29、
  无归属观测 639→743——帧更多地走"看懂了但不构成事"的出口，正是 WP4 的方向。归约层：occurrence
  348→432、消费判断 945→936。**碎裂全在单判断条目**（260/75% → 343/79%，+83），多判断链纹丝不动
  （≥2 判断 88→89、≥5 判断 26→26、最长 82→98），即长行为没有被切碎。
  代价来自 v18 C 行块头那句"不要立条也不要延续"：``continues`` 边 1,054→739（−30%），模型把"不要
  延续姿态条目"泛化到了一般延续上。**已裁定（用户，2026-08-31）：接受这个取舍，不收窄措辞、不再
  重跑对照**——"这部分逻辑还是需要等到正式的接入情况下再改，现在不断优化没有意义"。理由：一条长
  记录还是多条短记录哪个更好，取决于预测层/语义关联层真正怎么消费（"一天用 20 次手机"与"一次 30
  分钟"对预测是不同的事实）；消费方接入之前任何"更优"都是假设。**重审的触发点**：预测层或语义关联层
  正式接入、能说出它要的粒度之后，拿那时的真实需求回头看这条，再决定要不要收窄措辞。
  【周尺度验收】默认 600s TTL、零补丁：9,270 occurrence + 3 gap，4.5 分钟发布完，与原树按
  (started_at × kind) 多重集 **0 差异**，第二轮幂等为零；发布速率稳定 ~2,000 条/分**无衰减**
  （①修复前是 2,465 → 155 条/分的单调衰减）。
仍登记未做：刷新持续失败无退避（每轮重试）；``_release_unreferenced``/前沿/入队每轮各扫一次观测
存储（BHV-LIFECYCLE-001 欠账）；归约 sweep 同步 IO 与融合心跳共用事件循环（大数据量下应下沉线程）；
从未命中的词表条目不过期；``coverage_window_days`` 与 ``context_lookback`` 的跨域不等式未硬拒。

── 同一轮实测顺带撞到的其他问题（不属于融合本层，先登记在此，修时各归各处）────────────
- 观测投递：同一批内两条内容完全相同的观测（同一秒同一句，ASR 重复吐出——benchmark 里
  dirty-double-emission 正是这种脏数据）被观测存储以 "observation IDs must be unique within
  a batch" 整批拒收。上游真会这样发；方案：投递口对同身份观测确定性去重（保留一条、来源都记），
  而不是拒收整个交付。
- kinds 归一（behavior/kinds）：benchmark 70 条上已出现实质归错——「与医生通话」吞掉了
  打电话/接电话/接打电话（给母亲打电话、谈合同、做饭时接电话全成了"与医生通话"，8 条），
  「坐在沙发上」吞掉看电视/开电视；反向 做饭/做晚饭/煮饭/准备食材/洗菜切菜 各自独立。它是
  预测树与一切按类统计的聚合键，归错即统计错。方案：解析提示词按纪律用真实词表实测调优
  （TODO(BHV-KINDS) 调优项），并给"别名吸收"加保守约束（别名不得比正名更泛）。
- results_from 无读者：融合按"看得清才标"产出的因果边写进树的 links 后，prediction/source.py
  只读 concurrent_with，语义层也不读——最可信的一类关联产出即丢弃。真实数据上 5/5 全对但
  单天只有 1 条，覆盖率不能指望它做主力。方案：语义关联层读它；不要为提高覆盖率调提示词。
- goal 字段承载力：真实产出里 goal 基本是行为名的同义复述（洗手→清洁双手、点外卖→点外卖），
  "晚饭未解决"这类留下的状态从不出现在 goal 里，而散落在 status（洗菜切菜 abandoned）、
  summary 自由文本（"决定不做饭，改为点外卖"）和日叙事里。设计语义关联时不能把 goal 当状态载体。
- 预测树无反向引用：source.read 把 occurrence 压成 (kind_token, started_at, day)，节点/边键
  上没有任何 occurrence URI；从候选回到"由哪几次发生统计出来"只能全树扫描。方案：重建时随代
  发布溯源旁册（格子/边 → URI 列表），零语义、与代钉死。
- 融合跨窗口引用只回看 1 小时：「三天前挂号 → 今天去医院」这种跨日因果在结构上永远产不出，
  跨日语义支撑没有生产者——这是设计边界不是 bug，登记以免语义关联层误以为能从树上读到。
- 模型偶发漏写 schema 必填字段（'basis' is a required property）：json_schema 校验拒后重试
  即过，不需处置，仅记录频次以便与厂商结构化输出模式对照。
- kinds 归一的调用形状（behavior/reduction/runner._resolve_kind_tokens）：一次 sweep 里对
  **每个不同的行为名各调一次模型、严格串行**，词表只在整个循环结束后 CAS 落盘一次。七天合并
  归约实测：3061 个不同名字 ≈ 3061 次调用（≈2 小时），中途一次网络断连让整个 sweep 抛出、
  几千次调用全部作废重来（首次归约就这样崩了）。方案：① 词表按批增量落盘（每 N 个名字一次
  CAS），已归一的名字走 token_for 快路径，重试不重复调用；② 单次调用对 ModelTransportError/
  ModelResponseError 有界重试，不让一次瞬态错误打掉整轮 sweep；③ 把"一个名字一次调用"改成
  批量归一（一次给模型一批候选名与当前词表），调用数降一到两个数量级——这一条会改变提示词，
  须按纪律实测。
- kinds 词表容量（behavior/kinds/config.BehaviorKindConfig.max_kinds=500）在一周真实数据上
  撞顶：归一到第 1000 个名字时词表已有 498 个 kind（新建率约 50%，别名吸收率低），随后
  BehaviorKindLimitError 让整轮归约失败。500 这个数建立在"一个人的行为类型天然收敛"的假设上，
  而在当前微动作折叠粒度下（转头/点头/扶眼镜各成一类）它不收敛——这是折叠问题的另一个量化
  证据，不是单纯调大容量能解决的；但容量撞顶不该让归约整轮失败，应降级为"超限名字暂用原始名
  作 token + 信号"，并把容量并入 Config。
- 归约 sweep 锁（behavior/reduction/runner._SWEEP_LOCK_TTL_SECONDS=600）只在阶段边界
  ``guard.checkpoint()`` 续约，而 kinds 归一整段（实测 45 分钟）与万条文档的发布各自在一个
  阶段里——七天合并归约实测 LockLostError，整轮作废。方案：长循环内按名字/文档计数周期性
  checkpoint（如每 25 个），或把 kinds 归一移出锁内（词表有自己的 CAS）。
- 单文档容量（behavior/document/config.BehaviorDocumentConfig.max_encoded_bytes=512KB）在
  七天合并归约的发布阶段被撞破（BehaviorDocumentLimitError）：一条跨小时的长链把成千上万个
  observation_ids/judgement_ids 与几百步 basis 全塞进一个 occurrence。两层原因都要查：
  ① 溯源 id 全量内联到 L2 文档不可伸缩，应改为引用（按链/回执存一份、文档只存链身份）；
  ② 长链本身可能是"continues 指向已 completed"降级放行后把多次行为并成一条造成的，落地前
  要拿这批数据量一下最长链的构成。任一情况下发布阶段不应因单个超限文档让整轮 sweep 失败，
  应降级（截断溯源列表留信号）并继续。
- **数据膨胀**（七天实测，单人）：观测 37MB、判断 47MB（11,534 条，从不释放——
  BHV-LIFECYCLE-001）、回执 14MB、行为树 43MB（9,270 条 occurrence，单文档中位 1.5KB、
  均值 2.5KB、最大 535KB，溯源 id 内联占普通文档约 15%）、归约目录 56MB（staged 检查点
  20MB + 消费账本）——**约 200MB/周，线性外推 ≈10GB/年、48 万条 occurrence/年**。而全部设计
  假设是"单人一年万条"（prediction/model.py："单人一年万条 occurrence 毫秒级"），差 **48 倍**：
  预测树每夜全量重建要读并解析 48 万个 Markdown+JSON 文档；入队/作业/封口的全量扫描（第 8 条）
  按同样倍数恶化；观测存储 max_files=100,000 按今晚密度（8.5 万条/周）**两周即满**；日目录
  max_children_per_directory=10,000 在 2,074 条/天的日子下一年内未必触顶，但 L1 概览
  已经在 900 条/天时失败。根因仍是第 17 条折叠粒度（每天 50–85 条真实行为被 1000+ 条微动作
  淹没），但即便粒度修好，观测与判断层的原始数据量不变，生命周期（释放/分区/归档）与溯源
  改引用是独立必须做的：① 观测/判断在被归约消费并出封口窗口后按保留期释放或归档到冷区；
  ② occurrence 只存链身份，溯源 id 列表落到旁侧账本；③ 全量扫描改按日期分区。
- 生产 worker 之外的驱动（脚本、运维手动跑）若不续租，长调用/网络重试超过 300s 就丢租约、
  作业被重新认领并计一次 attempts；worker 自带心跳没这个问题。运维脚本一律复用
  BehaviorFusionWorker._execute_with_heartbeat，不要裸调 runner.execute。
"""

from behavior.fusion.assembly import assemble_judgement_batch
from behavior.fusion.config import BehaviorFusionConfig
from behavior.fusion.coverage import BehaviorCoverageIndex
from behavior.fusion.derivation import (
    FUSION_IMPLEMENTATION_VERSION,
    FUSION_VERSION,
    DurableFact,
    DurableJudgement,
    derive_judgements,
    judgement_payload,
    without_unresolvable_relations,
)
from behavior.fusion.enqueue import (
    DEFAULT_QUIET_PERIOD_SECONDS,
    BehaviorFusionEnqueuer,
    BehaviorFusionEnqueueResult,
)
from behavior.fusion.errors import BehaviorFusionError, BehaviorFusionLimitError
from behavior.fusion.jobs import (
    BehaviorFusionJob,
    BehaviorFusionJobBlockedError,
    BehaviorFusionJobConfig,
    BehaviorFusionJobError,
    BehaviorFusionJobLease,
    BehaviorFusionJobLeaseLostError,
    BehaviorFusionJobNotReadyError,
    BehaviorFusionJobStatus,
    BehaviorFusionJobStore,
    BehaviorFusionQueueSnapshot,
    StagedFusion,
)
from behavior.fusion.judgement import (
    BehaviorClaim,
    BehaviorFact,
    BehaviorJudgement,
    BehaviorJudgementBatch,
    JudgementLink,
    JudgementRelation,
    JudgementStatus,
    JudgementStatusBasis,
)
from behavior.fusion.prompt import (
    FUSION_PROMPT_VERSION,
    FUSION_SYSTEM_PROMPT,
    render_context_judgements,
    render_fragments,
)
from behavior.fusion.receipt import (
    RECEIPT_SCHEMA_VERSION,
    BehaviorFusionReceipt,
    build_fusion_receipt,
    receipt_identity,
    segment_identity,
)
from behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from behavior.fusion.result import BehaviorFusionResult
from behavior.fusion.runner import BehaviorFusionRunner, BehaviorFusionRunResult
from behavior.fusion.schema import JUDGEMENT_FUSION_JSON_SCHEMA, fusion_json_schema
from behavior.fusion.segmentation import BehaviorFusionSegment, segment_observations
from behavior.fusion.service import BehaviorJudgementFuser
from behavior.fusion.store import JUDGEMENT_SCHEMA_VERSION, BehaviorJudgementStore
from behavior.fusion.validation import unreadable_ratio, validate_judgement_batch

__all__ = [
    "DEFAULT_QUIET_PERIOD_SECONDS",
    "JUDGEMENT_FUSION_JSON_SCHEMA",
    "FUSION_IMPLEMENTATION_VERSION",
    "FUSION_PROMPT_VERSION",
    "FUSION_SYSTEM_PROMPT",
    "FUSION_VERSION",
    "JUDGEMENT_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "BehaviorClaim",
    "BehaviorJudgementFuser",
    "BehaviorFact",
    "BehaviorFusionConfig",
    "BehaviorFusionEnqueueResult",
    "BehaviorFusionEnqueuer",
    "BehaviorFusionError",
    "BehaviorFusionJob",
    "BehaviorFusionJobBlockedError",
    "BehaviorFusionJobConfig",
    "BehaviorFusionJobError",
    "BehaviorFusionJobLease",
    "BehaviorFusionJobLeaseLostError",
    "BehaviorFusionJobNotReadyError",
    "BehaviorFusionJobStatus",
    "BehaviorFusionJobStore",
    "BehaviorFusionLimitError",
    "BehaviorFusionQueueSnapshot",
    "BehaviorFusionReceipt",
    "BehaviorFusionReceiptStore",
    "BehaviorFusionResult",
    "BehaviorFusionRunResult",
    "BehaviorFusionRunner",
    "BehaviorFusionSegment",
    "BehaviorJudgement",
    "BehaviorJudgementBatch",
    "BehaviorCoverageIndex",
    "BehaviorJudgementStore",
    "DurableFact",
    "DurableJudgement",
    "JudgementLink",
    "JudgementRelation",
    "JudgementStatus",
    "JudgementStatusBasis",
    "StagedFusion",
    "assemble_judgement_batch",
    "build_fusion_receipt",
    "derive_judgements",
    "fusion_json_schema",
    "judgement_payload",
    "without_unresolvable_relations",
    "receipt_identity",
    "render_context_judgements",
    "render_fragments",
    "segment_identity",
    "segment_observations",
    "unreadable_ratio",
    "validate_judgement_batch",
]
