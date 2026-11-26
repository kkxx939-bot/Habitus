适用：改 `habitus/memory/conversation/**`、`habitus/pre/conversation/**`，或 runtime 里的生命周期入口时读。各项默认值以 `habitus/config/example.yaml` 为准。

# Conversation

## 两层与归属

- Conversation 只有 `pre/conversation/messages` 与 `pre/conversation/summaries` 两层。
- messages 保存完整的原始会话事实，严格区分 prompt、completion、tool_call、tool_result。每个 tool_result 必须出现在同 ID 的已知 tool_call 之后、逐字复用它的 tool_name，一个调用只能保存一个终态结果；否则不得进入 Conversation。
- summaries 保存绑定不可变消息片段或连续摘要范围的宽语义历史，不能替代原文参与记忆解析。
- Conversation Schema 只放在 `pre/conversation`，`pre` 里不写存储逻辑（架构测试 `test_pre_contains_only_conversation_schema_and_no_storage_dependency` 守着）。正式路径、live 追加、读取、history 封存、Segment Summary 和自动切段统一由 `memory/conversation` 实现，复用通用 PathLock 与耐久原子文件能力。
- 旧 SessionArchive projector、旧本地文件读写和单会话覆盖式摘要已经退出，不得恢复旧入口。
- 原子替换被强制中断后，只能忽略与该写入器严格命名规则吻合的有界普通临时文件；未知隐藏项和符号链接仍必须失败。
- 锁的获取、fencing 入口与幂等 release 可以有界等待，但不得重放已经进入过的临界区正文；release 的错误不得覆盖正文的原始异常。
- 长期记忆写入只走 `MemoryJob -> Summary（生成即耐久）与 MemoryEditor 纯规划 -> MemoryCommitWriter -> Job 记"卡片已写完" -> L0/L1 refresh -> Job COMMITTED` 这一条主链（没有向量同步与回执，裁定 110、111）。
- 周期压缩和跨层保留期删除只走统一的 Conversation 生命周期管理器。Runtime 的显式单 Conversation 维护入口和独立的 LifecycleWorker 都只调用它；维护循环不混进 MemoryWorker。

## 生命周期主链与保留期

- 主链：`live.jsonl -> history/{segment_id}.jsonl -> summaries/{segment_id}.json`。消息先进有界 live 窗口，完整片段封存为不可变 history，再从 history 派生 summary。
- history 是有保留期的原文层，不是永久层：一段的 MemoryJob 到终态（COMMITTED、SKIPPED、FAILED）或被人工放弃后，过 `workflow.lifecycle.history_retention_days` 天释放。每段各算各的，不要求最旧连续前缀，不等摘要压缩或退休，没有摘要的段同样适用（裁定 111）。
- 原文保留期间，history 与 segment summary 通过 `conversation_id`、`segment_id`、`source_message_digest` 一一绑定；summary 在生成期间允许暂时不存在或处于失败状态。
- History Segment 由相同 `conversation_id` 与连续 sequence 确定性关联，不得让 LLM 生成这类结构关系。

## 摘要层级与压缩

- 层级固定：第一阶段 `SegmentSummary -> RangeSummary`，第二阶段 `RangeSummary -> ArchiveRangeSummary`。ArchiveRangeSummary 不再自动压缩。禁止可配置的无限层级或原地覆盖式滚动摘要。
- Segment Summary 与来源 Segment 一一绑定。Range Summary 只能覆盖同一 Conversation 中至少两份有序、不重叠、同阶段且达到年龄与数量门槛的不可变摘要；两份之间只允许隔着永久没有摘要、原文已释放且没有别的摘要覆盖的段（裁定 111）。
- 压缩用连续消息序号，不用固定日历分桶；年龄按来源覆盖的结束时间算，不按文件生成时间算。
- 低频 Conversation 最老来源等满等待期、且至少有 2 份连续摘要时，允许提前形成较小的 RangeSummary；只有 1 份时保留这份 Segment Summary 作为语义终态。
- 每份父摘要保存来源 ID、摘要值和覆盖范围。父摘要耐久、完整回读并验证后，来源立即退出可重建的活跃检索前沿，再保留 30 天才允许物理删除。
- 删除普通 RangeSummary 前，必须先清理或确认不存在它覆盖的 Segment Summary，避免失去后续清理依据。
- LifecycleWorker 每一拍取得全局 lease 运行闲置封存；每满一个维护周期的那一拍再运行其余维护，以配置的有界批量轮转全部 Conversation。

## 封存（裁定 107）

- 自动封存沿用 OpenViking 的 afterTurn、pending-token 与闲置封存思路。除 afterTurn 与闲置封存外，只有显式 flush 封存。
- afterTurn 封存只在宿主明确进入 afterTurn、当前逻辑轮次以最终 completion 结束、且该轮全部 tool_call 都有唯一终态 tool_result 时发生。
- 闲置封存由 LifecycleWorker 按 `conversation.lifecycle.idle_*` 配置的节拍与批量、按内存游标轮转：live 有未封存消息、且最后一条消息的 `occurred_at` 已过 `idle_commit_seconds`（没有关闭档）的，当作对话到此为止，把 live 全部封存（不留最近轮次）。
- 闲置封存前系统如实收尾："修补悬空的工具调用"——没有结果的 tool_call 补一条同名的失败终态 tool_result；没有完整最终回复的最后一轮补一条说明中断的最终 completion（用现有形态，不新造类型）；走现有追加主链写入 live，再走现有 flush 封存、按完整的轮抽取。不为闲置另开分支。
- 收尾之后宿主送来的批由服务端接上：开头序号落在系统收尾消息上的整批顺延到它们之后；系统已补过终态结果（补在本批之前）的调用，其迟到的 tool_result 丢弃并记观测，本批后续消息顺补序号；只看盘上的收尾消息，重放判出同一批；引用从没出现过的调用的结果照旧拒收。
- 封存时持该地址 Memory 有序 Consumer 顺序锁，不与交付并发。
- 封存阈值与保留预算（`commit_token_threshold`、`keep_recent_turn_count`、`retained_message_token_budget`、`max_segment_tokens`）都从唯一 YAML 读取。
- 显式操作只叫 flush/commit，不建立 OPEN/CLOSING/CLOSED 状态，也不阻止同一 `conversation_id` 后续继续追加。

## 切段

- live 自动切段用显式 Config 和纯规划器：按 `prompt` 锚点形成逻辑轮次；未完成的最新轮次不得因容量阈值被封存。消息数、物理字节数、Token 预算、语义候选窗口、最近轮次保留数和单个 Segment 上限均由 Config 控制。
- 正常大小的内容优先在完整轮次边界封存。单轮或单条文本超过上限时，先把超长 prompt/completion 无损拆成带 `logical_message_id + logical_part_index + logical_part_count` 的连续物理消息，再在 60%–100% Token 窗口内按完整轮次、完整 assistant step、完整消息、段落/句子语义块的顺序选择安全边界。
- tool_call 与对应的终态 tool_result 永远不得被切开。候选边界只按结构排序，不用 Embedding（裁定 110）。
- 已闭合的超大逻辑轮次允许形成多个连续普通 Segment；Segment/Range Summary 必须保留 `starts_mid_turn` 与 `ends_mid_turn` 系统边界，按连续序号重新衔接语义。
- 每个结果都是普通不可变 History Segment，与同范围的 Segment Summary 一一绑定；不得建立 checkpoint/oversized 旁路。
- 超大文本工具结果在进入 Conversation 前只能转成有界的 `summarized`；媒体、二进制、下载文件和无法安全概括的超大结果只能保存 `omitted` 描述、大小与摘要值；不得在 Conversation 或旁路 blob 中保存原始载荷。
- 一组切不开的工具交换（未闭合的调用与它们的结果）合起来超过单个 Segment 上限时，按当前大小从大到小，把组里的工具结果与调用参数用同一套确定性关键信息抽取收成 `summarized`（调用参数记原参数规范 JSON 的大小与 sha256），直到整组放得进一段。不允许超限的原子组单独成段。
- history 原文释放、summary 多周期压缩与来源删除策略使用同一组生命周期 Config。
