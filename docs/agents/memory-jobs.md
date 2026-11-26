适用：改 `habitus/memory/workflow/**`、`habitus/runtime/lifecycle_memory.py`、`habitus/runtime/lifecycle_retry.py` 时读。

# 记忆任务：排序、outbox、失败与保留

## 序号（裁定 111）

- 全局 `memory_sequence` 由 `jobs/state.json` 里永久保存、永不回退的 high-watermark 分配，不得由仍存 Job 的最大值推导。先耐久推进高水位再建 STAGED Job；崩溃只允许留下序号空洞，禁止重用。
- 写同一 `memory-root` 的所有 Job 用独立于对话内消息序号的单调 `memory_sequence`；正常锁竞争用显式配置的有界等待取得唯一序号。
- 后台只有在前一序号到终态（COMMITTED、SKIPPED、重试用完的 FAILED）后，才能读旧记忆并开始下一项；同一时刻只有一条 Job 持有有效租约。
- 当前没有 owner 层，排序范围就是单棵 memory-root；将来用户隔离只能在最外层把排序键扩为 `owner_id + memory_root`。
- 不同 memory-root 可以并行。同一 Job 内 Summary 与 Memory Editor 纯规划可并行，但两者都成功后才能发布任何 L2 变化。

## 两阶段耐久 outbox

- 发布 history 前先建不可执行的 `STAGED` Job 并分配序号；history 原子发布且来源摘要校验通过后才激活为 `QUEUED`。
- 写 STAGED 时可继续持有 Conversation 租约，但不得嵌套持有 SQLite fencing 写事务；返回发布阶段前必须重新校验租约和完整 Segment 摘要。
- Worker 遇到最早的 `STAGED` Job 必须从 Conversation live/history 幂等补完封存并核对 `segment_id` 与摘要，不能静默跳过。目的：任一侧中断都不让 Segment 永久漏记。
- 前台保存和后续对话不等 LLM。
- Job 文件里的改动与删除 URI 清单只能由系统按实际写入结果生成，不能由 LLM 输出，不复制对话或记忆正文。

## 失败与终态（裁定 108、111）

- 失败 Job 在自动有界重试期间保持后续 Job 阻塞（放行会让后一段先写、前一段重试时盖回去）；重试用完判 FAILED 终态，后续照常推进，健康面报失败条数与最早一条，等人工重试或放弃。
- 生成摘要与记忆计划时碰到模型统一规则里"这一条输入有问题"（请求不合法、太长、内容安全）：这一段不进长期记忆，Job 进 SKIPPED，`last_error` 写明原因，后续照常推进（裁定 108）。长期记忆写成之后才碰到的输入问题不跳过，按普通失败处理。

## 租约与接管

- RUNNING Job 的有效租约阻止其他 Worker 接管。租约过期后只能在耐久队列锁内递增 `claim_generation` 产生新 claim，再看 Job 文件里的阶段：已记"卡片已写完"就只补 L0/L1、冷热状态和终态，不能重新规划；没有就整条重做。
- 陈旧 claim 的心跳、失败、完成结算全部拒绝。

## 人工运维（裁定 111）

- FAILED Job 的人工入口只在 Runtime：先读 FAILED 完整快照，再带着它调 Store 的版本校验重试或放弃。`retry_failed()` 必须匹配当前完整失败版本，陈旧恢复对象不得覆盖后来 Worker 的状态。
- 重开的 Job 按序号排回队列，由单并发 Worker 照常执行（另一条正持有有效租约时先等它做完）；不要求 Worker 被挡住。
- 这一段 history 已释放的 FAILED Job 不能重试。人工重试旧任务可能把卡片改回旧说法，这一代价已接受。

## 保留与删除（裁定 111）

- 终态 Job（COMMITTED、SKIPPED、FAILED）到终态满 `job_retention_days` 且对应 History 已释放后才可删。
- 人工放弃记录只为给 History 计时：放弃满 `history_retention_days` 且这一段 History 已不在（释放了或从没发布）就随之删。
- 原文还在的 FAILED 不按时间删（人工还能重试）；STAGED、QUEUED、RUNNING 不按时间删。
