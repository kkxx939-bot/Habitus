# m2bOS 新记忆链测试覆盖矩阵

本目录只针对当前 `Conversation -> MemoryJob -> Memory Editor -> MemoryTree -> Semantic/Vector -> SearchService`
主链编写测试，不恢复任何旧记忆接口或旧测试资产。每项先以领域不变量为测试依据，再覆盖正向、拒绝、幂等、边界和组合场景。

## 用例有效性准入

- 每条用例必须对应当前源码中可到达的公开行为、领域不变量、耐久数据损坏风险或外部依赖故障；不存在的 Agent、CLI、HTTP/API、Skill 和旧记忆兼容能力不写测试。
- 参数化只用于有不同失效机理的等价类，例如 Python `bool` 冒充 `int`、数字字符串被宽松转换、`null` 绕过容量上限；不适用组合在参数生成时排除，不以运行时 `skip` 增加数量。
- 测试替身只隔离模型、远程向量服务、时钟、锁竞争或故障注入。关键流程必须另有真实领域对象、真实文件存储或 Runtime 组装后的集成用例交叉验证。
- 不绕过领域构造器伪造生产路径不可能产生的数据对象来追求分支覆盖；防御分支只有在耐久损坏或外部输入确实可到达时才测试。
- 覆盖率用于发现未观察区域，不以行数或参数化实例数证明业务充分性。

| 业务模块 | 主要风险 | 用例位置 | 场景层次 |
| --- | --- | --- | --- |
| `foundation` | ID、规范 JSON、摘要不稳定 | `unit/foundation/` | 正向、非法类型、规范化 |
| `pre/conversation/messages` | 角色混淆、工具调用不闭合、序号错误 | `unit/pre/test_conversation_messages.py` | 正向、反向、工具组合 |
| `pre/conversation/summaries` | 摘要冒充原文、范围不连续、来源摘要不匹配 | `unit/pre/test_conversation_summaries.py` | Segment、Range、Archive 组合 |
| `Config` | 未知字段、宽松类型、跨域容量冲突、秘密进入 YAML | `unit/config/` | 文件、环境选择、跨模块组合 |
| `ModelClient` | 协议差异、重试越界、流式重放、JSON 有损修复 | `unit/model_client/` | Chat、Embedding、结构化输出、Adapter |
| 通用快照与锁 | 读集不完整、锁丢失、临界区被重放 | `unit/infrastructure/test_snapshot_and_locks.py`, `test_sqlite_lock_store.py` | 缺失快照、容量、进程内/SQLite fencing、过期接管、损坏布局 |
| 耐久文件 | 路径逃逸、符号链接、半写文件 | `unit/infrastructure/test_durable_files.py` | 原子创建/替换、损坏与恢复 |
| 远程向量抽象 | provider/adapter 混淆、发布中间态可见、checkpoint 回退 | `unit/infrastructure/test_vector_*`, `test_vikingdb_protocol.py` | 工厂、Schema、全量/增量发布、VikingDB 认证/重试/编解码/分页完整性 |
| Conversation Journal | 重放冲突、history 非连续、释放后序号回退 | `unit/conversation/test_journal.py` | append、seal、release、枚举 |
| Conversation 切段 | 拆断完整轮次或 tool_call/result、超大载荷长期保存 | `unit/conversation/test_retention.py` | afterTurn、flush、软硬阈值、降载 |
| Summary 与压缩 | 来源错配、范围重叠、过早清理、无限层级 | `unit/conversation/test_summary_and_compaction.py`, `test_summary_generation.py` | 真实结构化生成、完整原文上限、幂等、Range、Archive、清理 |
| 记忆 URI/树/文档 | 树外路径、旧协议、Schema/正文不一致、关系方向错误 | `unit/memory/` | 六类 L2、L0/L1、读写损坏、URI |
| Intention | completed 误召回、时间自动失效、确认时间误刷新 | `unit/memory/test_intention.py` | active/completed、30/60/180 复核 |
| 候选 Schema | LLM 越权输出、page_id 重定向、工具知识无真实结果 | `unit/editor/test_candidates_and_page_ids.py` | 六类、严格字段、工具恢复、完成意图 |
| 受控提取循环 | 检索不足仍继续、候选带病提交、无限再生成 | `unit/editor/test_extraction_*` | ReAct、审查、再生成、耗尽失败 |
| 节点变更与身份 | 同址覆盖、字段策略失真、误 MERGE/DELETE | `unit/editor/test_mutation_and_identity.py`, `test_identity_planner_rules.py` | CREATE/UPDATE/NOOP、同类合并、跨类/事件拒绝、整节点删除、PATCH/REPLACE |
| Links/Backlinks | 单边关系、删除未迁移、自环、显式 REMOVE 误用 | `unit/editor/test_relationships.py` | ADD/REMOVE、双向派生、MERGE 迁移、损坏拒绝 |
| 事务提交 | 内容与关系分批发布、旧快照覆盖新状态、日志残留 | `integration/test_memory_commit_chain.py`, `test_transaction_recovery.py` | CREATE/UPDATE/MERGE/DELETE、部分发布回滚、完整发布恢复、未知后续状态拒绝 |
| L0/L1 与 Memory Vector | 派生层混入权威字段、索引与树不一致 | `unit/semantic/` | 刷新、重建、增量、过滤、陈旧源 |
| 记忆检索 SearchService | Summary 抢占 Memory、kinds 截断关系、completed 占名额 | `unit/retrieval/` | 主召回、多查询失败、一跳非递归、关系损坏拒绝、充分性判断、条件后备 |
| Job/Receipt | 跨会话乱序、租约 ABA、失败跳过、回执与实写不一致 | `unit/workflow/`, `integration/test_change_receipt_chain.py`, `integration/test_memory_job_full_chain.py` | 顺序、lease、退避、STAGED/COMMITTED 恢复、实际事务回执、人工恢复、History 释放后过期 Job/Receipt 清理 |
| Worker/Runtime | 重复启动、无心跳执行、失败 Job 无运维入口、错误组装 | `unit/runtime/`, `integration/test_runtime_assembly.py` | Job Worker、LifecycleWorker、全局 lease、耐久游标、局部失败、共享依赖、Runtime 启停/重启/关闭 |
| 架构边界 | 旧 Evidence/Context 复活、领域反向依赖、双轨 Schema | `architecture/` | 源码静态约束 |
| Conversation 主链 | History 已发布但 Job 丢失，或 Job 重复 | `integration/test_conversation_to_job.py` | afterTurn、outbox、重放、flush |
| 完整 MemoryJob 主链 | 组件分别正确但发布顺序或恢复边界错误 | `integration/test_memory_job_full_chain.py` | Summary/Editor 并行、Receipt、事务、双向量 checkpoint、Job 终态、Journal 清理、L2 已提交后向量超时的无重规划恢复 |

## 主链验收顺序

1. 原始消息按角色和序号进入 `live.jsonl`。
2. 只在完整轮次边界封存不可变 History，先 STAGED、后 QUEUED。
3. 同一 Segment 独立生成过程 Summary 和长期记忆计划；任一失败不发布 L2。
4. 旧记忆读取、候选审查、字段计划、身份裁决和关系计划全部完成后才提交。
5. L2 内容与 Links/Backlinks 在一笔可恢复事务中发布，随后刷新 L0/L1 和远程索引。
6. Memory 是检索主结果；只有 Memory 不足时，Summary 才作为单独的历史细节后备。
7. Job、Receipt、History、Summary 和 Transaction Journal 按安全门槛清理，不允许时间直接改变记忆业务语义。

## 外部系统验证边界

- VikingDB 用例覆盖 HTTP 请求、认证头、错误分类、记录编解码、过滤和分页完整性，但不在单元测试中访问真实云账号。
- 模型 Adapter 用 MockTransport/受控 Provider 覆盖协议与结构化语义，不消耗真实 API Key。
- 真实供应商连通性、配额、服务端索引最终一致时间属于独立 smoke/contract 环境，不伪装成离线单元测试。
- 当前仓库没有 Agent 执行入口、CLI 或 HTTP/API 入口，因此本测试集只覆盖 Runtime 暴露的记忆检索门面和 SearchService，不能宣称覆盖“用户请求 -> Agent -> 模型/工具 -> 返回”的 Agent 主链。
- 参数化展开数量只表示同一契约的输入等价类，不作为独立业务场景数量，也不作为测试充分性的判断标准。
