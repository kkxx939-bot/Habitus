适用：改 `habitus/memory/editor/**`、`habitus/infrastructure/editor/**`、`habitus/memory/snapshot.py`、`habitus/memory/field_ops.py` 时读。

# Memory Editor：从对话到 L2

## 唯一写入口

- 新长期记忆只能以不可变 `ConversationSegment` 为事实来源，只能由按 memory-root 排序的 MemoryJob 调 Memory Editor 生成或修改。
- 除冷热压缩与退休（走同一个写入器）外，任何入口不得绕过 Memory Editor 的写入器直接编辑 L2。
- 旧数据契约已退出：不留 `MemoryEditSource`、旧 `MemoryEditBatch` 或兼容入口。

## 四类内容的边界

- 长期记忆记"现在是什么样、以后怎么做"，不记"这次对话先做了什么后做了什么"。日期换算规则写在提取提示词里（`habitus/memory/editor/extraction/prompt.py`）。
- `profile.md` 只答"用户是谁"：职业、背景、长期角色、稳定能力、稳定客观状态。`preferences/{topic}.md` 只答"用户如何选择"：沟通风格、输出格式、工作流习惯、工具偏好、明确的喜恶。
  - 沟通方式、做事习惯、选择倾向不得写进 profile；职业身份、客观能力、事实状态不得因常出现而写成 preference。
  - 只有用户表达为"以后都这样"的才算 preference；只针对这一次任务的具体要求不算。
- `entities/{category}/{name}.md` 只给有可识别稳定名称、且至少有一条以后可能再用的稳定事实/现状/关系的对象建卡。
  - 临时举例、顺带提及、无可复用事实的普通名词不建。
  - 卡片写现状：变了就改写，不留变化过程；仍成立的具体事实照留。
  - 一次发生的事、会议或操作、待办和计划不是实体，类别也不得用来装它们；只在它们改变了某个有稳定名称的东西的现状时写进那张卡，没处挂就不记。
  - `summary` 必填，承载已知的最小完整事实；`details` 可选。来源不足只写 summary，不得重复 summary、扩写常识或推测细节。
- `tools/{tool_name}.md` 只存真实工具调用形成的稳定工具知识，不存调用流水、次数、成功率或 Skill。
  - `tool_name` 必须逐字来自当前 Segment 的真实 `tool_call.name`，不依赖注册表、不由 LLM 改名；必须有同名、调用 ID 对应的成功 `tool_result`。
  - 本段新写或改写的 `failure_recovery` 还须有同名失败结果及其后的成功结果；与旧工具记忆逐值相同、原样带过来的视为已有来源。
  - 其余知识字段至少一个非空；口头描述不能冒充已验证知识，一次偶然成功不能泛化成规律。

## 候选批次 `MemoryCandidateBatch`

- 内容 Schema 四类独立声明（字段角色、类型、路径模板、Markdown 模板、合并策略）；四类都可更新，没有只增类型，也没有字段取值枚举。
- 批次由四类 YAML 派生且严格：顶层固定 `profile`、`preferences`、`entities`、`tools` 四个候选数组 + `identity_proposals`、`relations` 两个临时提议数组；未知顶层字段、未知业务字段一律拒绝；`profile` 最多一项。
- 临时 `page_id` 只在本次解析有效：系统给完整读取的旧节点分 `1..99`，模型给新节点分不重复的 `100+`，更新旧节点必须复用系统编号；不得持久化为文档身份。
- `identity_proposals` 只允许 `same_memory`（受控 `duplicate_identity`，引一个完整读取的旧来源编号 + 一个存活目标编号）或 `remove_memory`（原因只能 `explicit_forget` / `fully_invalidated`，引一个完整读取的旧编号）；不含 URI、revision、置信度或耐久操作。
- `relations` 每项只有受控 `ADD`/`REMOVE`、两个 `page_id`、受控关系类型；不含最终 URI、反向链接、权重、描述或文本锚点。`REMOVE` 只表示两节点都保留而关系被完整对话明确否定，不能由"本轮没再提"推出。
- 除 `page_id` 外，候选只含业务字段：不得含 URI、路径、耐久 create/update/delete、revision、时间戳、owner、tag、置信度或 EvidenceSlice。来源、旧快照、编号映射由系统在模型输出之外绑定；地址、关系、合并、耐久操作由 Memory Editor 确定性决定。

## 读旧记忆与受控 ReAct（裁定 110）

- 旧记忆读取两层：`infrastructure/editor/snapshot` 提供领域无关的版本快照；`memory/snapshot.py` 只接受严格的卡片 `habitus://` URI，读完整 `MemoryDocument`。
- 不靠向量检索：档案与本段真实 `tool_call.name` 对应的工具卡固定直读完整快照；偏好主题与实体名字清单直接放进第一轮输入并计入固定输入预算。读取失败、越界地址、完整性错误不得伪装成"没有旧记忆"。
- 每轮只能选 `finish`、一次 `memory_list`（列一个目录）、一次 `memory_search`（按模型给的词在卡片字段文字里字面匹配，可限定目录，不打语义分、不返回退休卡片，返回地址、名字与命中片段）或一次 `memory_read`（读名字清单、列过的目录、搜索结果、一跳关系或已读快照里出现过的若干地址）。系统只执行这三种只读动作；读未暴露过的地址明确失败。
- 总轮次、每次读取个数、搜索条数、完整旧记忆数、名字清单预算、上下文字符数、工具结果大小都由 Config 显式限制；轮数默认按逻辑下限定为 3。
- 轮次用完：最后一轮只允许 `finish`；仍判不足或结构纠错用完仍不合格时不判失败，记为在上限处结束（`MemoryExtractionResult.retrieval_ended_at_limit`，并写进 `workflow/memory_plan` 观察事件），按已读到的旧记忆生成候选。列目录与读取本身的失败仍明确失败。
- 纳入旧节点时可按 Links/Backlinks 有界补一跳邻居；不递归、不改关系。
- 不实现 BM25、Embedding、Reranker、知识图谱。
- Retrieval Grader 只返回临时充分性决策与至多一个下一步查询或待读 URI；不产生置信度，不输出候选、合并、删除或写入。

## 候选生成与来源校验

- 模型基于完整 Segment、全部完整旧 L2 快照和临时 `page_id` 一次生成严格候选。代码只做确定性校验（Schema、类型、快照、page_id、关系、字段规划），不用关键词、正则、封闭语法重新解释自然语言，不做第二遍语义审查。
- `context_violations` 不过：把哪条、违反哪条发回模型修补一次（同一结构化客户端与输入预算）。修补后仍不合格的条目连同引用它的关系与身份提议从本批剔除，其余照常；剔除条数与原因写进 `workflow/memory_plan` 观察事件（`source_rejected_count`、`source_rejected_reasons`）；全被剔即按空批提交。
- 新编号候选写到已有、这次没完整读过的旧卡地址：系统把那张旧卡读进上下文，并入同一次修补轮要求基于它重写；修补后仍写到该地址的只剔这一条（裁定 110）。不替模型按字段合并。
- 其余字段规划与身份规划失败明确终止，禁止带病写入。

## 节点规划（纯步骤）

- 候选生成后先读明确存在或缺失的精确目标快照，按 YAML 形成初步 CREATE/UPDATE/NOOP 完整字段计划；再由确定性 `MemoryIdentityPlanner` 验证身份提议，产出 `FinalIdentityMap`。
- `FinalIdentityMap` 覆盖相关旧节点与本批候选，明确每个编号是 CREATE/UPDATE/NOOP/MERGE/DELETE，并绑定最终 L2 URI 或删除终态。多个编号可指向同一 URI；旧编号只有显式 MERGE 才可重定向；DELETE 不得有最终 URI。关系层不得自行猜测节点处置。
- 明确失败：未读到的同址旧文件、未声明的旧编号改址、快照变化、类型不一致。
- 字段合并严格按 YAML：`immutable` 冲突即失败；`patch` 接受完整目标值、缺席保留旧值、不接受 SEARCH/REPLACE；`replace` 表示最新完整状态、缺席的可选字段移除。
- MERGE：只能把完整读取且不再接收字段变更的旧节点并入同类、单跳、由字段计划覆盖的存活目标；默认只允许 preference 与 entity，禁止 profile 与 tool。
- DELETE：只能删完整读取且不再接收字段变更的旧节点，原因必须 `explicit_forget` / `fully_invalidated`；未提及、陈旧、低相似度、部分纠正、状态变化都不构成删除。
- 提议来源不得同时被关系候选引用；合并链、循环、多目标、退休目标、信息丢失必须失败。
- 节点规划、关系规划、最终文档组装都是纯步骤：不加锁、不推进时间、不落盘。原 `MemoryMutationCommitter`、`MemoryLinkCommitter` 的独立发布职责已退出。

## 关系规划（裁定 111）

- 在 `FinalIdentityMap` 之后执行，顺序固定：已有端点映射到最终身份 → MERGE 迁移关系 → DELETE 清理关系 → 去自环与重复 → 校验并应用显式 `REMOVE` → 幂等应用 `ADD`。
- `REMOVE` 两端必须是本批完整读取的旧节点，且规范关系至少在一边旧快照中真实存在；结构性清理不伪装成 REMOVE。同一规范关系不能同批既 ADD 又 REMOVE。
- 对称的 `related_to`、`contradicts` 按 URI 顺序规范方向；统一按 `(from_uri, to_uri, link_type)` 去重。
- 正向 Links 是唯一输入，Backlinks 由最终正向集合确定性派生，不能由模型生成。
- 未知编号、树外 URI、未完成身份、本批显式关系的悬空端点明确失败。
- 两边分开写、放松一致性：检索补一跳、提取补一跳、关系规划、冷热退休四条读路径遇到单边或指向不存在文件的旧关系，跳过并记日志。关系规划对本批要写的节点顺手修：指向不存在文件的丢掉；正向有、对方缺反向的补上；反向有、对方没正向的丢掉（以正向为准）。为此读集包含要写节点的一跳邻居。

## 写入（裁定 111）

- 没有多文件事务：无预备日志、回滚、崩溃恢复、变更回执。
- 先在内存里构造全部最终 `MemoryDocument` 与删除集合；`MemoryCommitWriter` 对读集全部 URI（受影响节点、关系端点、一跳邻居）稳定排序一次拿齐锁，锁内做 state、revision、完整内容摘要 CAS。同一文档的内容与关系变化合成一次写、只推进一次 revision。
- 逐个原子单文件写，顺序固定"新建和更新 → 只改关系的邻居 → 删除"；合并目标没写成就不删来源。
- 中途失败就停，已写成的不回滚，按可重试失败整条重做。CAS 表明旧记忆已变时同样整条重跑 Memory Editor，不复用旧语义计划。
- 全部写成后先在任务文件里耐久记下"卡片已写完"与改动、删除的 URI 清单，再刷 L0/L1、清冷热、标完成；刷新失败不回滚 L2。崩溃后有这一阶段就只补后面、不再调模型，没有就整条重做；进度只记到步骤，不记到单个文件。
- 写入只复用快照、CAS、锁键、单文件原子写入原语；冷热的压缩与退休走同一个写入器。
