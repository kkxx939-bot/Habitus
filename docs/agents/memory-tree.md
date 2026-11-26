适用：改 `habitus/infrastructure/tree/**`、`habitus/memory/tree/**`、`habitus/memory/document/**`、`habitus/memory/uri.py`、`habitus/memory/schema/**` 时读。

# habitus 树：布局、地址、文件格式

## 一棵树（裁定 112）

- 长期记忆与行为侧的事件、空白在同一棵树里。树根是存储根下固定的 `habitus/`，不可配置。
- 树根下只有九样：`profile.md`、`preferences/`、`entities/`、`tools/`、`occurrences/<lane>/`、`gaps/<lane>/`、`habits/<lane>/`、`reminders/<lane>/`、`relations/<lane>/`；`<lane>` 是 `session` 或 `physical`。
- 行为侧关着时不建事件与空白两样；先开后关的原样留着、不读、不报错。
- 习惯、提醒、关系三类系统记录的规矩见 `behavior-scene-foresight.md`（裁定 113、115）。
- 共用存储代码在 `habitus/infrastructure/tree/`：它不认识任何领域类型，只引用 `foundation` 与 `infrastructure`（架构测试守）。分类定义、字段规则、写入纪律留在 `memory` 与 `behavior` 各自那边，不并成一套。

## 长期记忆只有四类（裁定 109）

- L2 权威目录：`profile.md`、`preferences/{topic}.md`、`entities/{category}/{name}.md`、`tools/{tool_name}.md`（架构测试守）。
- Event 与 Intention 已退出："发生过什么"只看行为侧每一轮的记录，"打算做的事"不再有地方记。不得以别名、兼容读取或替代卡片恢复它们。
- 集合目录允许可重建的 `.abstract.md`（L0）与 `.overview.md`（L1）；`profile.md` 是直接读取的单文件 L2。

## 旧盘面不兼容、不迁移

- 存储根下还有合并前的 `memory/` 记忆根或 `behavior/<lane>/tree/` 行为树时，启动在任何 Worker 起来之前明确报错。
- `MemoryTree.initialize()` 检查树根只含上面九样，遇到别的明确报错，不等到第一条记忆任务中途才失败。
- 不得加入旧实现或兼容入口。

## 地址

- 协议固定为 `habitus://`，协议名只在 `infrastructure/tree/address.py` 定义一次；其后路径与树根下的相对路径一一对应。卡片的根、合法目录及其 `.abstract.md`/`.overview.md` 也可确定性寻址。
- 原来的 `memory`、`behavior` 两个协议名已退出，生产代码里连注释都不出现（架构测试守）。
- URI 不含 owner、用户空间、稳定文档 ID 或旧 Context scope；只接受完整协议形式，不兼容短路径、`viking://`、`memoryos://`。
- 卡片地址解析复用 `MemoryAddress`、`MemoryDirectory`、`MemoryLevel` 的树约束；事件与空白地址复用 `BehaviorAddress` 与支名。禁止产生树外路径或静默清洗改名。
- 事件与空白的叶名和百分号编码与合并前逐字节相同。

## L2 与 L0/L1（裁定 110、111）

- L2 是唯一权威记忆，只能由 Memory Editor 的确定性写入阶段修改。
- L0/L1 从直接 L2 文件和子目录派生，必须能删掉重建，不是新的长期记忆字段；没有向量索引。
- L0/L1 刷新失败不得回滚或改写已写成的 L2；MemoryJob 已记"卡片已写完"时，有界重试只补刷派生层，用完判 FAILED。

## 文件格式（裁定 111、112）

- L2 统一为 Schema 渲染的 Markdown 正文 + 文件末尾唯一的 `HABITUS_FIELDS` JSON 注释；卡片与事件、空白同一种外壳，类型键 `type`；`links`、`backlinks` 只有卡片有。正文与结构字段在同一个原子文件里。
- 卡片注释顶层只存：`type`、`revision`、`created_at`、`updated_at`、业务 `fields`、系统维护的 `links`/`backlinks`。版本、时间、持久关系不得由 LLM 或内容 Schema 生成。
- 每条关系只存规范 `from_uri`、`to_uri`、受控 `link_type`；源的 `links` 与目标的 `backlinks` 存同一关系对象，统一身份 `(from_uri, to_uri, link_type)`。关系不嵌入正文。
- 读取校验：Schema、真实路径、本文件这一端的关系方向、规范序列化、正文重渲染完全一致。关系另一端的文件在不在、记没记同一条关系，不是读失败的理由（裁定 111）。
- 不兼容：旧裸 Markdown、缺系统字段、重复注释、宽松 JSON、双轨写入。
- 内部注释和关系元数据不得进入 L0/L1 提示词。

## 尺寸与写入职责

- L2 正文字符上限与物理文件字节上限统一由 `MemoryDocumentConfig` 管；记忆树写前、读后执行同一组检查。语义层不得另设冲突的单文件限制。
- `MemoryTree.write()` 只落盘已构造好的 `MemoryDocument`：不读旧记忆、不做 YAML 合并、不推进版本或生成时间——这些只属于 Memory Editor。
