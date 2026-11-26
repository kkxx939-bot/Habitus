# Habitus Repository Constraints

本文件约束仓库内所有人工和 Agent 变更。当前用户决策优先于历史实现，不得通过兼容层恢复已经删除的架构。

全部 Python 源码在唯一顶层包 `habitus/` 下，import 一律写 `habitus.<子包>`。设计以 `docs/rulings/README.md` 的编号裁定为准。

裁定条文由 AI 根据讨论整理、措辞是 AI 的；用户原话在桌面讨论记录（2026-11-18 起即裁定 104 以后见 `~/Desktop/90days/03-决定记录.md`，更早的已删）。条文有歧义或与用户最新说法冲突时，以用户为准并提出更新条文。

## 总原则

- 先在源头保证正常链路；兜底只留一层：拦住、报出、不卡死。不为罕见情形加多级兜底，不拿旧数据顶替。
- 同一条上游记录对全部消费者是同一个事实；重叠或矛盾在一处确定性消解，再分发。
- 语义判断交给模型，代码只做 Schema、类型、身份等结构校验，不用关键词、正则、映射表近似；模型做不好的全局结构约束由输出形状或确定性代码保证，不靠提示词或重试。
- 新包先在 `tests/architecture/test_dependency_boundaries.py` 登记允许和禁止的 import。
- 会显示给最终用户的句子用大白话、不出内部术语；只进日志或 trace 的不限。
- 延后的改造把方案、影响、例子写进桌面 `~/Desktop/Habitus/notes/待办.md`，代码里只留一行 `TODO(待办#N)`；已有的 TODO 改到所在文件时顺手迁移，不单独成批。

## 改哪里先读哪份

各模块的约束在 `docs/agents/` 下，改到对应路径之前先读那一份：

| 改这些 | 先读 |
|---|---|
| `habitus/behavior`、`scene`、`foresight`、`series`、`temporal`、`conditions`；`habitus/runtime/` 里的 generation、scene、foresight、model_queue 与三类记录卡片 | `docs/agents/behavior-scene-foresight.md` |
| `habitus/runtime/` 其余部分（Worker、租约、锁顺序、启停） | `docs/agents/runtime.md` |
| `habitus/config/`，或给任何领域加可调参数 | `docs/agents/config.md` |
| `habitus/model_client/`，或任何调用模型、处理结构化输出的代码 | `docs/agents/model-client.md` |
| `habitus/memory/conversation/`、`habitus/pre/conversation/` | `docs/agents/conversation.md` |
| `habitus/infrastructure/tree/`、`habitus/memory/{tree,document,schema}/`、`habitus/memory/uri.py` | `docs/agents/memory-tree.md` |
| `habitus/memory/editor/`、`habitus/infrastructure/editor/`、`habitus/memory/snapshot.py`、`field_ops.py` | `docs/agents/memory-editor.md` |
| `habitus/memory/{semantic,retrieval,recall}/`、`habitus/memory/browse.py` | `docs/agents/memory-retrieval.md` |
| `habitus/memory/workflow/`、`habitus/runtime/lifecycle_memory.py`、`lifecycle_retry.py` | `docs/agents/memory-jobs.md` |
| `habitus/integrations/`、`plugins/` | `docs/agents/integrations.md` |
| 各处的提示词文件（`prompt.py`、`prompts.py`） | `docs/agents/prompts.md` |

## 已退出的架构

旧长期记忆树与写入链、SessionArchive、旧 Behavior 第一层（现行 `habitus/behavior` 是重做的新设计）、ActionPolicy、Resource、Skill、通用 Context/ContextURI/ContextProjectionSanitizer、`memoryos://` 协议、`pre.session`、SourceStore、旧 Catalog/FTS/Vector/Trace/L0/L1 链路都已退出。不保留别名、迁移读取、禁用分支、占位实现或新旧双轨；退出领域的调用者（存储、运行时、SDK、HTTP、MCP、CLI、集成）一并删除。确实被新代码独立使用的底层原语（原子文件、路径安全、锁）可以保留，但不得暴露旧领域语义。ActionPolicy、SessionArchive、Resource、Skill 若要重做，作为新设计另行确认。Python 发行名统一为 `habitus`。

## 工作区安全

- 修改前记录分支、HEAD、工作区状态、未跟踪文件和相关公开入口。
- 用户已有修改不得回退、覆盖或顺手格式化。
- 不做 git 写操作（commit、add、push、checkout、stash push 等），改动留在工作区由用户提交；README 仅在用户明确授权时修改。
- 代码注释和 docstring 只写这段代码做什么、不明显的为什么（一两句）；讨论过程、用户原话、"用户定 / 用户说"、方案取舍不写进代码，记在桌面讨论文档里（现在是 `~/Desktop/90days/`），需要指向依据时只写「裁定 N」。已有的长注释改到所在文件时顺手精简，不单独成批。
- 源码中的说明性注释和 docstring 统一使用中文；许可证声明、协议名称、公开标识符以及 `noqa`、`type: ignore` 等机器指令保持原样。
- 调整契约时检查所有调用者、运行时装配、Schema、Worker、SDK、HTTP、MCP、CLI 和外部集成。

## 验证

- 完整的 `scripts/ci.sh`（compileall、Ruff、MyPy、Pyright、pytest 含分支覆盖率、插件 `node --test`）串行要 30–43 分钟。一件事只在全部改完、审核的修补也落地之后统一跑一次；中途和小改动只跑相关检查：改过文件的 compileall、ruff check、ruff format --check、mypy、pyright，加相关单测、`tests/architecture/` 和受影响的集成测试。汇报里写明跑了哪些。
- 远端 `.github/workflows/ci.yml` 另做 wheel 内容检查与装包冒烟。涉及真实运行时或存储的改动，另做相应的 smoke check。
- 工具缺失或外部服务不可用时记录真实错误，不得伪造通过，也不得为运行检查随意修改依赖。
