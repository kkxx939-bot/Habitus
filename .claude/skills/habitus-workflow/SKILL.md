---
name: habitus-workflow
description: Habitus 里执行一条改动、审核代码或方案、写交给别的 session 的提示词时用。触发词：执行、实现、改造、写提示词、review、审核、核对、体检、评审方案。
argument-hint: "change <清单条目> | review <范围> | fix <问题号> | audit <模块> | plan <方案文档> | deepen <题目>"
---

# Habitus 工作流程

## 总则

- "应该是什么"看裁定和方案；"实际是什么"只认本次读过的源码，每句带 `file:line`（附函数名）。注释、TODO、README、memory、旧报告、子代理或 codex 的转述都只是线索。
- 写设计前读完整个模块；改原则前先 grep 盘点全部落点（调用方、配置键、测试、文档）。
- 先写死预期再跑；测试走源码里的真实链路。只报有"输入 → 真实输出 → 和设计差在哪"的问题。
- 要换视角就换模型（codex），不换人设、不编排多个子代理。
- 先判会不会影响主流程，再决定这一轮修不修；边角进 `待办.md`。

## 何时用哪个

| 模式 | 什么时候 | 看哪里 |
|---|---|---|
| `change` | 执行一条改动 | 下面的执行流程 |
| `review` / `audit` | 一批改动交给用户前 / 给一个模块体检 | 下面的审核流程 |
| `fix` | 每个修复落地时 | 审核流程 0 → 1 → 2 → 5 → 7 → 8 |
| `deepen` | 新方向写方案之前先把题目问深 | [deepen-topic.md](references/deepen-topic.md) |
| `plan` | 动代码前评审方案或拟定的裁定 | [plan-review.md](references/plan-review.md) |
| 写提示词 | 任何交给另一个 session / 子代理的提示词 | [writing-prompts.md](references/writing-prompts.md) + [prompt-template.md](references/prompt-template.md) |

不改行为的改动（错别字、日志措辞）用 `/code-review low`。

## 执行流程（`change`）

主 session 只讨论、立项、写提示词、收口核对；产品代码的实现、实验、跑数写成一份提示词，由用户另开执行 session 做。规矩、文档、配置可由主 session 直接改。阶段内一路往下走，只在阶段边界给用户看。

| 步 | 谁 | 做什么 | 过闸 |
|---|---|---|---|
| 0 立项 | 主 session | 问题与数据支撑；读完模块、盘点落点；写死预期；按 [test-matrix.md](references/test-matrix.md) 选测试；按 [prompt-template.md](references/prompt-template.md) 写一份提示词，写明完整 CI 由谁、何时跑那一次；下一条要在上一条改完的代码上重读源码再写 | 以上都已存盘 |
| 1 执行 | 执行 session | 先写测试并在改动前的代码上看结果（复现用例红 / 重构快照绿，需要 A/B 先存 A 侧）；再实现；先写的测试不为迁就实现而改；按矩阵做撤回即失败、A/B、对抗、宿主实测 | 相关检查与本条测试通过，有证据 |
| 2 codex 审核 | 执行 session 发起 | 只给需求、共享说明和改动，不给执行报告；全部报出并标严重度（[codex-cross-check.md](references/codex-cross-check.md)） | 每条有处理结果 |
| 3 修补 | 执行 session | 修确认的问题，只跑相关测试；不发起第二轮复核 | 相关检查通过 |
| 4 收口 | 主 session | 对着设计和源码核报告；全部落地后跑一次完整 `scripts/ci.sh`；按 [report-format.md](references/report-format.md) 记问题、给用户汇报；问题总清单里同一类问题第二次出现，就装一道闸（架构测试 / lint / deny / hook）、清掉已有的同类代码，再删掉对应的规矩文字 | CI 全绿，用户看过 |

## 审核流程（`review` / `fix` / `audit`）

0 定范围 → 1 共享说明（[brief-template.md](references/brief-template.md)）→ 2 场景预期对比（[scenarios.md](references/scenarios.md)，最重要）、3 对照设计读源码（[checklist.md](references/checklist.md)）、4 codex 复核，三步并行 → 5 汇总与最小复现 → 6 分级（[severity.md](references/severity.md)）→ 7 修复与防复发 → 8 收口。审核只做一轮，修完不复审；完整 CI 整件事一次。每一步的出口条件见 [review-steps.md](references/review-steps.md)。

## 最常犯的

- 没跑场景就说"读了几遍没问题"；先跑再定预期。
- 改完再补测试，或改先写的测试让它过。
- 没复现就修"很具体的"报告；修完没做撤回即失败。
- 重构不做 A/B；改原则只改最明显的两处。
- 中途再跑一遍完整 CI。

完整红线、角色隔离、产物路径与实验记录规矩见 [roles-and-paths.md](references/roles-and-paths.md)。git 操作全部由用户做；一次性脚本不进仓库。
