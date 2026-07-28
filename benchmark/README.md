# m2bOS Memory Benchmark

这是项目一级目录中的数据集驱动长期记忆基准，不是 pytest 测试集。

主链路与 OpenViking 的 LoCoMo/LongMemEval benchmark 对齐：

- 对照版本：OpenViking `f0445e0cce5b703cd955ba19378f12b0bcbcd00b`。
- 公开 QA 默认使用 Top-10 和 4000 字符回答内容预算；超预算时整块跳过 Memory、关系节点或 Summary，不截断正文。
- LoCoMo、LongMemEval 使用各自的回答语义规则；Judge 默认复现 OpenViking 的宽松口径，也可显式切换严格口径。
- m2bOS 原生数据默认使用严格 Judge，因为 OpenViking 没有六类树、关系迁移、Summary fallback 等对应金标准。

```text
公开数据集完整 Sessions
        ↓
m2bOS Conversation
        ↓
MemoryJob → Summary / Memory Editor → MemoryTree → VectorStore
        ↓
SearchService 检索完整 L2、Links/Backlinks 和必要时的 Summary fallback
        ↓
Answer Model
        ↓
独立 Judge
        ↓
按类别统计准确率、检索行为、耗时和 Token
```

## 支持的数据集

- `locomo`：读取官方 `locomo10.json`。保留 Session 顺序、角色、时间、QA category 和 evidence。与 OpenViking 一样默认排除 category 5，只有 `--include-adversarial` 才纳入。
- `longmemeval`：读取官方 LongMemEval JSON。每个问题保留自己的 haystack sessions、session dates、question type、question date 和参考答案。
- `m2bos`：原生扩展协议，用于真实 `prompt/completion/tool_call/tool_result` 数据。它只描述自然会话和 QA，不声明 CREATE、UPDATE、DELETE 等内部预期操作。

仓库不复制第三方数据集；命令直接读取用户取得的官方 JSON。

## 1. 检查数据集

```bash
python -m benchmark inspect \
  --dataset locomo \
  --input /path/to/locomo10.json
```

可用 `--sample 0 --sample 1` 选择样本，使用 `--question-limit 20` 做小规模真实调用。

## 2. 运行完整基准

```bash
python -m benchmark run \
  --dataset locomo \
  --input /path/to/locomo10.json \
  --config Config/example.yaml \
  --output benchmark/results/locomo-run \
  --work /path/to/isolated/benchmark-work \
  --top-k 10 \
  --max-answer-context-chars 4000 \
  --question-concurrency 8
```

这会直接使用配置中的真实 Chat、Embedding、Reranker（如果配置）和 VectorStore。不存在 scripted/fake benchmark 模式。

每个样本有独立的本地 Storage Root。一次 benchmark run 只使用两个 run-scoped 远程 Collection；样本切换时从该样本自己的 MemoryTree 真相源执行完整 rebuild，既避免跨样本污染，也避免 LongMemEval 为每道题创建 Collection。

结果目录包含：

- `run.json`：数据集 SHA-256、选定样本摘要、代码 commit、模型/向量路由、配置指纹和运行状态。
- `ingest.jsonl`：每个样本真实导入的 Session、Message、Job、Receipt 和六类 Memory 文档数量。
- `answers.jsonl`：每道题的检索 URI、关系扩展、Summary fallback、回答、耗时和 Token。
- `answers.jsonl` 同时区分“SearchService 召回了什么”和“4000 字符预算内实际交给 Answer Model 什么”，并记录被预算跳过的节点。
- `summary.json`：机器可读汇总。
- `summary.md`：用于人工比较的汇总。

`--resume` 只在样本数据摘要与既有工作目录完全一致时续跑；不会对不完整导入猜测恢复位置。

## 3. 独立 Judge

正式分数应使用独立 Judge 配置，避免回答模型评判自己：

```bash
python -m benchmark judge \
  --dataset locomo \
  --input /path/to/locomo10.json \
  --answers benchmark/results/locomo-run/answers.jsonl \
  --config /path/to/judge-config.yaml \
  --output benchmark/results/locomo-run/judge.jsonl \
  --judge-policy dataset-default \
  --concurrency 16
```

LoCoMo 可显式添加 `--with-evidence`，把数据集 evidence 原文交给 Judge 作为辅助；默认只依据问题、参考答案和模型回答评分。
Judge 同时生成相邻的 `judge.jsonl.manifest.json`，记录答案文件摘要、Judge 模型和配置指纹。

`--judge-policy dataset-default` 的解析规则是：LoCoMo/LongMemEval 使用 `openviking-default`，m2BOS native 使用 `strict`。如需和 OpenViking 的 `--strict-prompt` 对照，显式传 `--judge-policy strict`。最终解析后的固定策略会写入 Judge manifest，续跑时不允许悄悄改变。

也可以在 `run` 时增加 `--judge-config` 一次完成回答、Judge 和报告。

## 4. 重新生成报告

```bash
python -m benchmark report \
  --answers benchmark/results/locomo-run/answers.jsonl \
  --judge benchmark/results/locomo-run/judge.jsonl \
  --ingest benchmark/results/locomo-run/ingest.jsonl \
  --output benchmark/results/locomo-run
```

报告包括：

- 总准确率及 LoCoMo category / LongMemEval question type 分项准确率；
- 回答链路错误和 Judge 错误，二者不会混为一类；
- 平均直接 Memory、Links/Backlinks 一跳节点、空召回率及 Answer-context evidence recall；
- Summary fallback 尝试率和命中率；
- Retrieval、Answer、Judge 和端到端 p50/p95/p99；
- Answer/Judge Token；
- Conversation、Job、Receipt 和六类 Memory 的导入规模。

公开数据集没有 Memory URI 金标准，因此本基准不会伪造 URI Precision/Recall。最终质量指标与 OpenViking 同类基准一样，以数据集 QA 的独立 Judge 准确率为主。

## m2bOS 原生数据协议

根节点是样本数组：

```json
[
  {
    "sample_id": "stable-source-id",
    "sessions": [
      {
        "session_id": "source-session-id",
        "started_at": "2026-01-01T10:00:00+00:00",
        "messages": [
          {"role": "prompt", "content": "..."},
          {"role": "completion", "content": "..."},
          {
            "role": "tool_call",
            "tool_call_id": "call-1",
            "tool_name": "search",
            "content": {"query": "..."}
          },
          {
            "role": "tool_result",
            "tool_call_id": "call-1",
            "tool_name": "search",
            "tool_status": "completed",
            "content": "..."
          }
        ]
      }
    ],
    "questions": [
      {
        "question": "...",
        "answer": "...",
        "question_type": "tool_knowledge"
      }
    ]
  }
]
```

原生数据仍必须形成可提交的完整会话轮次；benchmark 不会添加虚构 assistant 消息修补不完整数据。

## 5. 覆盖审计

```bash
python -m benchmark coverage \
  --dataset m2bos \
  --input benchmark/datasets/m2bos_core_v1.json \
  --strict
```

该命令同时输出 OpenViking 对照 revision、公开 QA 协议和 suite matrix。`--strict` 检查的是数据是否实际覆盖 28 类 m2bOS 记忆场景，不会因为代码中存在某个分支就冒充已覆盖。

## 6. SearchService 与写入争用

对应 OpenViking 的 Session contention / 检索性能维度，但调用的是 m2bOS 的真实 Runtime：

```bash
python -m benchmark load \
  --dataset m2bos \
  --input benchmark/datasets/m2bos_core_v1.json \
  --config /path/to/config.yaml \
  --output benchmark/results/load \
  --work /path/to/empty/load-work \
  --search-operations 100 \
  --write-operations 20 \
  --concurrency 16
```

输出搜索和写入 QPS、错误率、p50/p95/p99，并等待真实 MemoryJob 队列排空。

## 7. VectorStore 基准

```bash
python -m benchmark vector \
  --input benchmark/datasets/vector_core_v1.json \
  --config /path/to/config.yaml \
  --output benchmark/results/vector \
  --work /path/to/empty/vector-work \
  --top-k 10 \
  --concurrency 8 \
  --repeats 10
```

它通过正式 Embedder 和可配置 VectorStore Adapter 执行全量发布、目录过滤检索、Recall/Precision/NDCG、QPS、p50/p95/p99、增量更新删除及最终可见性检查。仓库自带的 12 文档数据仅用于低成本 smoke；性能结论必须换用具有真实相关性标注的大规模数据集。

OpenViking 的 `cuvs` 是本地索引专项，而 m2bOS 当前只允许远程 VectorStore，因此不会伪造一个本地 cuVS 对标。OpenViking `vectordb_perf` 的预计算向量和 read-only/write-only 运维模式也不等同于 m2bOS 的端到端 Embedder + VectorStore 基准；两者应分别报告，不能把数字放在同一列直接比较。

## 8. 生命周期与故障恢复

```bash
python -m benchmark lifecycle \
  --dataset m2bos \
  --input benchmark/datasets/m2bos_core_v1.json \
  --config /path/to/config.yaml \
  --output benchmark/results/lifecycle \
  --work /path/to/empty/lifecycle-work

python -m benchmark recovery \
  --dataset m2bos \
  --input benchmark/datasets/m2bos_core_v1.json \
  --config /path/to/config.yaml \
  --output benchmark/results/recovery \
  --work /path/to/empty/recovery-work
```

生命周期基准覆盖 Segment → Range → Archive 以及 History/Job/Receipt 清理。恢复基准在真实主链上注入一次瞬态 VectorStore 故障，测量 Job 重试、事务恢复和最终提交，不使用 scripted/fake Adapter。

## 对齐边界

- 已直接对齐：LoCoMo、LongMemEval 的样本隔离、默认排除 LoCoMo category 5、Top-10、4000 字符回答预算、Answer/Judge、分类准确率、耗时和 Token。
- m2bOS 增补：六类记忆树、工具结果、Intention 状态、Links/Backlinks、Summary fallback、生命周期和 Job 恢复。
- 不适用：OpenViking Resource RAG、SkillsBench、TAU2 Agent trajectory、grep/BM25 和本地 cuVS；这些能力不在当前 m2bOS 主链中，不能为了“数量对齐”伪造实现。
