"""行为融合的数据集驱动评测。

与同目录下的长期记忆基准并列，但**不共享数据协议**：那边是"会话 + 问答"，这边是"观测片段序列 →
判断"。共享的只有入口、真实配置加载与报告形态。

只测确定性可判的性质（结构合规、关系命中与误标、主体完整、读不懂、粒度）。需要语义评判的部分
（目标准不准、分解合不合理）等这一层跑稳、看清哪里真的不稳定之后再加 Judge。

每一类都配对照组：只量"该标的标了没有"会漏掉滥用，只量"不该标的没标"会漏掉能力被压死——后者
实测发生过，一句保守的措辞就让模型对所有真因果都不敢标。

## TODO(BHV-BENCH-GOLD)：人工标注黄金集接入（等用户交付标注，2026-08-31 登记）

**现状与问题**：真实切片用例（realdata 类）的期望是拿模型自己的 5 次输出"定档"的——它度量的是
**稳定性**（同输入产出不漂移），不是**正确性**（产出符合"可提醒/可代劳"判据）。用户明确指出过孤立
切片没有上下文、无法人工判对错，所以正确性基准一直缺位。

**已交付的标注材料**（生成器在 benchmark/data/egolife_week/tools/，数据不进 git）：
- `make_vocab_review.py` → 词表归错率标注（100 对名字，每对附各自 1-2 条真实判断实例：时间/摘要/
  目标/步骤；判据=提醒句测试，选项 同/不同/判不了）；
- `make_review_page.py` → DAY1 事件级标注（348 条 occurrence 按整天时间线排开，每条附 ±90s 原始
  字幕+转写；判据=这件事能提醒/能代劳吗，选项 对/不是一件事/该拆/该合）。
  两页标注存 localStorage，导出 JSON 形如 {条目id: {"v": 选项, "n": 备注}}。

**用户交付标注后要做的事**（触发条件：用户说标注完成并给出两份导出 JSON）：
1. 词表 100 对：对照表中"模型判定"列计算**归错率**（错误合并/错误分裂分开算——错误合并更重，
   见 kinds 的"宁分勿并"）；把判错的对连同其实例证据整理成 kinds 归一提示词的调优集，按纪律用
   真实模型对照实验调（不凭推理改措辞）；"判不了"的对单独列出与用户讨论判据边界。
2. DAY1 348 条：算**折叠正确率**并按 不是一件事/该拆/该合 分桶——"不是一件事"多=噪声进树（对应
   无归属出口没走够），"该拆"多=过折，"该合"多=欠折；把 realdata 用例的期望改为以人工裁定为准
   （人工与模型分歧的场景优先做成新用例），此后 stability 口径的"N/N 通过"不得再表述为"正确"。
3. 两份归错样本都要回填到 behavior/fusion/__init__.py 的审计登记（含具体例子），供下一轮提示词
   调优引用；调优本身遵守"真实模型对照 + 只跑 DAY1"的既有纪律。

**影响面**：只动 benchmark 期望与（若调优）融合/kinds 提示词，不动行为树 schema 与归约逻辑；
提示词若变，须重做 DAY1 对照并复核 continues 取舍的已裁定登记。
"""

from habitus.benchmark.fusion.dataset import (
    FUSION_CASE_SCHEMA,
    FusionBenchmarkError,
    FusionCase,
    FusionExpectation,
    FusionFragment,
    load_cases,
)
from habitus.benchmark.fusion.report import aggregate, evaluate, render_markdown
from habitus.benchmark.fusion.runner import FusionCaseRun, run_case

__all__ = [
    "FUSION_CASE_SCHEMA",
    "FusionBenchmarkError",
    "FusionCase",
    "FusionCaseRun",
    "FusionExpectation",
    "FusionFragment",
    "aggregate",
    "evaluate",
    "load_cases",
    "render_markdown",
    "run_case",
]
