"""模型四问（``19-预测层方案.md`` 第四节、第二节「后果」）：到了该说的时候，模型只看到那几个候选，一个几千字的小包——

1. 今天这次和历史那几次像不像？  2. 先说哪个？（外加一句话）  另有按账判一次的：这件事后果重不重？

每个答案指向具体哪条历史记录；问两遍（候选顺序反过来）不一致 → 按算法顺序、轻重缓急取较轻的一级；模型可去掉候选、可改先后，
不可凭空加候选。模型触点只在 ``prompt``（提示词与 schema）与 ``service``（调一次、问两遍、核对）；``package`` 是纯函数装的材料。
本包不从 ``habitus.foresight`` 包根导出：包根对模型客户端零知识（架构测试钉着）。
"""

from habitus.foresight.judge.package import (
    Candidate,
    HistoryRow,
    Package,
    UnitInput,
    build_package,
    plain_text,
    sources_of,
)
from habitus.foresight.judge.service import Judge, Judgement, SeverityJudgement, Verdict

__all__ = [
    "Candidate",
    "HistoryRow",
    "Judge",
    "Judgement",
    "Package",
    "SeverityJudgement",
    "UnitInput",
    "Verdict",
    "build_package",
    "plain_text",
    "sources_of",
]
