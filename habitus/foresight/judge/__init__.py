"""判断：预测层的模型触点。读渲染好的证据包，对每个摊开的候选说会不会、什么时候、接下来、依据。

包内分工按"一个变化理由"：``model`` 是产物形状，``schema`` 是输出契约，``prompt`` 是措辞，``assembly``
是自洽校验与降级，``service`` 是调用与重试。模型客户端只在 ``prompt`` 与 ``service`` 出现。
"""

from habitus.foresight.judge.assembly import ASSEMBLY_VERSION, JudgeAssemblyError, assemble_judgement
from habitus.foresight.judge.model import DAY_STATES, VERDICTS, CandidateVerdict, Judgement
from habitus.foresight.judge.prompt import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT, build_request
from habitus.foresight.judge.schema import JUDGE_JSON_SCHEMA, SCHEMA_FINGERPRINT, judge_json_schema
from habitus.foresight.judge.service import JUDGE_VERSION, Judge, JudgeConfig, LLMJudge

__all__ = [
    "ASSEMBLY_VERSION",
    "DAY_STATES",
    "JUDGE_JSON_SCHEMA",
    "JUDGE_PROMPT_VERSION",
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_VERSION",
    "SCHEMA_FINGERPRINT",
    "VERDICTS",
    "CandidateVerdict",
    "Judge",
    "JudgeAssemblyError",
    "JudgeConfig",
    "Judgement",
    "LLMJudge",
    "assemble_judgement",
    "build_request",
    "judge_json_schema",
]
