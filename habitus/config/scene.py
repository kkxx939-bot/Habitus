"""语义关联层（情景树）的运行配置。

本模块只有纯标量、不 import ``scene``（与 ``config/behavior.py`` 同一纪律）：领域内的自洽
校验仍在 ``scene`` 各自的 config 值对象里，这里只管取值范围与"有没有给全"。

情景树每封口日调一次模型，从行为树派生；行为侧没开就没有输入（跨域自洽在 ``root.py``）。
数值全部是**启动档**：回看天数与待用前提的过期期限最终由生命周期算法按 needs 边的滞后
分布定，现在先按用户裁定的初值走。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from habitus.config.loader import construct_config

_INT_BOUNDS = (
    ("max_prompt_chars", 1_000, 4_000_000),
    ("transient_retries", 0, 20),
    ("max_attempts_per_input", 1, 100),
    ("max_model_calls_per_run", 1, 1_000),
    ("max_targets_per_call", 1, 500),
    ("association_per_candidate", 1, 100),
    ("association_max_tasks_per_run", 1, 10_000),
    ("max_cause_rows", 1, 200),
    ("max_pending_rows", 1, 200),
)


@dataclass(frozen=True)
class SceneConfig:
    """语义关联层是否启用，以及调用边界与各项预算。

    按天归组那一套的配置已经全部删掉：``lookback_days`` / ``pending_expiry_days`` 是日历窗口与
    过期判断（周频行为的相邻两次正好卡在 7 天边界上，而"到期没到期"是预测层的判断）；
    ``max_occurrences_per_call`` 是按天的单位；``retained_generations`` 是按天情景文档的留代数，
    随日情景树一起删（规律级按候选累积、只增不改，没有"代"这个概念）。

    这里的数值分两类。**运维数**（一轮做几件、每个候选几件、一次调用几个目标）决定一夜的模型
    开销与哪个候选会被饿死，必须可调。**渲染预算**（前因几行、未兑现前提几行）与
    ``max_prompt_chars`` 是同一个预算的两端，分居两处会出现"字符没超但信息已砍光"或者反过来，
    所以也放在这一组。至于提示词内部的渲染细节（摘要截断、情形日期几个）留在领域对象里——改它
    要跑真实模型对照，不该让人在 yaml 里拧。
    """

    enabled: bool = False
    max_prompt_chars: int = 200_000
    # 传输层瞬态错误的有界重试（路由层自己还有一层，这里保持很小）。
    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0
    # 同一输入连续失败几次后封锁它（换关联版本自动作废）。
    max_attempts_per_input: int = 3
    # 一轮最多几次模型调用。**这是吞吐的真天花板**：一次调用推进一个 (候选, 日期)，所以它小于
    # 每天新增的 (候选, 日期) 数时，语义层就永远追不平行为树，而且那几次总是给最老的日子——
    # 判断者看到的背景会越来越旧。它要与下面的任务上限一起定，两个数错配等于白排队。
    # 注意它的单位是"每一轮"而不是"每一夜"：进程重启会立刻跑一拍。
    max_model_calls_per_run: int = 40
    # 一次调用最多问这个候选那天的几次发生。单位是"一个候选一天发生几次"，与旧归组那个"一整天的
    # 行为数"不是一回事。2026-09-14 按 DAY1 真实分布定为 40（桌面「关联探针」：108 个候选里 105 个
    # 一天 ≤6 次，其余 14 / 35 / 100 次），只挡住一天 100 次的"与某人交谈"这类词表噪声 token。
    # 超限的那一批不切块（同一天的几次要一起看），封锁并留信号；数值等自采数据再定。
    max_targets_per_call: int = 40
    # 每个候选一轮最多取几件（各自取最早的），以及整轮的总上限。两级预算：候选内部必须升序，
    # 候选之间不必，而全局截断会饿死新候选。
    association_per_candidate: int = 2
    association_max_tasks_per_run: int = 40  # 与 max_model_calls_per_run 同量级，否则多排的白排
    # 渲染预算：一次调用摊开几行前因候选、几条未兑现前提。截断都会留信号，不是失效判断。
    max_cause_rows: int = 12
    max_pending_rows: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("scene.enabled must be a boolean")
        for name, lower, upper in _INT_BOUNDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"scene.{name} must be an integer between {lower} and {upper}")
        delay = self.transient_retry_delay_seconds
        if isinstance(delay, bool) or not isinstance(delay, int | float) or not 0.0 <= float(delay) <= 600.0:
            raise ValueError("scene.transient_retry_delay_seconds must be between 0 and 600")

    @classmethod
    def from_mapping(cls, value: Any) -> SceneConfig:
        return construct_config(cls, value, "config.scene")


__all__ = ["SceneConfig"]
