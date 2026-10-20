"""生命周期共用的纯函数：按名次遗忘、精确压实、连续水位、退避、确定性失败的标记。

只依赖标准库。各层的生命周期（预测层的账、语义树的检验、行为侧与会话源的释放、memory 的扫描退避）
都从这里取同一份算法，不各自再写一遍（裁定 15：生命周期统一设定；裁定 57：按自己的机会数遗忘）。
"""

from habitus.foundation.lifecycle.backoff import BackoffPolicy
from habitus.foundation.lifecycle.compaction import Tally, advance, fold, fold_all, rewind, step_factor
from habitus.foundation.lifecycle.failures import DeterministicFailure
from habitus.foundation.lifecycle.ranks import Ranked, decay, effective_cap, rank_by
from habitus.foundation.lifecycle.watermark import Watermark, covers, mark

__all__ = [
    "BackoffPolicy",
    "DeterministicFailure",
    "Ranked",
    "Tally",
    "Watermark",
    "advance",
    "covers",
    "decay",
    "effective_cap",
    "fold",
    "fold_all",
    "mark",
    "rank_by",
    "rewind",
    "step_factor",
]
