"""归约写入层的错误类型。"""

from __future__ import annotations


class BehaviorReductionError(RuntimeError):
    """归约的输入或不变量被破坏；本层拒绝带病产出。"""


class BehaviorReductionBusyError(TimeoutError):
    """sweep 锁被另一持有者占用；这是多实例场景的正常让路，不是故障。

    单独成类是为了让调用方（Worker 节拍）把"锁忙跳拍"与真正的 TimeoutError
    （租约续期失败、文档锁竞争等）区分开——后者必须记录，前者只是让路。
    """


__all__ = ["BehaviorReductionBusyError", "BehaviorReductionError"]
