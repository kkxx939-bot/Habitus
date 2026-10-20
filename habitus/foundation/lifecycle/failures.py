"""确定性失败的标记基类：同样的输入再试多少次也是同样的失败（提示词超长、条数超限、校验不过）。

各层自己的异常多继承它；重试账按 ``isinstance`` 判"进隔离、不再重试"，不靠比对异常名或消息文本。
不是它的一律当可重试处理，次数用完再转确定性。
"""

from __future__ import annotations


class DeterministicFailure(Exception):  # noqa: N818 - 标记基类，名字表达"这类失败"，各层的具体异常才叫 *Error
    """标记：重试也不会好。"""


__all__ = ["DeterministicFailure"]
