"""生命周期线程与归约线程改写同一份文件（26 号清单 L10.3）："读整份、改、整份写回"的日志，截断与追加交错时
不许丢掉另一边刚写下的那一笔。

每条用例让截断停在"读完、写回之前"，另一个线程这时去追加；有锁时追加等截断写完再进来，没锁时追加的那一笔被截断写回的
旧内容盖掉。
"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path

from habitus.behavior.tree.dirty_days import DirtyDayLog

_PAUSE_SECONDS = 0.3


def _race(pause_inside, writer) -> None:  # type: ignore[no-untyped-def]
    """``pause_inside`` 是截断那一边（会在读完之后停住，直到放行）；``writer`` 在它停住时从另一个线程追加。"""

    inside, go = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def truncating() -> None:
        try:
            pause_inside(inside, go)
        except BaseException as exc:  # noqa: BLE001 - 交回主线程断言
            errors.append(exc)

    first = threading.Thread(target=truncating)
    first.start()
    assert inside.wait(2.0)
    second = threading.Thread(target=writer)
    second.start()
    second.join(_PAUSE_SECONDS)  # 没锁时追加在这段时间里就做完了；有锁时它还在等
    go.set()
    first.join(2.0)
    second.join(2.0)
    assert not errors, errors


def test_truncating_the_dirty_day_log_never_loses_a_concurrent_record(tmp_path: Path) -> None:
    log = DirtyDayLog(tmp_path / "reduction")
    first = log.record({date(2026, 10, 1)})
    original = log._entries

    def truncate(inside: threading.Event, go: threading.Event) -> None:
        def paused():  # type: ignore[no-untyped-def]
            found = original()
            if not inside.is_set():
                inside.set()
                go.wait(2.0)
            return found

        log._entries = paused  # type: ignore[method-assign]
        try:
            log.truncate_through(first)
        finally:
            log._entries = original  # type: ignore[method-assign]

    late = date(2026, 10, 2)
    _race(truncate, lambda: DirtyDayLog(tmp_path / "reduction").record({late}))

    _last, days = log.after(first)
    assert days == frozenset({late})
