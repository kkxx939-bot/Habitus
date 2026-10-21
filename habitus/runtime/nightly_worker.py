"""整晚那一趟的常驻节拍（裁定 59，清单 5.1；裁定 67 第 7 条）：每天主体本地的 ``scene.nightly_at`` 跑一次 ``Nightly``。

固定时刻，不是"进程一启动先跑一趟、之后每隔一天"：那一趟开头要把白天读不懂的重试一次、只试一次，重启插件那一刻多半额度
还没恢复。上一趟开跑的时刻落盘（``last_run``）：启动时只在错过了最近一个预定时刻（上一趟早于它）时补跑一趟，否则等到下一个。
上一趟抛了也算跑过——失败的那一步第二晚带着新数据再来，不在白天反复重跑。

原来挂在预测树的重建 worker 上（预测树一拍、接上语义树后整晚代替重建）；预测树删了，整晚那一趟自己有一个 worker。
正确性全在 ``runtime.nightly.Nightly``：哪一晚、闸、先后顺序都在那里。这里只管节奏、"循环别死"、留下可查的痕迹——
一夜失败只设 ``last_error``、记一笔观测，下一拍照常来（派生层不做级联）。整晚跑完了、但其中的结算或开头的重试抛了，也设
``last_error``：被整晚吞掉、照跑后面几步，不留在 worker 上的话健康面一直报好。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from datetime import time as clock_time
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

from habitus.foundation.observability import ObservationStatus, Observer
from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes
from habitus.runtime.nightly import NightOutcome
from habitus.runtime.resident import ResidentWorker

_LAST_RUN_BYTES = 4_096
#: 等下一个预定时刻时，一次最多睡多久就醒来重看墙钟：Mac 合盖时单调时钟不走（``mach_absolute_time``），一口气睡到
#: 预定时刻会被推迟到开盖之后好几个小时（裁定 68 第 4 条）。
MAX_NAP_SECONDS = 300.0


class NightRunner(Protocol):
    """ "第 N 晚"整趟；``runtime.nightly.Nightly`` 就是这个形状。"""

    async def run_once(self) -> NightOutcome: ...


def latest_due(now: datetime, *, at: clock_time, zone: ZoneInfo) -> datetime:
    """``now`` 那一刻之前（含）最近的一个预定时刻（主体本地的 ``at``）。"""

    local = now.astimezone(zone)
    due = datetime.combine(local.date(), at, tzinfo=zone)
    if due > local:
        due = datetime.combine(local.date() - timedelta(days=1), at, tzinfo=zone)
    return due


def next_due(now: datetime, *, at: clock_time, zone: ZoneInfo) -> datetime:
    """``now`` 之后最近的一个预定时刻。"""

    local = now.astimezone(zone)
    due = datetime.combine(local.date(), at, tzinfo=zone)
    if due <= local:
        due = datetime.combine(local.date() + timedelta(days=1), at, tzinfo=zone)
    return due


class LastRun:
    """上一趟开跑的时刻，落在一个小文件里（重启之后据它判错过没有）；``path`` 为 None 时只记在内存里（测试）。"""

    def __init__(self, path: str | Path | None = None, *, root: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self.root = None if root is None else Path(root)
        self._value: datetime | None = None

    def read(self) -> datetime | None:
        if self.path is None or self._value is not None:
            return self._value
        try:
            encoded = read_regular_bytes(
                self.path, artifact_root=self.root or self.path.parent, max_bytes=_LAST_RUN_BYTES
            )
            value = datetime.fromisoformat(str(json.loads(encoded.decode("utf-8"))["started_at"]))
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
            return None  # 读不出就当没跑过：补跑一趟，代价只是多跑一次
        self._value = value if value.utcoffset() is not None else None
        return self._value

    def write(self, started_at: datetime) -> None:
        self._value = started_at
        if self.path is None:
            return
        payload = json.dumps({"started_at": started_at.isoformat(timespec="seconds")}).encode("utf-8")
        atomic_replace_bytes(self.path, payload, artifact_root=self.root or self.path.parent)


class NightlyWorker(ResidentWorker):
    _task_name = "habitus-nightly"
    _observation_category = "nightly"

    def __init__(
        self,
        nightly: NightRunner,
        *,
        at: clock_time,
        zone: ZoneInfo,
        shutdown_timeout_seconds: float,
        last_run: LastRun | None = None,
        clock: Callable[[], datetime] | None = None,
        observer: Observer | None = None,
    ) -> None:
        if not hasattr(nightly, "run_once") or not callable(nightly.run_once):
            raise TypeError("nightly must expose an async run_once()")
        if not isinstance(at, clock_time) or not isinstance(zone, ZoneInfo):
            raise TypeError("at must be a time and zone a ZoneInfo")
        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        self.nightly = nightly
        self.at = at
        self.zone = zone
        self.last_run = last_run if last_run is not None else LastRun()
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        #: 最近一趟的结果（还没跑过、或最近一趟抛了是上一次成功的那份 / None）。
        self.last_outcome: NightOutcome | None = None

    async def run_once(self) -> NightOutcome:
        """手动跑一趟（运维与测试用）；不许和常驻循环抢。"""

        if self.running:
            raise RuntimeError("manual run_once cannot race the resident worker loop")
        outcome = await self.nightly.run_once()
        self.last_outcome = outcome
        return outcome

    @property
    def period_seconds(self) -> float:
        """一天一趟。"""

        return 86_400.0

    def missed(self) -> bool:
        """最近一个预定时刻之后还没跑过（刚装上从没跑过也算；记下的时刻晚于此刻——时钟回拨过——当没跑过）。"""

        now = self._clock()
        last = self.last_run.read()
        return last is None or last > now or last < latest_due(now, at=self.at, zone=self.zone)

    async def _run_loop(self) -> None:
        clean = getattr(self.nightly, "clean_leftovers", None)
        if callable(clean):
            try:
                await asyncio.to_thread(clean)  # 上一晚中途被杀剩下的重试项：现在就撤回，不等下一晚
            except Exception as exc:  # noqa: BLE001 - 常驻循环必须活过基础设施抖动
                self.last_error = exc
        while not self._stop_requested.is_set():
            if self.missed():
                await self._night()
            now = self._clock()
            remaining = (next_due(now, at=self.at, zone=self.zone) - now).total_seconds()
            await self._wait(min(MAX_NAP_SECONDS, max(1.0, remaining)))

    async def _night(self) -> None:
        started = time.monotonic()
        problems: list[str] = []
        try:
            self.last_run.write(self._clock())
        except Exception as exc:  # noqa: BLE001 - 记不下来照样跑：最坏是重启后多补跑一趟
            problems.append(f"nightly last-run marker not written: {type(exc).__name__}")
        try:
            outcome = await self.nightly.run_once()
        except Exception as exc:  # noqa: BLE001 - 常驻循环必须活过基础设施抖动
            self.last_error = exc
            self._observe("night", ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started=started)
            return
        self.last_outcome = outcome
        self._succeeded()
        # 整晚照跑完了、但其中几步没成：都列出来（同一晚结算与重试都失败，两条都要看得见），健康面据此报降级
        problems.extend(_problems(outcome))
        if problems:
            self.last_error = RuntimeError("; ".join(problems))
        self._observe("night", ObservationStatus.SUCCESS, _attributes(outcome), started=started)


def _problems(outcome: NightOutcome) -> list[str]:
    found: list[str] = []
    if outcome.settlement_failed is not None:
        found.append(f"nightly settlement failed: {outcome.settlement_failed}")
    if outcome.retry_failed is not None:
        # 白天读不懂的这一晚没重试成（裁定 67 第 4 条）
        found.append(f"nightly retry failed: {outcome.retry_failed}")
    retried = outcome.retried
    if retried is not None and retried.unavailable and not retried.recovered:
        # 整批因模型用不了丢掉：这些轮永久是「没读懂」（裁定 68 第 4 条）
        found.append(f"nightly retry dropped {retried.unavailable} items: model unavailable")
    if outcome.opportunities is not None and outcome.opportunities.failed:
        # 机会表或单元历史某条 lane 没做成（26 号清单 2.5.1）
        found.append(f"nightly opportunities failed: {'; '.join(outcome.opportunities.failed)}")
    return found


def _attributes(outcome: NightOutcome) -> dict[str, str | int | float | bool]:
    """这一趟的读数：哪一晚、为什么没跑、序列几条、语义树 / 机会表 / 结算跑没跑；开头的重试四类各几项（裁定 67 第 12 条）。"""

    retried = outcome.retried
    return {
        "night": outcome.night.isoformat(),
        "skipped": outcome.skipped or "",
        "records": len(outcome.series.records),
        "scene": outcome.scene is not None,
        "opportunities": outcome.opportunities is not None,
        "settled": outcome.settled is not None,
        "settlement_failed": outcome.settlement_failed or "",
        "opportunities_failed": len(outcome.opportunities.failed) if outcome.opportunities is not None else 0,
        "retry_failed": outcome.retry_failed or "",
        "retry_taken": retried.taken if retried is not None else 0,
        "retry_recovered": retried.recovered if retried is not None else 0,
        "retry_unavailable": retried.unavailable if retried is not None else 0,
        "retry_invalid": retried.invalid if retried is not None else 0,
        "retry_deterministic": retried.deterministic if retried is not None else 0,
        "retry_leftover": retried.leftover if retried is not None else 0,
    }


__all__ = ["LastRun", "NightRunner", "NightlyWorker", "latest_due", "next_due"]
