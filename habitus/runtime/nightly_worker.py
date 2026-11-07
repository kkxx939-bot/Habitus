"""整晚那一趟的常驻节拍（裁定 59，清单 5.1；裁定 94 第 1、8、9 条）：前一天定稿完成就接着跑一次 ``Nightly``，一天一次，不挂固定时刻。

夜里是一条线（前一天定稿 → 前置处理 → 结算 → 词表（每晚新增、定期拆改）→ 语义树 → 机会表 → 单元 → 当天的日子预期），前一步完成即接
下一步。定稿由归约在它的轮次里定（链都落树、封口视界已过那天的本地结束），这里每隔一会儿看一眼 ``Nightly.ready``：昨天定稿了、
今天这一晚还没跑过，就跑。

**只有整条线跑完、或这一晚停下了，才记这一晚跑过**（``NightOutcome.complete`` / ``stopped``），跑完了没有分开记（``LastRun``：
结算据它认停掉的那天）。线上每成一段就落盘（``LastRun.step``）：某一段还没准备好（词表迁移中、锁被占——含前置处理里重试那一支的
锁），下一眼带着记下的进度从那一段接着跑（``Nightly.run_once(resume=…)``；前置处理里的重试从它停下的地方接着跑，见
``behavior.retry.stage``）。一段出错（bug）或模型整个调不通，这一晚停下、记成跑过——不每隔几分钟整段重跑，下一晚照常（裁定 94
第 8 条）。服务夜里没跑的（关机、合盖、进程没起），起来时昨天已定稿就接着跑。进程被杀之后重试堆里剩下的项属于已推过的日子，
起来时直接撤回丢掉（裁定 68 第 3 条，``clean_leftovers``）。从断点接着跑的那一趟不再做前置处理，那一晚前置处理的计数（"错误·看不到"
几项）沿用之前那一趟的，健康面不丢（同一进程里；重启之后那一晚的计数只在观测里）。

正确性全在 ``runtime.nightly.Nightly``：哪一晚、闸、先后顺序、哪一段算成都在那里。这里只管节奏、"循环别死"、记进度、留下可查的
痕迹——停下、没跑完、前置处理里保持「错误·看不到」的都设 ``last_error``、记一笔观测，健康面据此报降级。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

from habitus.behavior.retry import RetryReport
from habitus.foundation.observability import ObservationStatus, Observer
from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes
from habitus.runtime.nightly import PREPARE, NightOutcome
from habitus.runtime.resident import ResidentWorker

_LAST_RUN_BYTES = 16_384
#: "哪几晚跑完了"留多久（一年）：结算认停掉的那天只往回看到过期待决的预期还没裁的那一段，账本整天释放之后更早的不再有人问。
KEPT_NIGHTS = 366
#: 隔多久看一眼昨天定稿了没有（归约本身也是几分钟一轮）。按墙钟醒来重看：Mac 合盖时单调时钟不走（``mach_absolute_time``），
#: 一口气睡长了会被推迟到开盖之后好几个小时（裁定 68 第 4 条）。
POLL_SECONDS = 300.0


class NightRunner(Protocol):
    """ "第 N 晚"整趟；``runtime.nightly.Nightly`` 就是这个形状。"""

    def ready(self) -> date | None: ...

    async def run_once(
        self, *, resume: Iterable[str] = (), on_step: Callable[[str], None] | None = None
    ) -> NightOutcome: ...


class LastRun:
    """这条线自己的记录，落在一个小文件里（``path`` 为 None 时只记在内存里，测试用）：

    - 最近跑过（跑完或停下）的那一晚：重启之后据它判这一晚跑过没有（``read``）；
    - 正在跑的那一晚已成的段：下一眼从哪里接着跑（``steps``）；
    - 哪几晚整条线跑完了：从第一次跑完的那一晚（``since``）起，最近跑完的那一晚（``finished``）之前没跑完的几晚（``missed``，
      停下了、没跑到、没开机都在里面）。"这一晚的线跑完了没有"只有这一个判断（``finished``，裁定 94 第 1、9 条：只有整条线跑完
      这一晚才算跑过，日子预期那一段没跑完即算停）：结算据它认停掉的那天，白天那一拍据它决定今天建不建预期。

    旧格式只有 ``night``（与 ``partial``）：照样读成"那一晚跑过"，跑完了没有不知道（``finished`` 答 None）。"""

    def __init__(self, path: str | Path | None = None, *, root: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self.root = None if root is None else Path(root)
        self._value: date | None = None
        self._progress: tuple[date, tuple[str, ...]] | None = None
        self._since: date | None = None
        self._finished: date | None = None
        self._missed: frozenset[date] = frozenset()
        self._loaded = path is None
        self._lock = threading.Lock()

    def read(self) -> date | None:
        """最近跑完（或停下）的那一晚。"""

        with self._lock:
            self._load()
            return self._value

    def steps(self, night: date) -> frozenset[str]:
        """这一晚已经成了的段（不是这一晚的进度就是空）。"""

        with self._lock:
            self._load()
            return self._steps(night)

    def finished(self, night: date) -> bool | None:
        """这一晚的线整条跑完了没有：跑完为真；停下了、没跑到、没开机为假；记录够不着的那一晚（第一次跑完之前、或留存之外）
        是 None——认不出，调用方各按各的保守口径（结算不认它停了，白天那一拍不当它跑完了）。"""

        with self._lock:
            self._load()
            if self._since is None or self._finished is None or night < self._since:
                return None
            return not (night > self._finished or night in self._missed)

    def step(self, night: date, step: str) -> None:
        """这一晚又成了一段。"""

        with self._lock:
            self._load()
            done = tuple(self._steps(night))
            self._progress = (night, (*done, step) if step not in done else done)
            self._save()

    def write(self, night: date, *, finished: bool) -> None:
        """这一晚跑完了（``finished``）或停下了：进度清掉；跑完的那一晚记进"哪几晚跑完了"。"""

        with self._lock:
            self._load()
            self._value, self._progress = night, None
            if finished:
                self._finish(night)
            self._save()

    def _steps(self, night: date) -> frozenset[str]:
        if self._progress is None or self._progress[0] != night:
            return frozenset()
        return frozenset(self._progress[1])

    def _finish(self, night: date) -> None:
        if self._finished is None or self._since is None:
            self._since = self._finished = night
        elif night > self._finished:
            gap = (night - self._finished).days
            self._missed |= {self._finished + timedelta(days=offset) for offset in range(1, gap)}
            self._finished = night
        else:
            self._missed -= {night}  # 时钟回拨过、那一晚补跑完了
        floor = self._finished - timedelta(days=KEPT_NIGHTS)
        self._since = max(self._since, floor)
        self._missed = frozenset(day for day in self._missed if day >= self._since)

    def _load(self) -> None:
        if self._loaded:
            return
        assert self.path is not None
        try:
            encoded = read_regular_bytes(
                self.path, artifact_root=self.root or self.path.parent, max_bytes=_LAST_RUN_BYTES
            )
            raw = json.loads(encoded.decode("utf-8"))
            self._value = _day(raw.get("night"))
            partial = raw.get("partial")
            if partial is not None:
                self._progress = (date.fromisoformat(str(partial["night"])), tuple(str(s) for s in partial["steps"]))
            self._since, self._finished = _day(raw.get("since")), _day(raw.get("finished"))
            self._missed = frozenset(date.fromisoformat(str(day)) for day in raw.get("missed", ()))
        except FileNotFoundError:
            # 还没有记录就是一晚都没跑过：保持初始状态
            pass
        except (ValueError, KeyError, TypeError, AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            # 读不出就当没跑过：再跑一趟，每一段都幂等，代价只是多跑一次；哪几晚跑完了也不知道，结算不认停掉的那天
            self._value = self._progress = self._since = self._finished = None
            self._missed = frozenset()
        self._loaded = True

    def _save(self) -> None:
        if self.path is None:
            return
        raw: dict[str, object] = {"night": None if self._value is None else self._value.isoformat()}
        if self._progress is not None:
            raw["partial"] = {"night": self._progress[0].isoformat(), "steps": list(self._progress[1])}
        if self._since is not None and self._finished is not None:
            raw["since"], raw["finished"] = self._since.isoformat(), self._finished.isoformat()
            raw["missed"] = sorted(day.isoformat() for day in self._missed)
        atomic_replace_bytes(self.path, json.dumps(raw).encode("utf-8"), artifact_root=self.root or self.path.parent)


def _day(value: object) -> date | None:
    return None if value is None else date.fromisoformat(str(value))


class NightlyWorker(ResidentWorker):
    _task_name = "habitus-nightly"
    _observation_category = "nightly"

    def __init__(
        self,
        nightly: NightRunner,
        *,
        shutdown_timeout_seconds: float,
        last_run: LastRun | None = None,
        observer: Observer | None = None,
    ) -> None:
        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        self.nightly = nightly
        self.last_run = last_run if last_run is not None else LastRun()
        #: 最近一趟的结果（还没跑过、或最近一趟抛了是上一次成功的那份 / None）与它是哪一晚的（``_night`` 收到的那一晚）。
        self.last_outcome: NightOutcome | None = None
        self._last_night: date | None = None

    async def run_once(self) -> NightOutcome:
        """手动从头跑一趟（运维与测试用），不读写进度；不许和常驻循环抢。"""

        if self.running:
            raise RuntimeError("manual run_once cannot race the resident worker loop")
        outcome = await self.nightly.run_once()
        self.last_outcome = outcome
        return outcome

    def due(self) -> date | None:
        """现在该跑的那一晚：昨天定稿了、这一晚还没跑完（记下的那一晚不是它——时钟回拨过也再跑一趟）；否则 None。"""

        night = self.nightly.ready()
        return night if night is not None and self.last_run.read() != night else None

    async def _run_loop(self) -> None:
        clean = getattr(self.nightly, "clean_leftovers", None)
        if callable(clean):
            try:
                await asyncio.to_thread(clean)  # 上一晚中途被杀剩下的重试项：现在就撤回，不等下一晚
            except Exception as exc:  # noqa: BLE001 - 常驻循环必须活过基础设施抖动
                self.last_error = exc
        while not self._stop_requested.is_set():
            try:
                night = await asyncio.to_thread(self.due)
            except Exception as exc:  # noqa: BLE001 - 读不了定稿日照样活着，下一眼再看
                self.last_error = exc
                night = None
            if night is not None:
                await self._night(night)
            await self._wait(POLL_SECONDS)

    async def _night(self, night: date) -> None:
        started = time.monotonic()
        problems: list[str] = []
        try:
            resume = self.last_run.steps(night)
        except Exception:  # noqa: BLE001 - 读不出进度就从头跑：每一段都幂等
            resume = frozenset()
        try:
            outcome = await self.nightly.run_once(resume=resume, on_step=lambda step: self._step(night, step, problems))
        except Exception as exc:  # noqa: BLE001 - 常驻循环必须活过基础设施抖动；不记跑过，下一眼从断点接着跑
            self.last_error = exc
            self._observe("night", ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started=started)
            return
        outcome = self._carry_prepared(night, resume, outcome)
        if outcome.complete or outcome.stopped is not None:
            problems.extend(self._mark(night, finished=outcome.complete))
        self.last_outcome = outcome
        self._last_night = night
        self._succeeded()
        # 停下、没跑完、或前置处理里有保持「错误·看不到」的：都列出来，健康面据此报降级
        problems.extend(_problems(outcome))
        if problems:
            self.last_error = RuntimeError("; ".join(problems))
        self._observe("night", ObservationStatus.SUCCESS, _attributes(outcome), started=started)

    def _carry_prepared(self, night: date, resume: frozenset[str], outcome: NightOutcome) -> NightOutcome:
        """从断点接着跑的这一趟没做前置处理：那一晚的前置处理结果沿用之前那一趟的（续跑不丢健康面上的"错误·看不到"）。"""

        previous = self.last_outcome
        if PREPARE not in resume or self._last_night != night or previous is None:
            return outcome
        return replace(outcome, retried=previous.retried, calendar_failed=previous.calendar_failed)

    def _step(self, night: date, step: str, problems: list[str]) -> None:
        try:
            self.last_run.step(night, step)
        except Exception as exc:  # noqa: BLE001 - 记不下来照样活着：最坏是下一眼从更前面接着跑
            problems.append(f"nightly progress not written: {type(exc).__name__}")

    def _mark(self, night: date, *, finished: bool) -> list[str]:
        try:
            self.last_run.write(night, finished=finished)
        except Exception as exc:  # noqa: BLE001 - 记不下来照样活着：最坏是重启后多跑一趟
            return [f"nightly last-run marker not written: {type(exc).__name__}"]
        return []


def _problems(outcome: NightOutcome) -> list[str]:
    found: list[str] = []
    if outcome.stopped is not None:
        # 模型整个调不通、或某一段出错：这一晚停下、不拿旧数据顶（裁定 94 第 4、8 条）
        found.append(f"nightly stopped: {outcome.stopped}")
    elif not outcome.complete:
        found.append(
            f"nightly line not finished (done: {', '.join(outcome.done) or '-'}; skipped: {outcome.skipped or '-'})"
        )
    if outcome.calendar_failed is not None:
        found.append(f"nightly calendar refresh failed: {outcome.calendar_failed}")
    kept = kept_errors(outcome.retried)
    if kept is not None:
        found.append(f"nightly errors kept unseen: {kept}")
    return found


def kept_errors(retried: RetryReport | None) -> str | None:
    """前置处理之后仍保持「错误·看不到」、要人看的（裁定 94 第 4 条）：我们自己处理不了的是 bug，内容安全拦下的单独列，格式不合
    夜里再试仍不合的也列出来；都没有是 None。锁被占不在这里：那是还没准备好，线停在前置处理、下一眼接着跑。"""

    if retried is None or not (retried.deterministic or retried.refused or retried.invalid):
        return None
    return f"处理不了 {retried.deterministic}，内容安全 {retried.refused}，格式不合 {retried.invalid}"


def _attributes(outcome: NightOutcome) -> dict[str, str | int | float | bool]:
    """这一趟的读数：哪一晚、停在哪、为什么停、序列几条、语义树 / 机会表 / 结算跑没跑；开头的重试各类几项（裁定 67 第 12 条）。"""

    retried = outcome.retried
    return {
        "night": outcome.night.isoformat(),
        "skipped": outcome.skipped or "",
        "complete": outcome.complete,
        "stopped": outcome.stopped or "",
        "records": len(outcome.series.records),
        "scene": outcome.scene is not None,
        "opportunities": outcome.opportunities is not None,
        "settled": outcome.settled is not None,
        "calendar_failed": outcome.calendar_failed or "",
        "days_recorded": outcome.days.formed.recorded if outcome.days is not None else 0,
        "days_skipped": (outcome.days.skipped or "") if outcome.days is not None else "",
        "retry_taken": retried.taken if retried is not None else 0,
        "retry_recovered": retried.recovered if retried is not None else 0,
        "retry_unavailable": retried.unavailable if retried is not None else 0,
        "retry_invalid": retried.invalid if retried is not None else 0,
        "retry_deterministic": retried.deterministic if retried is not None else 0,
        "retry_leftover": retried.leftover if retried is not None else 0,
        "retry_refused": retried.refused if retried is not None else 0,
        "retry_waiting": ",".join(retried.waiting) if retried is not None else "",
    }


__all__ = ["KEPT_NIGHTS", "POLL_SECONDS", "LastRun", "NightRunner", "NightlyWorker", "kept_errors"]
