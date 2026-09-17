"""判断的产物：每个摊开的候选一条判决，外加"今天到此刻正不正常"。

这是判断者说出来的话的形状，对 scene 零知识，也不认识证据包——包是它读的材料，不是它的一部分。
``basis`` 里放的是被引用的历史卡的 occurrence URI（装配层从卡的编号换回来），所以一条判决能被
原样核对："他说此刻像那几次"指的是行为树上哪几条记录。

不确定性用"少说一点"表达：``说不准`` 是合格答案，``window`` 可以是 None，``next`` 可以为空，
``day_state`` 为 None 是"没答"；没有 confidence 这类元字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from habitus.foresight.errors import ForesightError
from habitus.foresight.model import Moment

#: 三种判决。``会``：此刻像它历史上的那几次，接下来会做；``不会``：此刻已经走到了别的路上；
#: ``说不准``：此刻场景与历次都对不上，或者材料不够。
VERDICTS = ("会", "不会", "说不准")
#: 今天到此刻为止整体像不像往常的这个时候；判断里为 None 表示模型没给出可用的答案（不是"正常"）。
DAY_STATES = ("正常", "反常")


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ForesightError(f"{label} must be a single non-empty line")


@dataclass(frozen=True)
class CandidateVerdict:
    """一个候选的判决。

    ``window`` 是今天的槽号区间（含两端）：判断者认为它会在这段里开始；``不会`` 的判决没有时窗。
    ``next`` 是它之后紧跟着的行为名字，只能来自被引用的卡的"之后"那段。``basis`` 是引用的卡；
    ``会`` 必须引用至少一张——只看数字不看卡的"会"在装配层就降成了说不准。
    """

    kind_token: str
    verdict: str
    window: tuple[int, int] | None
    next: tuple[str, ...]
    basis: tuple[str, ...]
    note: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind_token, str) or not self.kind_token:
            raise ForesightError("verdict kind_token must be non-empty text")
        if self.verdict not in VERDICTS:
            raise ForesightError(f"unknown verdict: {self.verdict!r}")
        if self.window is not None:
            start, end = self.window
            for value in (start, end):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ForesightError("verdict window must be a pair of non-negative slot numbers")
            if start > end:
                raise ForesightError("verdict window must not end before it starts")
        if self.verdict == "不会" and self.window is not None:
            raise ForesightError("a candidate that will not happen has no window")
        if self.verdict == "会" and not self.basis:
            raise ForesightError("a 会 verdict must cite at least one card")
        for label, series in (("basis", self.basis), ("next", self.next)):
            if any(not isinstance(item, str) or not item for item in series):
                raise ForesightError(f"verdict {label} must be non-empty strings")
            if len(set(series)) != len(series):
                raise ForesightError(f"verdict {label} repeats an item")
        if self.next and not self.basis:
            raise ForesightError("next steps can only come from cited cards")
        if not isinstance(self.note, str) or "\n" in self.note or "\r" in self.note:
            raise ForesightError("verdict note must be a single line")


@dataclass(frozen=True)
class Judgement:
    """一刻的判断：对每个摊开的候选各一条判决，今天整体正常与否，以及装配时降级的信号。

    ``moment`` 与 ``generation`` 是它读的那一包的，``judge_version`` 是产出它的判断者（提示词 + schema + 装配
    纪律）；判断存起来之后要能说清"这是对着哪一代、哪一刻、由哪一版判断者说的"。
    """

    judged_at: datetime
    generation: str
    moment: Moment
    verdicts: tuple[CandidateVerdict, ...]
    day_state: str | None
    day_note: str | None
    judge_version: str
    signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.judged_at, datetime) or self.judged_at.utcoffset() is None:
            raise ForesightError("judged_at must be a timezone-aware datetime")
        if not isinstance(self.generation, str) or not self.generation:
            raise ForesightError("judgement generation must be non-empty text")
        if not isinstance(self.moment, Moment):
            raise ForesightError("judgement moment must be a Moment")
        names = [item.kind_token for item in self.verdicts]
        if names != sorted(set(names)):
            raise ForesightError("judgement verdicts must be unique and sorted by kind")
        if self.day_state is not None and self.day_state not in DAY_STATES:
            raise ForesightError(f"unknown day state: {self.day_state!r}")
        if not isinstance(self.judge_version, str) or not self.judge_version:
            raise ForesightError("judgement judge_version must be non-empty text")
        if self.day_note is not None:
            _text(self.day_note, "judgement day_note")
        if any(not isinstance(note, str) for note in self.signals):
            raise ForesightError("judgement signals must be strings")

    def verdict_for(self, kind_token: str) -> CandidateVerdict | None:
        return next((item for item in self.verdicts if item.kind_token == kind_token), None)

    @property
    def expected(self) -> tuple[CandidateVerdict, ...]:
        """判为"会"的那些。"""

        return tuple(item for item in self.verdicts if item.verdict == "会")


__all__ = ["DAY_STATES", "VERDICTS", "CandidateVerdict", "Judgement"]
