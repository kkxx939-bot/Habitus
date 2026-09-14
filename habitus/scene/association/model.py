"""关联的输入与产出形状。

与归组的区别在**单位**：归组问的是"这天的行为分成哪几件事"（单位是一天），关联问的是"这个候选
这一次为什么发生、和它历史上哪几次同源"（单位是一个候选的一次发生）。两个问题混进一个提示词
两头都做不好，所以拆成两套。

输入的六份材料里，**前四份是确定性算出来的**（当天的流、情境事实、现有情境、未兑现的前提），
第五份前因候选由预测树的数字筛出（全天基线越低越罕见、树上已有转移边的优先），第六份是这次
发生本身。每一行都带编号与 URI——模型只能**引用**这些编号，不能编：产出里的每个引用都要能
在输入里找回去，找不回去的当场作废（见 ``assembly``）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol


def _number(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _ascending(values: object, label: str) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    numbers = tuple(_number(value, label) for value in values)
    if list(numbers) != sorted(set(numbers)):
        raise ValueError(f"{label} must be strictly ascending without repeats")
    return numbers


class _Numbered(Protocol):
    """带编号的输入行。用 Protocol 而不是 ``getattr``，让类型检查真的看得见这个字段。"""

    @property
    def no(self) -> int: ...


def _positive_series(values: Sequence[_Numbered], label: str) -> tuple[int, ...]:
    """行号必须是 1..N。口径与下游取值一致——下游拿的是**原始** ``row.no``，所以这里也不做
    ``int()`` 转换：一个 ``no=1.5`` 的行会 ``int()`` 成 1 混过去，再以 ``#4.5`` 渲染出去，
    模型无论写什么整数都引用不到它。"""

    numbers = tuple(_number(item.no, f"{label} no") for item in values)
    if numbers != tuple(range(1, len(numbers) + 1)):
        raise ValueError(f"{label} must be numbered 1..N in order")
    return numbers


def _aware(value: object, label: str) -> datetime:
    """时刻必须带时区。提示词把时刻当事实渲染给模型看，而 naive 值会被按系统本地时区
    重新解释，渲染出的钟点与输入不是同一个时刻。"""

    if not isinstance(value, datetime) or value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    return value


@dataclass(frozen=True)
class OccurrenceRow:
    """当天时间线上的一条行为。目标那几条也在里面，靠 ``AssociationInput.targets`` 指出来。"""

    no: int
    uri: str
    name: str
    kind_token: str
    started_at: datetime
    summary: str
    goal: str | None = None


@dataclass(frozen=True)
class CauseRow:
    """一条前因候选：上一次这个候选发生到这一次之间、值得看的那些行为。

    **不带任何统计数字。**挑哪些行给模型看，由预测树的全天基线与转移边计数决定（见
    ``scene.backlog.CauseFacts``），但那是**排序依据**，排完就留在装配层。把数字摆进提示词会让
    模型拿转移计数倒推语义前因，统计被洗成语义再喂回预测层，形成自证；而语义关联层的维度必须是
    语义的表现，不能把预测树候选已经带着的统计维度重新表达一遍。
    """

    no: int
    uri: str
    name: str
    started_at: datetime
    summary: str


@dataclass(frozen=True)
class SituationRow:
    """现有的一种情境（L1 里的一条）：它是什么、覆盖了哪些日期。"""

    no: int
    text: str
    days: tuple[date, ...] = ()


@dataclass(frozen=True)
class PendingRow:
    """一条尚未兑现的待用前提；``consumed_by`` 是它在等的那个 kind。"""

    no: int
    text: str
    producer_uri: str
    created_on: date
    consumed_by: str | None = None


@dataclass(frozen=True)
class DayFacts:
    """情境事实，**全部确定性算出**，不经模型。

    它们是判断者做逐次比对时的硬抓手：周几对不对得上、日型同不同、几月差多远，先有这些，
    再轮到语义像不像。
    """

    weekday: int
    month: int
    day_note: str | None = None
    observed_gaps: tuple[tuple[datetime, datetime], ...] = ()

    def agrees_with(self, day: date) -> bool:
        return self.weekday == day.weekday() and self.month == day.month


@dataclass(frozen=True)
class AssociationInput:
    """一次关联调用的全部材料。"""

    kind_token: str
    day: date
    targets: tuple[int, ...]
    occurrences: tuple[OccurrenceRow, ...]
    facts: DayFacts
    situations: tuple[SituationRow, ...] = ()
    causes: tuple[CauseRow, ...] = ()
    pending: tuple[PendingRow, ...] = ()
    origin: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind_token, str) or not self.kind_token:
            raise ValueError("association input kind_token must be non-empty text")
        if isinstance(self.day, datetime) or not isinstance(self.day, date):
            raise TypeError("association input day must be a date without a time")
        if not self.occurrences:
            raise ValueError("association input must contain the day's occurrences")
        _positive_series(self.occurrences, "occurrence rows")
        _positive_series(self.causes, "cause rows")
        _positive_series(self.situations, "situation rows")
        _positive_series(self.pending, "pending rows")
        if not self.targets:
            raise ValueError("association input must name at least one target occurrence")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("association targets must not repeat")
        by_no = {row.no: row for row in self.occurrences}
        for target in self.targets:
            targeted = by_no.get(target)
            if targeted is None:
                raise ValueError("an association target must be one of the day's occurrences")
            if targeted.kind_token != self.kind_token:
                raise ValueError("an association target must be an occurrence of this candidate")
        for occurrence in self.occurrences:
            _aware(occurrence.started_at, "association row started_at")
        for cause in self.causes:
            _aware(cause.started_at, "association row started_at")
        for start, end in self.facts.observed_gaps:
            _aware(start, "observed gap start")
            _aware(end, "observed gap end")
        if not self.facts.agrees_with(self.day):
            # 上游一个 off-by-one 会让模型为周二的发生写下"周五晚上"，而那是一句通顺的话，
            # 装配层看不出来——只能在这里拦。
            raise ValueError("association day facts must describe the same day as `day`")
        instants = [row.started_at for row in self.occurrences]
        if instants != sorted(instants):
            raise ValueError("occurrence rows must be ordered by started_at")

    @property
    def citable(self) -> frozenset[int]:
        """``context`` 可以引用的编号：当天的流与前因候选。

        分成两组编号会让提示词多一层解释成本；这里用**同一套编号**（前因候选是当天之外的行为，
        编号接在当天的流后面），所以模型只需要记住"引用你看到的那个 #n"。
        """

        return frozenset(row.no for row in self.occurrences) | frozenset(
            row.no + len(self.occurrences) for row in self.causes
        )

    def row(self, no: int) -> OccurrenceRow:
        """按编号取当天的一行。编排层要把编号还原成 URI，给它一个显式入口。"""

        if not 1 <= no <= len(self.occurrences):
            raise KeyError(f"occurrence #{no} is not part of this day")
        return self.occurrences[no - 1]

    @property
    def instants(self) -> dict[int, datetime]:
        """每个可引用编号对应的开始时刻，供装配层核对"前因必须更早"。"""

        offset = len(self.occurrences)
        return {row.no: row.started_at for row in self.occurrences} | {
            row.no + offset: row.started_at for row in self.causes
        }

    def cause_no(self, cited: int) -> int | None:
        """把 ``citable`` 里的编号还原成前因候选自己的编号；不是前因就返回 None。"""

        offset = cited - len(self.occurrences)
        return offset if 1 <= offset <= len(self.causes) else None


@dataclass(frozen=True)
class AssociationDraft:
    """一条发生的关联结果（尚未落盘）。"""

    occurrence_no: int
    context: str
    cites: tuple[int, ...]
    causes: tuple[int, ...] = ()
    situation_no: int | None = None
    new_situation: str | None = None
    consumed: tuple[int, ...] = ()
    left: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _number(self.occurrence_no, "association draft occurrence_no")
        for name in ("context", "new_situation"):
            value = getattr(self, name)
            if value is None and name == "new_situation":
                continue
            if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
                raise ValueError(f"association draft {name} must be a single non-empty line")
        for name in ("cites", "causes", "consumed"):
            _ascending(getattr(self, name), f"association draft {name}")
        if not self.cites:
            raise ValueError("association draft must cite at least one row")
        if not set(self.causes) <= set(self.cites):
            raise ValueError("association draft causes must be a subset of its cites")
        if self.situation_no is not None:
            _number(self.situation_no, "association draft situation_no")
            if self.new_situation is not None:
                raise ValueError("association draft names an existing situation and a new one at once")
        if not isinstance(self.left, tuple):
            raise TypeError("association draft left must be a tuple")
        for item in self.left:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or any(not isinstance(part, str) or not part.strip() for part in item)
            ):
                raise ValueError("association draft left must contain (text, consumed_by) pairs of non-empty text")


@dataclass(frozen=True)
class AssociationAssembly:
    """一次调用的装配结果；``signals`` 记下被降级或丢弃的部分，不静默。

    ``unanswered`` 是这次没能拿到草稿的目标编号。编排层要据此决定那几次发生算不算做过，
    让它去正则匹配中文信号串是不可接受的——所以给一个结构化的出口。
    """

    drafts: tuple[AssociationDraft, ...]
    unanswered: tuple[int, ...] = ()
    signals: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.drafts, tuple) or any(not isinstance(item, AssociationDraft) for item in self.drafts):
            raise TypeError("association assembly drafts must be a tuple of AssociationDraft")
        answered = [draft.occurrence_no for draft in self.drafts]
        if len(set(answered)) != len(answered):
            raise ValueError("association assembly answers an occurrence more than once")
        _ascending(self.unanswered, "association assembly unanswered")
        if set(answered) & set(self.unanswered):
            raise ValueError("association assembly cannot both answer and skip an occurrence")
        if not isinstance(self.signals, tuple) or any(not isinstance(note, str) for note in self.signals):
            raise TypeError("association assembly signals must be a tuple of strings")


__all__ = [
    "AssociationAssembly",
    "AssociationDraft",
    "AssociationInput",
    "CauseRow",
    "DayFacts",
    "OccurrenceRow",
    "PendingRow",
    "SituationRow",
]


# TODO(ASSOC-CONFIG-001): 第 4 步接线时把数值搬进配置的单一出处。``AssociationConfig`` 现在只有
# 类上的默认值，没有生产调用方——这不算违规（领域对象带默认值是本仓库的既定做法），但一旦编排层
# 上线，数值就必须只从 ``Config.scene`` 进入。
#
# 改造：复用 ``scene:`` 这一组、不另起新组——``max_prompt_chars`` / ``transient_retries`` /
# ``transient_retry_delay_seconds`` 与归组语义完全相同，直接沿用同一批值；只新增
# ``max_targets_per_call``、``association_per_candidate``、``association_max_tasks_per_run`` 三项，
# 写进 ``SceneConfig`` + ``_INT_BOUNDS`` + ``example.yaml``，由 ``runtime/behavior.py`` 按字段注入。
# 归组已经删掉，``max_occurrences_per_call`` 也随之摘掉了（2026-09-13）。
# 影响：小（一组配置字段 + 一处注入 + 配置契约矩阵登记）。现在就补等于加一组空转的旋钮，
# 反而制造"配了但不生效"的假象。
