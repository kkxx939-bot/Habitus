"""结算账本的形状：承诺与结算。

承诺（``Claim``）是判断者说出的话里**可核对的那一条**：某个候选、在今天哪一段槽里会开始、依据是哪几张卡。
只有带时窗的「会」进账——「不会」没有可核对的时点，「说不准」是弃权、也没有；说不出时窗的「会」同样进不来。
结算（``Settlement``）是那天定稿之后对着行为树写下的事实：在窗里做了、偏离了几槽、还是没做。

账本只存事实：门槛、幅度、怎么折算都不在这里，它们是读时投影（``gate``），公式改了账本也不作废。
本模块对 scene 零知识、不做 IO；落盘由组合根的存储做（foresight 不许 import infrastructure）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from habitus.foresight.errors import ForesightError
from habitus.foresight.model import CandidateNumbers

#: 一刻的外部条件：键值对，按键升序、不重复。形状与 ``scene.facts.Conditions`` 相同，但账本不 import scene——
#: 条件到这里已经是一组纯文本对了，谁给的、怎么来的是组合根的事（架构测试钉死账本对 scene 零知识）。
Conditions = tuple[tuple[str, str], ...]

#: 结算的三种事实：窗里做了、做了但不在窗里（``slot_offset`` 记差几槽，负数是早于窗）、那天没再做。
OUTCOMES = ("验证", "偏离", "落空")
#: 主体对提醒的回应。提醒通道还没有，现在恒为 None——但字段第一天就在：被提醒之后的发生必须能与自然发生
#: 分开，事后分不开（用户裁定，时间不可逆）。
RESPONSES = ("接受", "拒绝")


def _aware(value: object, label: str) -> None:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ForesightError(f"{label} must be a timezone-aware datetime")


def _texts(values: object, label: str) -> None:
    if not isinstance(values, tuple) or any(not isinstance(item, str) or not item for item in values):
        raise ForesightError(f"{label} must be a tuple of non-empty strings")
    if len(set(values)) != len(values):
        raise ForesightError(f"{label} repeats an item")


@dataclass(frozen=True)
class Claim:
    """判断者的一条可核对的话：这个候选会在今天 ``window`` 这段槽里开始。

    ``slot`` 是说这话时的槽，``slot_minutes`` 是说话那一代树的槽宽——**槽号只有配上槽宽才有意义**，
    换了参数重建的新一代不能拿来核对旧承诺。``situations`` 是引用的卡所属的情形（去重），账按
    (行为, 情形) 分家就靠它；卡没有关联记录时为空，记在 (行为, "") 那本账上。

    ``conditions`` / ``condition_keys`` / ``facts_version`` 是说这话那一刻的外部条件：答了什么、问了哪些键、
    谁按什么口径答的。三样要一起存——只看答案分不清"这个源那天离线"与"从来没有这个键"，也分不清"日型"
    是名义日历答的还是真日历答的；这些事后都补不回来。现在还没有任何真实的条件源，所以三样分别是
    ``()``、``()``、``"none"``（见 ``TODO(FORESIGHT-LOSS-001)``）。
    """

    claim_id: str
    kind_token: str
    day: date
    slot: int
    slot_minutes: int
    window: tuple[int, int]
    judged_at: datetime
    generation: str
    judge_version: str
    basis: tuple[str, ...]
    situations: tuple[str, ...]
    numbers: CandidateNumbers
    conditions: Conditions = ()
    condition_keys: tuple[str, ...] = ()
    facts_version: str = "none"

    def __post_init__(self) -> None:
        for label in ("claim_id", "kind_token", "generation", "judge_version"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise ForesightError(f"claim {label} must be non-empty text")
        if isinstance(self.day, datetime) or not isinstance(self.day, date):
            raise ForesightError("claim day must be a date")
        if isinstance(self.slot_minutes, bool) or not isinstance(self.slot_minutes, int) or not 0 < self.slot_minutes <= 1440:
            raise ForesightError("claim slot_minutes must be a positive integer of at most 1440")
        slots = 1440 // self.slot_minutes
        start, end = self.window
        for label, value in (("slot", self.slot), ("window start", start), ("window end", end)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < slots:
                raise ForesightError(f"claim {label} must be a slot number on its own clock face")
        if start > end:
            raise ForesightError("claim window must not end before it starts")
        _aware(self.judged_at, "claim judged_at")
        _texts(self.basis, "claim basis")
        if not self.basis:
            raise ForesightError("a claim must cite at least one card")
        _texts(self.situations, "claim situations")
        if not isinstance(self.numbers, CandidateNumbers):
            raise ForesightError("claim numbers must be CandidateNumbers")
        if not isinstance(self.conditions, tuple) or any(
            not isinstance(pair, tuple) or len(pair) != 2 or not all(isinstance(x, str) and x for x in pair)
            for pair in self.conditions
        ):
            raise ForesightError("claim conditions must be (key, value) text pairs")
        keys = [key for key, _value in self.conditions]
        if keys != sorted(set(keys)):
            # 第三刀按键取份额，重键或乱序会让"这个键的取值"取到哪一个都对不上。
            raise ForesightError("claim conditions must be sorted by key without repeats")
        _texts(self.condition_keys, "claim condition_keys")
        if not set(keys) <= set(self.condition_keys):
            raise ForesightError("a claim cannot answer a condition key it did not ask for")
        if not isinstance(self.facts_version, str) or not self.facts_version:
            raise ForesightError("claim facts_version must be non-empty text")

    def overlaps(self, window: tuple[int, int]) -> bool:
        """两段时窗说的是不是同一次即将发生：有交集就算。判断者每槽重判，同一件事不能记成十条。"""

        return self.window[0] <= window[1] and window[0] <= self.window[1]


@dataclass(frozen=True)
class Settlement:
    """一条承诺对上事实之后的记录。

    ``slot_offset`` 只在「偏离」时有值：实际开始槽减去最近的窗边（晚是正、早是负）。**账本不判"偏离多少
    还算数"**——那是读时的事，用户还没定（见 ``TODO(FORESIGHT-LOSS-001)``）。
    """

    claim_id: str
    kind_token: str
    day: date
    situations: tuple[str, ...]
    outcome: str
    occurrence_uri: str | None
    slot_offset: int | None
    settled_at: datetime
    reminded: bool = False
    response: str | None = None

    def __post_init__(self) -> None:
        for label in ("claim_id", "kind_token"):
            if not isinstance(getattr(self, label), str) or not getattr(self, label):
                raise ForesightError(f"settlement {label} must be non-empty text")
        if self.outcome not in OUTCOMES:
            raise ForesightError(f"unknown settlement outcome: {self.outcome!r}")
        if isinstance(self.day, datetime) or not isinstance(self.day, date):
            raise ForesightError("settlement day must be a date")
        _texts(self.situations, "settlement situations")
        if (self.outcome == "落空") != (self.occurrence_uri is None):
            raise ForesightError("a settlement names an occurrence exactly when the behaviour happened")
        if (self.outcome == "偏离") != (self.slot_offset is not None):
            raise ForesightError("slot_offset is set exactly for a deviating settlement")
        if self.slot_offset is not None and (isinstance(self.slot_offset, bool) or self.slot_offset == 0):
            raise ForesightError("slot_offset must be a non-zero integer")
        _aware(self.settled_at, "settlement settled_at")
        if not isinstance(self.reminded, bool):
            raise ForesightError("settlement reminded must be a boolean")
        if self.response is not None and self.response not in RESPONSES:
            raise ForesightError(f"unknown settlement response: {self.response!r}")

    @property
    def verified(self) -> bool:
        return self.outcome == "验证"


__all__ = ["OUTCOMES", "RESPONSES", "Claim", "Settlement"]
