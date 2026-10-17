"""账本的形状（``19-预测层方案.md`` 第二、五节；裁定 48–51）：按预期的六种记录（预期、递送、回话、结算、追加裁决、
漏报）与按账的四种记录（事实、重轻、设置、级的变化）。只增不改，每条一份记录。

- **预期**是"此刻对得上历史"时建的一条：什么事、预计什么时候、依据是行为树上哪几次、此刻它的前因在不在场。
  说没说都记（影子账是唯一没有选择偏差的准确率来源）。身份不含槽号与时刻：同一个触发实例只有一条。
  **每个机会都建**（裁定 49）：形成时这件事已经做了的也建，``observed_from`` 是从什么时候起的发生算这条预期说中的
  （钟面 / 日子：当天零点；事件：前因做完那一刻），结算照常对答案——不然账上只剩没做的那几次。
- **递送**是对一条预期"说不说、按哪一级说"的决定，连同为什么没说、被说出的概率（随机扣发时不是 0 或 1）。
- **回话**是主体对一次递送的回应；"拒绝"要拆开：已做过是观测缺口，这次不用是预测错了，稍后是时窗错了。
- **结算**对着行为树给预期一个结果；到最晚时限没发生只是「过期待决」，之后由**追加裁决**补一条（后补 / 回话 / 未判），
  旧记录不改。
- **漏报**不挂在预期上：发生了、而发生前没有任何一条在时的预期。
- **事实**是一本账上的一笔：哪条轴（预测 / 反应）、好坏、强度、因为什么。递法只读事实；同一条预期同一条轴后写的覆盖
  先写的（后补、回话改判一条过期待决时不改旧记录，再写一条）。
- **重轻**是这本账的属性：模型判一次、落账、到期重判；人可以改（裁定 48）。**设置**是人对账的两个动作
  （「以后都提醒我」「以后别提了」，「照常」撤销）。**级的变化**记账从影子到提醒、从提醒回影子的那一刻（滞回要知道上一次在哪一级）。

本模块对 scene、存储零知识；时刻一律带时区。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256

from habitus.foresight.errors import ForesightError

#: 预期从哪种触发来（``19-预测层方案.md`` 第三节）。
SOURCES = ("事件", "钟面", "日子", "没做完", "约定", "情境")
#: 递法两档（裁定 48）：影子只记账；提醒在该来了还没来时主动说。
LEVELS = (0, 1)
#: 一次递送为什么说了 / 没说。
#: 「不可达」= 没有送达渠道（裁定 47：接上渠道之前提醒档的决定都是它，结算按没说过处理）；「已发生」= 到了时窗时这件事
#: 已经做了，不递；「同类已说」= 同一类今天已经说过一句（几本账同一件事只说一次）；「等时窗末」= 轻的事话准备好了，
#: 到时窗末还没做再说（「该来了还没来」是提醒的说话时机，「重」的时窗一开就说）。
DELIVERY_REASONS = (
    "说了",
    "影子",
    "名额已满",
    "随机扣发",
    "不可达",
    "模型判不像",
    "已发生",
    "同类已说",
    "等时窗末",
    "前因不成立",
    "上次的原因今天又在",
    "单元已选",
)
#: 主体的回话：人能说的只剩这三句（裁定 50）；「稍后」不进账。
RESPONSES = ("这次不用", "稍后", "已做过")
#: 结算的结果。「过期待决」不是终态。每个机会都建，形成时已做的照常结准时 / 早到（裁定 49）。
OUTCOMES = ("准时", "早到", "晚了但做了", "提醒后发生", "过期待决")
#: 过期待决之后的追加裁决。「后补」是唯一能从行为记录自动得出的"抓到缺失"。
RULINGS = ("后补", "用户说忘了", "用户说这次不用", "用户说早做过了", "未判")
#: 不说的后果重不重：这本账的属性，模型判一次、到期重判，人可改（裁定 48）。没判过按轻。
#: 这件事本身的代价比 c 能取的范围（裁定 58 第 1 条）：漏一次相当于错提醒几次的代价。范围是保护闸（挡住模型答出的离谱数），
#: 不是分档；线由 c 与提醒的效果连续地算（``ladder.Tolerance``）。
COST_RATIO_RANGE = (0.1, 10.0)
SEVERITY_ORIGINS = ("模型", "人")
#: 事实的三条轴（裁定 50，裁定 58 第 5–7 条改）：**预测**（他自己做了没、离窗多近）管升；**提醒组**（说了之后做了没、有没有
#: 明确拒绝；单位权重）管降——它就是"说了而他没做"的那个误提醒率；**扣发组**（该说而随机扣下没说之后做了没；单位权重）是
#: 提醒组的对照，提醒的效果 = 两者完成率之差。原来的"反应轴"把拒绝（强度 1）、落空（强度 p）、提醒后做了（强度 = 快慢分位）
#: 加进同一个 Beta，量纲不同、区间宽度失真，删掉。
FACT_AXES = ("预测", "提醒组", "扣发组")
#: 事实因为什么（给人看，也给强度找同类历史）。
#: 顺序进事实文件的身份（种类号），只许往后加。「作废」：前因不成立，这次机会不存在（权重 0，裁定 54）。
FACT_KINDS = (
    "准时",
    "早到",
    "晚了但做了",
    "提醒后发生",
    "过期待决",
    "后补",
    "落空",
    "这次不用",
    "已做过",
    "忘了",
    "作废",
    "扣发后发生",
    "扣发后落空",
)
FACT_ORIGINS = ("观测", "人说")
#: 一次落空找到的原因（裁定 55，方案第五节「纠错」）。「推迟」不另记：落空后补做了，那条事实就是「后补」。
#: 「还没进关联」= 下一步 X 不在任何语义关联里，记下来攒次数（不是终点）；「没看到下一步」= 到最晚时限没有任何动作（观测的边界）。
#: 单次落空不说"习惯在变"：那是连着落空、账判出"规律断了"的事（裁定 54）。
CAUSE_KINDS = ("同一件事", "线索没了", "别的起因", "分支有条件", "分支条件不够", "还没进关联", "没看到下一步")
#: 人对账的设置：「照常」撤销前两者。
SETTINGS = ("以后都提醒我", "以后别提了", "照常")


def _aware(value: object, label: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ForesightError(f"{label} must be a timezone-aware datetime")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ForesightError(f"{label} must be non-empty text")
    return value


def _texts(values: object, label: str) -> None:
    if not isinstance(values, tuple) or any(not isinstance(item, str) or not item for item in values):
        raise ForesightError(f"{label} must be a tuple of non-empty text")


def _day(value: object, label: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ForesightError(f"{label} must be a date")
    return value


def _count(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ForesightError(f"{label} must be an integer of at least {minimum}")
    return value


def _one_of(value: object, choices: tuple[str, ...], label: str) -> str:
    if value not in choices:
        raise ForesightError(f"unknown {label}: {value!r}")
    assert isinstance(value, str)
    return value


def expectation_identity(kind_token: str, source: str, trigger: str, trigger_instance: str) -> str:
    """预期的身份：类、触发种类、触发身份、这一次的触发实例。**不含槽号、时刻与日子**——同一个触发实例每拍、每天重算
    得到同一个身份，于是同一条预期全账本只写一次（账本按身份索引；旧账本的身份含槽与说话时刻，是每拍重问的根源）。"""

    parts = (kind_token, source, trigger, trigger_instance, "")
    return sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]


def anchor_uri(trigger_instance: str) -> str:
    """触发实例里那条记录的出处。「离上次够久」的实例是 ``<上次的 URI>@<这一回合的第一天>``（同一个上次能分几回合到期），
    其余触发的实例原样返回（事件：前因的开始时刻，按时刻到最近几天的行里找；钟面、几号：今天的日期）。"""

    uri, separator, _tail = trigger_instance.rpartition("@")
    return uri if separator else trigger_instance


@dataclass(frozen=True)
class Antecedent:
    """反向看到的一条前因：语义树里以这个候选为后果的成立关系，此刻它在不在场、是抬高还是压低。"""

    relation: str
    present: bool
    upward: bool
    #: 在场时前因这一次的出处（已封口记录的 URI，或 ``unsealed:<开始时刻>``）；不在场为空。
    source: str = ""

    def __post_init__(self) -> None:
        _text(self.relation, "antecedent relation")
        if not isinstance(self.present, bool) or not isinstance(self.upward, bool):
            raise ForesightError("antecedent present and upward must be booleans")
        if not isinstance(self.source, str):
            raise ForesightError("antecedent source must be text")
        if self.source and not self.present:
            raise ForesightError("an absent antecedent has no source")


@dataclass(frozen=True)
class Expectation:
    """一条预期。``window`` 来自历史那几次的实际时刻；``deadline`` 是最晚时限（这条触发自己那一段的边界）；``basis`` 是命中的那几次。

    ``opportunities`` / ``hits`` 是依据的历史机会数与命中数，摆出来让人掂量，不是门。``level`` 是形成时这本账所在的级。
    ``observed_from`` 是从什么时候起的发生算这条预期说中的（不晚于形成时刻）：时窗是历史那几次的实际时刻，可以早于形成时刻
    （钟面到了钟点才建、形成时他可能已经做了），已做的照常结算（裁定 49）。
    """

    expectation_id: str
    kind_token: str
    day: date
    formed_at: datetime
    source: str
    trigger: str
    trigger_instance: str
    window: tuple[datetime, datetime]
    deadline: datetime
    basis: tuple[str, ...]
    opportunities: int
    hits: int
    observed_from: datetime
    antecedents: tuple[Antecedent, ...] = ()
    level: int = 0
    generation: str = ""
    night: date | None = None
    version: str = ""
    #: 预测层实现的指纹（形成这条预期的那份代码；修了 bug 之后按它把预期分成前后两批对比，裁定 54）。
    implementation: str = ""

    def __post_init__(self) -> None:
        for label in ("expectation_id", "kind_token", "trigger", "trigger_instance"):
            _text(getattr(self, label), f"expectation {label}")
        _day(self.day, "expectation day")
        _aware(self.formed_at, "expectation formed_at")
        _one_of(self.source, SOURCES, "expectation source")
        if not isinstance(self.window, tuple) or len(self.window) != 2:
            raise ForesightError("expectation window must be a (start, end) pair")
        start, end = self.window
        _aware(start, "expectation window start")
        _aware(end, "expectation window end")
        if end < start:
            raise ForesightError("expectation window must not end before it starts")
        _aware(self.deadline, "expectation deadline")
        if self.deadline < end:
            raise ForesightError("expectation deadline must not come before the window ends")
        _aware(self.observed_from, "expectation observed_from")
        if self.observed_from > self.formed_at:
            raise ForesightError("an expectation counts occurrences from no later than when it was formed")
        if self.deadline < self.formed_at:
            raise ForesightError("an expectation cannot be formed after its deadline")
        _texts(self.basis, "expectation basis")
        _count(self.opportunities, "expectation opportunities")
        _count(self.hits, "expectation hits")
        if self.hits > self.opportunities:
            raise ForesightError("expectation hits cannot exceed its opportunities")
        if len(self.basis) > self.hits:
            raise ForesightError("expectation basis lists more records than it counts as hits")
        if not isinstance(self.antecedents, tuple) or any(
            not isinstance(item, Antecedent) for item in self.antecedents
        ):
            raise ForesightError("expectation antecedents must be Antecedent records")
        if self.level not in LEVELS:
            raise ForesightError(f"unknown expectation level: {self.level!r}")
        for label in ("generation", "version", "implementation"):
            if not isinstance(getattr(self, label), str):
                raise ForesightError(f"expectation {label} must be text")
        if self.night is not None:
            _day(self.night, "expectation night")

    @property
    def account(self) -> str:
        """这条预期记在哪本账上：事 + 触发种类 + 触发身份（不含触发实例：每次机会都记在同一本上）。"""

        return account_key(self.kind_token, self.source, self.trigger)

    def covers(self, at: datetime) -> bool:
        """这一刻发生的事算不算这条预期说中的：``observed_from`` 之后、最晚时限之前。"""

        return self.observed_from < at <= self.deadline


@dataclass(frozen=True)
class Delivery:
    """对一条预期的递送决定。``p_speak`` 是这条被说出的概率：确定性规则下是 0 或 1，随机扣发时是扣发前的概率。
    ``evidence`` 是模型指的那几条历史记录。重轻不在这里：它是账的属性（``AccountSeverity``）。"""

    expectation_id: str
    decided_at: datetime
    level: int
    spoken: bool
    reason: str
    p_speak: float
    text: str | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.expectation_id, "delivery expectation_id")
        _aware(self.decided_at, "delivery decided_at")
        if self.level not in LEVELS:
            raise ForesightError(f"unknown delivery level: {self.level!r}")
        if not isinstance(self.spoken, bool):
            raise ForesightError("delivery spoken must be a boolean")
        _one_of(self.reason, DELIVERY_REASONS, "delivery reason")
        if self.spoken != (self.reason == "说了"):
            raise ForesightError("a spoken delivery has reason 说了 and an unspoken one does not")
        if (
            isinstance(self.p_speak, bool)
            or not isinstance(self.p_speak, int | float)
            or not 0.0 <= self.p_speak <= 1.0
        ):
            raise ForesightError("delivery p_speak must be a number between 0 and 1")
        if self.text is not None and not self.spoken and self.reason != "等时窗末":
            raise ForesightError("an unspoken delivery carries no text unless it waits for the window to end")
        if self.text is not None:
            _text(self.text, "delivery text")
        _texts(self.evidence, "delivery evidence")


@dataclass(frozen=True)
class Response:
    """主体对一次递送的回话。"""

    expectation_id: str
    responded_at: datetime
    response: str
    note: str | None = None

    def __post_init__(self) -> None:
        _text(self.expectation_id, "response expectation_id")
        _aware(self.responded_at, "response responded_at")
        _one_of(self.response, RESPONSES, "response")
        if self.note is not None:
            _text(self.note, "response note")


@dataclass(frozen=True)
class Settlement:
    """对着行为树给一条预期的结果。``offset_minutes``：早到为负（早了几分钟）、晚了但做了为正、提醒后发生是提醒到发生的延迟。"""

    expectation_id: str
    kind_token: str
    day: date
    settled_at: datetime
    outcome: str
    occurrence_uri: str | None = None
    offset_minutes: int | None = None

    def __post_init__(self) -> None:
        _text(self.expectation_id, "settlement expectation_id")
        _text(self.kind_token, "settlement kind_token")
        _day(self.day, "settlement day")
        _aware(self.settled_at, "settlement settled_at")
        _one_of(self.outcome, OUTCOMES, "settlement outcome")
        found = self.outcome in ("准时", "早到", "晚了但做了", "提醒后发生")
        if found != (self.occurrence_uri is not None):
            raise ForesightError("a settlement names the occurrence exactly when it found one")
        if self.occurrence_uri is not None:
            _text(self.occurrence_uri, "settlement occurrence_uri")
        offset = self.outcome in ("早到", "晚了但做了", "提醒后发生")
        if offset != (self.offset_minutes is not None):
            raise ForesightError("a settlement carries an offset exactly for 早到 / 晚了但做了 / 提醒后发生")
        if self.offset_minutes is not None and (
            isinstance(self.offset_minutes, bool) or not isinstance(self.offset_minutes, int)
        ):
            raise ForesightError("settlement offset_minutes must be an integer")

    @property
    def hit(self) -> bool:
        """算不算说准了（自然准确率只用没被提醒的：准时、早到、晚了但做了）。"""

        return self.outcome in ("准时", "早到", "晚了但做了")


@dataclass(frozen=True)
class Ruling:
    """过期待决之后的追加裁决。「后补」带着补做的那条与晚了多少分钟。"""

    expectation_id: str
    ruled_at: datetime
    ruling: str
    occurrence_uri: str | None = None
    delay_minutes: int | None = None

    def __post_init__(self) -> None:
        _text(self.expectation_id, "ruling expectation_id")
        _aware(self.ruled_at, "ruling ruled_at")
        _one_of(self.ruling, RULINGS, "ruling")
        if (self.ruling == "后补") != (self.occurrence_uri is not None and self.delay_minutes is not None):
            raise ForesightError("a 后补 ruling names the occurrence and its delay; other rulings carry neither")
        if self.occurrence_uri is not None:
            _text(self.occurrence_uri, "ruling occurrence_uri")
        if self.delay_minutes is not None:
            _count(self.delay_minutes, "ruling delay_minutes", minimum=1)


@dataclass(frozen=True)
class Omission:
    """漏报：这条记录发生时没有任何一条在时的预期。``had_expectation`` 是那天这一类有没有过别的预期（排序问题还是机会问题）。"""

    occurrence_uri: str
    kind_token: str
    day: date
    noticed_at: datetime
    had_expectation: bool

    def __post_init__(self) -> None:
        _text(self.occurrence_uri, "omission occurrence_uri")
        _text(self.kind_token, "omission kind_token")
        _day(self.day, "omission day")
        _aware(self.noticed_at, "omission noticed_at")
        if not isinstance(self.had_expectation, bool):
            raise ForesightError("omission had_expectation must be a boolean")


def account_key(kind_token: str, source: str, trigger: str) -> str:
    """一本账的身份：事 + 触发种类 + 触发身份（不含触发实例：每次机会都记在同一本上）。"""

    return "\n".join((kind_token, source, trigger))


def parse_account(account: str) -> tuple[str, str, str]:
    """账身份 → (类, 触发种类, 触发身份)。"""

    parts = account.split("\n")
    if len(parts) != 3 or not all(parts):
        raise ForesightError(f"malformed account key: {account!r}")
    return parts[0], parts[1], parts[2]


def _account(value: object) -> str:
    text = _text(value, "account")
    parse_account(text)
    return text


@dataclass(frozen=True)
class Cause:
    """一次落空对照答案找到的原因（裁定 55）。里面的东西一律落成类编号与行为树编号：白天那一拍不读语义树也核得了
    "上次的原因今天还在不在"。

    - 同一件事：``classes`` = 他这次做的那一类（与该做的同在一个汇总概念下），``relation`` = 那个汇总概念。
    - 线索没了：``relation`` = 该做的那件事惯常的前因关系（键文本），这次它的前因没来；或 ``condition`` = ``日型=<备注>``（这天的日型
      与依据那几次都不同）。
    - 别的起因：``classes`` = 起因那一类，``relation`` / ``segment`` = 起因 → 他这次做的那件事的关系与跨度段。
    - 分支有条件：``condition`` = 条件文字，``holds_when`` = 条件取这个值时该做的那件事不来。
    - 分支条件不够 / 还没进关联：``classes`` = 他接下来做的那一类；没看到下一步：什么都不带。
    ``uris`` 是依据的行为树编号（他这次做的、起因那一次）。"""

    kind: str
    classes: tuple[str, ...] = ()
    uris: tuple[str, ...] = ()
    relation: str = ""
    segment: str = ""
    condition: str = ""
    holds_when: bool | None = None

    def __post_init__(self) -> None:
        _one_of(self.kind, CAUSE_KINDS, "cause kind")
        for label in ("classes", "uris"):
            value = getattr(self, label)
            if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
                raise ForesightError(f"cause {label} must be a tuple of non-empty text")
        for label in ("relation", "segment", "condition"):
            if not isinstance(getattr(self, label), str):
                raise ForesightError(f"cause {label} must be text")
        if self.holds_when is not None and not isinstance(self.holds_when, bool):
            raise ForesightError("cause holds_when must be a boolean or None")
        if self.kind == "别的起因" and not (self.classes and self.segment):
            raise ForesightError("another cause names its class and the span it acts over")
        if self.kind == "分支有条件" and (not self.condition or self.holds_when is None):
            raise ForesightError("a conditioned branch names its condition and the value that keeps the matter away")


@dataclass(frozen=True)
class AccountFact:
    """一本账上的一笔：哪条轴、好坏、强度、因为什么、来自哪条预期。``offset_minutes``：预测轴是偏差（窗内 0、早到为负、
    晚到为正），提醒组是说完到发生隔了几分钟，补做是最晚时限之后几分钟；别的没有。

    两个时刻各管各的：``at`` 是这次机会的时刻（这条预期形成的时刻）——遗忘、"规律断了"的先后都按它；``written_at`` 是改判
    写下的时刻（后补、回话、重结），同一条预期同一条轴谁覆盖谁按它（None = 结算时写的原判，按 ``at``）。改判若取写下的
    时刻当 ``at``，几天前那次机会就会被当成最新的证据：遗忘被重置，连错的先后也排错。"""

    account: str
    at: datetime
    axis: str
    good: bool
    strength: float
    kind: str
    expectation_id: str
    origin: str = "观测"
    offset_minutes: int | None = None
    written_at: datetime | None = None
    #: 这次落空找到的原因（裁定 55）：机会表那一拍追加一条同值的改判带上它，不动读数。
    cause: Cause | None = None

    def __post_init__(self) -> None:
        _account(self.account)
        _aware(self.at, "fact at")
        _one_of(self.axis, FACT_AXES, "fact axis")
        if not isinstance(self.good, bool):
            raise ForesightError("fact good must be a boolean")
        if (
            isinstance(self.strength, bool)
            or not isinstance(self.strength, int | float)
            or not 0.0 <= self.strength <= 1.0
        ):
            raise ForesightError("fact strength must be a number in [0, 1]")
        _one_of(self.kind, FACT_KINDS, "fact kind")
        _text(self.expectation_id, "fact expectation_id")
        _one_of(self.origin, FACT_ORIGINS, "fact origin")
        if self.offset_minutes is not None and (
            isinstance(self.offset_minutes, bool) or not isinstance(self.offset_minutes, int)
        ):
            raise ForesightError("fact offset_minutes must be an integer")
        if self.cause is not None and not isinstance(self.cause, Cause):
            raise ForesightError("fact cause must be a Cause or None")
        if self.written_at is not None:
            _aware(self.written_at, "fact written_at")
            if self.written_at < self.at:
                raise ForesightError("a fact cannot be revised before its opportunity formed")

    @property
    def precedence(self) -> datetime:
        """同一条预期同一条轴谁覆盖谁：后写的覆盖先写的。"""

        return self.at if self.written_at is None else self.written_at


@dataclass(frozen=True)
class AccountSeverity:
    """这本账的后果有多重：代价比 c（漏一次相当于错提醒几次，裁定 58 第 1 条）、谁判的、依据、什么时候判的、到什么时候该重判。"""

    account: str
    judged_at: datetime
    cost_ratio: float
    basis: str
    expires_at: datetime
    origin: str = "模型"

    def __post_init__(self) -> None:
        _account(self.account)
        _aware(self.judged_at, "severity judged_at")
        low, high = COST_RATIO_RANGE
        if (
            isinstance(self.cost_ratio, bool)
            or not isinstance(self.cost_ratio, int | float)
            or not low <= float(self.cost_ratio) <= high
        ):
            raise ForesightError(f"severity cost_ratio must be a number in [{low}, {high}]")
        if not isinstance(self.basis, str):
            raise ForesightError("severity basis must be text")
        _aware(self.expires_at, "severity expires_at")
        if self.expires_at <= self.judged_at:
            raise ForesightError("a severity must expire after it was judged")
        _one_of(self.origin, SEVERITY_ORIGINS, "severity origin")

    def current(self, at: datetime) -> bool:
        return self.judged_at <= at < self.expires_at


@dataclass(frozen=True)
class AccountSetting:
    """人对这本账的设置；最新的一条说了算。"""

    account: str
    at: datetime
    setting: str

    def __post_init__(self) -> None:
        _account(self.account)
        _aware(self.at, "setting at")
        _one_of(self.setting, SETTINGS, "setting")


@dataclass(frozen=True)
class AccountLevel:
    """这本账的级在 ``at`` 变成了 ``level``（只在变的时候记）。"""

    account: str
    at: datetime
    level: int

    def __post_init__(self) -> None:
        _account(self.account)
        _aware(self.at, "level at")
        if self.level not in LEVELS:
            raise ForesightError(f"unknown account level: {self.level!r}")


__all__ = [
    "CAUSE_KINDS",
    "DELIVERY_REASONS",
    "FACT_AXES",
    "FACT_KINDS",
    "FACT_ORIGINS",
    "SETTINGS",
    "SEVERITY_ORIGINS",
    "AccountFact",
    "AccountLevel",
    "AccountSetting",
    "AccountSeverity",
    "account_key",
    "parse_account",
    "LEVELS",
    "OUTCOMES",
    "RESPONSES",
    "RULINGS",
    "COST_RATIO_RANGE",
    "SOURCES",
    "Antecedent",
    "Cause",
    "Delivery",
    "Expectation",
    "Omission",
    "Response",
    "Ruling",
    "Settlement",
    "anchor_uri",
    "expectation_identity",
]
