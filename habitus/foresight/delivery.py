"""递送原因（方案第二节、第四节后半；裁定 48–51、54、58、64、67、68）：一条到了时窗的预期这一拍记哪一种原因。纯函数、对存储零知识。

组合根（``runtime/foresight_delivery.py``）分拨（时窗开 / 时窗末 / 等宿主）、取材料、问模型、落账；记哪一种原因、先后怎么排都在这里。
最后三种（影子、名额已满、待取 / 说了）由 ``ladder.decide`` 定，这里在它之前把别的原因按顺序过一遍：

- **时窗开**：已发生 → 影子（档 0）→ 久别回来 → 过了时限 → 等时窗末（提醒档一律等到时窗末还没做再说，裁定 58 第 1 条）；
- **时窗末**：已发生 → 说不准（窗里有「没读懂」，裁定 67 第 14 条）→ 影子（窗里被压回影子或关了账）→ 久别回来 → 前因不成立 →
  模型判不像 → 同类已说 → 不可达（没有宿主活着，方案 32 第 3.3 条）→ 过了时限 → ``decide``；
- **过了时限**（用户 11-10）：提醒档轮到决定时此刻已过最晚时限——不说（不记「待取」、不占名额、出口不收），终局，结算照常结它；
- **等宿主**（最近一条是「待取」）：做了追加已发生；过了最晚时限还没人问出口追加宿主没取（终局，结算按没说过处理）。
- **依据没说成**（``unsaid``，裁定 92 第 2 条②）：依据记了「不可达」「宿主没取」「过了时限」，或停在「等时窗末」「待取」过了最晚时限，
  或依据本来就不能开口（候选关系、巧合，裁定 56 第 12 条；这回已经做了的除外）——话没对人说出口，被它「单元已选」吸收的
  成员时限还在的照常走时窗末（组合根接手）。依据能开口而自己决定不说的（已发生、单元读数定的影子、同类已说、名额已满、
  模型判不像……）不算：那是单元替这一刻做的决定。

材料由组合根准备成 ``Circumstances``：每一项问到才算——走到那一步才读账、读树，前面的原因成立了后面的就不读。
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from habitus.foresight.ladder import Standing, decide
from habitus.foresight.ledger.model import (
    AWAITING_HOST,
    BASIS_WITHDRAWN,
    HOST_NEVER_CLAIMED,
    PAST_DEADLINE,
    RETURNED,
    UNOBSERVED,
    Delivery,
    Expectation,
)
from habitus.foresight.ledger.settle import counts_as_done
from habitus.foresight.model import Seen
from habitus.foresight.units import UNIT

#: 不是终局的递送原因：之后还会再决定一次。
PENDING = "等时窗末"
#: 时窗末要说、这一拍没有宿主活着（方案 32 第 3.3 条）。
UNREACHABLE = "不可达"
#: 到了时窗时（或时窗末）这件事已经做了。
HAPPENED = "已发生"
#: 优先于久别的设置：人说了"以后都提醒我"，就不因为久别回来而先看一次。
_ALWAYS = "以后都提醒我"


def _no() -> bool:
    return False


def _yes() -> bool:
    return True


@dataclass(frozen=True)
class Circumstances:
    """一条预期这一拍的材料（组合根准备，问到才算）：这件事这回是不是已经做了；依据那本账是不是久别回来（不看设置，设置在
    ``quiet_after_absence`` 里优先）；说中的范围里（起算 → 此刻）有没有「没读懂」；事件触发的前因此刻还成不成立。"""

    happened: Callable[[], bool]
    absent: Callable[[], bool]
    unobserved: Callable[[], bool] = _no
    antecedent_holds: Callable[[], bool] = _yes


def quiet_after_absence(standing: Standing, absent: Callable[[], bool]) -> bool:
    """依据那本账久别回来：这一次只记影子，不改账的级、不写单元的档（26 号清单 3.3）；「以后都提醒我」优先。"""

    if standing.setting == _ALWAYS:
        return False
    return absent()


def on_opening(item: Expectation, standing: Standing, *, now: datetime, facts: Circumstances) -> Delivery:
    """时窗开了：按档决定；依据那本账久别回来的记「久别回来」；提醒档一律记「等时窗末」——到时窗末还没做再问模型、再写话（这一版
    不分急缓，裁定 58 第 1 条）。有没有宿主活着到时窗末才看：时窗开时没人、末了打开了宿主的那条照样要问。"""

    level = standing.level
    if facts.happened():
        return Delivery(item.expectation_id, now, level, False, HAPPENED, 0.0)
    if level == 0:
        return Delivery(item.expectation_id, now, 0, False, "影子", 0.0)
    if quiet_after_absence(standing, facts.absent):
        return Delivery(item.expectation_id, now, level, False, RETURNED, 0.0)
    if now > item.deadline:  # 补看到的拍（被挤掉的那一拍）开时窗时就已过了时限：不说
        return Delivery(item.expectation_id, now, level, False, PAST_DEADLINE, 0.0)
    return Delivery(item.expectation_id, now, level, False, PENDING, 0.0)


def on_closing(
    item: Expectation,
    pending: Delivery,
    standing: Standing,
    *,
    now: datetime,
    facts: Circumstances,
    kept: bool | None,
    evidence: Sequence[str],
    spoken_kinds: Collection[str],
    alive: bool,
    spoken_today: int,
    daily_cap: int,
    text: Callable[[], str],
) -> Delivery:
    """时窗末到了：还没做、此刻还在提醒档就说；做了记「已发生」；窗里有「没读懂」的记「说不准」（他可能已经做了、我们没看到，
    不把"看了没读懂"读成"没做"，裁定 67 第 14 条）；窗里被压回影子或关了账的记「影子」；久别回来的记「久别回来」（时窗开时判过
    一次；跨夜的时窗换了机会表，这里再判一次）。

    ``pending`` 是时窗开时记的那条「等时窗末」；``kept`` 是模型判像不像（没问是 None），``evidence`` 是模型指的那几条历史记录；
    ``spoken_kinds`` 是今天已经说过的类（「待取」也算）；``alive`` 是这一拍有没有宿主活着；``text`` 问到才写那句话。"""

    if facts.happened():
        return Delivery(item.expectation_id, now, pending.level, False, HAPPENED, 0.0)
    if facts.unobserved():
        return Delivery(item.expectation_id, now, pending.level, False, UNOBSERVED, 0.0)
    if standing.level == 0:
        return Delivery(item.expectation_id, now, 0, False, "影子", 0.0)
    if quiet_after_absence(standing, facts.absent):
        return Delivery(item.expectation_id, now, standing.level, False, RETURNED, 0.0)
    level = standing.level
    cited = tuple(evidence)
    if not facts.antecedent_holds():
        return Delivery(item.expectation_id, now, level, False, "前因不成立", 0.0, evidence=cited)
    if kept is False:
        return Delivery(item.expectation_id, now, level, False, "模型判不像", 0.0, evidence=cited)
    if item.kind_token in spoken_kinds:
        return Delivery(item.expectation_id, now, level, False, "同类已说", 0.0, evidence=cited)
    if not alive:
        return Delivery(item.expectation_id, now, level, False, UNREACHABLE, 0.0, evidence=cited)
    if now > item.deadline:  # 时窗末那一拍才轮到、已过了时限：不说，不记「待取」占名额
        return Delivery(item.expectation_id, now, level, False, PAST_DEADLINE, 0.0, evidence=cited)
    return decide(
        item,
        level,
        now=now,
        spoken_today=spoken_today,
        daily_cap=daily_cap,
        text=text(),
        evidence=cited,
        awaiting_host=True,
    )


def on_awaiting(item: Expectation, pending: Delivery, *, now: datetime, happened: bool) -> Delivery | None:
    """「待取」之后每拍只做两件事：做了追加「已发生」；过了最晚时限还没人问出口追加「宿主没取」（终局，结算按没说过处理）。
    都不是就不记（None）。``happened`` 与结算同一个口径（``done_since``）：最晚时限之后才做的不算这一回做了——结算把它结成
    过期待决，这里记「宿主没取」，两边说的是同一件事。"""

    if happened:
        return Delivery(item.expectation_id, now, pending.level, False, HAPPENED, 0.0)
    if now > item.deadline:
        return Delivery(item.expectation_id, now, pending.level, False, HOST_NEVER_CLAIMED, 0.0)
    return None


def on_withdrawn(item: Expectation, pending: Delivery, *, now: datetime, voiced: bool, taken: bool) -> Delivery | None:
    """「待取」之后依据失效（实时化方案第 7 节）：话已写好、宿主还没取走（``taken`` 为假），而它此刻已不能当开口的依据
    （``voiced`` 为假：换代之后关系变回候选、从机会表上消失）——撤下请求，记「依据失效」（终局，结算按没说过处理）；被它吸收的
    成员照现有规则接手（``unsaid`` 认它没说成）。已被取走的不撤（宿主可能已在问），依据还能开口的不动：都是 None。"""

    if voiced or taken:
        return None
    return Delivery(item.expectation_id, now, pending.level, False, BASIS_WITHDRAWN, 0.0)


def unsaid(
    item: Expectation, made: Sequence[Delivery], *, now: datetime, voiceless: bool = False, taken: bool = False
) -> bool:
    """依据这条预期的话没对人说成、也说不成了（``made`` 是它的全部递送）：没有「说了」，而且最近一条是「不可达」「宿主没取」「过了时限」，
    或还停在「等时窗末」「待取」却已过了最晚时限（中间停过机、时窗末那一拍没跑到）。``voiceless`` 是它本来就不能当开口的依据
    （候选关系、巧合，``units.voiced``）：单元只能从它里面取、记了影子——也是没说成（裁定 56 第 12 条、裁定 92 第 2 条②），
    这回已经做了（「已发生」）的除外：那是单元替这一刻做的决定。``taken`` 是它的请求已被宿主取走（还没说出口）：取走等同说出口
    （用户 11-10），成员不接手，同一刻最多一条请求。"""

    if not made or taken or any(item_.spoken for item_ in made):
        return False
    if voiceless and all(item_.reason != HAPPENED for item_ in made):
        return True
    latest = max(made, key=lambda delivery: delivery.decided_at)
    if latest.reason in (UNREACHABLE, HOST_NEVER_CLAIMED, PAST_DEADLINE):
        return True
    return latest.reason in (PENDING, AWAITING_HOST) and item.deadline.astimezone(UTC) < now.astimezone(UTC)


def hand_over(
    known: Mapping[str, Expectation],
    decided: Mapping[str, Sequence[Delivery]],
    members: Mapping[str, Sequence[Expectation]],
    instant: datetime,
    *,
    voiced: Callable[[Expectation], bool] = lambda item: True,
    taken: Callable[[str], bool] = lambda request_id: False,
) -> tuple[tuple[tuple[tuple[Expectation, ...], tuple[str, ...]], ...], dict[str, Delivery]]:
    """依据没说成（``delivery.unsaid``）的那几刻里，能接手的成员：只决定过「单元已选」、自己最晚时限还没到的。已经有成员接手过、
    它还没说成或自己决定了不说（不是 ``unsaid``）的，这一刻不再交给别的成员——同一刻只说一次。依据或成员已经结掉（不在账上的
    还没结那几天里）的不动。

    依据本来就不能开口（``voiced`` 为假：候选关系、巧合——只进读数、不当依据，裁定 56 第 12 条）也算没说成（裁定 92 第 2 条②）：
    同一刻只有它先到了时窗，单元只能从它里面取、记了影子，被吸收的成员（还没到时窗时拉进来的、后来并进来的）到了自己的时窗末
    按同一组输入重算单元、依据换成能开口的那条。依据的请求已被宿主取走（``taken``）等同说出口：成员不接手。

    （原在 ``runtime.foresight_delivery._handovers``，纯规则，挪到这里与 ``unsaid`` 放在一起，判法不变。）"""

    found: list[tuple[tuple[Expectation, ...], tuple[str, ...]]] = []
    subsumed: dict[str, Delivery] = {}
    for basis_id in sorted(members):
        basis = known.get(basis_id)
        if basis is None:
            continue
        # 依据的请求已被宿主取走（还没说出口）：取走等同说出口，成员不接手（用户 11-10，F8）
        spoken = decided.get(basis_id, ())
        if not unsaid(basis, spoken, now=instant, voiceless=not voiced(basis), taken=taken(basis_id)):
            continue
        group = [item for item in members[basis_id] if item.expectation_id in known]
        took = [item for item in group if any(made.reason != UNIT for made in decided[item.expectation_id])]
        if any(not unsaid(item, decided[item.expectation_id], now=instant) for item in took):
            continue
        free = [
            item
            for item in group
            if item not in took and item.deadline.astimezone(UTC) > instant and decided[item.expectation_id]
        ]
        if not free:
            continue
        for item in free:
            subsumed[item.expectation_id] = max(decided[item.expectation_id], key=lambda made: made.decided_at)
        found.append(((basis, *group), tuple(item.expectation_id for item in free)))
    return tuple(found), subsumed


def done_since(
    item: Expectation, seen: Iterable[Seen], *, now: datetime, current: Callable[[str], Sequence[str]]
) -> bool:
    """递送判"这件事这回已经做了"（做了就不递：那句话说的事他刚做完）：这一拍读到的行（已封口的与还没封口、现场归上类的）里，
    有到此刻为止开始、按结算同一个口径算做了的（``ledger.settle.counts_as_done``：说中的范围里——起算之后、最晚时限之前——
    这一类按 ``current`` 认、没放弃）。过了最晚时限才做的不算这一回做了：结算也不认它（结成过期待决）。"""

    return any(
        row.started_at <= now
        and counts_as_done(item, row.kind_token, row.started_at, abandoned=row.abandoned, current=current)
        for row in seen
    )


def reminded_before(
    item: Expectation,
    decided_at: datetime,
    *,
    kind_token: str,
    started_at: datetime,
    current: Callable[[str], Sequence[str]],
) -> bool:
    """一条 ``started_at`` 开始、类为 ``kind_token`` 的记录，是不是被这条说出口的预期（``decided_at`` 决定说的）提醒过：决定在它
    开始之前、它落在那条预期说中的范围里（起算之后、最晚时限之前），而且那条预期认的类现在对应着它的类（``current`` 与结算同一个
    映射：词表拆改之后两边认的是同一件事）。"""

    return decided_at < started_at and item.covers(started_at) and kind_token in current(item.kind_token)


__all__ = [
    "PENDING",
    "UNREACHABLE",
    "Circumstances",
    "done_since",
    "hand_over",
    "on_awaiting",
    "on_closing",
    "on_opening",
    "on_withdrawn",
    "quiet_after_absence",
    "reminded_before",
    "unsaid",
]
