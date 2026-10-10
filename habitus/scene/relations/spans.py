"""跨度段、扎堆合并与配对的层（语义树新方案 ``13`` ①）。

**四段互不重叠，按预测树的槽对齐**：A（合并后的那条链）在第 ``e`` 槽做完（最后所见），

1. **同一条链**：``[e, e + W]``（W = 转移窗口的槽数）；实验组只算不是 A 那条链自己的记录。从**做完**起量，不从开始起量：
   人一件接一件做事，A 做着的时候别的事要等它做完；从开始量，做得久的 A 会对一切后果都"更少"（第四轮评审算法 1，合成数据
   各类独立、只有开会 50–70 分钟，10/10 读出「开会 → 其余各类 ↓」）；
2. **当天剩下的**：``e + W`` 之后当天人在场的槽；
3. **次日**：A 那天的下一个日历日；
4. **第 2–7 天**：A 那天之后第 2 到第 7 个日历日。

每段各自报，一个效应不会在几档重复显著；允许"不随时间单调"的形状。

**扎堆合并**：同一个概念相邻两次隔不到 W 槽的算同一条链里的一次（与预测树对"链"同一个定义），锚在链头。
后面几段窗口仍重叠的再合一次（"窗口重叠才合并"，结构规则、没有常数）：当天剩下的与次日，同一天的几条链合成一次；
第 2–7 天，隔不到 6 天的合成一次（两个窗口 [d+2, d+7] 与 [d'+2, d'+7] 重叠当且仅当 d' − d ≤ 5）。

**只量人在场的时间**：窗口里人不在的那段不是"B 没来"，是没有机会来。不这么量，"A 之后人就下线了"会被读成
"A 让一切都更难发生"（合成数据实测：17:50 收工前写文档，于是"写文档 → 当天剩下的修改代码 / 调研 / 讨论方案"全部显著为负；
周五做的事"让次日什么都不发生"）。那是"A 之后收工"，预测树的"没有后继"已经记着，不是两件事之间的关系。所以：

- 同一条链的窗口要整段人在场（A 是收工前最后一件的，这一次结果未知）；
- 当天剩下的，量的是 A 之后到当天结束人在场的槽；
- 次日、第 2–7 天只算人在场的日子；窗口里一天都不在场的，结果未知；
- 观测空白不在"人在场"里，自然不算机会。

**对照（病例交叉）**：每次（合并后的）A 是一层；对照取同一层里与 A 同一种时刻、量法与实验组一样。
零假设下 A 那一刻只是这一层里随便一个时刻，检验在层内比（``stats``）：

- **同一条链**：同一天**别的事做完**的时刻（含「待定」那些叫不出名的事），只取各自那一类的链做完的时刻、且做完的那件不是 B 本身——
  两边都是"刚做完一件不是 B 的事、它那一类暂时不再来、人空出来了"，比的只是接下来做什么；同一天也把"那天在赶项目、所以什么都多"
  一起扣掉；
- **当天剩下的**：**别的周、同一个周几**、人在场的日子、**同一个钟点**，各自量到当天结束。不能取同一天的别的钟点：窗口长达几小时，
  早晚不同的钟点后面接的本来就不一样（合成数据实测：17:50 固定写文档，"任何事 → 当天剩下的写文档"全部显著为正）；
- **次日、第 2–7 天**：别的周、同一个周几、人在场的日子。

跨天与当天之后用"同一个周几"（裁定 27 第 2 条）：取同一周别的日子扣不掉周几，会把两件事各自的周节律（周四开会、周五写周报，
彼此无关）读成关系，而周几 × 钟点本来就是预测树的基线；周与周之间的慢漂移由按块的检验与维持检验兜。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from habitus.scene.relations.timeline import LaneTimeline, Mark


class Segment(str, Enum):
    """跨度：前四段是短跨度（A 之后多久），后五段是长跨度（"标志之间的状态"，``longspan``）。"""

    CHAIN = "chain"
    REST_OF_DAY = "rest_of_day"
    NEXT_DAY = "next_day"
    DAYS_2_7 = "days_2_7"
    #: 新类第一次出现之后（记录开始一阵子之后才算"第一次"）。
    FIRST_SEEN = "first_seen"
    #: 一段明显多于 / 少于它自己常态的日子（变点检测）。
    HIGH_STATE = "high_state"
    LOW_STATE = "low_state"
    #: 久别重来之后（隔了超过预测树的复发窗口）。
    RETURN = "return"
    #: 情境概念成立的日子（如「出差中」「周末」）。
    SITUATION = "situation"

    @property
    def daily(self) -> bool:
        """短跨度里，对照按日子取（后两段）还是按槽取（前两段）。"""

        return self in (Segment.NEXT_DAY, Segment.DAYS_2_7)

    @property
    def long(self) -> bool:
        return self not in SEGMENTS


#: 短跨度的四段（按层配对照）。
SEGMENTS = (Segment.CHAIN, Segment.REST_OF_DAY, Segment.NEXT_DAY, Segment.DAYS_2_7)
#: 前因是行为概念时的长跨度标志。
MARKER_SEGMENTS = (Segment.FIRST_SEEN, Segment.HIGH_STATE, Segment.LOW_STATE, Segment.RETURN)


@dataclass(frozen=True)
class Chain:
    """合并后的一次前因：锚在链头；``end`` 是链里最晚做完的那一槽；``uris`` 是链里全部记录（实验组的窗口里不算它们自己）。"""

    anchor: Mark
    uris: frozenset[str]
    goals: frozenset[str]
    end: int

    @property
    def day(self) -> date:
        return self.anchor.day


@dataclass(frozen=True)
class Stratum:
    """一层：A 那一刻来没来（``treated``），对照里定义得了的 ``controls`` 个时刻有几个来了（``positives``）。

    ``block`` 是这次 A 落在第几个块；``same_event`` 是实验组来了的话，来的那条 B 与 A 是不是同一件事（目标相同）。
    """

    chain: Chain
    treated: int
    positives: int
    controls: int
    block: int = 0
    same_event: bool = False

    @property
    def null_probability(self) -> float:
        """零假设下 A 那一刻是阳性的概率：这一层 ``controls + 1`` 个时刻里阳性的比例。"""

        return (self.treated + self.positives) / (self.controls + 1)

    @property
    def control_rate(self) -> float:
        return self.positives / self.controls


def chains_of(line: LaneTimeline, concept: str, window: int) -> tuple[Chain, ...]:
    """扎堆合并：相邻两次隔不到 ``window`` 槽的并进同一条链。"""

    found: list[Chain] = []
    members: list[Mark] = []
    for mark in line.marks.get(concept, []):
        if members and mark.slot - members[-1].slot > window:
            found.append(_chain(members))
            members = []
        members.append(mark)
    if members:
        found.append(_chain(members))
    return tuple(found)


def merge_for(segment: Segment, chains: tuple[Chain, ...]) -> tuple[Chain, ...]:
    """后几段窗口仍重叠的再合一次；锚取最早那条链。"""

    if segment is Segment.CHAIN:
        return chains
    kept: list[Chain] = []
    for chain in chains:
        if kept and _overlapping(segment, kept[-1].day, chain.day):
            continue
        kept.append(chain)
    return tuple(kept)


class Controls:
    """量一层：实验组那一刻与同一层的对照时刻，都按人在场的时间量。

    对照的结果只取决于（后果, 段, 日子, 窗口长短），与是哪个前因无关：每个（后果, 日子）建一次按在场槽排的前缀和，
    所有前因共用，一个对照窗口 O(1)。
    """

    def __init__(self, line: LaneTimeline, window: int, *, blind_as: int | None = None) -> None:
        self.line = line
        self.window = window
        #: 后果"判不了"（细分概念映射答看不到、或窗口里有「待定」）时当作什么：None = 结果未知（这一层不要），0 / 1 = 上下界
        #: 检验的两头（裁定 27 第 6 条：细分概念的关系两遍都过才算，结论不依赖模型什么时候答"看不到"）
        self.blind_as = blind_as
        self._days: dict[tuple[str, Segment, date], int | None] = {}
        self._prefix: dict[tuple[str, date], tuple[list[int], list[int]]] = {}
        self._on_day: dict[tuple[str, date], tuple[tuple[tuple[int, str], ...], frozenset[int]]] = {}

    def stratum(self, segment: Segment, chain: Chain, consequent: str) -> Stratum | None:
        """这一次 A 的一层；A 那一刻的结果未知、或同一层里没有可比的对照时刻，返回 None。"""

        if segment.daily:
            treated = self._day(consequent, segment, chain.day)
            results = self._daily_controls(segment, chain, consequent) if treated is not None else []
        else:
            treated = self._treated_slots(segment, chain, consequent)
            results = self._slot_controls(segment, chain, consequent) if treated is not None else []
        defined = [result for result in results if result is not None]
        if treated is None or not defined:
            return None
        same = treated == 1 and _same_event(self.line, consequent, segment, chain, self.window)
        return Stratum(chain=chain, treated=treated, positives=sum(defined), controls=len(defined), same_event=same)

    # ── 前两段：按槽 ───────────────────────────────────────────────────────

    def _treated_slots(self, segment: Segment, chain: Chain, consequent: str) -> int | None:
        line, day = self.line, chain.day
        present = line.present.get(day, ())
        own = chain.end - line.day_start(day)
        if segment is Segment.CHAIN:
            if not self._fully_present(present, own):
                return None
            return self._after(consequent, day, own, excluded=chain.uris)
        wanted = set(present[bisect_right(present, own + self.window) :])
        if not wanted:
            return None
        start = line.day_start(day)
        for mark in line.marks.get(consequent, []):
            if mark.slot - start in wanted and mark.uri not in chain.uris:
                return 1
        blind = {slot - start for slot in line.unknown.get(consequent, [])}
        return self.blind_as if blind & wanted else 0

    def _slot_controls(self, segment: Segment, chain: Chain, consequent: str) -> list[int | None]:
        line, day = self.line, chain.day
        present = line.present.get(day, ())
        start = line.day_start(day)
        own = chain.end - start
        found: list[int | None] = []
        if segment is Segment.CHAIN:
            # 同一天别的事做完的时刻（同一槽做完的几件算一个时刻）。做完的那件本身是 B 的不算：它那一类之后暂时不会再来，
            # 对照里的 B 会被压低（实验组的前因不是 B，没有这一层）——两边都得是"一件不是 B 的事做完、它那一类暂时不再来"。
            itself = {uri for _slot, uri in self._day_marks(consequent, day)[0]}
            ending: dict[int, set[str]] = {}
            for slot, uri in line.ends.get(day, ()):
                if uri not in chain.uris and uri not in itself:
                    ending.setdefault(slot - start, set()).add(uri)
            for slot, uris in sorted(ending.items()):
                if slot != own and self._fully_present(present, slot):
                    found.append(self._after(consequent, day, slot, excluded=uris))
            return found
        for other in self._peers(day):
            others = line.present.get(other, ())
            if own not in others:
                continue  # 那天这个钟点人不在：不是可比的时刻
            other_hits, other_blind = self._prefix_of(consequent, other)
            found.append(
                _count(other_hits, other_blind, bisect_right(others, own + self.window), len(others) - 1, self.blind_as)
            )
        return found

    def _after(self, consequent: str, day: date, slot: int, *, excluded: frozenset[str] | set[str]) -> int | None:
        """``[slot, slot + W]``（当天槽号）里有没有 B 开始（``excluded`` 那几件自己不算）；没有但有判不了的，未知。"""

        hits, blind = self._day_marks(consequent, day)
        last = slot + self.window
        if any(slot <= at <= last and uri not in excluded for at, uri in hits):
            return 1
        return self.blind_as if any(slot <= at <= last for at in blind) else 0

    def _day_marks(self, consequent: str, day: date) -> tuple[tuple[tuple[int, str], ...], frozenset[int]]:
        key = (consequent, day)
        if key not in self._on_day:
            line = self.line
            start, end = line.day_start(day), line.day_start(day) + line.slots_per_day
            hits = tuple((mark.slot - start, mark.uri) for mark in line.marks.get(consequent, []) if start <= mark.slot < end)
            blind = frozenset(slot - start for slot in line.unknown.get(consequent, []) if start <= slot < end)
            self._on_day[key] = (hits, blind)
        return self._on_day[key]

    def _peers(self, day: date) -> tuple[date, ...]:
        """别的周、同一个周几、人在场的日子。"""

        return tuple(
            other for other, slots in self.line.present.items() if slots and other != day and other.weekday() == day.weekday()
        )

    def _fully_present(self, present: tuple[int, ...], slot: int) -> bool:
        """``[slot, slot + W]`` 整段在场（同一天之内）。"""

        position = bisect_left(present, slot)
        last = position + self.window
        return last < len(present) and present[position] == slot and present[last] == slot + self.window

    def _prefix_of(self, consequent: str, day: date) -> tuple[list[int], list[int]]:
        """按当天在场槽的位置排的两条前缀和：B 命中、B 判不了。"""

        key = (consequent, day)
        if key not in self._prefix:
            line = self.line
            start = line.day_start(day)
            present = line.present.get(day, ())
            position = {slot: index for index, slot in enumerate(present)}
            hit = [0] * len(present)
            blind = [0] * len(present)
            for mark in line.marks.get(consequent, []):
                index = position.get(mark.slot - start)
                if index is not None:
                    hit[index] = 1
            for slot in line.unknown.get(consequent, []):
                index = position.get(slot - start)
                if index is not None:
                    blind[index] = 1
            self._prefix[key] = (_running(hit), _running(blind))
        return self._prefix[key]

    # ── 后两段：按日子 ─────────────────────────────────────────────────────

    def _daily_controls(self, segment: Segment, chain: Chain, consequent: str) -> list[int | None]:
        return [self._day(consequent, segment, other) for other in self._peers(chain.day)]

    def _day(self, consequent: str, segment: Segment, day: date) -> int | None:
        """从 ``day`` 起量的这一段：窗口里人在场的日子有没有 B。伸过截止日、或一天都不在场的，未知。"""

        key = (consequent, segment, day)
        if key not in self._days:
            self._days[key] = self._measure_days(consequent, segment, day)
        return self._days[key]

    def _measure_days(self, consequent: str, segment: Segment, day: date) -> int | None:
        line = self.line
        offsets = (1,) if segment is Segment.NEXT_DAY else range(2, 8)
        window = [day + timedelta(days=offset) for offset in offsets]
        if line.day_start(window[-1]) >= line.horizon:
            return None
        days = {item for item in window if line.present.get(item)}
        if not days:
            return None
        if any(mark.day in days for mark in line.marks.get(consequent, [])):
            return 1
        per_day = line.slots_per_day
        blind = any(date.fromordinal(slot // per_day) in days for slot in line.unknown.get(consequent, []))
        return self.blind_as if blind else 0


def _count(hits: list[int], blind: list[int], first: int, last: int, blind_as: int | None) -> int | None:
    """在场槽位置 ``[first, last]`` 里有没有 B；位置超出当天的，未知。"""

    if first < 0 or last < first or last + 1 >= len(hits):
        return None
    if hits[last + 1] - hits[first] > 0:
        return 1
    return blind_as if blind[last + 1] - blind[first] > 0 else 0


def _running(values: list[int]) -> list[int]:
    total = [0]
    for value in values:
        total.append(total[-1] + value)
    return total


def _same_event(line: LaneTimeline, consequent: str, segment: Segment, chain: Chain, window: int) -> bool:
    """来的那条 B 与 A 目标相同（只用结构字段；"同一件事占比"高的关系对融合口径敏感，算法不看它）。只看同一天的两段。"""

    if not chain.goals or segment.daily:
        return False
    low = chain.end if segment is Segment.CHAIN else chain.end + window + 1
    high = chain.end + window if segment is Segment.CHAIN else line.day_start(chain.day) + line.slots_per_day - 1
    return any(
        low <= item.slot <= high and item.uri not in chain.uris and item.goal in chain.goals
        for item in line.marks.get(consequent, [])
    )


def _chain(members: list[Mark]) -> Chain:
    return Chain(
        anchor=members[0],
        uris=frozenset(item.uri for item in members),
        goals=frozenset(item.goal for item in members if item.goal),
        end=max(item.end for item in members),
    )


def _overlapping(segment: Segment, kept: date, later: date) -> bool:
    if segment in (Segment.REST_OF_DAY, Segment.NEXT_DAY):
        return later == kept
    return (later - kept).days <= 5


__all__ = ["MARKER_SEGMENTS", "SEGMENTS", "Chain", "Controls", "Segment", "Stratum", "chains_of", "merge_for"]
