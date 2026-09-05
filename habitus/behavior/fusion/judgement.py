"""行为判断：融合层的唯一产物。

## 它是判断，不是事实

融合产出的不是"一件已经发生完的事"，而是**我们在某个时刻对一段观测做出的判断**。同一段行为
在不同时刻可以有不同判断，它们都为真，只是知道的时间不同：

    20:00:10  「有人在水池边，在做什么还看不出来」
    20:00:30  「刚才那段是洗手，已经洗完了」

后来的判断**不修改**先前的判断，只是新增一条指向它。旧判断永远原样保留——因为"我们当时是怎么
判断的"必须可复原，否则防标签泄漏就没有真相可依。

## 一条判断 = 一个可提醒或可代劳的行为单位

融合的产物单位不是"任何可命名的动作"，而是**在原则上能对主体说"该做 X 了"、或替他做 X 的那种
行为**（吃饭、交谈、给人充电、洗餐具、锁门、喝水、吃药）。这是行为事件在本系统里的操作化判据
（用户裁定，2026-08-30，依据 EgoLife 七天真实数据：按"任何动作"产出时一天 900–2000 条、85% 是
转头点头扶眼镜）。"实际规律不规律"是跨多次观察的统计，留给预测层；这里只判单位性。

    behavior 有               →  一个行为单位；goal 可有可无，basis 可有可无
    behavior 空               →  这段观测没读懂        （只剩 covers）
    帧不属于任何判断           →  看到了、看懂了、不构成任何事（无意识小动作、过渡帧）

``goal`` 是语义面的可读字段，不再决定"算不算一条"；``basis`` 是构成这件事的步骤，有目标的事
通常写得出步骤，说不出目标的单位（锁门、喝水）同样可以有步骤。两者的空非空都不是校验对象——
那些是现实的形状。不确定性一律靠**少说**来表达：融不成就退一层，读不懂就整条留空。

## ``basis`` 的条目不是判断

它们没有身份、没有状态、没有关系、没有主体，只有一层，不能再往下。拿起、放下、走到桌前、
瞥一眼手机这些"改变了世界"的动作若只是某件事的一步，就在这里，不单独成条。

## 主体是多值

一件事可以由多个人一起做（两个人抬桌子）。曾经把 ``subjects`` 设计成单值，实测下来模型一半把
两个人拼成一个字符串（随后被校验拒绝）、一半只留一个人——不稳定且丢人。这与"多人分离暂不考虑"
无关：不考虑的是**区分谁是谁**，不是**只准记一个人**。

## 什么不在这里

不约束"现实中的行为该长什么样"——一个动作的证据帧连不连续、一帧属不属于多个判断，都是上游数据
的事实，不是本层能规定的策略。这里只保证**我们自己的产物内部自洽**。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from habitus.behavior.fusion.errors import BehaviorFusionError
from habitus.behavior.model import MAX_BEHAVIOR_NAME_UTF8_BYTES, semantic_name


class JudgementStatus(str, Enum):
    """这次行为在这条判断成立的时刻处于什么状态。"""

    ONGOING = "ongoing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    ABANDONED = "abandoned"


class JudgementStatusBasis(str, Enum):
    """这个状态是怎么知道的。

    ``INTERRUPTED``（行为真的没做完）与 ``OBSERVATION_LOST``（可能做完了只是没看见）必须分开：
    归约层提取结果时，前者是"没达成"，后者是"不知道"。把两者混成一个值，结果就提不出来了。
    """

    OBSERVED = "observed"
    INFERRED = "inferred"
    OBSERVATION_LOST = "observation_lost"


class JudgementRelation(str, Enum):
    """一条判断与另一条判断的关系；这是"渐进披露"唯一的表达途径。

    ``RESULTS_FROM`` 与其它三种的地位不同：延续、修正、并行都是**这一次观察内部**就能确定的
    事实，而因果是**跨多次观察才成立的规律**。所以它是"看得清清楚楚才标"的加分项，不追求覆盖
    率——"开空调之后常常脱外套"这种要看很多次才知道，属于预测层的统计。让模型在单次观察里硬判
    因果，它只能靠常识猜，而那正是幻觉的来源。

    实测印证过这条：模型稳定标出「开空调→脱外套」「烧水→泡茶」，稳定不标「做饭→吃饭」
    「洗手→吃面包」「开灯→看书」。曾经把"没标"当成缺口去找分界、想调提示词逼它多标，方向是
    反的——真正该保住的是"标出来的都可信"。
    """

    CONTINUES = "continues"
    SUPERSEDES = "supersedes"
    CONCURRENT_WITH = "concurrent_with"
    # 结果指回原因，而不是原因指向结果：先前的判断已经落盘且不可变，只有后来的那条能指回去。
    # 这也正好对上"有些结果不是立刻的、是后来才看到的"——结果本来就晚于原因。
    RESULTS_FROM = "results_from"


@dataclass(frozen=True)
class JudgementLink:
    """指向另一条判断的一条关系。

    目标有两个来源，恰好用一个：``target_no`` 指本批新产出的判断，``context_no`` 指先前已经落盘、
    在提示词里以序号呈现的判断。模型不可能写对 64 位的内容身份，所以两边都用小整数，由系统映射
    回真实身份。
    """

    kind: JudgementRelation
    target_no: int | None
    context_no: int | None

    def __post_init__(self) -> None:
        if (self.target_no is None) == (self.context_no is None):
            raise BehaviorFusionError(
                "a relation must name exactly one target: either target_no or context_no"
            )
        for name in ("target_no", "context_no"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise BehaviorFusionError(f"{name} must be a positive integer")

    @property
    def is_cross_window(self) -> bool:
        return self.context_no is not None


@dataclass(frozen=True)
class BehaviorFact:
    """构成一个目标行为事件的一条行为事实。

    折叠的产物：八条"人在搓手"归成一条「打肥皂搓手」。折叠是判断，从原始帧推不出来，所以必须
    记下——归约层要看「冲水关龙头」到底发生了没有，只给它八条原始语义等于让它重判一遍。
    """

    semantics: str
    fragment_nos: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.semantics.strip():
            raise BehaviorFusionError("a behavior fact must carry semantics")
        if not self.fragment_nos:
            raise BehaviorFusionError("a behavior fact must reference at least one fragment")


@dataclass(frozen=True)
class BehaviorClaim:
    """判断的内容。三层递减完全由 ``behavior`` / ``goal`` 的空与非空决定。"""

    behavior: str | None
    goal: str | None
    summary: str | None
    basis: tuple[BehaviorFact, ...]

    def __post_init__(self) -> None:
        if self.behavior is None:
            # 读不懂的那一层：不能同时又说得出目标或摘要。
            if self.goal is not None or self.summary is not None or self.basis:
                raise BehaviorFusionError("an unreadable claim must not carry goal, summary or basis")
            return
        # behavior 名将来就是行为树的地址身份；不合格的名字若放行落盘，会在归约封口后才炸、
        # 且无自愈路径（正确性不依赖模型：在这里打回，模型换个说法重试即可）。
        try:
            semantic_name(self.behavior, "behavior")
        except (TypeError, ValueError) as exc:
            raise BehaviorFusionError(
                f"behavior name is not usable as a tree address ({exc}); rephrase it"
            ) from exc
        if len(self.behavior.encode("utf-8")) > MAX_BEHAVIOR_NAME_UTF8_BYTES:
            raise BehaviorFusionError(
                "behavior name is too long for a tree address; describe the behaviour in fewer words"
            )
        if self.summary is None:
            raise BehaviorFusionError("a readable claim must carry a summary")
        # goal 与 basis 的空非空互不约束：锁门、喝水说不出目标却有步骤；一次交谈有目标却未必
        # 分得出步骤。它们描述的是现实的形状，不是我们产物的自洽。

    @property
    def is_readable(self) -> bool:
        return self.behavior is not None

    @property
    def is_goal_directed(self) -> bool:
        return self.goal is not None


@dataclass(frozen=True)
class BehaviorJudgement:
    """一条行为判断。

    ``judgement_no`` 与 ``covers`` 里的编号都是**解析期临时编号**，不持久化——耐久记录里换成
    观测的内容身份（见 ``derivation.py``）。
    """

    judgement_no: int
    covers: tuple[int, ...]
    subjects: tuple[str, ...]
    claim: BehaviorClaim
    status: JudgementStatus | None
    status_basis: JudgementStatusBasis | None
    # 关系是**列表**：一条判断可以同时延续前一段又与另一条并行。空列表就是独立。
    # 早先把它设计成单值字段，于是"做饭"既已声明独立就再也挂不上"与接电话并行"——那是设计
    # 缺陷，不是模型的错。
    relations: tuple[JudgementLink, ...]

    def __post_init__(self) -> None:
        if self.judgement_no <= 0:
            raise BehaviorFusionError("judgement_no must be a positive integer")
        if not self.covers:
            raise BehaviorFusionError("a judgement must cover at least one fragment")
        if not self.claim.is_readable:
            if self.subjects or self.status is not None or self.status_basis is not None:
                raise BehaviorFusionError(
                    "an unreadable judgement must not carry subjects or a status"
                )
            if self.relations:
                raise BehaviorFusionError("an unreadable judgement cannot relate to another one")
        else:
            if not self.subjects:
                raise BehaviorFusionError("a readable judgement must name at least one subject")
            if len(set(self.subjects)) != len(self.subjects):
                raise BehaviorFusionError("subjects must not repeat")
            if self.status is None or self.status_basis is None:
                raise BehaviorFusionError("a readable judgement must carry a status and its basis")
        seen: set[tuple[str, int | None, int | None]] = set()
        for link in self.relations:
            if link.target_no == self.judgement_no:
                raise BehaviorFusionError("a judgement cannot relate to itself")
            key = (link.kind.value, link.target_no, link.context_no)
            if key in seen:
                raise BehaviorFusionError("a judgement declares the same relation twice")
            seen.add(key)
        # ``basis`` 引用的帧必须落在本判断覆盖的范围内——这是我们自己的产物是否自洽，不是在
        # 规定现实中的动作该覆盖哪些帧。
        covered = set(self.covers)
        for fact in self.claim.basis:
            outside = sorted(set(fact.fragment_nos) - covered)
            if outside:
                raise BehaviorFusionError(
                    f"judgement[{self.judgement_no}] basis references fragments it does not cover: {outside}"
                )

    @property
    def earliest_fragment_no(self) -> int:
        return self.covers[0]


@dataclass(frozen=True)
class BehaviorJudgementBatch:
    """一次融合产出的全部判断，按最早覆盖片段排序。

    ``degradations`` 是装配层对模型记账疏漏做的降级记录（去重、剔名、丢边、goal 置空），每条
    带可定位的坐标。它们是信号不是语义：不进判断本体、不进树，只供回执/可观测面板统计——
    真实数据实测（BHV-REALDATA-001）这类疏漏一周上千次，整批拒绝会把串行队列封死。
    """

    judgements: tuple[BehaviorJudgement, ...]
    degradations: tuple[str, ...] = ()
    # 读得懂却不属于任何判断的片段编号（无意识小动作、过渡帧）：树上不写任何东西，曝光分母
    # 照常算这段在观测。它是"允许模型不产出"的出口，所以占比要作为信号报出来。
    unowned_fragment_nos: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # 空批是合法的：整段都是无意识小动作/过渡（5 分钟静坐）时，模型把每帧填 []、一条判断都不出
        # ——逼它"至少声明一条"等于逼它发明（WP4 裁定）。回执照记、覆盖照走、树上无痕。
        if not isinstance(self.judgements, tuple):
            raise BehaviorFusionError("judgements must be a tuple")
        if not isinstance(self.unowned_fragment_nos, tuple) or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in self.unowned_fragment_nos
        ):
            raise BehaviorFusionError("unowned_fragment_nos must be a tuple of positive integers")
        if not isinstance(self.degradations, tuple) or any(
            not isinstance(item, str) or not item for item in self.degradations
        ):
            raise BehaviorFusionError("degradations must be a tuple of non-empty strings")

    @property
    def readable(self) -> tuple[BehaviorJudgement, ...]:
        return tuple(item for item in self.judgements if item.claim.is_readable)

    @property
    def unreadable_fragment_count(self) -> int:
        """一条读不懂的片段都没有被别的判断读懂——取**并集**，不是求和。

        一帧允许被多条判断认领（并行时本来如此），求和会把同一帧数好几遍，算出大于 1 的
        "比例"，而这个数的用途正是阈值告警。同理，一帧若同时被某条可读判断认领，它就不算
        没读懂——只有完全没人读懂的帧才计入。
        """

        readable = {no for item in self.judgements if item.claim.is_readable for no in item.covers}
        unreadable = {
            no for item in self.judgements if not item.claim.is_readable for no in item.covers
        }
        return len(unreadable - readable)


__all__ = [
    "BehaviorClaim",
    "BehaviorFact",
    "BehaviorJudgement",
    "BehaviorJudgementBatch",
    "JudgementLink",
    "JudgementRelation",
    "JudgementStatus",
    "JudgementStatusBasis",
]
