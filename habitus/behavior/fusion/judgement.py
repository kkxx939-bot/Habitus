"""行为判断：融合层的唯一产物。

## 它是判断，不是事实

融合产出的不是"一件已经发生完的事"，而是**我们在某个时刻对一段观测做出的判断**（带着判断的时刻
``judged_at``）。判断落盘后不修改——"我们当时是怎么判断的"必须可复原，否则防标签泄漏就没有真相可依。
两条 lane 都是一个单位出一条判断，判断之间不互相指向。

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
from habitus.behavior.model import BehaviorNameTooLongError, behavior_name


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
    # 只用在 behavior 为空的判断上：这段时间不是"看了没读懂"，而是上游说它**没在看**（没戴、充电、没数据）。
    # 归约据此把这段落成「未观测」的空白，而不是「没读懂」。
    UNOBSERVED = "unobserved"
    # 只用在 behavior 为空的判断上：在看，但系统出了问题没读成（模型调不通、抛异常、材料坏，裁定 94 第 3 条）。
    # 归约据此落成「错误」的空白，而不是「没读懂」（模型读了、给不出结论）。
    ERROR = "error"


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
            behavior_name(self.behavior)
        except BehaviorNameTooLongError as exc:
            raise BehaviorFusionError(
                "behavior name is too long for a tree address; describe the behaviour in fewer words"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise BehaviorFusionError(f"behavior name is not usable as a tree address ({exc}); rephrase it") from exc
        if self.summary is None:
            raise BehaviorFusionError("a readable claim must carry a summary")
        # goal 与 basis 的空非空互不约束：锁门、喝水说不出目标却有步骤；一次交谈有目标却未必
        # 分得出步骤。它们描述的是现实的形状，不是我们产物的自洽。

    @property
    def is_readable(self) -> bool:
        return self.behavior is not None


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

    def __post_init__(self) -> None:
        if self.judgement_no <= 0:
            raise BehaviorFusionError("judgement_no must be a positive integer")
        if not self.covers:
            raise BehaviorFusionError("a judgement must cover at least one fragment")
        if not self.claim.is_readable:
            if self.subjects or self.status is not None:
                raise BehaviorFusionError("an unreadable judgement must not carry subjects or a status")
            if self.status_basis not in (None, JudgementStatusBasis.UNOBSERVED, JudgementStatusBasis.ERROR):
                raise BehaviorFusionError("a judgement without a behaviour can only say it was unobserved or failed")
        else:
            if not self.subjects:
                raise BehaviorFusionError("a readable judgement must name at least one subject")
            if len(set(self.subjects)) != len(self.subjects):
                raise BehaviorFusionError("subjects must not repeat")
            if self.status is None or self.status_basis is None:
                raise BehaviorFusionError("a readable judgement must carry a status and its basis")
            if self.status_basis in (JudgementStatusBasis.UNOBSERVED, JudgementStatusBasis.ERROR):
                raise BehaviorFusionError("a readable judgement cannot be unobserved or failed")
        # ``basis`` 引用的帧必须落在本判断覆盖的范围内——这是我们自己的产物是否自洽，不是在
        # 规定现实中的动作该覆盖哪些帧。
        covered = set(self.covers)
        for fact in self.claim.basis:
            outside = sorted(set(fact.fragment_nos) - covered)
            if outside:
                raise BehaviorFusionError(
                    f"judgement[{self.judgement_no}] basis references fragments it does not cover: {outside}"
                )


@dataclass(frozen=True)
class BehaviorJudgementBatch:
    """一次融合产出的全部判断。"""

    judgements: tuple[BehaviorJudgement, ...]

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
        unreadable = {no for item in self.judgements if not item.claim.is_readable for no in item.covers}
        return len(unreadable - readable)


__all__ = [
    "BehaviorClaim",
    "BehaviorFact",
    "BehaviorJudgement",
    "BehaviorJudgementBatch",
    "JudgementStatus",
    "JudgementStatusBasis",
]
