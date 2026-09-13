"""预测层在组合根的组装。

预测层站在两棵派生树之上：数字取自预测树的一代，与之对应的历史背景取自情景树。本模块负责
把它们接起来，并且**把三件只有组合根知道的事定死**：

1. **钉住一代**。每次装配从 ``store.load()` 取当前一代并核对它的 ``config_digest`` 与现行配置
   一致——参数变了就是另一套统计，混读出来的数字互不一致而且看不出来。不一致时硬拒，等夜批
   按新参数重建。
2. **此刻是本地时刻**。时区来自 ``config.locale``，不用进程默认时区：读侧的每个"今天是哪一天、
   此刻在哪个槽"都从它的本地时分算，错一个时区就整体错一天。
3. **一次查询一份缓存**。``DayIndexCache`` 的契约是"在它的生命周期里同一天只读一次"，所以每次
   装配新建一个；跨次复用会读到过期的今天。

现阶段**不起 worker**：判断的节奏还没定（见 TODO(PRED-DOWNSTREAM-001)），装配入口给脚本、
测试与后续的判断段调用。本模块不接 ``StructuredChatClient``，也不碰 memory。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from habitus.behavior.tree import BehaviorTree
from habitus.config import HabitusConfig
from habitus.foresight import CellIndex, ForesightError
from habitus.foresight.assemble import CandidateEvidence, candidates_evidence, moment_at
from habitus.prediction import builder
from habitus.prediction.config import PredictionTreeConfig
from habitus.prediction.store import PredictionTreeStore
from habitus.scene import DayTypeCalendar, NominalCalendar, SceneTree
from habitus.scene.views import DayIndexCache


@dataclass(frozen=True)
class ForesightRuntimeComponents:
    """预测层的全部已组装部件。现在只有装配器，没有节拍。"""

    assembler: EvidenceAssembler


class EvidenceAssembler:
    """按此刻装配候选的证据：四层数字，每层配算出它的那几天的历史背景。"""

    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        scene_tree: SceneTree,
        store: PredictionTreeStore,
        subject: str,
        zone: ZoneInfo,
        calendar: DayTypeCalendar,
        expected_digest: str,
        half_width: int,
        window_days: int,
        transition_window_seconds: float,
        max_days_per_layer: int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if not isinstance(scene_tree, SceneTree):
            raise TypeError("scene_tree must be a SceneTree")
        if not isinstance(store, PredictionTreeStore):
            raise TypeError("store must be a PredictionTreeStore")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("subject must be non-empty text")
        # zone 是本类唯一真正在守的那条纪律（"此刻必须是主体本地时刻"），却最容易被漏检：
        # ``astimezone(None)`` 会安安静静地落到**进程默认时区**，产出一个看起来完全正常的
        # Moment，而今天是哪一天、此刻在哪个槽已经错了。
        if not isinstance(zone, ZoneInfo):
            raise TypeError("zone must be a ZoneInfo")
        if not isinstance(calendar, DayTypeCalendar):
            raise TypeError("calendar must implement DayTypeCalendar")
        if not isinstance(expected_digest, str) or not expected_digest:
            raise ValueError("expected_digest must be non-empty text")
        self.behavior_tree = behavior_tree
        self.scene_tree = scene_tree
        self.store = store
        self.subject = subject
        self.zone = zone
        self.calendar = calendar
        self.expected_digest = expected_digest
        self.half_width = half_width
        self.window_days = window_days
        self.transition_window_seconds = transition_window_seconds
        self.max_days_per_layer = max_days_per_layer
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def assemble(
        self, kind_tokens: Sequence[str], *, now: datetime | None = None
    ) -> tuple[CandidateEvidence, ...]:
        """装配这一刻这些候选的证据。

        ``now`` 缺省取注入的时钟；无论哪来，都换算到主体的时区之后再落钟面——传进来的可以是
        UTC，但落到"今天是哪一天、此刻在哪个槽"之前必须先变成本地时刻。
        """

        tree = self.store.load()
        if tree is None:
            # 还没有发布过任何一代。这不是"什么都不会发生"，是"还没算过"——必须能分清。
            raise ForesightError("no prediction generation has been published yet")
        if tree.config_digest != self.expected_digest:
            raise ForesightError(
                "the published prediction generation was built with different estimation "
                "parameters; wait for the nightly rebuild instead of mixing two sets of statistics"
            )
        at = (now if now is not None else self._clock()).astimezone(self.zone)
        cache = DayIndexCache(
            self.behavior_tree, self.scene_tree, subject=self.subject, calendar=self.calendar
        )
        moment = moment_at(at, slot_minutes=tree.slot_minutes, day_note=cache.day(at.date()).day_note)
        return candidates_evidence(
            CellIndex.of(tree),
            kind_tokens,
            moment,
            cache,
            half_width=self.half_width,
            window_days=self.window_days,
            transition_window_seconds=self.transition_window_seconds,
            max_days_per_layer=self.max_days_per_layer,
        )


def build_foresight_components(
    config: HabitusConfig,
    *,
    behavior_tree: BehaviorTree,
    scene_tree: SceneTree | None,
    store: PredictionTreeStore | None,
    clock: Callable[[], datetime] | None = None,
) -> ForesightRuntimeComponents | None:
    """组装预测层；未启用时返回 None。

    两棵派生树缺任何一棵都返回 None 而不是半个装配器：跨域校验已经在配置层拒过这种组合，
    这里是第二道——组合根拿到的部件本来就可能因为上游未启用而是 None。
    """

    if not config.foresight.enabled:
        return None
    if scene_tree is None or store is None:
        return None
    tree_config = PredictionTreeConfig(**config.prediction.tree_parameters())
    assembler = EvidenceAssembler(
        behavior_tree=behavior_tree,
        scene_tree=scene_tree,
        store=store,
        subject=config.behavior.primary_subject,
        zone=config.locale.zone(),
        calendar=_calendar(config),
        # 指纹在这里算一次：读侧每次装配都拿它核对钉住的那一代，防的是跨参数混读。
        expected_digest=builder.config_digest(tree_config),
        # 窗口就是树的池化邻域与转移窗，不另配一份——配两份会让数字与背景取自不同的范围。
        half_width=tree_config.pool_half_width,
        transition_window_seconds=tree_config.transition_window_seconds,
        window_days=config.scene.lookback_days,
        max_days_per_layer=config.foresight.max_days_per_layer,
        clock=clock,
    )
    return ForesightRuntimeComponents(assembler=assembler)


def _calendar(config: HabitusConfig) -> DayTypeCalendar:
    """当地日历。数据文件的格式还没定，所以现在只有名义日历一种。

    配置层已经拒过非空路径；这里是第二道——组合根是唯一知道"这个路径该交给谁解析"的地方，
    将来多一种实现时也从这里分叉。
    """

    if config.locale.calendar_path is not None:
        raise ForesightError("locale.calendar_path is not supported yet")
    return NominalCalendar()


__all__ = [
    "EvidenceAssembler",
    "ForesightRuntimeComponents",
    "build_foresight_components",
]
