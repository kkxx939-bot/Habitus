"""预测层在组合根的组装（``19-预测层方案.md`` 第九节，按步接）。

现在接上的：账本（``ForesightLedgerStore``）与结算这一拍（``SettlementStage``，挂在预测树夜批之后）；机会表的存储
（``OpportunityStore``）与算它的那一拍（``ForesightNightlyRun``，由 ``build_foresight_nightly`` 装、挂在
``runtime.nightly.Nightly`` 的语义树之后——语义树夜批还没接进常驻 worker，眼下由脚本串）；白天那一拍（``ForesightTick``，由
``build_foresight_tick`` 装：触发、候选、前因清单，落账；到了时窗按账的级决定说不说）；常驻 worker（``ForesightWorker``，由
``build_foresight_worker`` 装，assembly 接进 ``worker``，Runtime 起停它）。读账的两个借用值（α、遗忘半衰期）由组合根从
关系检验与预测树的配置装成 ``Evidence`` 传进来（裁定 49：全系统一个"多大把握"、一种"过去多快淡掉"）。

三件只有组合根知道的事定在这里：此刻是主体本地时刻（时区来自 ``config.locale``）；预测层只接会话一支（按不按 lane 分
还没定，见 assembly 的 TODO）；账本在 ``storage.root/foresight/<branch>``。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from habitus.behavior.tree import BehaviorTree
from habitus.config import HabitusConfig
from habitus.foresight import Evidence, ForesightError, Tolerance, UnsealedReader
from habitus.foresight.judge import Judge
from habitus.foundation.observability import Observer
from habitus.model_client import StructuredChatClient
from habitus.prediction.store import PredictionTreeStore
from habitus.runtime.behavior import BehaviorRuntimeComponents
from habitus.runtime.foresight_causes import CauseStage
from habitus.runtime.foresight_delivery import DeliveryStage
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_nightly import ForesightNightlyRun
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_reminded import RemindedLookup
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.foresight_tick import ForesightTick
from habitus.runtime.foresight_worker import ForesightWorker
from habitus.runtime.resident import ResidentWorker
from habitus.scene import DayTypeCalendar, NominalCalendar
from habitus.scene.relations import RelationConfig
from habitus.scene.relations.store import RelationStore


@dataclass(frozen=True)
class ForesightReadSide:
    """预测层读行为树的那一侧：哪棵树、谁的、按什么日历、什么时区。"""

    behavior_tree: BehaviorTree
    subject: str
    calendar: DayTypeCalendar
    zone: ZoneInfo

    def __post_init__(self) -> None:
        if not isinstance(self.behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("subject must be non-empty text")
        if not isinstance(self.calendar, DayTypeCalendar):
            raise TypeError("calendar must implement DayTypeCalendar")
        if not isinstance(self.zone, ZoneInfo):
            raise TypeError("zone must be a ZoneInfo")


@dataclass(frozen=True)
class ForesightRuntimeComponents:
    """预测层的全部已组装部件。``worker`` 在触发与递法接上之前为 None。"""

    read_side: ForesightReadSide
    ledger: ForesightLedgerStore
    settlement: SettlementStage
    opportunities: OpportunityStore
    worker: ResidentWorker | None = None

    def __post_init__(self) -> None:
        if self.settlement.ledger is not self.ledger:
            raise ValueError("the settlement stage must use the assembled ledger")
        if self.settlement.read_side is not self.read_side:
            raise ValueError("the settlement stage must read the assembled behaviour tree")
        if not isinstance(self.opportunities, OpportunityStore):
            raise TypeError("opportunities must be an OpportunityStore")
        if self.opportunities.root != self.ledger.root:
            raise ValueError("the opportunity tables live beside the ledger of the same branch")
        if self.worker is not None and not isinstance(self.worker, ResidentWorker):
            raise TypeError("worker must be a ResidentWorker or None")

    def assert_attached_to(self, *, behavior: BehaviorRuntimeComponents) -> None:
        """实例同一性：预测层必须读**这个** Runtime 的那棵行为树、按它的定稿日结算。"""

        if self.read_side.behavior_tree is not behavior.tree:
            raise ValueError("foresight must read the assembled behaviour tree")
        if self.settlement.closed_days != behavior.reduction_runner.closed_days:
            raise ValueError("foresight must settle against the reduction's own closed days")


def build_foresight_components(
    config: HabitusConfig,
    *,
    behavior_tree: BehaviorTree,
    current_classes: Callable[[str], Sequence[str]],
    vocabulary_migrating: Callable[[], bool],
    closed_days: Callable[[], Iterable[date]],
    evidence: Evidence,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
    branch: str = "session",
) -> ForesightRuntimeComponents | None:
    """组装预测层；未启用时返回 None。``branch`` 是它接的那一支：账本取这一支的目录。"""

    if not config.foresight.enabled:
        return None
    read_side = ForesightReadSide(
        behavior_tree=behavior_tree,
        subject=config.behavior.primary_subject,
        calendar=_calendar(config),
        zone=config.locale.zone(),
    )
    ledger = ForesightLedgerStore(config.foresight_root / branch)
    opportunities = OpportunityStore(config.foresight_root / branch)
    settlement = SettlementStage(
        ledger,
        read_side,
        # 定稿日只有归约说了算；承诺记的是形成时的类编号，结算按"这个编号现在对应哪些编号"认
        closed_days=closed_days,
        current=current_classes,
        migrating=vocabulary_migrating,
        evidence=evidence,
        opportunities=opportunities,
        lane=branch,
        observer=observer,
        clock=clock,
        tolerance=_tolerance(config),
    )
    return ForesightRuntimeComponents(
        read_side=read_side, ledger=ledger, settlement=settlement, opportunities=opportunities
    )


def build_foresight_nightly(
    components: ForesightRuntimeComponents,
    *,
    relations: RelationStore,
    relation_config: RelationConfig,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
) -> ForesightNightlyRun:
    """机会表那一拍：读语义树的关系表（``relations``）、按关系检验同一份时间参数（``relation_config``，由
    ``runtime.scene_night.relation_config`` 换算）、写进这一支的机会表存储；同一拍给这一支已结的落空找原因（裁定 55）。
    交给 ``runtime.nightly.Nightly`` 的 ``foresight``。"""

    return ForesightNightlyRun(
        components.opportunities,
        relations,
        relation_config,
        observer=observer,
        clock=clock,
        causes=CauseStage(
            components.ledger,
            lane=components.settlement.lane,
            current=components.settlement.current,
            calendar=components.read_side.calendar,
            clock=clock,
        ),
    )


def build_foresight_tick(
    components: ForesightRuntimeComponents,
    *,
    config: HabitusConfig,
    prediction: PredictionTreeStore,
    unsealed: UnsealedReader,
    evidence: Evidence,
    structured_chat: StructuredChatClient | None = None,
    vocabulary_version: Callable[[], int] | None = None,
    vocabulary_migrating: Callable[[], bool] | None = None,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
    lane: str = "session",
) -> ForesightTick:
    """白天那一拍：读这一支的机会表与正在服务的预测树、行为树今天与前 7 天、还没封口的判断（``unsealed``），把此刻对得上
    历史的预期写进这一支的账本；再把到了时窗的按账的级决定说不说（``structured_chat`` 给了就带模型四问，不给就按算法与模板）。
    ``shadow_uses_unestablished`` 开着时候选关系也进影子账；词表迁移中（``vocabulary_migrating``）整拍不做。

    送达渠道还没有（方案第二节：本层只产出"递送请求 + 档 + 话"，送达在本层之外），所以 ``reachable`` 为假：提醒档的决定都
    记「不可达」，结算按没说过处理，模型也不问（裁定 47）。接上渠道时把它改成真，这是组合根唯一要改的一行。"""

    foresight = config.foresight
    delivery = DeliveryStage(
        components.ledger,
        tolerance=_tolerance(config),
        evidence=evidence,
        daily_cap=foresight.daily_reminder_cap,
        severity_review_days=foresight.severity_review_days,
        holdout_rate=foresight.holdout_rate,
        holdout_rate_heavy=foresight.holdout_rate_heavy,
        unit_prior_strength=foresight.unit_prior_strength,
        judge=(
            None
            if structured_chat is None
            else Judge(
                structured_chat,
                retries=foresight.judge_transient_retries,
                delay_seconds=foresight.judge_transient_retry_delay_seconds,
            )
        ),
        tick_seconds=foresight.tick_interval_seconds,
        reachable=False,
    )
    return ForesightTick(
        components.ledger,
        components.opportunities,
        prediction,
        components.read_side,
        unsealed,
        lane=lane,
        shadow_unestablished=config.foresight.shadow_uses_unestablished,
        vocabulary_version=vocabulary_version,
        migrating=vocabulary_migrating,
        observer=observer,
        clock=clock,
        delivery=delivery,
        tick_seconds=foresight.tick_interval_seconds,
    )


def build_foresight_worker(
    tick: ForesightTick, *, config: HabitusConfig, observer: Observer | None = None
) -> ForesightWorker:
    """常驻 worker：每 ``tick_interval_seconds`` 跑一拍。"""

    return ForesightWorker(
        tick,
        interval_seconds=config.foresight.tick_interval_seconds,
        shutdown_timeout_seconds=config.foresight.worker_shutdown_timeout_seconds,
        observer=observer,
    )


def build_reminded_lookup(components: ForesightRuntimeComponents) -> RemindedLookup:
    """归约封口时盖 ``reminded`` 的查法（第 0 步）：交给 ``BehaviorReductionRunner.attach_reminder_lookup``。"""

    return RemindedLookup(components.ledger, current=components.settlement.current)


def _tolerance(config: HabitusConfig) -> Tolerance:
    foresight = config.foresight
    return Tolerance(floor=foresight.line_floor, ceiling=foresight.line_ceiling, unjudged=foresight.cost_ratio_unjudged)


def _calendar(config: HabitusConfig) -> DayTypeCalendar:
    """当地日历。数据文件的格式还没定，所以现在只有名义日历一种。"""

    if config.locale.calendar_path is not None:
        raise ForesightError("locale.calendar_path is not supported yet")
    return NominalCalendar()


__all__ = [
    "ForesightReadSide",
    "ForesightRuntimeComponents",
    "build_foresight_components",
    "build_foresight_nightly",
    "build_foresight_tick",
    "build_foresight_worker",
    "build_reminded_lookup",
]
