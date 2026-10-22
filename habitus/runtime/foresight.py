"""预测层在组合根的组装（``19-预测层方案.md`` 第九节，按步接）。

现在接上的：账本（``ForesightLedgerStore``）与结算这一拍（``SettlementStage``，整晚那一趟的第一步）；机会表的存储
（``OpportunityStore``）与算它的那一拍（``ForesightNightlyRun``，由 ``build_foresight_nightly`` 装、挂在
``runtime.nightly.Nightly`` 的语义树之后，同一拍重标单元的历史 ``HistoryStage``，整晚那一趟由 ``NightlyWorker`` 定拍）；白天那一拍（``ForesightTick``，由
``build_foresight_tick`` 装：触发、候选、前因清单，落账；到了时窗按账的级决定说不说）；常驻 worker（``ForesightWorker``，由
``build_foresight_worker`` 装，assembly 接进 ``worker``，Runtime 起停它）。读账的两个借用值（α、遗忘半衰期）由组合根从
关系检验与 ``config.foresight`` 的配置装成 ``Evidence`` 传进来（裁定 49：全系统一个"多大把握"、一种"过去多快淡掉"）。

三件只有组合根知道的事定在这里：此刻是主体本地时刻（时区来自 ``config.locale``）；预测层只接会话一支（按不按 lane 分
还没定，见 assembly 的 TODO）；账本在 ``storage.root/foresight/<branch>``。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from habitus.behavior.tree import BehaviorTree
from habitus.conditions import DayTypeCalendar
from habitus.config import HabitusConfig
from habitus.foresight import Evidence, Tolerance, UnsealedReader
from habitus.foresight.judge import Judge
from habitus.foresight.triggers.assemble import relation_statuses
from habitus.foundation.observability import Observer
from habitus.model_client import StructuredChatClient
from habitus.pre.conversation import QuestionTool
from habitus.runtime.behavior import BehaviorRuntimeComponents
from habitus.runtime.conditions import local_calendar
from habitus.runtime.foresight_compaction import LedgerCompaction
from habitus.runtime.foresight_delivery import DeliveryStage
from habitus.runtime.foresight_history import HistoryStage
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_nightly import ForesightNightlyRun
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_outlet import ReminderOutlet
from habitus.runtime.foresight_outlet_consumer import OutletObservingConsumer
from habitus.runtime.foresight_outlet_store import OutletStore
from habitus.runtime.foresight_reminded import RemindedLookup
from habitus.runtime.foresight_responses import ReminderResponses
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.foresight_tick import ForesightTick
from habitus.runtime.foresight_worker import ForesightWorker
from habitus.runtime.resident import ResidentWorker
from habitus.runtime.session_lane import BehaviorSessionConsumer
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
    """预测层的全部已组装部件。``history`` 是夜里重标单元历史的那一步（与它的落盘）；``worker`` 在触发与递法接上之前为 None。"""

    read_side: ForesightReadSide
    ledger: ForesightLedgerStore
    settlement: SettlementStage
    opportunities: OpportunityStore
    history: HistoryStage
    worker: ResidentWorker | None = None
    #: 夜里的账本压实（26 号清单 4.8）；不给就不压。
    compaction: LedgerCompaction | None = None
    #: 提醒出口（方案 32）：宿主来取、从转写认提问写「说了」；递送那一拍按它的活性决定问不问模型。
    outlet: ReminderOutlet | None = None

    def __post_init__(self) -> None:
        if self.compaction is not None and self.compaction.ledger is not self.ledger:
            raise ValueError("the compaction must use the assembled ledger")
        if self.outlet is not None and self.outlet.ledger is not self.ledger:
            raise ValueError("the reminder outlet must write the assembled ledger")
        if self.settlement.ledger is not self.ledger:
            raise ValueError("the settlement stage must use the assembled ledger")
        if self.settlement.read_side is not self.read_side:
            raise ValueError("the settlement stage must read the assembled behaviour tree")
        if not isinstance(self.opportunities, OpportunityStore):
            raise TypeError("opportunities must be an OpportunityStore")
        if self.opportunities.root != self.ledger.root:
            raise ValueError("the opportunity tables live beside the ledger of the same branch")
        if not isinstance(self.history, HistoryStage):
            raise TypeError("history must be a HistoryStage")
        if self.history.ledger is not self.ledger or self.history.store.root != self.ledger.root:
            raise ValueError("the unit history is read from and lives beside the ledger of the same branch")
        if self.settlement.history is not None and self.settlement.history is not self.history.store:
            raise ValueError("the settlement stage reads the same unit history the nightly stage writes")
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
    calendar: DayTypeCalendar | None = None,
    kind_label: Callable[[str], str | None] | None = None,
    question_tools: Sequence[QuestionTool] | None = None,
) -> ForesightRuntimeComponents | None:
    """组装预测层；未启用时返回 None。``branch`` 是它接的那一支：账本取这一支的目录。``calendar`` 是组合根建好、与语义树共用的
    那一份（不给就自己建）。``kind_label`` 是类编号 → 给人看的类名（词表），出口认提问时拿它配题目；不给就只按话配。"""

    if not config.foresight.enabled:
        return None
    read_side = ForesightReadSide(
        behavior_tree=behavior_tree,
        subject=config.behavior.primary_subject,
        calendar=calendar if calendar is not None else local_calendar(config),
        zone=config.locale.zone(),
    )
    ledger = ForesightLedgerStore(config.foresight_root / branch)
    # 派生物留当前一代与 ``fallback_generations`` 代备用（26 号清单 6.12）
    generations = 1 + config.lifecycle.fallback_generations
    opportunities = OpportunityStore(config.foresight_root / branch, retained_nights=generations)
    history_store = HistoryStore(config.foresight_root / branch, retained_nights=generations)
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
        history=history_store,
    )
    history = HistoryStage(
        ledger,
        history_store,
        evidence=evidence,
        statuses=relation_statuses(config.foresight.shadow_uses_unestablished),
    )
    foresight = config.foresight
    outlet = ReminderOutlet(
        OutletStore(config.foresight_root / branch / "outlet"),
        ledger,
        liveness=timedelta(hours=foresight.outlet_liveness_hours),
        lease=timedelta(minutes=foresight.outlet_lease_minutes),
        max_attempts=foresight.outlet_max_attempts,
        claim_limit=foresight.outlet_claim_limit,
        similarity=foresight.outlet_match_similarity,
        kind_label=kind_label,
        tools=question_tools,
        observer=observer,
    )
    return ForesightRuntimeComponents(
        read_side=read_side,
        ledger=ledger,
        settlement=settlement,
        opportunities=opportunities,
        history=history,
        compaction=LedgerCompaction(ledger, evidence=evidence, lane=branch, clock=clock),
        outlet=outlet,
    )


def build_foresight_nightly(
    components: ForesightRuntimeComponents,
    *,
    relations: RelationStore,
    relation_config: RelationConfig,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
    unseen_day_types: str = "skip",
) -> ForesightNightlyRun:
    """机会表那一拍：读语义树的关系表（``relations``）、按关系检验同一份时间参数（``relation_config``，由
    ``runtime.scene_night.relation_config`` 换算）、写进这一支的机会表存储；同一拍重标这一支的单元历史（裁定 55 新版第 2 条）。
    交给 ``runtime.nightly.Nightly`` 的 ``foresight``。"""

    return ForesightNightlyRun(
        components.opportunities,
        relations,
        relation_config,
        observer=observer,
        clock=clock,
        history=components.history,
        calendar=components.read_side.calendar,
        unseen_day_types=unseen_day_types,
        compaction=components.compaction,
    )


def build_foresight_tick(
    components: ForesightRuntimeComponents,
    *,
    config: HabitusConfig,
    unsealed: UnsealedReader,
    evidence: Evidence,
    structured_chat: StructuredChatClient | None = None,
    vocabulary_version: Callable[[], int] | None = None,
    vocabulary_migrating: Callable[[], bool] | None = None,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
    lane: str = "session",
    reread: Callable[[datetime, datetime], Awaitable[bool]] | None = None,
) -> ForesightTick:
    """白天那一拍：读这一支的机会表、今天的日历条件、行为树今天与前 7 天、还没封口的判断（``unsealed``），把此刻对得上
    历史的预期写进这一支的账本；再把到了时窗的按账的级决定说不说（``structured_chat`` 给了就带模型四问，不给就按算法与模板）。
    ``shadow_uses_unestablished`` 开着时候选关系也进影子账；词表迁移中（``vocabulary_migrating``）整拍不做。

    送达走提醒出口（方案 32）：递送那一拍按出口的活性决定问不问模型——最近一段有宿主来取过才写话、记「待取」，否则记「不可达」；
    每拍末尾把可取的请求清单交给出口。装了插件的用户自动"接通"，没装的一分钱不多花，组合根不再有写死的一行。"""

    foresight = config.foresight
    delivery = DeliveryStage(
        components.ledger,
        tolerance=_tolerance(config),
        evidence=evidence,
        daily_cap=foresight.daily_reminder_cap,
        unit_prior_share=foresight.unit_prior_share,
        history=components.history.store,
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
        outlet=components.outlet,
        lane=lane,
    )
    return ForesightTick(
        components.ledger,
        components.opportunities,
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
        recent=evidence.window,
        reread=reread,
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


def build_outlet_consumer(
    components: ForesightRuntimeComponents,
    inner: BehaviorSessionConsumer,
    *,
    observer: Observer | None = None,
    host_of: Callable[[str], str | None] | None = None,
) -> OutletObservingConsumer:
    """会话源的行为侧消费者外面包出口那一层（方案 32 第 4.2 条）：同一份会话先过 ``outlet.observe``，再进会话 lane。"""

    if components.outlet is None:
        raise ValueError("the foresight components carry no reminder outlet")
    return OutletObservingConsumer(inner, components.outlet, observer=observer, host_of=host_of)


def build_reminder_responses(
    components: ForesightRuntimeComponents, *, config: HabitusConfig, observer: Observer | None = None
) -> ReminderResponses:
    """会话 lane 每轮带开着的提醒、写回回应的那座桥（方案 30 第五节）：交给 ``SessionLane.attach_reminders``。"""

    return ReminderResponses(components.ledger, limit=config.foresight.open_reminders_judged, observer=observer)


def _tolerance(config: HabitusConfig) -> Tolerance:
    foresight = config.foresight
    return Tolerance(line=1.0 - foresight.false_alarm_rate)


__all__ = [
    "ForesightReadSide",
    "ForesightRuntimeComponents",
    "build_foresight_components",
    "build_foresight_nightly",
    "build_foresight_tick",
    "build_foresight_worker",
    "build_outlet_consumer",
    "build_reminded_lookup",
    "build_reminder_responses",
]
