"""预测层在组合根的组装（``19-预测层方案.md`` 第九节，按步接）。

现在接上的：账本（``ForesightLedgerStore``）与结算这一拍（``SettlementStage``，换代里的一步）；机会表的存储
（``OpportunityStore``）与算它的那一拍（``ForesightNightlyRun``，由 ``build_foresight_nightly`` 装、挂在
``runtime.generation.Generation`` 的语义树之后，同一拍重标单元的历史 ``HistoryStage``，换代由预测 worker 每槽先看一眼、在连续
定稿到昨天之后做）；当天的日子预期（``DayExpectationStage``，由 ``build_foresight_days`` 装，换代必须成的一步）；
预测拍（``ForesightTick``，由 ``build_foresight_tick`` 装：事件与情境的触发、候选、前因清单，落账；到了时窗按账的级决定
说不说；落盘的水位、没在看的时段、预测根租约）；常驻 worker（``ForesightWorker``，在 ``runtime.scene.assemble_generation`` 里
与换代一起装：拍点对齐槽边界，每槽先看一眼换代、再做这一拍；``with_slot_worker`` 接进 ``worker``，Runtime 起停它）。读账的两个借用值（α、遗忘半衰期）由组合根从
关系检验与 ``config.foresight`` 的配置装成 ``Evidence`` 传进来（裁定 49：全系统一个"多大把握"、一种"过去多快淡掉"）。

三件只有组合根知道的事定在这里：此刻是主体本地时刻（时区来自 ``config.locale``）；预测层只接会话一支（按不按 lane 分
还没定，见 assembly 的 TODO）；账本在 ``storage.root/foresight/<branch>``。
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from habitus.behavior.tree import BehaviorTree
from habitus.conditions import DayTypeCalendar
from habitus.config import HabitusConfig
from habitus.foresight import Evidence, Tolerance, UnsealedReader
from habitus.foresight.judge import Judge
from habitus.foresight.triggers.assemble import relation_statuses
from habitus.foundation.observability import Observer
from habitus.foundation.tracing import TraceSink
from habitus.infrastructure.store.contracts import LockStore
from habitus.model_client import StructuredChatClient
from habitus.pre.conversation import QuestionTool
from habitus.runtime.behavior import BehaviorRuntimeComponents
from habitus.runtime.conditions import local_calendar
from habitus.runtime.foresight_compaction import LedgerCompaction
from habitus.runtime.foresight_days import DayExpectationStage
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
from habitus.runtime.generation_lease import Lease
from habitus.runtime.generation_watermark import TickTrack, TickWatermarkStore
from habitus.runtime.resident import ResidentWorker
from habitus.runtime.session_lane import BehaviorSessionConsumer
from habitus.runtime.unsealed import UnsealedFromJudgements
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
        if not self.subject.strip():
            raise ValueError("subject must be non-empty text")


@dataclass(frozen=True)
class ForesightRuntimeComponents:
    """预测层的全部已组装部件。``history`` 是换代里重标单元历史的那一步（与它的落盘）；``worker`` 在触发与递法接上之前为 None。
    ``watermark`` 是预测拍的水位（换代据它判"没在看的空白落树了没有"）。"""

    read_side: ForesightReadSide
    ledger: ForesightLedgerStore
    settlement: SettlementStage
    opportunities: OpportunityStore
    history: HistoryStage
    worker: ResidentWorker | None = None
    #: 换代里的账本压实（26 号清单 4.8）；不给就不压。
    compaction: LedgerCompaction | None = None
    #: 提醒出口（方案 32）：宿主来取、从转写认提问写「说了」；递送那一拍按它的活性决定问不问模型。
    outlet: ReminderOutlet | None = None
    #: 换代里建当天日子预期的那一步（必须成）；与 ``worker`` 一起在未封口的桥装好之后接上。
    days: DayExpectationStage | None = None
    #: 预测拍的水位（``ticked_at``、``unwatched_until``、``formed_through``、``formed_generation``）。
    watermark: TickWatermarkStore | None = None

    def __post_init__(self) -> None:
        if self.compaction is not None and self.compaction.ledger is not self.ledger:
            raise ValueError("the compaction must use the assembled ledger")
        if self.outlet is not None and self.outlet.ledger is not self.ledger:
            raise ValueError("the reminder outlet must write the assembled ledger")
        if self.settlement.ledger is not self.ledger:
            raise ValueError("the settlement stage must use the assembled ledger")
        if self.settlement.read_side is not self.read_side:
            raise ValueError("the settlement stage must read the assembled behaviour tree")
        if self.opportunities.root != self.ledger.root:
            raise ValueError("the opportunity tables live beside the ledger of the same branch")
        if self.history.ledger is not self.ledger or self.history.store.root != self.ledger.root:
            raise ValueError("the unit history is read from and lives beside the ledger of the same branch")
        if self.settlement.history is not None and self.settlement.history is not self.history.store:
            raise ValueError("the settlement stage reads the same unit history the generation writes")
        if self.days is not None and (
            self.days.ledger is not self.ledger or self.days.opportunities is not self.opportunities
        ):
            raise ValueError("the day expectations are built from and written to this branch's table and ledger")

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
    finished: Callable[[date], bool | None] | None = None,
) -> ForesightRuntimeComponents | None:
    """组装预测层；未启用时返回 None。``branch`` 是它接的那一支：账本取这一支的目录。``calendar`` 是组合根建好、与语义树共用的
    那一份（不给就自己建）。``kind_label`` 是类编号 → 给人看的类名（词表），出口认提问时拿它配题目；``finished`` 是换代的完成标记
    （``GenerationMarks.finished``），结算据它认没有这一代的那天（裁定 94 第 9 条）。"""

    if not config.foresight.enabled:
        return None
    read_side = ForesightReadSide(
        behavior_tree=behavior_tree,
        subject=config.behavior.primary_subject,
        calendar=calendar if calendar is not None else local_calendar(config),
        zone=config.locale.zone(),
    )
    limits = config.foresight.store
    ledger = ForesightLedgerStore(config.foresight_root / branch, limits=limits)
    generations = 1 + config.lifecycle.fallback_generations  # 派生物留当前一代与几代备用（26 号清单 6.12）
    opportunities = OpportunityStore(config.foresight_root / branch, retained_nights=generations, limits=limits)
    history_store = HistoryStore(config.foresight_root / branch, retained_nights=generations, limits=limits)
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
        finished=finished,
    )
    history = HistoryStage(
        ledger,
        history_store,
        evidence=evidence,
        statuses=relation_statuses(config.foresight.shadow_uses_unestablished),
    )
    foresight = config.foresight
    outlet = ReminderOutlet(
        OutletStore(config.foresight_root / branch / "outlet", limits=limits),
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
        watermark=TickWatermarkStore(config.foresight_root / branch),
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
    ``runtime.scene_config.relation_config`` 换算）、写进这一支的机会表存储；同一拍重标这一支的单元历史（裁定 55 新版第 2 条）。
    交给 ``runtime.generation.Generation`` 的 ``foresight``。"""

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
    trace: Callable[[date], TraceSink] | None = None,
    unwatched: Callable[[datetime, datetime], object] | None = None,
    lease: Lease | None = None,
) -> ForesightTick:
    """预测拍：读这一支的机会表、今天的日历条件、行为树今天与前 7 天、还没封口的判断（``unsealed``），把此刻对得上历史的预期写进
    这一支的账本；再把到了时窗的按账的级决定说不说（``structured_chat`` 给了就带模型四问，不给就按算法与模板）。``shadow_uses_unestablished``
    开着时候选关系也进影子账；词表迁移中整拍不做；``trace`` 给了每拍一个 ``foresight.tick`` span。送达走提醒出口（方案 32）：最近一段有宿主
    来取过才写话、记「待取」，否则记「不可达」；每拍末尾把可取的请求清单交给出口（装了插件的自动"接通"，没装的一分钱不多花）。
    往回看读落盘的水位，没在看的那一段交 ``unwatched`` 写成「未观测」，整拍持预测根租约（``lease``）。"""

    foresight = config.foresight
    delivery = DeliveryStage(
        components.ledger,
        tolerance=_tolerance(config),
        evidence=evidence,
        daily_cap=foresight.daily_reminder_cap,
        unit_prior_share=foresight.unit_prior_share,
        history=components.history.store,
        judge=None if structured_chat is None else _judge(structured_chat, config),
        tick_seconds=foresight.tick_interval_seconds,
        outlet=components.outlet,
        claimed=components.outlet.store.claimed if components.outlet is not None else lambda request_id: False,
        # 与结算同一个"编号现在对应哪些编号"：开口前核前因、判做没做两边认的是同一件事
        current=components.settlement.current,
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
        tolerance=_tolerance(config),
        reread=reread,
        trace=trace,
        track=_track(components, config, unwatched),
        lease=lease,
        finished=components.settlement.finished,  # 白天那道闸：今天这一代完成了才建（完成标记，与结算同一个判断）
    )


def build_foresight_days(
    components: ForesightRuntimeComponents,
    *,
    config: HabitusConfig,
    unsealed: UnsealedReader,
    evidence: Evidence,
    vocabulary_version: Callable[[], int] | None = None,
    vocabulary_migrating: Callable[[], bool] | None = None,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
    lane: str = "session",
) -> DayExpectationStage:
    """当天的日子预期：换代写完这一代的机会表与单元历史之后，用这张含昨天的表建（裁定 94 第 1 条）；读材料与预测拍同一套
    （行为树今天与前 7 天、还没封口的判断 ``unsealed``）。交给 ``runtime.generation.Generation`` 的 ``days``。"""

    return DayExpectationStage(
        components.ledger,
        components.opportunities,
        components.read_side,
        unsealed,
        lane=lane,
        recent=evidence.window,
        tolerance=_tolerance(config),
        shadow_unestablished=config.foresight.shadow_uses_unestablished,
        vocabulary_version=vocabulary_version,
        migrating=vocabulary_migrating,
        observer=observer,
        clock=clock,
    )


def with_slot_worker(
    components: ForesightRuntimeComponents | None, worker: ResidentWorker | None
) -> ForesightRuntimeComponents | None:
    """把每槽先看一眼换代、再做预测拍的那个常驻 worker 接进预测层的部件（没开预测层就没有这份部件，原样返回）。"""

    return components if components is None or worker is None else replace(components, worker=worker)


def foresight_lease_key(config: HabitusConfig, branch: str = "session") -> str:
    """预测根租约的键：由这一支的预测根派生。"""

    digest = hashlib.sha256(str((config.foresight_root / branch).expanduser().absolute()).encode("utf-8")).hexdigest()
    return f"foresight:root:{digest[:24]}"


def foresight_ledger_lease(config: HabitusConfig, lock_store: LockStore, branch: str = "session") -> Lease:
    """预测根租约：预测 worker 每槽持一次，在它里面先看一眼换代（要就做今天这一代）、再做预测拍——账本只有一个写者，两个
    进程挂同一个根时同一天只换代一次。租期与续租间隔是三把锁共用的那两个数（``scene.generation_lease_*``）。"""

    return Lease(
        lock_store,
        foresight_lease_key(config, branch),
        ttl_seconds=config.scene.generation_lease_ttl_seconds,
        heartbeat_seconds=config.scene.generation_heartbeat_seconds,
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


def _judge(structured_chat: StructuredChatClient, config: HabitusConfig) -> Judge:
    foresight = config.foresight
    return Judge(
        structured_chat,
        retries=foresight.judge_transient_retries,
        delay_seconds=foresight.judge_transient_retry_delay_seconds,
    )


def _track(
    components: ForesightRuntimeComponents,
    config: HabitusConfig,
    unwatched: Callable[[datetime, datetime], object] | None,
) -> TickTrack | None:
    if components.watermark is None:
        return None
    slot = timedelta(seconds=config.foresight.tick_interval_seconds)
    return TickTrack(
        components.watermark, slot=slot, tolerance_slots=config.foresight.unwatched_tolerance_slots, unwatched=unwatched
    )


def _tolerance(config: HabitusConfig) -> Tolerance:
    foresight = config.foresight
    return Tolerance(line=1.0 - foresight.false_alarm_rate)


def unsealed_bridge(branch: BehaviorRuntimeComponents) -> UnsealedFromJudgements:
    """预测层那座未封口的桥，取自这一支：判断存储 + 消费账本 + 归约的白天归类。"""

    runner = branch.reduction_runner
    return UnsealedFromJudgements(branch.judgements, runner.ledger, runner.kinds)


__all__ = [
    "ForesightReadSide",
    "ForesightRuntimeComponents",
    "build_foresight_components",
    "build_foresight_days",
    "build_foresight_nightly",
    "build_foresight_tick",
    "build_outlet_consumer",
    "foresight_lease_key",
    "foresight_ledger_lease",
    "build_reminded_lookup",
    "build_reminder_responses",
    "unsealed_bridge",
    "with_slot_worker",
]
