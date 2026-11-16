"""预测层在组合根的组装（``19-预测层方案.md`` 第九节，按步接）。

现在接上的：账本（``ForesightLedgerStore``）与结算这一拍（``SettlementStage``，换代里的一步）；机会表的存储
（``OpportunityStore``）与算它的那一拍（``ForesightNightlyRun``，由 ``build_foresight_nightly`` 装、挂在
``runtime.generation.Generation`` 的语义树之后，同一拍重标单元的历史 ``HistoryStage``，换代由预测 worker 每槽先看一眼、在连续
定稿到昨天之后做）；当天的日子预期（``DayExpectationStage``，由 ``build_foresight_days`` 装，换代必须成的一步）；
预测拍（``ForesightTick``，由 ``build_foresight_tick`` 装：事件与情境的触发、候选、前因清单，落账；到了时窗按账的级决定
说不说；落盘的水位、没在看的时段、预测根租约）；常驻 worker（``ForesightWorker``，``runtime.scene.build_workers`` 装：两支共用一个，
拍点对齐槽边界，每槽按支依次先看一眼这一支的换代、再做这一支的一拍；Runtime 起停它）。读账的两个借用值（α、遗忘半衰期）由组合根从
关系检验与 ``config.foresight`` 的配置装成 ``Evidence`` 传进来（裁定 49：全系统一个"多大把握"、一种"过去多快淡掉"）。

三件只有组合根知道的事定在这里：此刻是主体本地时刻（时区来自 ``config.locale``）；两支各一份预测层（裁定 42：两条独立管线），
账本、机会表、单元历史、水位、出口的传输状态在 ``storage.root/foresight/<lane>``，预测根租约由它派生；提醒出口只有一个（宿主里的
会话），两支的请求都从它取、各带是哪一支，账按支记（``build_outlet``）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.tree import BehaviorTree
from habitus.conditions import DayTypeCalendar
from habitus.config import HabitusConfig
from habitus.foresight import Evidence, Tolerance, UnsealedReader
from habitus.foresight.hierarchy import KindHierarchyPort
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
from habitus.runtime.foresight_outlet_ledgers import OUTLET_DIRECTORY, BranchOutlet
from habitus.runtime.foresight_outlet_store import OutletStore
from habitus.runtime.foresight_reminded import RemindedLookup
from habitus.runtime.foresight_responses import BranchReminders, ReminderResponses
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.foresight_tick import ForesightTick
from habitus.runtime.generation_lease import Lease
from habitus.runtime.generation_watermark import TickTrack, TickWatermarkStore
from habitus.runtime.scene_vocabulary import kind_hierarchy, kind_label, kind_resolution
from habitus.runtime.session_lane import BehaviorSessionConsumer
from habitus.runtime.unsealed import UnsealedFromJudgements
from habitus.scene.relations import RelationConfig
from habitus.scene.relations.store import RelationStore
from habitus.series.model import KindResolution


@dataclass(frozen=True)
class ForesightReadSide:
    """预测层读行为树的那一侧：哪棵树、谁的、按什么日历、什么时区，树上的 ``kind_token`` 怎么换算（删掉的节点换成最近的
    活着的上级，与事件序列同一个口；不给是没有删掉的节点）。"""

    behavior_tree: BehaviorTree
    subject: str
    calendar: DayTypeCalendar
    zone: ZoneInfo
    kinds: KindResolution | None = None
    #: 词表的上下级与在用的词（组合根从词表状态装，每拍开头取一份；不给是没有上下级、全都在用）。
    hierarchy: KindHierarchyPort | None = None

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise ValueError("subject must be non-empty text")


@dataclass(frozen=True)
class ForesightRuntimeComponents:
    """一支预测层的全部已组装部件。``history`` 是换代里重标单元历史的那一步（与它的落盘）；``watermark`` 是预测拍的水位（换代据它
    判"没在看的空白落树了没有"）。驱动它的预测 worker 两支共用一个，在 ``RuntimeComponents.slot_worker``。"""

    #: 这是哪一支（裁定 42：两支各一份，不共用任何东西）。
    lane: str
    read_side: ForesightReadSide
    ledger: ForesightLedgerStore
    settlement: SettlementStage
    opportunities: OpportunityStore
    history: HistoryStage
    #: 换代里的账本压实（26 号清单 4.8）；不给就不压。
    compaction: LedgerCompaction | None = None
    #: 提醒出口（方案 32）：宿主来取、从转写认提问写「说了」；递送那一拍按它的活性决定问不问模型。两支共用这一个，账按支记。
    outlet: ReminderOutlet | None = None
    #: 换代里建当天日子预期的那一步（必须成）；与 ``worker`` 一起在未封口的桥装好之后接上。
    days: DayExpectationStage | None = None
    #: 预测拍的水位（``ticked_at``、``unwatched_until``、``formed_through``、``formed_generation``）。
    watermark: TickWatermarkStore | None = None

    def __post_init__(self) -> None:
        if self.compaction is not None and self.compaction.ledger is not self.ledger:
            raise ValueError("the compaction must use the assembled ledger")
        if self.outlet is not None and self.outlet.ledgers.get(self.lane) is not self.ledger:
            raise ValueError("the reminder outlet must write this branch's assembled ledger")
        if self.settlement.lane != self.lane:
            raise ValueError("the settlement stage must settle this branch")
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


def foresight_ledger(config: HabitusConfig, lane: str) -> ForesightLedgerStore:
    """一支的账本（``<foresight_root>/<lane>``）。组合根先建两支的账本，再建共用的出口、各支的预测层。"""

    return ForesightLedgerStore(config.foresight_root / lane, limits=config.foresight.store)


def build_outlet(
    config: HabitusConfig,
    ledgers: Mapping[str, ForesightLedgerStore],
    *,
    kind_stores: Mapping[str, BehaviorKindStore] | None = None,
    question_tools: Sequence[QuestionTool] | None = None,
    observer: Observer | None = None,
) -> ReminderOutlet:
    """提醒出口（方案 32）：两支共用一个（出口只有宿主里的会话，丙二 D4），传输状态按支存在 ``<foresight_root>/<lane>/outlet``；
    每条请求带是哪一支，「说了」与回话写回那一支的账本，类名按那一支的词表（``kind_stores``）取。"""

    foresight = config.foresight
    return ReminderOutlet(
        {
            lane: OutletStore(config.foresight_root / lane / OUTLET_DIRECTORY, limits=foresight.store)
            for lane in ledgers
        },
        ledgers,
        liveness=timedelta(hours=foresight.outlet_liveness_hours),
        lease=timedelta(minutes=foresight.outlet_lease_minutes),
        max_attempts=foresight.outlet_max_attempts,
        claim_limit=foresight.outlet_claim_limit,
        similarity=foresight.outlet_match_similarity,
        kind_labels=None if kind_stores is None else {lane: kind_label(store) for lane, store in kind_stores.items()},
        tools=question_tools,
        observer=observer,
    )


def build_foresight_components(
    config: HabitusConfig,
    *,
    lane: str,
    ledger: ForesightLedgerStore,
    behavior_tree: BehaviorTree,
    closed_days: Callable[[], Iterable[date]],
    evidence: Evidence,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
    calendar: DayTypeCalendar | None = None,
    kind_store: BehaviorKindStore | None = None,
    outlet: ReminderOutlet | None = None,
    finished: Callable[[date], bool | None] | None = None,
) -> ForesightRuntimeComponents | None:
    """组装一支的预测层；未启用时返回 None。``lane``：这一支（目录 ``<foresight_root>/<lane>``）；``ledger``：这一支的账本
    （``foresight_ledger``）；``calendar``：与语义树共用的那一份（不给自己建）；``kind_store``：这一支的词表状态（编号换算）；
    ``outlet``：两支共用的提醒出口（``build_outlet``）；``finished``：这一支的换代完成标记，结算据它认没有这一代的那天（裁定 94）。"""

    if not config.foresight.enabled:
        return None
    root = config.foresight_root / lane
    if ledger.root != root:
        raise ValueError(f"the {lane} foresight layer keeps its ledger at {root}")
    read_side = ForesightReadSide(
        behavior_tree=behavior_tree,
        subject=config.behavior.primary_subject,
        calendar=calendar if calendar is not None else local_calendar(config),
        zone=config.locale.zone(),
        kinds=None if kind_store is None else kind_resolution(kind_store),
        hierarchy=None if kind_store is None else kind_hierarchy(kind_store),
    )
    limits = config.foresight.store
    generations = 1 + config.lifecycle.fallback_generations  # 派生物留当前一代与几代备用（26 号清单 6.12）
    opportunities = OpportunityStore(root, retained_nights=generations, limits=limits)
    history_store = HistoryStore(root, retained_nights=generations, limits=limits)
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=closed_days,  # 定稿日只有归约说了算
        evidence=evidence,
        opportunities=opportunities,
        lane=lane,
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
    return ForesightRuntimeComponents(
        lane=lane,
        read_side=read_side,
        ledger=ledger,
        settlement=settlement,
        opportunities=opportunities,
        history=history,
        compaction=LedgerCompaction(ledger, evidence=evidence, lane=lane, clock=clock),
        outlet=outlet,
        watermark=TickWatermarkStore(root),
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
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
    reread: Callable[[datetime, datetime], Awaitable[bool]] | None = None,
    trace: Callable[[date], TraceSink] | None = None,
    unwatched: Callable[[datetime, datetime], object] | None = None,
    lease: Lease | None = None,
) -> ForesightTick:
    """预测拍：读这一支的机会表、今天的日历条件、行为树今天与前 7 天、还没封口的判断（``unsealed``），把此刻对得上历史的预期写进
    这一支的账本；再把到了时窗的按账的级决定说不说（``structured_chat`` 给了就带模型四问，不给就按算法与模板）。``shadow_uses_unestablished``
    开着时候选关系也进影子账；``trace`` 给了每拍一个 ``foresight.tick`` span。送达走提醒出口（方案 32）：最近一段有宿主
    来取过才写话、记「待取」，否则记「不可达」；每拍末尾把这一支可取的请求交给出口（只换这一支的那几行，另一支的照留）。
    往回看读落盘的水位，没在看的那一段交 ``unwatched`` 写成「未观测」（只有会话一支给；物理一支在没在看是上游的逻辑，
    裁定 43，只推水位），整拍持这一支的预测根租约（``lease``）。"""

    foresight = config.foresight
    lane = components.lane
    delivery = DeliveryStage(
        components.ledger,
        tolerance=_tolerance(config),
        evidence=evidence,
        daily_cap=foresight.daily_reminder_cap,
        unit_prior_share=foresight.unit_prior_share,
        history=components.history.store,
        judge=None if structured_chat is None else _judge(structured_chat, config),
        tick_seconds=foresight.tick_interval_seconds,
        outlet=None if components.outlet is None else BranchOutlet(components.outlet, lane),
        claimed=components.outlet.store.claimed if components.outlet is not None else lambda request_id: False,
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
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
) -> DayExpectationStage:
    """当天的日子预期：换代写完这一代的机会表与单元历史之后，用这张含昨天的表建（裁定 94 第 1 条）；读材料与预测拍同一套
    （行为树今天与前 7 天、还没封口的判断 ``unsealed``）。交给 ``runtime.generation.Generation`` 的 ``days``。"""

    return DayExpectationStage(
        components.ledger,
        components.opportunities,
        components.read_side,
        unsealed,
        lane=components.lane,
        recent=evidence.window,
        tolerance=_tolerance(config),
        shadow_unestablished=config.foresight.shadow_uses_unestablished,
        vocabulary_version=vocabulary_version,
        observer=observer,
        clock=clock,
    )


def foresight_lease_key(config: HabitusConfig, lane: str) -> str:
    """预测根租约的键：由这一支的预测根派生（两支不同键）。"""

    digest = hashlib.sha256(str((config.foresight_root / lane).expanduser().absolute()).encode("utf-8")).hexdigest()
    return f"foresight:root:{digest[:24]}"


def foresight_ledger_lease(config: HabitusConfig, lock_store: LockStore, lane: str) -> Lease:
    """一支的预测根租约：预测 worker 每槽替这一支持一次，在它里面先看一眼这一支的换代（要就做今天这一代）、再做这一支的预测拍
    ——账本只有一个写者，两个进程挂同一个根时同一支同一天只换代一次。租期与续租间隔是三把锁共用的那两个数（``scene.generation_lease_*``）。"""

    return Lease(
        lock_store,
        foresight_lease_key(config, lane),
        ttl_seconds=config.scene.generation_lease_ttl_seconds,
        heartbeat_seconds=config.scene.generation_heartbeat_seconds,
    )


def build_reminded_lookup(components: ForesightRuntimeComponents) -> RemindedLookup:
    """归约封口时盖 ``reminded`` 的查法（第 0 步）：交给 ``BehaviorReductionRunner.attach_reminder_lookup``。"""

    return RemindedLookup(components.ledger, components.read_side.hierarchy)


def build_outlet_consumer(
    outlet: ReminderOutlet,
    inner: BehaviorSessionConsumer,
    *,
    observer: Observer | None = None,
    host_of: Callable[[str], str | None] | None = None,
) -> OutletObservingConsumer:
    """会话源的行为侧消费者外面包出口那一层（方案 32 第 4.2 条）：同一份会话先过 ``outlet.observe``，再进会话 lane。出口两支共用。"""

    return OutletObservingConsumer(inner, outlet, observer=observer, host_of=host_of)


def build_reminder_responses(
    branches: Sequence[ForesightRuntimeComponents], *, config: HabitusConfig, observer: Observer | None = None
) -> BranchReminders:
    """会话 lane 每轮带开着的提醒、写回回应的那座桥（方案 30 第五节）：交给 ``SessionLane.attach_reminders``。两支的提醒都从
    宿主里的会话说出口，所以会话 lane 看两支的账；回应写回提醒所属的那一支（丙二 D4，共用的是出口不是数据）。"""

    return BranchReminders(
        tuple(
            ReminderResponses(item.ledger, limit=config.foresight.open_reminders_judged, observer=observer)
            for item in branches
        ),
        limit=config.foresight.open_reminders_judged,
    )


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
    """预测层那座未封口的桥，取自这一支：判断存储 + 消费账本 + 归约的选词器 + 这一支词表的换算口。"""

    runner = branch.reduction_runner
    return UnsealedFromJudgements(
        branch.judgements, runner.ledger, runner.kinds, kinds=kind_resolution(branch.kind_store)
    )


__all__ = [
    "ForesightReadSide",
    "ForesightRuntimeComponents",
    "build_foresight_components",
    "build_outlet",
    "foresight_ledger",
    "build_foresight_days",
    "build_foresight_nightly",
    "build_foresight_tick",
    "build_outlet_consumer",
    "foresight_lease_key",
    "foresight_ledger_lease",
    "build_reminded_lookup",
    "build_reminder_responses",
    "unsealed_bridge",
]
