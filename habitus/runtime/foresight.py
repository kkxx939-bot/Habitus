"""预测层在组合根的组装。

预测层站在两棵派生树之上：数字取自预测树的一代，历史卡取自行为树的读时投影加规律树的关联记录，
此刻场景取自行为树的今天加判断存储里还没封口的最近一段。本模块负责把它们接起来，并且**把四件只有
组合根知道的事定死**：

1. **钉住一代**。每次装配从指针取当前一代并核对它的 ``config_digest`` 与现行配置一致——参数变了就是
   另一套统计，混读出来的数字互不一致而且看不出来。不一致时硬拒，等夜批按新参数重建。
2. **此刻是本地时刻**。时区来自 ``config.locale``，不用进程默认时区：读侧的每个"今天是哪一天、
   此刻在哪个槽"都从它的本地时分算，错一个时区就整体错一天。传 naive 的时刻进来同样硬拒。
3. **一次查询一份缓存**。``DayIndexCache`` 的契约是"在它的生命周期里同一天只读一次"，所以每次
   装配新建一个；跨次复用会读到过期的今天。
4. **关联记录与"已关联"用同一把尺子**。两边都从同一个事实源（``AssociationLedger``）取版本：它既回答
   "这个候选哪几天关联完成了"，也说"按哪个版本算的"，没有第二个旋钮。

未封口的判断经 ``UnsealedReader`` 注入；生产实现是 ``runtime.unsealed.UnsealedFromJudgements``（判断存储 +
消费账本 + 词表），没注入时用显式的空实现，明说"最近一小时没补"。

节奏：``ForesightWorker`` 每个槽一拍（对齐到槽边界，槽宽就是树的 ``slot_minutes``），每拍经
``JudgementRunner`` 装配 → 判断。同一槽内此刻场景没变（``NowScene.fingerprint`` 相同）就复用上一次的
判断，不再调模型。本轮判断**只返回与记录**（观测事件 + ``runner.last``），不落盘、不结算、不开口。
本模块不碰 memory。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.reduction.ledger import BehaviorReductionLedger
from habitus.behavior.tree import BehaviorTree
from habitus.config import HabitusConfig
from habitus.foresight import EvidencePack, ForesightError, NoUnsealed, UnsealedReader, assemble, moment_at
from habitus.foresight.judge import Judge, JudgeConfig, Judgement, LLMJudge
from habitus.foundation.observability import ObservationStatus, Observer
from habitus.model_client import StructuredChatClient
from habitus.prediction import builder
from habitus.prediction.config import PredictionTreeConfig
from habitus.prediction.errors import PredictionTreeStoreError
from habitus.prediction.model import PredictionTree
from habitus.prediction.store import PredictionTreeStore, PublishedGeneration
from habitus.runtime.behavior import BehaviorRuntimeComponents
from habitus.runtime.prediction import PredictionRuntimeComponents
from habitus.runtime.resident import ResidentWorker
from habitus.runtime.unsealed import UnsealedFromJudgements
from habitus.scene import AssociationLedger, DayTypeCalendar, NominalCalendar
from habitus.scene.regularity import RegularityTree
from habitus.scene.views import DayIndexCache, association_glosses, situations_of


@dataclass(frozen=True)
class ForesightRuntimeComponents:
    """预测层的全部已组装部件：装配器、判断的执行者、每槽一拍的 worker。"""

    assembler: EvidenceAssembler
    runner: JudgementRunner
    worker: ForesightWorker

    def __post_init__(self) -> None:
        if self.runner.assembler is not self.assembler:
            raise ValueError("the foresight runner must drive the assembled evidence assembler")
        if self.worker.runner is not self.runner:
            raise ValueError("the foresight worker must drive the assembled runner")
        if self.worker.zone is not self.assembler.zone:
            raise ValueError("the foresight worker must tick in the assembler's time zone")

    def assert_attached_to(
        self,
        *,
        behavior: BehaviorRuntimeComponents,
        prediction: PredictionRuntimeComponents,
        structured_chat: StructuredChatClient,
    ) -> None:
        """实例同一性：预测层必须读**这个** Runtime 的三份存储、经这个 Runtime 的模型客户端判断、按这棵树的参数取窗。

        换一份同路径的实例，数字与背景会来自两批不同的日子、"封没封口"会读成另一个口径，而且看不出来。
        用 ``is`` 比对象，用鸭子取属性：这里只该认"它有没有读对那份东西"，不该认具体类。
        """

        assembler = self.assembler
        if assembler.behavior_tree is not behavior.tree:
            raise ValueError("foresight must read the assembled behaviour tree")
        if behavior.regularity_tree is None or assembler.regularity_tree is not behavior.regularity_tree:
            raise ValueError("foresight must read association records from the assembled regularity tree")
        if getattr(assembler.associated, "tree", None) is not behavior.regularity_tree:
            raise ValueError("foresight must read the assembled regularity tree")
        refresher = behavior.association_refresher
        if refresher is None or assembler.associated.version != refresher.associator.version:
            raise ValueError("foresight must judge association by the assembled associator's version")
        if assembler.store is not prediction.store:
            raise ValueError("foresight must read the assembled prediction store")
        unsealed = assembler.unsealed
        if getattr(unsealed, "judgements", None) is not behavior.judgements:
            raise ValueError("foresight must read unsealed judgements from the assembled judgement store")
        if getattr(unsealed, "ledger", None) is not behavior.reduction_runner.ledger:
            raise ValueError("foresight must read the assembled reduction ledger")
        if getattr(unsealed, "kinds", None) is not behavior.kind_store:
            raise ValueError("foresight must read the assembled kind store")
        if getattr(self.runner.judge, "client", None) is not structured_chat:
            raise ValueError("foresight must judge through the assembled structured chat client")
        # 窗口只有一处出处：邻域宽度、转移窗、指纹、槽宽、时区都是那棵树与那个 Runtime 的。
        tree_config = prediction.tree_config
        if assembler.half_width != tree_config.pool_half_width:
            raise ValueError("foresight must take its neighbourhood width from the prediction tree")
        if assembler.transition_window_seconds != tree_config.transition_window_seconds:
            raise ValueError("foresight must take its transition window from the prediction tree")
        if assembler.expected_digest != builder.config_digest(tree_config):
            raise ValueError("foresight must pin the generation built with the assembled tree parameters")
        if self.worker.slot_minutes != tree_config.slot_minutes:
            raise ValueError("the foresight worker must tick at the prediction tree's slot width")
        if assembler.zone is not prediction.rebuilder.zone:
            raise ValueError("foresight must use the same time zone as the prediction rebuild")


class EvidenceAssembler:
    """按此刻装配证据包：候选、四层数字、历史卡、此刻场景。"""

    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        regularity_tree: RegularityTree,
        associated: AssociationLedger,
        store: PredictionTreeStore,
        subject: str,
        zone: ZoneInfo,
        calendar: DayTypeCalendar,
        expected_digest: str,
        half_width: int,
        window_days: int,
        transition_window_seconds: float,
        max_days_per_layer: int,
        unsealed: UnsealedReader | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if not isinstance(regularity_tree, RegularityTree):
            raise TypeError("regularity_tree must be a RegularityTree")
        if not callable(getattr(associated, "days_for", None)) or not isinstance(
            getattr(associated, "version", None), str
        ):
            raise TypeError("associated must answer days_for(kind) and carry the association version")
        if not isinstance(store, PredictionTreeStore):
            raise TypeError("store must be a PredictionTreeStore")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("subject must be non-empty text")
        # zone 是本类真正在守的那条纪律（"此刻必须是主体本地时刻"），却最容易被漏检：
        # ``astimezone(None)`` 会安安静静地落到**进程默认时区**，产出一个看起来完全正常的
        # Moment，而今天是哪一天、此刻在哪个槽已经错了。
        if not isinstance(zone, ZoneInfo):
            raise TypeError("zone must be a ZoneInfo")
        if not isinstance(calendar, DayTypeCalendar):
            raise TypeError("calendar must implement DayTypeCalendar")
        if not isinstance(expected_digest, str) or not expected_digest:
            raise ValueError("expected_digest must be non-empty text")
        if unsealed is not None and not callable(getattr(unsealed, "rows", None)):
            raise TypeError("unsealed must implement UnsealedReader")
        self.behavior_tree = behavior_tree
        self.regularity_tree = regularity_tree
        self.associated = associated
        self.store = store
        self.subject = subject
        self.zone = zone
        self.calendar = calendar
        self.expected_digest = expected_digest
        self.half_width = half_width
        self.window_days = window_days
        self.transition_window_seconds = transition_window_seconds
        self.max_days_per_layer = max_days_per_layer
        self.unsealed: UnsealedReader = NoUnsealed() if unsealed is None else unsealed
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def assemble(self, *, now: datetime | None = None) -> EvidencePack:
        """装配这一刻的证据包。

        ``now`` 缺省取注入的时钟；无论哪来，都必须带时区，并换算到主体的时区之后再落钟面——传进来的
        可以是 UTC，但落到"今天是哪一天、此刻在哪个槽"之前必须先变成本地时刻。
        """

        published, tree = self._pinned_generation()
        if tree.config_digest != self.expected_digest:
            raise ForesightError(
                "the published prediction generation was built with different estimation "
                "parameters; wait for the nightly rebuild instead of mixing two sets of statistics"
            )
        given = now if now is not None else self._clock()
        if not isinstance(given, datetime) or given.utcoffset() is None:
            raise ForesightError(
                "now must be a timezone-aware datetime; a naive value would be read in the process time zone"
            )
        at = given.astimezone(self.zone)
        cache = DayIndexCache(self.behavior_tree, subject=self.subject, calendar=self.calendar)
        moment = moment_at(at, slot_minutes=tree.slot_minutes, day_note=cache.day(at.date()).day_note)
        # 未封口从今天零点读到此刻：窗口内的进流，更早的只计入"今天做过"（与树上今天的行同一口径）。
        version = self.associated.version
        return assemble(
            tree,
            moment,
            cache,
            generation=published.generation,
            unsealed=self.unsealed.rows(since=at.replace(hour=0, minute=0, second=0, microsecond=0), until=at),
            glosses_for=lambda kind, days: association_glosses(self.regularity_tree, kind, days, version=version),
            situations_for=lambda kind, weekday: situations_of(self.regularity_tree, kind, weekday=weekday),
            associated=self.associated.days_for,
            half_width=self.half_width,
            window_days=self.window_days,
            transition_window_seconds=self.transition_window_seconds,
            max_days_per_layer=self.max_days_per_layer,
        )

    def _pinned_generation(self) -> tuple[PublishedGeneration, PredictionTree]:
        try:
            published = self.store.active()
            if published is None:
                # 还没有发布过任何一代。这不是"什么都不会发生"，是"还没算过"——必须能分清。
                raise ForesightError("no prediction generation has been published yet")
            tree = self.store.load_generation(published.generation, expected_digest=published.digest)
        except PredictionTreeStoreError as exc:
            # 本层把下层的失败归一成自己的类型：调用方只该认识 ForesightError。
            raise ForesightError(f"the published prediction generation cannot be read: {exc}") from exc
        return published, tree


@dataclass(frozen=True)
class JudgementRun:
    """一拍的结果：读的那一包、判断、以及这次是不是复用了上一拍（没调模型）。"""

    pack: EvidencePack
    judgement: Judgement
    reused: bool


class JudgementRunner:
    """一拍：装配 → 判断。同一槽内此刻场景没变就复用上一次的判断。

    复用的判据是 ``(一代, 日期, 槽, 场景指纹, 各候选的卡数与未关联日)`` 相同：同一代同一槽候选集合不变，
    场景指纹不含此刻的时分——钟在走不算场景在变；夜批关联在槽中间做完了（卡上多了记录）算材料变了。
    ``last`` 是最近一拍，供健康面与后续步骤读；复用时 ``last.judgement.moment`` 是上一次判断那一刻，
    ``last.pack.moment`` 才是这一拍的此刻。
    """

    def __init__(
        self, assembler: EvidenceAssembler, judge: Judge, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        if not isinstance(assembler, EvidenceAssembler):
            raise TypeError("assembler must be an EvidenceAssembler")
        if not callable(getattr(judge, "judge", None)) or not isinstance(getattr(judge, "version", None), str):
            raise TypeError("judge must implement Judge")
        self.assembler = assembler
        self.judge = judge
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self.last: JudgementRun | None = None
        self._last_key: tuple[object, ...] | None = None

    async def run_once(self, *, now: datetime | None = None) -> JudgementRun:
        at = now if now is not None else self._clock()
        pack = await asyncio.to_thread(self.assembler.assemble, now=at)
        key = _reuse_key(pack)
        previous = self.last
        if previous is not None and key == self._last_key:
            run = JudgementRun(pack=pack, judgement=previous.judgement, reused=True)
        else:
            run = JudgementRun(pack=pack, judgement=await self.judge.judge(pack), reused=False)
        self.last = run
        self._last_key = key
        return run


def _reuse_key(pack: EvidencePack) -> tuple[object, ...]:
    return (
        pack.generation,
        pack.moment.day.isoformat(),
        pack.moment.slot,
        pack.now.fingerprint,
        tuple((item.kind_token, len(item.background.cards), item.unassociated) for item in pack.expanded),
    )


class ForesightWorker(ResidentWorker):
    """每个槽一拍：起来先判一次，之后对齐到主体本地时区的槽边界。"""

    _task_name = "habitus-foresight-judge"
    _observation_category = "foresight"

    def __init__(
        self,
        runner: JudgementRunner,
        *,
        slot_minutes: int,
        zone: ZoneInfo,
        shutdown_timeout_seconds: float,
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        if not isinstance(runner, JudgementRunner):
            raise TypeError("runner must be a JudgementRunner")
        if isinstance(slot_minutes, bool) or not isinstance(slot_minutes, int) or not 0 < slot_minutes <= 1440:
            raise ValueError("slot_minutes must be a positive integer of at most 1440")
        if not isinstance(zone, ZoneInfo):
            raise TypeError("zone must be a ZoneInfo")
        self.runner = runner
        self.slot_minutes = slot_minutes
        self.zone = zone
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    async def run_once(self) -> JudgementRun:
        if self.running:
            raise RuntimeError("manual run_once cannot race the resident worker loop")
        return await self._tick()

    def seconds_until_next_slot(self, *, at: datetime | None = None) -> float:
        """从 ``at``（缺省此刻）到下一个槽边界还有几秒（主体本地时区，槽从当天 00:00 起数）。正好在边界上算下一个。"""

        local = (at if at is not None else self._clock()).astimezone(self.zone)
        midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
        slot_seconds = self.slot_minutes * 60
        elapsed = (local - midnight).total_seconds() % slot_seconds
        return slot_seconds - elapsed

    async def _tick(self) -> JudgementRun:
        started = time.monotonic()
        try:
            run = await self.runner.run_once()
        except Exception as exc:
            self.last_error = exc
            self._observe("judgement", ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started=started)
            raise
        self._succeeded()
        self._observe("judgement", ObservationStatus.SUCCESS, _run_attributes(run), started=started)
        return run

    async def _run_loop(self) -> None:
        while not self._stop_requested.is_set():
            # 下一拍的边界从**这一拍开始**的时刻算：一拍跑过了槽边界，下一槽立刻判，不跳过它。
            tick_started = self._clock()
            try:
                await self._tick()
            except Exception:  # noqa: BLE001 - 一拍失败只留 last_error 与观测，下一槽照常
                pass
            spent = (self._clock() - tick_started).total_seconds()
            await self._wait(max(0.0, self.seconds_until_next_slot(at=tick_started) - spent))


def _run_attributes(run: JudgementRun) -> dict[str, str | int | float | bool]:
    judgement = run.judgement
    return {
        "reused": run.reused,
        "generation": run.pack.generation,
        "slot": run.pack.moment.slot,
        "candidates": len(run.pack.candidates),
        "expanded": len(run.pack.expanded),
        "expected": len(judgement.expected),
        "day_state": judgement.day_state or "未答",
        "signals": len(judgement.signals),
    }


def build_foresight_components(
    config: HabitusConfig,
    *,
    behavior_tree: BehaviorTree,
    regularity_tree: RegularityTree | None,
    associated: AssociationLedger | None,
    store: PredictionTreeStore | None,
    judge: Judge | None = None,
    structured_chat: StructuredChatClient | None = None,
    judgements: BehaviorJudgementStore | None = None,
    ledger: BehaviorReductionLedger | None = None,
    kinds: BehaviorKindStore | None = None,
    unsealed: UnsealedReader | None = None,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
) -> ForesightRuntimeComponents | None:
    """组装预测层；未启用时返回 None。

    两棵派生树缺任何一棵都返回 None 而不是半个装配器：跨域校验已经在配置层拒过这种组合，
    这里是第二道——组合根拿到的部件本来就可能因为上游未启用而是 None。

    判断者二选一：注入一个 ``judge``（测试、DAY1 脚本），或给 ``structured_chat`` 由这里装
    ``LLMJudge``。未封口的读法同理：给判断存储 + 消费账本 + 词表三份就装生产实现，或直接注入
    一个 ``unsealed``；都不给就是显式的空实现。
    """

    if not config.foresight.enabled:
        return None
    if regularity_tree is None or associated is None or store is None:
        return None
    resolved_judge = _judge(config, judge=judge, structured_chat=structured_chat, clock=clock)
    resolved_unsealed = _unsealed(unsealed=unsealed, judgements=judgements, ledger=ledger, kinds=kinds)
    tree_config = PredictionTreeConfig(**config.prediction.tree_parameters())
    zone = config.locale.zone()
    assembler = EvidenceAssembler(
        behavior_tree=behavior_tree,
        regularity_tree=regularity_tree,
        associated=associated,
        store=store,
        subject=config.behavior.primary_subject,
        zone=zone,
        calendar=_calendar(config),
        # 指纹在这里算一次：读侧每次装配都拿它核对钉住的那一代，防的是跨参数混读。
        expected_digest=builder.config_digest(tree_config),
        # 窗口就是树的池化邻域与转移窗，不另配一份——配两份会让数字与背景取自不同的范围。
        half_width=tree_config.pool_half_width,
        transition_window_seconds=tree_config.transition_window_seconds,
        window_days=config.foresight.window_days,
        max_days_per_layer=config.foresight.max_days_per_layer,
        unsealed=resolved_unsealed,
        clock=clock,
    )
    runner = JudgementRunner(assembler, resolved_judge, clock=clock)
    worker = ForesightWorker(
        runner,
        # 节奏就是树的槽宽：一槽一拍，与"此刻在哪个槽"用同一个数。
        slot_minutes=tree_config.slot_minutes,
        zone=zone,
        shutdown_timeout_seconds=config.foresight.worker_shutdown_timeout_seconds,
        observer=observer,
        clock=clock,
    )
    return ForesightRuntimeComponents(assembler=assembler, runner=runner, worker=worker)


def _judge(
    config: HabitusConfig,
    *,
    judge: Judge | None,
    structured_chat: StructuredChatClient | None,
    clock: Callable[[], datetime] | None,
) -> Judge:
    """二选一，两个都给是接线错误——静默偏向其中一个会让另一份被以为在用。"""

    if judge is not None and structured_chat is not None:
        raise ValueError("pass either an explicit judge or a structured_chat to build one, not both")
    if judge is not None:
        return judge
    if structured_chat is None:
        raise ValueError("foresight needs a judge: pass structured_chat or an explicit judge")
    return LLMJudge(
        structured_chat,
        config=JudgeConfig(
            transient_retries=config.foresight.judge_transient_retries,
            transient_retry_delay_seconds=config.foresight.judge_transient_retry_delay_seconds,
        ),
        clock=clock,
    )


def _unsealed(
    *,
    unsealed: UnsealedReader | None,
    judgements: BehaviorJudgementStore | None,
    ledger: BehaviorReductionLedger | None,
    kinds: BehaviorKindStore | None,
) -> UnsealedReader | None:
    """注入一个读口，或给三份存储装生产实现；都不给返回 None（装配器用显式的空实现）。"""

    stores = (judgements, ledger, kinds)
    if unsealed is not None and any(item is not None for item in stores):
        raise ValueError("pass either an explicit unsealed reader or the judgement/ledger/kind stores, not both")
    if unsealed is not None:
        return unsealed
    if all(item is None for item in stores):
        return None
    if judgements is None or ledger is None or kinds is None:
        raise ValueError(
            "reading unsealed judgements needs the judgement store, the reduction ledger and the kind store"
        )
    return UnsealedFromJudgements(judgements, ledger, kinds)


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
    "ForesightWorker",
    "JudgementRun",
    "JudgementRunner",
    "UnsealedFromJudgements",
    "build_foresight_components",
]
