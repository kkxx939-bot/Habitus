"""时间预测树夜批在组合根的组装。

预测侧是可选启用的：``config.prediction.enabled`` 为假时组合根完全跳过本模块。启用后接成的
是一条极短的链——

    行为树 → 第 N 晚的事件序列（只含 N 之前、已封口的日子）→ 快照 → 整棵重建 → 两阶段发布 → 清理老代

**全程零 LLM、零语义**：本模块不接 ``StructuredChatClient``，也不碰 memory；预测算法与执行层
（含那两个受控的在线模型调用点）都在本层之外，见 ``TODO(PRED-DOWNSTREAM-001)``。

调度目前是最简的定拍循环（``rebuild_interval_seconds``，启动档一天一拍）。真正的"夜间批处理"
应当挑用户睡着的时段跑，那需要作息数据本身——正好是这棵树建成之后才有的东西，所以先定拍，
等树上有数据再谈择时。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from habitus.behavior.tree import BehaviorTree
from habitus.config import HabitusConfig
from habitus.foundation.observability import ObservationStatus, Observer
from habitus.prediction import builder, source
from habitus.prediction.config import PredictionTreeConfig
from habitus.prediction.store import PredictionTreeStore, PublishedGeneration
from habitus.runtime.resident import ResidentWorker
from habitus.series import EventSeries
from habitus.series.reader import read_series


@dataclass(frozen=True)
class PredictionRuntimeComponents:
    """预测夜批的全部已组装部件。"""

    tree_config: PredictionTreeConfig
    store: PredictionTreeStore
    rebuilder: PredictionRebuilder
    worker: PredictionRebuildWorker

    def __post_init__(self) -> None:
        # 实例同一性：Worker 与 rebuilder 必须就是这里的那一份，否则手动触发与定拍循环
        # 会发布到两个不同的存储上，读侧看到的代来自谁全凭运气。
        if self.worker.rebuilder is not self.rebuilder:
            raise ValueError("prediction worker must drive the assembled rebuilder")
        if self.rebuilder.store is not self.store:
            raise ValueError("prediction rebuilder must publish into the assembled store")
        if self.rebuilder.config is not self.tree_config:
            raise ValueError("prediction rebuilder must use the assembled tree parameters")


class PredictionRebuilder:
    """一次全量重建：读第 N 晚的事件序列 → 建树 → 发布。

    **防前视**（语义树新方案第四节）：第 N 晚（N = 主体所在地的"今天"）只用 N 之前的日子建树，衰减的基准日是
    N 的前一天（最后一个封口的日子）。今天还没过完，算进来就是把"今天剩下的时段什么都没做"当成了观测。

    没有增量路径，也不打算有：重建成本低，而增量会让口径漂移（词表变了就要用新口径
    重数历史）。行为树上一条 occurrence 都没有时不发布——发布一棵空树只会让读侧把"还没有
    数据"误当成"什么都不会发生"。
    """

    def __init__(
        self,
        behavior_tree: BehaviorTree,
        store: PredictionTreeStore,
        *,
        config: PredictionTreeConfig,
        zone: ZoneInfo,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if not isinstance(store, PredictionTreeStore):
            raise TypeError("store must be a PredictionTreeStore")
        if not isinstance(config, PredictionTreeConfig):
            raise TypeError("config must be a PredictionTreeConfig")
        if not isinstance(zone, ZoneInfo):
            raise TypeError("zone must be a ZoneInfo")
        self.behavior_tree = behavior_tree
        self.store = store
        self.config = config
        self.zone = zone
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def run_once(self) -> PublishedGeneration | None:
        """同步跑一趟：读今晚的序列再建树。行为树还没有可用记录时返回 None。"""

        return self.build_from(read_series(self.behavior_tree, cutoff=self.tonight()))

    def tonight(self) -> date:
        """第几晚：主体所在地的"今天"（``config.locale.timezone``），不看进程默认时区——差一天，
        整棵树的衰减权重就整体挪一天，而每个数字看起来都还是对的。"""

        return self._clock().astimezone(self.zone).date()

    def build_from(self, series: EventSeries) -> PublishedGeneration | None:
        """用一份已经读好的序列建树发布（回放、回测按日截一份序列交进来，不重读行为树）。

        衰减的基准日取截止日前一天，而不是最后一条记录那天：取后者的话停记一个月之后算出来的还是一个月前的热度。
        """

        snapshot = source.snapshot_of(series)
        if not snapshot.actions:
            # 只有观测空白、一条 occurrence 都没有，是上游刚接入时的正常状态。发布出去的是一棵没有任何动作的树，
            # 读侧对任何动作都会拿到 0.0——正好把"还没有数据"说成"什么都不会发生"。
            return None
        tree = builder.build(snapshot, config=self.config, reference=series.last_day, built_at=self._clock())
        return self.store.publish(tree)


def _stage_attributes(outcome: object) -> dict[str, str | int | float | bool]:
    """把后置阶段的结果（一个 dataclass：结算报告、夜批报告……）摊平成观测属性：数值原样，列表 / 元组记条数。
    不按字段名认（以前认的是旧语义树的字段，新夜批报告挂上去就只剩一句"跑过了"；第四轮评审 E10）。不是 dataclass 的不猜。"""

    if not is_dataclass(outcome) or isinstance(outcome, type):
        return {}
    attributes: dict[str, str | int | float | bool] = {}
    for item in fields(outcome):
        value = getattr(outcome, item.name)
        if isinstance(value, bool | int | float):
            attributes[item.name] = value
        elif isinstance(value, tuple | list):
            attributes[item.name] = len(value)
    return attributes


class PredictionRebuildWorker(ResidentWorker):
    """重建节拍。正确性全在 rebuilder，这里只管节奏与"循环别死"。

    重建期间的同步 IO 会占用事件循环，所以整趟下沉到 ``asyncio.to_thread``——一次全量扫描
    的量级受 BHV-LIFECYCLE-001 的欠账支配，比行为侧的 sweep 大得多。
    """

    _task_name = "habitus-prediction-rebuild"
    _observation_category = "prediction"

    def __init__(
        self,
        rebuilder: PredictionRebuilder,
        *,
        interval_seconds: float,
        shutdown_timeout_seconds: float,
        observer: Observer | None = None,
        after_rebuild: Callable[[], Awaitable[object]] | None = None,
    ) -> None:
        """``after_rebuild`` 是夜批里排在重建**之后**的阶段（组合根注入，本模块不知道它是什么
        ——现状是承诺结算；新语义树的夜批（``runtime.nightly.Nightly``）在第 11 步接常驻 worker 时接进来）。

        排在重建之后，因为后面几拍要向这一代树要对照期望，必须等树落地。该阶段失败只记观测
        事件，不阻断下一轮重建——派生层不做级联。
        """

        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        if after_rebuild is not None and not callable(after_rebuild):
            raise TypeError("after_rebuild must be an async callable or None")
        self.rebuilder = rebuilder
        self.interval_seconds = float(interval_seconds)
        self.after_rebuild = after_rebuild

    async def run_once(self) -> PublishedGeneration | None:
        """手动触发一次夜批：全量重建 → 后置阶段（运维与测试用）。"""

        if self.running:
            raise RuntimeError("manual run_once cannot race the resident worker loop")
        published = await asyncio.to_thread(self.rebuilder.run_once)
        await self._run_after_rebuild()
        return published

    async def _run_after_rebuild(self) -> None:
        """后置阶段失败不阻断重建（派生层不做级联），但**必须留下可查的痕迹**。

        返回值要摊进观测属性：一夜 100% 失败而事件写着 SUCCESS、属性为空，运维就只能靠猜。
        失败还要设 ``last_error``，否则健康检查会一直报这个 worker 健康。
        """

        if self.after_rebuild is None:
            return
        started = time.monotonic()
        try:
            outcome = await self.after_rebuild()
        except Exception as exc:  # noqa: BLE001 - 后置阶段失败不阻断重建，只留观测
            self.last_error = exc
            self._observe(
                "nightly_stage", ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started=started
            )
        else:
            self._observe("nightly_stage", ObservationStatus.SUCCESS, _stage_attributes(outcome), started=started)

    async def _run_loop(self) -> None:
        while not self._stop_requested.is_set():
            started = time.monotonic()
            try:
                published = await asyncio.to_thread(self.rebuilder.run_once)
                await self._run_after_rebuild()
            except Exception as exc:  # noqa: BLE001 - 常驻循环必须活过基础设施抖动
                self.last_error = exc
                self._observe(
                    "tree_rebuild",
                    ObservationStatus.FAILURE,
                    {"error_type": type(exc).__name__},
                    started=started,
                )
            else:
                self._succeeded()
                self._observe(
                    "tree_rebuild",
                    ObservationStatus.SUCCESS,
                    {"published": published.generation if published else "none"},
                    started=started,
                )
            await self._wait(self.interval_seconds)


def build_prediction_components(
    config: HabitusConfig,
    *,
    behavior_tree: BehaviorTree,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
    after_rebuild: Callable[[], Awaitable[object]] | None = None,
) -> PredictionRuntimeComponents | None:
    """组装预测夜批；未启用时返回 None。

    无存储写入、无模型请求——与 ``build_runtime`` 同一纪律。取值范围的校验全部由
    ``PredictionTreeConfig`` 在这里完成，配置层只负责"给没给全"。
    """

    prediction_config = config.prediction
    if not prediction_config.enabled:
        return None
    tree_config = PredictionTreeConfig(**prediction_config.tree_parameters())
    store = PredictionTreeStore(config.prediction_root, retained_generations=tree_config.published_generations)
    rebuilder = PredictionRebuilder(behavior_tree, store, config=tree_config, zone=config.locale.zone(), clock=clock)
    worker = PredictionRebuildWorker(
        rebuilder,
        interval_seconds=tree_config.rebuild_interval_seconds,
        shutdown_timeout_seconds=prediction_config.worker_shutdown_timeout_seconds,
        observer=observer,
        after_rebuild=after_rebuild,
    )
    return PredictionRuntimeComponents(tree_config=tree_config, store=store, rebuilder=rebuilder, worker=worker)


__all__ = [
    "PredictionRebuildWorker",
    "PredictionRebuilder",
    "PredictionRuntimeComponents",
    "build_prediction_components",
]
