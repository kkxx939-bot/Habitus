"""把尚未处理的观测切段并登记成融合作业。

排队与执行分开：本模块只做确定性的"哪些观测还没人管、它们该怎么分组"，不调模型、不写行为树。

## 已处理的判定走作业而不是行为树

作业身份就是它未来那份回执的身份（``receipt_identity(片段, 实现版本, 提示词版本)``），所以
"这段观测处理过没有"是一次文件存在性判断。不能反过来去数产出了多少判断——一段观测完全读不懂时
只会留下一条 ``behavior`` 为空的判断，按产物反推会把"读不懂"与"没处理过"混为一谈。

## 尾部要留白，而且判据要看"到达"不看"发生"

最后一段观测常常是"行为还在进行中"——上游还会继续推语义过来。立刻把它切成一段送去融合，会把
一次完整行为拦腰截断。所以保留一个静默期。

判据必须建立在**观测何时到达我们**（``available_at``）上，不能建立在**行为何时发生**
（``occurred_at``）上。云侧 agent 重启后补发几小时前的观测是常态，那些观测的 ``occurred_at``
天生就"够老"，静默期一秒也拦不住：实测一次连续十分钟的做饭被逐条补发时，会被切成六个各覆盖一条
观测的作业，六次模型调用、六组互不相认的判断。而按到达时刻判，同一批补发数据会静静躺着直到补发
停止，然后作为完整的一段被处理。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from behavior.fusion.config import BehaviorFusionConfig
from behavior.fusion.coverage import BehaviorCoverageIndex
from behavior.fusion.derivation import FUSION_VERSION
from behavior.fusion.jobs import BehaviorFusionJob, BehaviorFusionJobStore
from behavior.fusion.prompt import FUSION_PROMPT_VERSION
from behavior.fusion.receipt import segment_identity
from behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from behavior.fusion.segmentation import BehaviorFusionSegment, segment_observations
from behavior.observation import BehaviorObservationStore

DEFAULT_QUIET_PERIOD_SECONDS = 300.0


@dataclass(frozen=True)
class BehaviorFusionEnqueueResult:
    """一次排队扫描的结果。"""

    enqueued: tuple[BehaviorFusionJob, ...]
    withheld_observations: int

    @property
    def count(self) -> int:
        return len(self.enqueued)


class BehaviorFusionEnqueuer:
    """扫描观测存储，把还没有作业覆盖的观测切段并登记。"""

    def __init__(
        self,
        observations: BehaviorObservationStore,
        jobs: BehaviorFusionJobStore,
        receipts: BehaviorFusionReceiptStore,
        *,
        config: BehaviorFusionConfig | None = None,
        coverage: BehaviorCoverageIndex | None = None,
        quiet_period_seconds: float = DEFAULT_QUIET_PERIOD_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(observations, BehaviorObservationStore):
            raise TypeError("observations must be BehaviorObservationStore")
        if not isinstance(jobs, BehaviorFusionJobStore):
            raise TypeError("jobs must be BehaviorFusionJobStore")
        if not isinstance(receipts, BehaviorFusionReceiptStore):
            raise TypeError("receipts must be BehaviorFusionReceiptStore")
        resolved = config or BehaviorFusionConfig()
        if not isinstance(resolved, BehaviorFusionConfig):
            raise TypeError("config must be BehaviorFusionConfig")
        if isinstance(quiet_period_seconds, bool) or not isinstance(quiet_period_seconds, int | float):
            raise ValueError("quiet_period_seconds must be a non-negative number")
        if quiet_period_seconds < 0:
            raise ValueError("quiet_period_seconds must be a non-negative number")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self.observations = observations
        self.jobs = jobs
        self.receipts = receipts
        self.config = resolved
        if coverage is not None and not isinstance(coverage, BehaviorCoverageIndex):
            raise TypeError("coverage must be BehaviorCoverageIndex")
        self.coverage = coverage or BehaviorCoverageIndex(observations.root)
        self.quiet_period = timedelta(seconds=float(quiet_period_seconds))
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def enqueue_ready(self) -> BehaviorFusionEnqueueResult:
        """把已经收尾且尚未被任何作业覆盖的观测登记成作业。"""

        # 整段扫描圈在同一个栅栏里：读覆盖、读观测、登记三步之间若有别的扫描插进来，两次扫描会
        # 按不同快照切出不同的段，``job_id`` 不同于是幂等失效，同一批观测被融合两次。
        with self.jobs.scan_fence():
            return self._scan()

    def _scan(self) -> BehaviorFusionEnqueueResult:
        now = self._timestamp()
        covered = set(self.jobs.covered_observation_ids(fenced=False))
        # 已融合的观测由覆盖索引回答（有窗口、按日过期），不再全量枚举回执目录。
        covered.update(self.coverage.covered_observation_ids(now))
        envelopes = self.observations.list()
        if not envelopes:
            return BehaviorFusionEnqueueResult((), 0)
        # 送达早于覆盖窗口的观测一律不再入队：它们的覆盖记录可能已按窗口过期，再融合一次只会
        # 产出第二套判断（BHV-REALDATA-001 审计：覆盖过期 → 重入队 → 重复 occurrence）。按契约窗口
        # 之外不再有补发，仍留在存储里的只可能是被隔离判断引用的交付——留给运维，不重跑模型。
        stale_before = now - timedelta(days=self.coverage.window_days)
        for envelope in envelopes:
            for observation in envelope.batch.observations:
                if observation.available_at < stale_before:
                    covered.add(observation.observation_id)
        segments = segment_observations(envelopes, config=self.config, exclude=covered)
        if not segments:
            return BehaviorFusionEnqueueResult((), 0)
        cutoff = self._timestamp() - self.quiet_period
        # 只取**前缀**而不是逐段过滤：作业严格串行，越过一段未收尾的观测去登记它后面的那段，
        # 会让后一段先融合、先落树，而前一段回来时再也接不上上下文。
        ready = 0
        while ready < len(segments) and self._is_settled(segments[ready], cutoff):
            ready += 1
        withheld = sum(len(item.fragments) for item in segments[ready:])
        return BehaviorFusionEnqueueResult(
            tuple(self._enqueue(item) for item in segments[:ready]), withheld
        )

    def _is_settled(self, segment: BehaviorFusionSegment, cutoff: datetime) -> bool:
        """这一段最近一次**收到**新观测是不是已经够久了。

        取 ``available_at`` 的 max 而不是末条片段的 ``occurred_at``：补发的旧观测发生得早、到达
        得晚，按发生时刻判会让它一落盘就被切走。
        """

        return max(item.available_at for item in segment.fragments) <= cutoff

    def _enqueue(self, segment: BehaviorFusionSegment) -> BehaviorFusionJob:
        return self.jobs.enqueue(
            segment_digest=segment_identity(segment.fragments),
            observation_ids=tuple(
                sorted({item.observation_id for item in segment.fragments})
            ),
            source_refs=tuple(segment.source_refs),
            fusion_version=FUSION_VERSION,
            prompt_version=FUSION_PROMPT_VERSION,
        )

    def _timestamp(self) -> datetime:
        now = self.clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise TypeError("clock must return timezone-aware datetimes")
        return now.astimezone(timezone.utc)


__all__ = [
    "DEFAULT_QUIET_PERIOD_SECONDS",
    "BehaviorFusionEnqueueResult",
    "BehaviorFusionEnqueuer",
]
