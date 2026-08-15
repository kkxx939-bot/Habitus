"""按耐久租约执行最早一个融合作业。

## 执行顺序不能换

    调模型 → 派生判断 → 建回执 → **stage（检查点）** → 落判断 → 落回执 → commit

模型调用在检查点之前，落盘在检查点之后。这个顺序是被"融合不是纯函数"逼出来的：同一段观测重跑
一次，模型换个说法就是另一批判断，落盘之后就是两套互不相认的记录。检查点之后的任何一次崩溃都
只重放落盘，模型恰好调用一次。

## 落盘天然幂等

判断按**内容身份**命名，所以重放写入同一条判断就是写同一个文件名同样的字节——存储层直接复用，
不需要"读回来验明正身"那类判据。这比发布进行为树简单得多，也少了一整类误判的可能。

## 融合不碰行为树

本层只产出判断。判断到行为树文档（事件、结果、长期事件）的归约是**另一层**的事，它需要事后才有
的信息（开空调的结果要等室温变化，长期事件要看完整段），节奏与融合相反。把它塞进融合，融合就
永远完不了。

## 认领与执行分离

``claim`` 与 ``execute`` 分开，好让上层 Worker 在长时间的模型调用期间独立维持租约心跳。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from behavior.fusion.derivation import derive_judgements, judgement_payload
from behavior.fusion.errors import BehaviorFusionError
from behavior.fusion.jobs import (
    BehaviorFusionJob,
    BehaviorFusionJobBlockedError,
    BehaviorFusionJobError,
    BehaviorFusionJobLease,
    BehaviorFusionJobNotReadyError,
    BehaviorFusionJobStore,
    StagedFusion,
)
from behavior.fusion.receipt import BehaviorFusionReceipt, build_fusion_receipt
from behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from behavior.fusion.segmentation import BehaviorFusionSegment
from behavior.fusion.service import BehaviorJudgementFuser
from behavior.fusion.store import BehaviorJudgementStore
from behavior.observation import BehaviorObservation, BehaviorObservationStore


@dataclass(frozen=True)
class BehaviorFusionRunResult:
    """一次作业执行实际产生的结果。"""

    job: BehaviorFusionJob
    receipt: BehaviorFusionReceipt
    judgement_ids: tuple[str, ...]
    fused: bool

    @property
    def replayed(self) -> bool:
        """本次执行只重放了落盘——说明上一次在检查点之后崩过。"""

        return not self.fused


class BehaviorFusionRunner:
    """把一个融合作业从认领跑到 COMMITTED。"""

    def __init__(
        self,
        jobs: BehaviorFusionJobStore,
        observations: BehaviorObservationStore,
        fuser: BehaviorJudgementFuser,
        judgements: BehaviorJudgementStore,
        receipts: BehaviorFusionReceiptStore,
        *,
        primary_subject: str,
        context_limit: int = 8,
        context_lookback_seconds: float = 21_600.0,
    ) -> None:
        for value, expected in (
            (jobs, BehaviorFusionJobStore),
            (observations, BehaviorObservationStore),
            (fuser, BehaviorJudgementFuser),
            (judgements, BehaviorJudgementStore),
            (receipts, BehaviorFusionReceiptStore),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{expected.__name__} is required")
        self.jobs = jobs
        self.observations = observations
        self.fuser = fuser
        self.judgements = judgements
        self.receipts = receipts
        # 个人版：画面里出现别人是常态，但只跟踪主体。谁是主体是**配置事实**，上游保证
        # ``participants`` 用的是跨批次稳定的标识，所以这里逐字比对即可。
        if not isinstance(primary_subject, str) or not primary_subject.strip():
            raise BehaviorFusionError("primary_subject must be non-empty text")
        self.primary_subject = primary_subject.strip()
        # 有界且确定性：按判断成立时刻取尾部，不按相关性排序——否则同一批输入会得到不同上下文，
        # 融合就不可重放。
        self.context_limit = context_limit
        self.context_lookback_seconds = context_lookback_seconds

    def claim(self, worker_id: str) -> BehaviorFusionJobLease | None:
        """认领最早未完成的作业；队列空返回 None，被阻塞则抛出。"""

        oldest = self.jobs.oldest_uncommitted()
        if oldest is None:
            return None
        return self.jobs.claim(oldest, worker_id)

    async def execute(
        self,
        lease: BehaviorFusionJobLease,
        *,
        judged_at: datetime | None = None,
    ) -> BehaviorFusionRunResult:
        """执行一次已认领的作业；任何失败都在同一租约内结算并记录退避。"""

        if not isinstance(lease, BehaviorFusionJobLease):
            raise TypeError("lease must be BehaviorFusionJobLease")
        try:
            fused = lease.job.needs_fusion
            if fused:
                lease = await self._fuse_and_stage(lease, judged_at=judged_at)
            staged = lease.job.staged
            if staged is None:  # pragma: no cover - stage 成功即非空
                raise BehaviorFusionJobError("fusion job reached persistence without a checkpoint")
            self._persist(staged)
            committed = self.jobs.commit(lease)
        except (BehaviorFusionJobBlockedError, BehaviorFusionJobNotReadyError):
            raise
        except Exception as exc:
            self.jobs.fail(lease, exc, retryable=_is_retryable(exc))
            raise
        return BehaviorFusionRunResult(
            job=committed,
            receipt=staged.receipt,
            judgement_ids=tuple(staged.receipt.judgement_ids),
            fused=fused,
        )

    async def _fuse_and_stage(
        self,
        lease: BehaviorFusionJobLease,
        *,
        judged_at: datetime | None,
    ) -> BehaviorFusionJobLease:
        job = lease.job
        segment = BehaviorFusionSegment(self._fragments(job), tuple(job.source_refs))
        context = self._context(segment)
        result = await self.fuser.fuse(
            segment, primary_subject=self.primary_subject, context_judgements=context
        )
        stamped = judged_at or self.jobs.clock()
        derived = derive_judgements(
            result.batch,
            segment.fragments,
            source_refs=segment.source_refs,
            judged_at=stamped,
            context_ids=tuple(item["judgement_id"] for item in context),
        )
        receipt = build_fusion_receipt(
            derived,
            segment.fragments,
            source_refs=segment.source_refs,
            prompt_version=result.prompt_version,
            validation_attempts=result.validation_attempts,
            primary_subject=self.primary_subject,
            judged_at=stamped,
        )
        if receipt.receipt_id != job.job_id:
            # 作业身份由片段与版本派生，回执同理。两者不等只可能是版本在作业排队之后改过，
            # 那份产物属于另一个版本谱系，不能混进这条作业。
            raise BehaviorFusionError(
                "fusion produced a receipt identity that does not match its job; "
                "the fusion or prompt version changed after this job was enqueued"
            )
        # 只落属于主体的判断——不属于的会把判断存储淹没，而下游拿它们毫无用处。它们的观测身份
        # 已经记在回执的 ``out_of_scope_observation_ids`` 里，缺失并没有被解释掉。
        staged = StagedFusion(
            receipt=receipt,
            judgements=tuple(
                judgement_payload(item)
                for item in derived
                if self.primary_subject in item.subjects
            ),
        )
        return self.jobs.stage(lease, staged)

    def _context(self, segment: BehaviorFusionSegment) -> tuple[Mapping[str, Any], ...]:
        """取本段之前已经成立的若干条判断，供模型跨窗口指回。

        截断点是本段**最早观测的可用时刻**：能给模型看的只能是在这段观测进入系统之前就已经成立
        的判断。用更晚的截断点等于把后来才知道的事喂回给更早的一段——那是标签泄漏。
        """

        cutoff = min(item.available_at for item in segment.fragments)
        return self.judgements.recent_before(
            cutoff,
            limit=self.context_limit,
            lookback_seconds=self.context_lookback_seconds,
        )

    def _fragments(self, job: BehaviorFusionJob) -> tuple[BehaviorObservation, ...]:
        """按作业钉住的观测身份重建片段；顺序由发生时刻决定，与交付顺序无关。"""

        wanted = set(job.observation_ids)
        found: dict[str, BehaviorObservation] = {}
        for envelope in self.observations.list():
            for observation in envelope.batch.observations:
                if observation.observation_id in wanted:
                    found.setdefault(observation.observation_id, observation)
        missing = wanted - set(found)
        if missing:
            raise BehaviorFusionError(
                f"fusion job references {len(missing)} observations that are no longer stored"
            )
        return tuple(
            sorted(found.values(), key=lambda item: (item.occurred_at, item.observation_id))
        )

    def _persist(self, staged: StagedFusion) -> None:
        """先落判断再落回执：回执指向判断，反过来会让回执一度指向不存在的东西。"""

        existing = self.receipts.read(staged.receipt.receipt_id)
        if existing is not None and existing.judgement_ids != staged.receipt.judgement_ids:
            # 同一片段同一版本已经有过一次融合，而这次产出了不同的判断。落下去会造出一批
            # 没有任何回执指向的孤儿判断，而回执存储撞车时复用先到的那份，调用方还会拿到一份
            # 磁盘上并不存在的回执。作业身份就是回执身份，出现这种情况说明有人绕过了排队层。
            raise BehaviorFusionError(
                f"segment {staged.receipt.segment_digest} was already fused into a different "
                f"set of judgements; refusing to orphan them"
            )
        for payload in staged.judgements:
            self.judgements.put_payload(payload)
        stored = self.receipts.put(staged.receipt)
        if stored.judgement_ids != staged.receipt.judgement_ids:  # pragma: no cover - 上面已拦
            raise BehaviorFusionError("stored receipt does not describe the judgements just written")


def _is_retryable(error: BaseException) -> bool:
    """契约违约不因重试而改变，只有环境性失败值得退避重来。"""

    return not isinstance(error, (TypeError, BehaviorFusionJobError))


__all__ = ["BehaviorFusionRunResult", "BehaviorFusionRunner"]
