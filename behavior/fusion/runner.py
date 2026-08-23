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

本层只产出判断。判断到行为树文档（occurrence 与 gap）的归约是**另一层**的事：链要等老出引用
窗口才封口（延续、修正、结果都可能在窗口内到达），节奏与融合相反。把它塞进融合，融合就永远
完不了。

## 认领与执行分离

``claim`` 与 ``execute`` 分开，好让上层 Worker 在长时间的模型调用期间独立维持租约心跳。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from behavior.fusion.config import FUSION_CONTEXT_LIMIT, FUSION_CONTEXT_LOOKBACK_SECONDS
from behavior.fusion.derivation import (
    FUSION_VERSION,
    derive_judgements,
    judgement_payload,
    persistable_judgements,
    without_unresolvable_relations,
)
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
from behavior.fusion.prompt import FUSION_PROMPT_VERSION
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
        context_limit: int = FUSION_CONTEXT_LIMIT,
        context_lookback_seconds: float = FUSION_CONTEXT_LOOKBACK_SECONDS,
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
        if oldest.fusion_version != FUSION_VERSION or oldest.prompt_version != FUSION_PROMPT_VERSION:
            # 队列是耐久的，升级重启不会清掉它。版本在排队时就已知，在这里纠正，而不是等到
            # 调完模型才发现回执身份对不上——那条路要白烧一次调用并把整条串行队列卡死。
            replacement = self.jobs.retarget(
                oldest, fusion_version=FUSION_VERSION, prompt_version=FUSION_PROMPT_VERSION
            )
            # 改挂后作业换了身份；``None`` 表示当前版本下这段已另有作业，队首要重新取。
            oldest = replacement if replacement is not None else self.jobs.oldest_uncommitted()
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
        except asyncio.CancelledError:
            # 优雅停机超时取消：把租约按可重试失败结算后再传播取消——否则作业停留
            # RUNNING+租约，重启后先 Blocked 等租约过期、再按"worker died"烧一次 attempt，
            # 反复部署能把队首烧成永久 FAILED（评审推演）。
            with suppress(Exception):
                self.jobs.fail(lease, RuntimeError("fusion execution cancelled"), retryable=True)
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
        # 落主体的判断与没读懂的观测段（后者归约层要物化成 gap 节点）；旁人的可读判断分流，
        # 其观测身份已记在回执的 ``out_of_scope_observation_ids`` 里，缺失并没有被解释掉。
        # 口径必须与回执共用同一个函数，否则 StagedFusion 的一致性校验会拦下整批。
        in_scope = persistable_judgements(derived, self.primary_subject)
        # 落盘之后能被解析的目标只有这两类。指向被分流掉的旁人判断的关系必须在这里剪掉，
        # 否则不可变存储里会永久留下一个指向不存在记录的 ``target_id``。
        visible = {item.judgement_id for item in in_scope}
        visible.update(item["judgement_id"] for item in context)
        staged = StagedFusion(
            receipt=receipt,
            judgements=tuple(
                judgement_payload(without_unresolvable_relations(item, visible))
                for item in in_scope
            ),
        )
        return self.jobs.stage(lease, staged)

    def _context(self, segment: BehaviorFusionSegment) -> tuple[Mapping[str, Any], ...]:
        """取本段之前已经成立的若干条判断，供模型跨窗口指回。

        截断点是本段**最早观测的可用时刻**：能给模型看的只能是在这段观测进入系统之前就已经成立
        的判断。用更晚的截断点等于把后来才知道的事喂回给更早的一段——那是标签泄漏。

        TODO(BHV-CONTEXT-CUTOFF-001): ``recent_before`` 用**严格小于**截断点，于是
        ``evidence_ready_at == cutoff`` 的那条判断被排除在外。

        - 具体场景：上游一次投递里的观测若共享同一个 ``available_at``（整批盖一个送达时间戳），
          而这次投递恰好横跨切段边界，那么前一段那条判断的 ``evidence_ready_at``（= 覆盖观测
          ``available_at`` 的 max）正好等于后一段的 ``min(available_at)``，于是被严格小于排除。
          被排除的恰恰是"被切段拦腰切断的前半截"——而"后半段要能指回前半段"是整条严格串行纪律
          唯一的存在理由。停机后补算时更彻底：全批共享一个 ``available_at``，所有段的上下文都是空。
        - 影响大小：取决于上游行为。若上游逐条标注送达时刻，这就只是一个理论边界，影响为零。
        - 改造方案：把 ``recent_before`` 的比较从 ``<`` 改成 ``<=``。取等号不构成泄漏——
          ``evidence_ready_at == cutoff`` 意味着这条判断的全部证据在本段最早观测可用的**同一时刻**
          就已齐备，它没有用到本段之后才存在的任何信息。同时补一条测试，构造两条观测共享
          ``available_at`` 且横跨切段边界的输入。
        - 为什么先不动：上游观测源尚未适配、甚至尚未开发，字段语义还没对齐（``available_at``
          到底按投递批次还是按单条标注，现在没有真实样本可查）。按项目纪律，判据要先用真实数据
          验证再定，不能凭推理改成 ``<=`` 然后用自己写的测试自证。
        - 时机：上游观测接入、拿到第一批真实交付之后。
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

    return not isinstance(error, TypeError | BehaviorFusionJobError)


__all__ = ["BehaviorFusionRunResult", "BehaviorFusionRunner"]
