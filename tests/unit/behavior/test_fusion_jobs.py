"""融合作业队列与 Runner：串行纪律、检查点、崩溃重放、排队。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from behavior.fusion import (
    FUSION_PROMPT_VERSION,
    FUSION_VERSION,
    BehaviorFusionEnqueuer,
    BehaviorFusionJobBlockedError,
    BehaviorFusionJobConfig,
    BehaviorFusionJobError,
    BehaviorFusionJobStatus,
    BehaviorFusionJobStore,
    BehaviorFusionReceiptStore,
    BehaviorFusionRunner,
    BehaviorFusionSegment,
    BehaviorJudgementFuser,
    BehaviorJudgementStore,
    segment_identity,
)
from behavior.observation import (
    BehaviorObservation,
    BehaviorObservationBatch,
    BehaviorObservationConfig,
    BehaviorObservationEnvelope,
    BehaviorObservationStore,
)
from foundation.integrity import canonical_digest
from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from ModelClient import (
    ChatClient,
    ChatModelConfig,
    ChatRequest,
    ModelResponse,
    PreparedChatRequest,
    ProviderCapabilities,
    ProviderConfig,
    StructuredChatClient,
)
from tests.unit.behavior.fusion_wire import SUBJECT, single_event_wire

TZ8 = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 14, 20, 0, tzinfo=TZ8)
LATER = NOW + timedelta(hours=1)
OBSERVATION_CONFIG = BehaviorObservationConfig()
OBSERVER = "home-a/hall"


def fragment(offset: int, semantics: str) -> BehaviorObservation:
    at = NOW + timedelta(seconds=offset)
    return BehaviorObservation.create(
        observer_id=OBSERVER,
        occurred_at=at,
        available_at=at + timedelta(milliseconds=800),
        modality="vision",
        semantics=semantics,
        participants=[SUBJECT],
        knowledge_state="observed",
        confidence=0.9,
        evidence_refs=[f"cam:{offset}"],
        config=OBSERVATION_CONFIG,
    )


FRAGMENTS = (fragment(0, "人走到水池边"), fragment(4, "手伸向水龙头"), fragment(8, "水流出"))
SEGMENT_DIGEST = segment_identity(FRAGMENTS)
OBSERVATION_IDS = tuple(sorted(item.observation_id for item in FRAGMENTS))


def envelope(
    observations: tuple[BehaviorObservation, ...],
    seed: str,
    *,
    recorded_at: datetime | None = None,
) -> BehaviorObservationEnvelope:
    return BehaviorObservationEnvelope.create(
        observer_id=OBSERVER,
        protocol="habitus_behavior_observation_v1",
        batch=BehaviorObservationBatch(observer_id=OBSERVER, observations=observations),
        delivery_id=canonical_digest(seed),
        recorded_at=recorded_at or (NOW + timedelta(minutes=10)),
        config=OBSERVATION_CONFIG,
    )


class ScriptedProvider:
    """按脚本依次返回响应，并记录调用次数——检查点的断言全靠它。"""

    provider_name = "fake"
    model = "fake-1"
    is_remote = False
    capabilities = ProviderCapabilities(
        async_completion=True,
        streaming=False,
        tools=False,
        structured_output_mode="json_schema",
        reasoning=False,
    )

    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        self.bodies = bodies
        self.calls = 0
        self.prompts: list[str] = []

    def prepare(self, request: ChatRequest, *, stream: bool) -> PreparedChatRequest:
        return PreparedChatRequest(
            request=request, body=b"{}", model_visible_body=b"{}", reserved_output_tokens=1_000, stream=stream
        )

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:
        self.prompts.append(request.request.messages[-1].content or "")
        body = self.bodies[min(self.calls, len(self.bodies) - 1)]
        self.calls += 1
        return ModelResponse(
            content=json.dumps(body, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            finish_reason="stop",
        )

    def complete(self, request: PreparedChatRequest) -> ModelResponse:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request: PreparedChatRequest) -> Iterator[Any]:  # pragma: no cover
        raise NotImplementedError

    def stream_async(self, request: PreparedChatRequest) -> AsyncIterator[Any]:  # pragma: no cover
        raise NotImplementedError

    def health_check(self) -> Mapping[str, object]:  # pragma: no cover
        return {}

    async def aclose(self) -> None:  # pragma: no cover
        return None


def build_fuser(bodies: list[dict[str, Any]]) -> tuple[BehaviorJudgementFuser, ScriptedProvider]:
    provider = ScriptedProvider(bodies)
    config = ChatModelConfig(
        route=ProviderConfig(
            provider="fake",
            adapter="openai_compatible_chat",
            model="fake-1",
            base_url="https://example.invalid",
            credential_ref="FAKE_KEY",
        ),
        context_window_tokens=128_000,
        max_output_tokens=8_000,
        structured_output_mode="json_schema",
    )
    return BehaviorJudgementFuser(StructuredChatClient(ChatClient(config, provider), validation_retries=1)), provider


class Harness:
    """一整套融合链路：观测存储、作业队列、判断存储、回执存储、Runner。"""

    def __init__(self, root, bodies: list[dict[str, Any]], *, now: datetime = LATER) -> None:
        self.now = now
        self.observations = BehaviorObservationStore(root)
        self.jobs = BehaviorFusionJobStore(root, PathLock(ProcessLocalLockStore()), clock=lambda: self.now)
        self.judgements = BehaviorJudgementStore(root)
        self.receipts = BehaviorFusionReceiptStore(root)
        self.fuser, self.provider = build_fuser(bodies)
        self.runner = BehaviorFusionRunner(
            self.jobs,
            self.observations,
            self.fuser,
            self.judgements,
            self.receipts,
            primary_subject=SUBJECT,
        )
        self.enqueuer = BehaviorFusionEnqueuer(self.observations, self.jobs, self.receipts, clock=lambda: self.now)

    def deliver(
        self,
        observations: tuple[BehaviorObservation, ...],
        seed: str,
        *,
        recorded_at: datetime | None = None,
    ) -> None:
        self.observations.put(envelope(observations, seed, recorded_at=recorded_at))

    def enqueue_job(self):
        return self.jobs.enqueue(
            segment_digest=SEGMENT_DIGEST,
            observation_ids=OBSERVATION_IDS,
            source_refs=("observation-delivery:d1",),
            fusion_version=FUSION_VERSION,
            prompt_version=FUSION_PROMPT_VERSION,
        )

    def run_once(self, worker_id: str = "worker-1"):
        lease = self.runner.claim(worker_id)
        assert lease is not None
        return asyncio.run(self.runner.execute(lease, judged_at=self.now))


@pytest.fixture()
def harness(tmp_path):
    return Harness(tmp_path, [single_event_wire(len(FRAGMENTS))])


# --- 端到端 ---------------------------------------------------------------------------


def test_a_job_runs_from_observations_to_judgements_and_a_receipt(harness) -> None:
    harness.deliver(FRAGMENTS, "d1")
    job = harness.enqueue_job()
    assert job.status is BehaviorFusionJobStatus.QUEUED
    result = harness.run_once()
    assert result.job.status is BehaviorFusionJobStatus.COMMITTED
    assert result.fused is True
    assert len(result.judgement_ids) == 1
    assert harness.judgements.read(result.judgement_ids[0]) is not None
    assert harness.receipts.read(result.receipt.receipt_id) is not None
    assert harness.jobs.oldest_uncommitted() is None


def test_the_job_identity_is_the_receipt_identity(harness) -> None:
    """ "这段观测处理过没有"因此是一次文件存在性判断，不必扫描判断存储反推。"""

    harness.deliver(FRAGMENTS, "d1")
    job = harness.enqueue_job()
    assert harness.run_once().receipt.receipt_id == job.job_id


def test_fusion_never_touches_the_behavior_tree(harness) -> None:
    """判断到树内文档的归约属于另一层；本层写完判断就结束。"""

    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    harness.run_once()
    assert not (harness.jobs.root / "behaviors").exists()


# --- 检查点：模型恰好调用一次 ---------------------------------------------------------


def test_a_crash_after_staging_replays_persistence_without_calling_the_model(harness) -> None:
    """融合不是纯函数：重跑一次就是另一批判断，落盘之后成两套互不相认的记录。"""

    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    lease = harness.runner.claim("worker-1")
    assert lease is not None
    staged_lease = asyncio.run(harness.runner._fuse_and_stage(lease, judged_at=harness.now))
    assert harness.provider.calls == 1
    assert staged_lease.job.staged is not None

    # 模拟崩溃：租约过期后由另一个 Worker 接管。
    harness.now = harness.now + timedelta(seconds=3_600)
    reclaimed = harness.runner.claim("worker-2")
    assert reclaimed is not None
    assert reclaimed.job.needs_fusion is False
    result = asyncio.run(harness.runner.execute(reclaimed, judged_at=harness.now))
    assert harness.provider.calls == 1
    assert result.replayed is True
    assert result.job.status is BehaviorFusionJobStatus.COMMITTED


def test_replaying_persistence_is_idempotent(harness) -> None:
    """判断按内容身份命名，所以重放写入就是写同一个文件名同样的字节。"""

    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    lease = harness.runner.claim("worker-1")
    assert lease is not None
    staged_lease = asyncio.run(harness.runner._fuse_and_stage(lease, judged_at=harness.now))
    staged = staged_lease.job.staged
    assert staged is not None
    harness.runner._persist(staged)
    harness.runner._persist(staged)
    assert len(harness.judgements.list()) == 1


def test_a_failure_after_staging_returns_to_staged_not_queued(harness) -> None:
    """回退到哪里由有没有暂存决定；回到 QUEUED 就等于允许再调一次模型。"""

    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    lease = harness.runner.claim("worker-1")
    assert lease is not None
    staged_lease = asyncio.run(harness.runner._fuse_and_stage(lease, judged_at=harness.now))
    failed = harness.jobs.fail(staged_lease, RuntimeError("disk full"))
    assert failed.status is BehaviorFusionJobStatus.STAGED
    assert failed.staged is not None
    assert failed.next_attempt_at is not None


# --- 串行纪律 -------------------------------------------------------------------------


def test_a_later_job_cannot_be_claimed_before_the_oldest_commits(harness) -> None:
    harness.deliver(FRAGMENTS, "d1")
    first = harness.enqueue_job()
    tail = (fragment(600, "人坐到沙发"),)
    harness.deliver(tail, "d2")
    second = harness.jobs.enqueue(
        segment_digest=segment_identity(tail),
        observation_ids=(tail[0].observation_id,),
        source_refs=("observation-delivery:d2",),
        fusion_version=FUSION_VERSION,
        prompt_version=FUSION_PROMPT_VERSION,
    )
    assert second.fusion_sequence == first.fusion_sequence + 1
    with pytest.raises(BehaviorFusionJobBlockedError):
        harness.jobs.claim(second, "worker-1")


def test_an_active_lease_blocks_a_second_worker(harness) -> None:
    harness.deliver(FRAGMENTS, "d1")
    job = harness.enqueue_job()
    harness.jobs.claim(job, "worker-1")
    with pytest.raises(BehaviorFusionJobBlockedError):
        harness.runner.claim("worker-2")


def test_retries_are_exhausted_into_a_terminal_failure(harness) -> None:
    harness.deliver(FRAGMENTS, "d1")
    job = harness.enqueue_job()
    store = BehaviorFusionJobStore(
        harness.jobs.root,
        PathLock(ProcessLocalLockStore()),
        config=BehaviorFusionJobConfig(max_attempts=2, retry_base_delay_seconds=1.0),
        clock=lambda: harness.now,
    )
    current = job
    for _ in range(2):
        lease = store.claim(current, "worker-1")
        current = store.fail(lease, RuntimeError("boom"))
        harness.now = harness.now + timedelta(seconds=60)
    assert current.status is BehaviorFusionJobStatus.FAILED
    with pytest.raises(BehaviorFusionJobBlockedError):
        store.claim(current, "worker-1")
    reopened = store.retry_failed(current)
    assert (reopened.status, reopened.attempts) == (BehaviorFusionJobStatus.QUEUED, 0)


# --- 排队 -----------------------------------------------------------------------------


def test_enqueue_withholds_observations_that_may_still_be_in_progress(harness) -> None:
    harness.now = NOW + timedelta(seconds=30)
    harness.deliver(FRAGMENTS, "d1")
    result = harness.enqueuer.enqueue_ready()
    assert (result.count, result.withheld_observations) == (0, 3)

    harness.now = NOW + timedelta(hours=1)
    result = harness.enqueuer.enqueue_ready()
    assert (result.count, result.withheld_observations) == (1, 0)


def test_enqueue_is_idempotent_across_scans(harness) -> None:
    harness.deliver(FRAGMENTS, "d1")
    assert harness.enqueuer.enqueue_ready().count == 1
    assert harness.enqueuer.enqueue_ready().count == 0
    assert harness.jobs.high_watermark() == 1


def test_sliding_window_deliveries_do_not_silently_drop_new_observations(harness) -> None:
    """滑窗让每个交付都与前一个重叠；按交付整体过滤会让新观测被静默丢掉。

    观测本来就不完整（人走出画面、设备离线），缺失只能记录、不能由我们自己制造。
    """

    tail = (fragment(600, "人坐到沙发"), fragment(604, "人拿起遥控器"))
    harness.deliver(FRAGMENTS, "d1")
    assert harness.enqueuer.enqueue_ready().count == 1
    harness.deliver((FRAGMENTS[-1], tail[0]), "d2")
    assert harness.enqueuer.enqueue_ready().count == 1
    harness.deliver((tail[0], tail[1]), "d3")
    assert harness.enqueuer.enqueue_ready().count == 1
    assert {item.observation_id for item in (*FRAGMENTS, *tail)} <= harness.jobs.covered_observation_ids()


def test_committed_observations_are_not_reenqueued_after_the_job_is_discarded(harness) -> None:
    """作业提交即清（队列记录里没有产物）；"处理过没有"由覆盖索引回答，不靠作业文件。"""

    harness.deliver(FRAGMENTS, "d1")
    harness.enqueuer.enqueue_ready()
    result = harness.run_once()
    # runner 在提交的同一步已经把作业清掉：再清一次是幂等的 False，目录里不再有它。
    assert harness.jobs.discard_committed(result.job) is False
    assert harness.jobs.try_read(result.job.job_id) is None
    assert harness.enqueuer.enqueue_ready().count == 0


def test_the_queue_snapshot_reports_state_without_leaking_content(harness) -> None:
    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    snapshot = harness.jobs.observability_snapshot()
    assert (snapshot.queued, snapshot.running, snapshot.committed) == (1, 0, 0)
    assert snapshot.high_watermark == 1
    assert snapshot.oldest_age_seconds >= 0.0


# --- 模型契约 -------------------------------------------------------------------------


def test_a_short_frame_table_is_caught_by_the_schema_and_retried(tmp_path) -> None:
    """穷尽性钉在 Schema 的 minItems/maxItems 上，所以少一行在结构阶段就被拦下。

    比装配层更早、反馈也更机械——模型看到的是"数组长度不对"，不需要它再判断该把哪一帧塞到哪。
    """

    short = single_event_wire(len(FRAGMENTS))
    short["frames"] = short["frames"][:2]
    harness = Harness(tmp_path, [short, single_event_wire(len(FRAGMENTS))])
    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    result = harness.run_once()
    assert harness.provider.calls == 2
    assert "json_schema" in harness.provider.prompts[1]
    assert result.job.status is BehaviorFusionJobStatus.COMMITTED


def test_a_semantically_inconsistent_output_is_fed_back_from_the_assembly_layer(tmp_path) -> None:
    """自洽性错误（引用了没声明的判断）Schema 拦不住，必须由装配层精确反馈。"""

    broken = single_event_wire(len(FRAGMENTS))
    broken["frames"][1]["assignments"] = [{"judgement_no": 9, "basis_no": None}]
    harness = Harness(tmp_path, [broken, single_event_wire(len(FRAGMENTS))])
    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    result = harness.run_once()
    assert harness.provider.calls == 2
    assert "undeclared judgement: 9" in harness.provider.prompts[1]
    assert result.job.status is BehaviorFusionJobStatus.COMMITTED


def test_the_prompt_carries_fragments_but_never_absolute_time(harness) -> None:
    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    harness.run_once()
    prompt = harness.provider.prompts[0]
    assert "#1 (+0s)" in prompt
    assert "#3 (+8s)" in prompt
    assert "2026" not in prompt
    assert "【先前的判断】" in prompt


# --- 补发与并发（评审实测出来的两个洞） -------------------------------------------------


def test_backfilled_observations_are_withheld_by_arrival_not_by_occurrence(tmp_path) -> None:
    """云侧 agent 重启后补发几小时前的观测是常态。

    按"行为何时发生"判收尾，这些观测一落盘就"够老"，静默期一秒也拦不住——一次连续的做饭会被
    逐条切成多个作业，多次模型调用、多组互不相认的判断。判据必须看"何时到达我们"。
    """

    cooking = tuple(fragment(index * 120, f"做饭第{index}步") for index in range(6))
    # 观测发生在两小时前，但现在才逐条补发到达。
    backfilled = tuple(
        BehaviorObservation.create(
            observer_id=OBSERVER,
            occurred_at=item.occurred_at,
            available_at=NOW + timedelta(hours=2, seconds=index * 20),
            modality="vision",
            semantics=item.semantics,
            participants=[SUBJECT],
            knowledge_state="observed",
            confidence=0.9,
            evidence_refs=[f"cam:backfill:{index}"],
            config=OBSERVATION_CONFIG,
        )
        for index, item in enumerate(cooking)
    )

    harness = Harness(tmp_path, [single_event_wire(len(FRAGMENTS))])
    for index, item in enumerate(backfilled):
        harness.now = NOW + timedelta(hours=2, seconds=index * 20 + 1)
        harness.deliver((item,), f"backfill-{index}", recorded_at=harness.now)
        assert harness.enqueuer.enqueue_ready().count == 0, f"第 {index} 条补发就被切走了"

    # 补发停止、静默期过去之后，整段作为一个作业进入队列。
    harness.now = NOW + timedelta(hours=3)
    result = harness.enqueuer.enqueue_ready()
    assert result.count == 1
    assert len(result.enqueued[0].observation_ids) == len(backfilled)


def test_the_whole_enqueue_scan_runs_inside_the_queue_fence(tmp_path) -> None:
    """读覆盖、读观测、登记必须在同一个栅栏内。

    三步之间若有别的扫描插进来，两次扫描会按不同快照切出不同的段，``segment_digest`` 不同则
    ``job_id`` 不同，``enqueue`` 的幂等完全不起作用，同一批观测被融合两次。
    """

    harness = Harness(tmp_path, [single_event_wire(len(FRAGMENTS))])
    harness.deliver(FRAGMENTS, "d1")

    held: list[bool] = []
    original = harness.observations.list

    def probe():
        # 扫描进行中，另一方取同一把队列锁必须拿不到。
        try:
            with harness.jobs._queue_fence():
                held.append(False)
        except TimeoutError:
            held.append(True)
        return original()

    harness.observations.list = probe  # type: ignore[method-assign]
    try:
        assert harness.enqueuer.enqueue_ready().count == 1
    finally:
        harness.observations.list = original  # type: ignore[method-assign]
    assert held == [True], "排队扫描读观测时并未持有队列锁"


def test_a_worker_that_never_settles_is_not_reclaimed_forever(tmp_path) -> None:
    """Worker 硬崩（OOM、被 kill）永远走不到 fail()，认领路径也必须判次数。

    否则作业在"接管 → 崩 → 租约过期 → 接管"里无限循环，而且每一轮都重调一次模型。
    """

    harness = Harness(tmp_path, [single_event_wire(len(FRAGMENTS))])
    harness.deliver(FRAGMENTS, "d1")
    job = harness.enqueue_job()
    store = BehaviorFusionJobStore(
        harness.jobs.root,
        PathLock(ProcessLocalLockStore()),
        config=BehaviorFusionJobConfig(max_attempts=2, lease_ttl_seconds=60),
        clock=lambda: harness.now,
    )
    current = job
    for _ in range(2):
        store.claim(current, "worker-1")  # 认领后进程直接消失
        harness.now = harness.now + timedelta(seconds=120)
        current = store.oldest_uncommitted()
        assert current is not None
    with pytest.raises(BehaviorFusionJobBlockedError, match="exhausted its attempts"):
        store.claim(current, "worker-1")
    assert store.oldest_uncommitted().status is BehaviorFusionJobStatus.FAILED


def test_persisting_a_second_fusion_of_the_same_segment_is_refused(tmp_path) -> None:
    """回执撞车时复用先到的那份，所以第二批判断会成为没有任何回执指向的孤儿。"""

    from behavior.fusion.errors import BehaviorFusionError

    harness = Harness(tmp_path, [single_event_wire(len(FRAGMENTS))])
    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    first = harness.run_once()

    harness.jobs.discard_committed(first.job)
    harness.enqueue_job()
    lease = harness.runner.claim("worker-2")
    assert lease is not None
    staged = asyncio.run(harness.runner._fuse_and_stage(lease, judged_at=harness.now + timedelta(minutes=5))).job.staged
    assert staged is not None
    with pytest.raises(BehaviorFusionError, match="already fused into a different"):
        harness.runner._persist(staged)
    assert len(harness.judgements.list()) == 1


# --- 跨融合窗口的关联 -------------------------------------------------------------------


def _continuation_wire(fragment_count: int) -> dict[str, Any]:
    """后半段指回【先前的判断】C1，而不是当成全新的一件事。"""

    from tests.unit.behavior.fusion_wire import judgement, wire

    return wire(
        [
            judgement(
                1,
                behavior="继续做饭",
                goal="准备餐食",
                summary="回到灶台继续做饭",
                basis=["翻炒并盛盘"],
                context_relations=[("continues", 1)],
            )
        ],
        [[(1, 1)]] * fragment_count,
    )


def test_a_later_segment_can_continue_a_judgement_from_an_earlier_one(tmp_path) -> None:
    """切段是机械的，一件事经常横跨两次融合。

    没有跨窗口关联时，后半段只能当成全新的一件事——一次做饭在数据上变成两件互不相干的事，中间还
    多出一条不存在的转移。
    """

    # 第一段是被切开的前半截，所以是 ongoing——completed 的事没有"后半段"。
    first = single_event_wire(len(FRAGMENTS), behavior="做饭", goal="准备餐食", status="ongoing")
    tail = (fragment(2_000, "人回到灶台"), fragment(2_004, "人翻炒"), fragment(2_008, "人盛盘"))
    harness = Harness(tmp_path, [first, _continuation_wire(len(tail))])

    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    earlier = harness.run_once()

    harness.deliver(tail, "d2", recorded_at=NOW + timedelta(seconds=2_100))
    harness.jobs.enqueue(
        segment_digest=segment_identity(tail),
        observation_ids=tuple(sorted(item.observation_id for item in tail)),
        source_refs=("observation-delivery:d2",),
        fusion_version=FUSION_VERSION,
        prompt_version=FUSION_PROMPT_VERSION,
    )
    harness.now = harness.now + timedelta(hours=1)
    later = harness.run_once("worker-2")

    # 后一段确实看到了前一段，并且指回了它的**真实身份**。
    assert "C1" in harness.provider.prompts[1]
    stored = harness.judgements.read(later.judgement_ids[0])
    assert stored is not None
    assert stored["relations"] == [{"kind": "continues", "target_id": earlier.judgement_ids[0]}]


def test_the_context_cutoff_never_shows_a_judgement_from_the_future(tmp_path) -> None:
    """能给模型看的只能是本段观测进入系统之前就已经成立的判断。

    用更晚的截断点等于把后来才知道的事喂回给更早的一段——那是标签泄漏，而且离线指标上看不出来。
    """

    harness = Harness(tmp_path, [single_event_wire(len(FRAGMENTS))])
    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    harness.run_once()

    # 这一段的观测比上面那条判断的证据更早，所以它不该看见任何上下文。
    earlier = (fragment(-600, "更早的观测"), fragment(-596, "更早的观测二"))
    segment = BehaviorFusionSegment(earlier, ("observation-delivery:d0",))
    assert harness.runner._context(segment) == ()

    # 而在它之后的段看得见。
    later = (fragment(3_000, "后来的观测"),)
    assert len(harness.runner._context(BehaviorFusionSegment(later, ("d2",)))) == 1

    # 【关键】一段**横跨**那条判断成立时刻的观测：最早的一条在它之前进入系统，最晚的一条在它
    # 之后。截断点取 min 还是取 max，只有这种输入能分辨——上面两段无论取哪个都得到同样的答案，
    # 于是这条测试曾经守不住任何东西（把 min 改成 max，全套 8873 条一条不红）。
    #
    # 取 max 就是泄漏：这一段最早那条观测流进来的时候，那条判断还不存在，模型当时不可能看见它。
    straddling = (fragment(-600, "跨越判断成立时刻的第一条"), fragment(3_000, "同一段里更晚的一条"))
    assert harness.runner._context(BehaviorFusionSegment(straddling, ("d3",))) == ()


def test_a_prompt_change_does_not_poison_the_jobs_already_queued(tmp_path) -> None:
    """升级会停服务、重装、再跑——但**队列是耐久的**，重启不会清掉盘上那批作业。

    作业身份在排队时由片段与版本钉死。旧版本的作业留到重启之后，执行方要等到"回执身份与
    作业身份不符"才发现，而那一步在调完模型之后：白烧一次调用、退避重试到 FAILED、卡死整条
    串行队列，而 ``retry_failed`` 重开后还会再烧一轮。版本在排队时就是已知的，必须在认领之前
    纠正。
    """

    harness = Harness(tmp_path, [single_event_wire(len(FRAGMENTS))])
    harness.deliver(FRAGMENTS, "d1")
    stale = harness.jobs.enqueue(
        segment_digest=SEGMENT_DIGEST,
        observation_ids=OBSERVATION_IDS,
        source_refs=("observation-delivery:d1",),
        fusion_version=FUSION_VERSION,
        prompt_version="behavior_judgement_prompt_v0",
    )

    lease = harness.runner.claim("worker-1")

    assert lease is not None
    assert lease.job.prompt_version == FUSION_PROMPT_VERSION
    # 队列位置必须保住：改挂后排到队尾，后面的段就会先于它融合，跨窗口指回随之失效。
    assert lease.job.fusion_sequence == stale.fusion_sequence
    assert harness.jobs.try_read(stale.job_id) is None
    assert harness.provider.calls == 0  # 纠正发生在调模型之前


def test_a_checkpointed_job_keeps_its_own_version(tmp_path) -> None:
    """已经到检查点的作业不改挂：它的判断是在旧版本下真做出来的，按自己的版本提交才诚实。"""

    harness = Harness(tmp_path, [single_event_wire(len(FRAGMENTS))])
    harness.deliver(FRAGMENTS, "d1")
    job = harness.enqueue_job()
    lease = harness.jobs.claim(job, "worker-1")
    lease = asyncio.run(harness.runner._fuse_and_stage(lease, judged_at=None))
    staged_version = lease.job.prompt_version

    unchanged = harness.jobs.retarget(
        lease.job, fusion_version=FUSION_VERSION, prompt_version="behavior_judgement_prompt_v0"
    )

    assert unchanged is not None
    assert unchanged.prompt_version == staged_version
    assert unchanged.staged is not None


def test_a_foreign_file_in_the_jobs_directory_does_not_brick_the_pipeline(tmp_path) -> None:
    """``.DS_Store`` 这类条目不是本存储写的，为它们整库硬失败等于让一次偶发污染瘫痪整条流水线。

    排队、认领、结算、可观测性都要枚举这个目录。判断与回执两个兄弟存储早就是跳过的，这里必须
    同口径；名字合规但内容损坏的仍然硬失败——那才是我们自己的东西。
    """

    harness = Harness(tmp_path, [single_event_wire(len(FRAGMENTS))])
    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    for junk in (".DS_Store", "notes.txt", "job.json.bak"):
        (harness.jobs.jobs_root / junk).write_bytes(b"\x00")

    assert harness.jobs.oldest_uncommitted() is not None
    assert harness.jobs.high_watermark() >= 1
    assert harness.jobs.covered_observation_ids()
    assert harness.jobs.observability_snapshot() is not None
    assert harness.runner.claim("worker-1") is not None

    (harness.jobs.jobs_root / f"{'a' * 64}.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(BehaviorFusionJobError):
        harness.jobs.oldest_uncommitted()


def test_cancelled_execution_settles_the_lease_before_propagating(tmp_path) -> None:
    """停机超时取消：租约按可重试失败结算再传播取消——否则作业停留 RUNNING+租约，
    重启后先 Blocked 等租约过期、再被计一次 attempt，反复部署能把队首烧成永久 FAILED。"""

    import asyncio

    harness = Harness(tmp_path, [single_event_wire()])
    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    lease = harness.runner.claim("worker-cancel")
    assert lease is not None

    class _HangingFuser:
        async def fuse(self, *args, **kwargs):
            await asyncio.sleep(3600)

    harness.runner.fuser = _HangingFuser()  # type: ignore[assignment]

    async def scenario() -> None:
        task = asyncio.ensure_future(harness.runner.execute(lease))
        await asyncio.sleep(0.05)
        task.cancel()
        with __import__("pytest").raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    job = harness.jobs.oldest_uncommitted()
    assert job is not None
    assert job.claim_id is None or job.status.value != "running"  # 租约已结算，不再是活跃持有
    assert job.attempts == 1
    assert job.last_error is not None and "cancelled" in job.last_error
    # 结算成可重试：退避后可再次认领，不需要等 300 秒租约过期
    harness.now = harness.now + timedelta(hours=1)
    relaimed = harness.runner.claim("worker-2")
    assert relaimed is not None


class _TruncatingProvider(ScriptedProvider):
    """模型输出被 length 截断：同一段重试还会截断，结构化客户端最终抛 truncated。"""

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            content='{"judgements": [', model=self.model, provider=self.provider_name, finish_reason="length"
        )


def test_a_truncated_segment_is_recorded_as_an_unreadable_gap_instead_of_blocking(tmp_path) -> None:
    """输出截断是确定性的（512/160/100 条的段实测都撞过），原样重试只会把串行队列封死。

    降级：整段记成一条没读懂判断——时间轴上留下"观测到了但没读懂"的空白，覆盖索引照记、
    队列照走、留信号。
    """

    harness = Harness(tmp_path, [])
    provider = _TruncatingProvider([])
    config = ChatModelConfig(
        route=ProviderConfig(
            provider="fake",
            adapter="openai_compatible_chat",
            model="fake-1",
            base_url="https://example.invalid",
            credential_ref="FAKE_KEY",
        ),
        context_window_tokens=128_000,
        max_output_tokens=8_000,
        structured_output_mode="json_schema",
    )
    harness.runner.fuser = BehaviorJudgementFuser(
        StructuredChatClient(ChatClient(config, provider), validation_retries=1)
    )
    harness.deliver(FRAGMENTS, "d1")
    harness.enqueuer.enqueue_ready()
    result = harness.run_once()
    assert result.fused is True
    assert result.degradations == ("segment_truncated fragments=3 recorded as unreadable",)
    (record,) = harness.judgements.list()
    assert record["behavior"] is None  # 一条没读懂判断覆盖整段
    assert set(record["observation_ids"]) == {item.observation_id for item in FRAGMENTS}
    # 覆盖索引记住了这段：重投不会再融合一次
    assert harness.enqueuer.enqueue_ready().count == 0
    assert harness.jobs.oldest_uncommitted() is None


def test_a_segment_with_no_judgement_still_commits_a_receipt_that_covers_it(tmp_path) -> None:
    """零判断的段照样结账：回执覆盖全部观测并把它们记为无归属，作业提交，观测不会被重新入队。"""

    from tests.unit.behavior.fusion_wire import wire

    harness = Harness(tmp_path, [wire([], [[] for _ in FRAGMENTS])])
    harness.deliver(FRAGMENTS, "d1")
    harness.enqueue_job()
    result = harness.run_once()
    assert result.job.status is BehaviorFusionJobStatus.COMMITTED
    assert result.judgement_ids == ()
    receipt = harness.receipts.read(result.receipt.receipt_id)
    assert receipt is not None
    assert set(receipt.unowned_observation_ids) == set(receipt.observation_ids)
    assert harness.jobs.oldest_uncommitted() is None
