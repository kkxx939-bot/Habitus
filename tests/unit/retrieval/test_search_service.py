"""助手查记忆：清单 → 挑 → 找 → 判，三处来源、总时间上限、记账退路、模型不可用只给档案。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from habitus.behavior import BehaviorKind, BehaviorTree
from habitus.behavior.tree.writer import BehaviorDocumentWriter
from habitus.foundation.observability import ObservationEvent
from habitus.memory.conversation import ConversationAddress, ConversationSummaryMatch, summary_reference
from habitus.memory.document import MemoryDocumentMetadata
from habitus.memory.model import MemoryAddress, MemoryKind, MemoryLevel
from habitus.memory.recall import (
    MemoryRecallLifecycle,
    MemoryRecallLifecycleConfig,
    MemoryRecallTarget,
    SQLiteMemoryRecallLifecycleStore,
)
from habitus.memory.retrieval import (
    BehaviorRecord,
    BehaviorRecordBatch,
    MemoryCatalogReader,
    MemorySearchDegradationStage,
    MemorySearchServiceConfig,
    NoBehaviorRecords,
    SearchService,
)
from habitus.memory.semantic import MemorySemanticConfig, MemorySemanticRefresher
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from habitus.model_client import ModelCircuitOpenError, ModelErrorKind
from habitus.runtime.behavior_records import BehaviorTreeRecords
from tests.helpers import codec, registry, segment_summary
from tests.lock_store import ProcessLocalLockStore
from tests.unit.behavior.tree_payloads import DAY, gap_payload, local, occurrence_payload

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 20, 4, 0, tzinfo=UTC)
PROFILE_URI = MemoryURI.from_address(MemoryAddress.profile())


@dataclass
class _Response:
    value: object


@dataclass
class ScriptedClient:
    """按顺序回答"挑"与"判"；每次记下提示词。``delay`` 让某一步慢下来，``error`` 让它失败。"""

    answers: list[object]
    prompts: list[tuple[str, str]] = field(default_factory=list)
    delays: dict[int, float] = field(default_factory=dict)
    error: BaseException | None = None

    async def complete_json_async(self, request, *, schema, name, validator, context):  # type: ignore[no-untyped-def]
        index = len(self.prompts)
        self.prompts.append((name, request.messages[-1].content))
        if self.error is not None:
            raise self.error
        if index in self.delays:
            await asyncio.sleep(self.delays[index])
        return _Response(validator(self.answers[index]))


def pick(read=(), card_keywords=(), records=None, summaries=None) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {"read": list(read), "card_keywords": list(card_keywords), "records": records, "summaries": summaries}


def judge(use, *, enough: bool = True, last: bool = False, following=None) -> dict[str, object]:  # type: ignore[no-untyped-def]
    value: dict[str, object] = {"enough": enough, "use": list(use)}
    if not last:
        value["next"] = following
    return value


def source(date_from: str | None = None, date_to: str | None = None, keywords=()) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {"date_from": date_from, "date_to": date_to, "keywords": list(keywords)}


@dataclass
class FakeRecords:
    records: tuple[BehaviorRecord, ...]
    failing: tuple[date, ...] = ()

    def days(self) -> tuple[date, ...]:
        return tuple(sorted({record.day for record in self.records}))

    def read_days(self, days: tuple[date, ...]) -> BehaviorRecordBatch:
        wanted = set(days) - set(self.failing)
        return BehaviorRecordBatch(
            tuple(record for record in self.records if record.day in wanted),
            tuple(sorted(set(days) & set(self.failing))),
        )


@dataclass
class FakeSummaries:
    matches: tuple[ConversationSummaryMatch, ...] = ()
    config: Any = field(default_factory=lambda: type("C", (), {"max_record_chars": 16_000})())

    def walk(self) -> tuple[ConversationSummaryMatch, ...]:
        return self.matches


@dataclass
class FakeContextReader:
    config: MemorySearchServiceConfig

    def read(self, _address):  # type: ignore[no-untyped-def]
        raise AssertionError("these tests search without a Conversation")


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def record(name: str, hour: int, *, day: date = date(2026, 3, 31), summary: str = "", goal: str = "") -> BehaviorRecord:
    started = datetime(day.year, day.month, day.day, hour, 0, tzinfo=CST)
    return BehaviorRecord(
        address=f"session:behavior://occurrences/{day:%Y/%m/%d}/{name}",
        lane="session",
        day=day,
        started_at=started,
        name=name,
        goal=goal or None,
        summary=summary or name,
        steps=(f"{name}的步骤",),
    )


@dataclass
class World:
    tree: MemoryTree
    refresher: MemorySemanticRefresher
    lifecycle: MemoryRecallLifecycle

    def put(
        self, kind: MemoryKind, fields: dict[str, object], *, updated: datetime = NOW - timedelta(days=1)
    ) -> MemoryURI:
        document = codec().build(
            kind, fields, metadata=MemoryDocumentMetadata(revision=1, created_at=updated, updated_at=updated)
        )
        self.tree.write(document)
        self.refresher.refresh_for(document.address)
        return MemoryURI.from_address(document.address)

    def service(
        self,
        client: ScriptedClient,
        *,
        records=None,  # type: ignore[no-untyped-def]
        summaries=None,  # type: ignore[no-untyped-def]
        config: MemorySearchServiceConfig | None = None,
        expander=None,  # type: ignore[no-untyped-def]
        observer: Recorder | None = None,
        catalog: MemoryCatalogReader | None = None,
    ) -> SearchService:
        resolved = config or MemorySearchServiceConfig()
        return SearchService(
            catalog=catalog or self.catalog(resolved),
            snapshot_reader=MemorySnapshotReader(self.tree),
            client=client,  # type: ignore[arg-type]
            recall_lifecycle=self.lifecycle,
            conversation_context=FakeContextReader(resolved),  # type: ignore[arg-type]
            summaries=summaries or FakeSummaries(),  # type: ignore[arg-type]
            records=records or NoBehaviorRecords(),
            zone=CST,
            config=resolved,
            cold_probe_expander=expander,
            clock=lambda: NOW,
            observer=observer,
        )

    def catalog(self, config: MemorySearchServiceConfig) -> MemoryCatalogReader:
        return MemoryCatalogReader(
            self.tree,
            registry(),
            self.lifecycle,
            max_catalog_chars=config.max_catalog_chars,
            overview_max_bytes=1_000_000,
            max_unlisted_scan=config.max_unlisted_scan,
        )


def world(tmp_path: Path, *, semantic: MemorySemanticConfig | None = None) -> World:
    from habitus.infrastructure.store.contracts import PathLock

    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    tree.initialize()
    refresher = MemorySemanticRefresher(tree, registry(), PathLock(ProcessLocalLockStore()), config=semantic)
    config = MemoryRecallLifecycleConfig()
    lifecycle = MemoryRecallLifecycle(
        SQLiteMemoryRecallLifecycleStore(tmp_path / "recall.sqlite3", config=config), config=config
    )
    return World(tree, refresher, lifecycle)


def search(service: SearchService, query: str = "问题", **kwargs: Any):  # type: ignore[no-untyped-def]
    return asyncio.run(service.search(query, **kwargs))


PREFERENCE = {"topic": "方案文档表述与结构", "content": "- 方案措辞偏正式，便于评审\n- 先写核心结论"}


# ── 卡片 ─────────────────────────────────────────────────────────────────────


def test_a_card_in_the_list_is_picked_read_handed_over_and_counted_once(tmp_path: Path) -> None:
    w = world(tmp_path)
    uri = w.put(MemoryKind.PREFERENCE, PREFERENCE)
    client = ScriptedClient([pick(read=["preferences/方案文档表述与结构.md"]), judge(["c1"])])

    result = search(w.service(client))

    assert [str(memory.uri) for memory in result.memories] == [str(uri)]
    assert "方案措辞偏正式" in result.context
    pick_prompt = client.prompts[0][1]
    assert "preferences/方案文档表述与结构.md — 方案措辞偏正式，便于评审 先写核心结论" in pick_prompt
    assert "revision" not in pick_prompt and "r1\t" not in pick_prompt  # 冷热用的版本、时间不给模型看
    (state,) = w.lifecycle.store.read_many((uri,))
    assert state.useful_recall_count == 1
    assert result.rounds == 1 and len(client.prompts) == 2


def test_reading_a_path_that_is_not_in_the_list_is_refused(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.put(MemoryKind.PREFERENCE, PREFERENCE)
    client = ScriptedClient([pick(read=["preferences/不存在.md", "entities/x/y.md"]), judge([])])

    result = search(w.service(client))

    assert result.memories == ()
    assert any("清单里没有 preferences/不存在.md" in note for note in result.notes)


def test_kinds_limit_the_list_and_limit_caps_the_cards_handed_over(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.put(MemoryKind.PREFERENCE, PREFERENCE)
    w.put(MemoryKind.PREFERENCE, {"topic": "通知措辞", "content": "- 通知要客气"})
    w.put(MemoryKind.ENTITY, {"category": "项目", "name": "五一拉新方案", "summary": "方案放在 plans/campaign.md。"})
    client = ScriptedClient(
        [pick(read=["preferences/方案文档表述与结构.md", "preferences/通知措辞.md"]), judge(["c1", "c2"])]
    )

    result = search(w.service(client), kinds=(MemoryKind.PREFERENCE,), limit=1)

    assert "五一拉新方案" not in client.prompts[0][1]
    assert len(result.memories) == 1


# ── 每一轮的记录 ───────────────────────────────────────────────────────────────


def test_an_answer_only_in_the_turn_records_is_found_by_date_and_keyword_in_its_own_section(tmp_path: Path) -> None:
    w = world(tmp_path)
    records = FakeRecords(
        (
            record("核对昨日广告花费与转化", 9, summary="花费 13670 元，转化 396 个"),
            record("记录体重", 7, day=date(2026, 4, 8), summary="60.8 公斤"),
        )
    )
    client = ScriptedClient([pick(records=source("2026-03-31", "2026-03-31", ["广告", "花费"])), judge(["r1"])])

    result = search(w.service(client, records=records))

    assert [item.name for item in result.records] == ["核对昨日广告花费与转化"]
    assert "【每一轮的记录】" in result.context and "13670" in result.context
    assert "60.8" not in result.context
    judge_prompt = client.prompts[1][1]
    assert "[r1] 记录 2026-03-31 09:00 核对昨日广告花费与转化" in judge_prompt


def test_with_the_behaviour_side_off_the_search_still_answers_from_the_other_sources(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.put(MemoryKind.PREFERENCE, PREFERENCE)
    client = ScriptedClient(
        [pick(read=["preferences/方案文档表述与结构.md"], records=source(None, None, ["方案"])), judge(["c1"])]
    )

    result = search(w.service(client, records=NoBehaviorRecords()))

    assert result.records == () and len(result.memories) == 1
    assert result.degradations == ()


def test_a_day_that_fails_to_read_is_skipped_and_the_rest_is_returned(tmp_path: Path) -> None:
    w = world(tmp_path)
    bad = date(2026, 3, 30)
    records = FakeRecords((record("周一例会", 9, day=bad), record("核对广告", 9)), failing=(bad,))
    client = ScriptedClient([pick(records=source("2026-03-30", "2026-03-31", ["广告", "例会"])), judge(["r1"])])

    result = search(w.service(client, records=records))

    assert [item.name for item in result.records] == ["核对广告"]
    assert any("2026-03-30" in note and "没读出来" in note for note in result.notes)


def test_unwatched_gaps_are_not_records_and_continuations_are_searchable(tmp_path: Path) -> None:
    """「未观测」「没读懂」是空白（gap），不进列出和搜索；名字照常的 occurrence（含续话这类）都读得到。"""

    tree = BehaviorTree(tmp_path / "behavior-tree")
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: local(23, 0))
    writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    writer.publish(BehaviorKind.GAP, gap_payload(gap_kind="未观测", started_at=local(21, 0), ended_at=local(21, 30)))
    source_records = BehaviorTreeRecords((("session", tree),))

    batch = source_records.read_days(source_records.days())

    assert [item.name for item in batch.records] == ["洗了手"]
    assert batch.records[0].day == DAY and batch.records[0].address.startswith("session:behavior://")
    w = world(tmp_path)
    client = ScriptedClient([pick(records=source(DAY.isoformat(), DAY.isoformat(), ["未观测"])), judge([])])
    result = search(w.service(client, records=source_records))
    assert result.records == ()
    assert "未观测" not in client.prompts[1][1].split("到目前为止找到的材料")[1].split("请判断")[0]


# ── 会话摘要 ──────────────────────────────────────────────────────────────────


def summary_match() -> ConversationSummaryMatch:
    summary = segment_summary()
    address = ConversationAddress(summary.conversation_id, summary.started_at.date())
    return ConversationSummaryMatch(
        summary_reference(address, summary), summary, "overview: 转写出的文字稿叫 meeting_transcript.txt"
    )


def test_an_answer_only_in_a_summary_is_found_in_the_first_round(tmp_path: Path) -> None:
    w = world(tmp_path)
    match = summary_match()
    client = ScriptedClient([pick(summaries=source(None, None, ["转写"])), judge(["s1"])])

    result = search(w.service(client, summaries=FakeSummaries((match,))))

    assert [item.reference.identity for item in result.summaries] == [match.reference.identity]
    assert result.rounds == 1 and len(client.prompts) == 2
    assert "【会话摘要】" in result.context and "meeting_transcript.txt" in result.context


def test_the_judge_is_told_the_later_material_wins_and_every_material_carries_its_time(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.put(MemoryKind.PREFERENCE, PREFERENCE)
    client = ScriptedClient(
        [
            pick(read=["preferences/方案文档表述与结构.md"], summaries=source(None, None, ["转写"])),
            judge(["c1", "s1"]),
        ]
    )

    search(w.service(client, summaries=FakeSummaries((summary_match(),))))

    judge_prompt = client.prompts[1][1]
    assert "以时间更晚的为准" in judge_prompt
    assert "（更新于 2026-08-19 12:00）" in judge_prompt
    assert "会话摘要（2026-07-01 16:00 到" in judge_prompt


def test_a_second_round_follows_the_judge_and_the_last_judge_cannot_ask_again(tmp_path: Path) -> None:
    w = world(tmp_path)
    records = FakeRecords((record("核对广告", 9),))
    client = ScriptedClient(
        [
            pick(),
            judge([], enough=False, following=pick(records=source(None, None, ["广告"]))),
            judge(["r1"], last=True),
        ]
    )

    result = search(w.service(client, records=records))

    assert result.rounds == 2 and len(client.prompts) == 3
    assert "最后一次可以再找" in client.prompts[1][1] and "已经不能再找了" in client.prompts[2][1]
    assert [item.name for item in result.records] == ["核对广告"]


# ── 时间上限与记账 ────────────────────────────────────────────────────────────


def test_at_the_time_limit_what_was_found_is_handed_over_within_the_limit(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.put(MemoryKind.PREFERENCE, PREFERENCE)
    config = MemorySearchServiceConfig(time_limit_seconds=1.0, accounting_reserve_seconds=0.2)
    client = ScriptedClient([pick(read=["preferences/方案文档表述与结构.md"]), judge(["c1"])], delays={1: 5.0})

    started = time.monotonic()
    result = search(w.service(client, config=config))
    elapsed = time.monotonic() - started

    assert elapsed <= 1.0 + 0.15
    assert result.timed_out
    assert len(result.memories) == 1 and result.context
    assert any(item.stage is MemorySearchDegradationStage.TIME_LIMIT for item in result.degradations)


class SlowExpander:
    """记账一直拿不到锁：等过给它的时间。"""

    def __init__(self, *, hang: bool) -> None:
        self.hang = hang

    def expand_for_probe(self, document):  # type: ignore[no-untyped-def]
        return document

    async def record_context_use(self, targets, *, used_at=None, lock_wait_seconds=5.0):  # type: ignore[no-untyped-def]
        if self.hang:
            await asyncio.sleep(60)
        return type(
            "Receipt", (), {"documents": (), "rejected_uris": (), "skipped_uris": tuple(t.uri for t in targets)}
        )()


@pytest.mark.parametrize("hang", [True, False])
def test_accounting_that_cannot_get_the_lock_is_skipped_observed_and_the_cards_still_come_back(
    tmp_path: Path, hang: bool
) -> None:
    w = world(tmp_path)
    w.put(MemoryKind.PREFERENCE, PREFERENCE)
    observer = Recorder()
    config = MemorySearchServiceConfig(time_limit_seconds=1.0, accounting_reserve_seconds=0.3)
    client = ScriptedClient([pick(read=["preferences/方案文档表述与结构.md"]), judge(["c1"])])

    started = time.monotonic()
    result = search(w.service(client, config=config, expander=SlowExpander(hang=hang), observer=observer))

    assert time.monotonic() - started <= 1.0 + 0.15
    assert len(result.memories) == 1
    assert any(item.stage is MemorySearchDegradationStage.RECALL_LIFECYCLE for item in result.degradations)
    assert any(event.operation == "use_accounting" and event.attributes["skipped"] == 1 for event in observer.events)


# ── 清单读不出、模型不可用 ────────────────────────────────────────────────────


class FlakyCatalog:
    def __init__(self, inner: MemoryCatalogReader, failures: int) -> None:
        self.inner = inner
        self.tree = inner.tree
        self.failures = failures
        self.calls = 0

    def read(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("memory overview cannot be read")
        return self.inner.read(**kwargs)

    def match_unlisted(self, *args: Any):  # type: ignore[no-untyped-def]
        return self.inner.match_unlisted(*args)


def test_an_unreadable_catalog_is_not_retried_and_the_other_sources_are_still_searched(tmp_path: Path) -> None:
    """阶段 3 去掉"提交窗口内拒读"后也去掉了为它加的短暂重试：清单读不出就读一次、照常查另外两处并说明。"""

    w = world(tmp_path)
    w.put(MemoryKind.PREFERENCE, PREFERENCE)
    config = MemorySearchServiceConfig()
    stuck = FlakyCatalog(w.catalog(config), failures=99)
    records = FakeRecords((record("核对广告", 9),))
    client = ScriptedClient([pick(records=source(None, None, ["广告"])), judge(["r1"])])
    result = search(w.service(client, config=config, catalog=stuck, records=records))  # type: ignore[arg-type]
    assert stuck.calls == 1
    assert "（没有卡片）" in client.prompts[0][1]
    assert [item.name for item in result.records] == ["核对广告"]
    assert any("清单这次读不出来" in note for note in result.notes)
    assert any(item.stage is MemorySearchDegradationStage.CATALOG for item in result.degradations)


def test_search_answers_normally_while_a_memory_job_is_writing(tmp_path: Path) -> None:
    """助手查记忆时正好有记忆任务写到一半（锁还拿着）：照常返回，没有拒读。"""

    from threading import Event, Thread

    from habitus.infrastructure.store.contracts import PathLock
    from habitus.memory.tree import MemoryDocumentLockKeyspace

    w = world(tmp_path)
    uri = w.put(MemoryKind.PREFERENCE, PREFERENCE)
    lock = PathLock(ProcessLocalLockStore())
    held, release = Event(), Event()

    def hold_the_card_lock() -> None:
        with lock.acquire(MemoryDocumentLockKeyspace(w.tree.root).key(uri), ttl_seconds=30):
            held.set()
            release.wait(timeout=5)

    writer = Thread(target=hold_the_card_lock)
    writer.start()
    assert held.wait(timeout=5)
    try:
        client = ScriptedClient([pick(read=["preferences/方案文档表述与结构.md"]), judge(["c1"])])
        result = search(w.service(client))
    finally:
        release.set()
        writer.join(timeout=5)
    assert [memory.uri for memory in result.memories] == [uri]
    assert not any(item.stage is MemorySearchDegradationStage.CATALOG for item in result.degradations)


def test_one_hop_expansion_skips_dangling_and_one_sided_relations_with_a_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """检索补一跳这条读路径：指向不存在文件的、对方没记另一边的关联跳过并记日志，好的那条照常带出。"""

    import logging

    from habitus.memory.document import MemoryLinkType, MemoryStoredLink

    w = world(tmp_path)
    seed = MemoryURI.parse("memory://preferences/方案文档表述与结构.md")
    good = MemoryURI.parse("memory://entities/项目/habitus.md")
    one_sided = MemoryURI.parse("memory://entities/项目/看板.md")
    ghost = MemoryURI.parse("memory://entities/项目/已删.md")
    links = tuple(
        sorted(
            (MemoryStoredLink(seed, target, MemoryLinkType.BELONGS_TO) for target in (good, one_sided, ghost)),
            key=lambda item: item.identity,
        )
    )
    meta = MemoryDocumentMetadata(revision=1, created_at=NOW - timedelta(days=1), updated_at=NOW - timedelta(days=1))
    w.tree.write(codec().build(MemoryKind.PREFERENCE, PREFERENCE, metadata=meta, links=links))
    w.tree.write(
        codec().build(
            MemoryKind.ENTITY,
            {"category": "项目", "name": "Habitus", "summary": "记忆系统。"},
            metadata=meta,
            backlinks=(MemoryStoredLink(seed, good, MemoryLinkType.BELONGS_TO),),
        )
    )
    w.tree.write(
        codec().build(MemoryKind.ENTITY, {"category": "项目", "name": "看板", "summary": "看板。"}, metadata=meta)
    )
    w.refresher.rebuild()
    client = ScriptedClient([pick(read=["preferences/方案文档表述与结构.md"]), judge(["c1"])])

    with caplog.at_level(logging.WARNING):
        result = search(w.service(client))

    assert [memory.uri for memory in result.memories][:1] == [seed]
    related = [item.relation.to_uri for memory in result.memories for item in memory.related]
    assert good in related and one_sided not in related and ghost not in related
    assert "missing L2 document" in caplog.text and "one-sided memory relation" in caplog.text


def test_with_the_model_breaker_open_only_the_profile_comes_back_with_the_reason(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.put(MemoryKind.PROFILE, {"content": "- 用户是产品经理"})
    w.put(MemoryKind.PREFERENCE, PREFERENCE)
    error = ModelCircuitOpenError("open", opened_by=ModelErrorKind.QUOTA, retry_after_seconds=60.0)
    client = ScriptedClient([], error=error)

    result = search(w.service(client, records=FakeRecords((record("核对广告", 9),))))

    assert [memory.uri for memory in result.memories] == [PROFILE_URI]
    assert result.records == () and result.summaries == ()
    assert "额度或余额用完了" in result.notes[0] and "只交出了档案" in result.notes[0]
    assert "额度或余额用完了" in result.context
    assert any(item.stage is MemorySearchDegradationStage.PICK for item in result.degradations)


# ── 冷热 ─────────────────────────────────────────────────────────────────────


def test_the_coldest_cards_are_listed_apart_by_name_one_per_search_and_warm_up_once_used(tmp_path: Path) -> None:
    w = world(tmp_path)
    old = NOW - timedelta(days=400)
    w.put(MemoryKind.ENTITY, {"category": "人", "name": "老王", "summary": "老王是前同事。"}, updated=old)
    w.put(MemoryKind.ENTITY, {"category": "人", "name": "老李", "summary": "老李是房东。"}, updated=old)
    client = ScriptedClient([pick(read=["entities/人/老王.md", "entities/人/老李.md"]), judge(["c1", "c2"])])

    result = search(w.service(client))

    pick_prompt = client.prompts[0][1]
    cold_section = pick_prompt.split("## 很久没用到的卡片")[1]
    assert "- entities/人/老王.md\n" in cold_section + "\n" and "老王是前同事" not in pick_prompt
    assert len(result.memories) == 1  # 最冷那一组每次最多一张
    picked = result.memories[0].uri
    client = ScriptedClient([pick(), judge([])])
    search(w.service(client))
    normal = client.prompts[0][1].split("## 很久没用到的卡片")[0]
    assert str(picked).removeprefix("memory://") in normal  # 用过一次回暖，回到普通清单


# ── 目录超出清单上限 ──────────────────────────────────────────────────────────


def test_a_directory_over_the_list_bound_says_how_many_are_missing_and_keywords_find_them(tmp_path: Path) -> None:
    w = world(tmp_path, semantic=MemorySemanticConfig(max_direct_entries=2))
    w.put(
        MemoryKind.PREFERENCE, {"topic": "最早的主题", "content": "- 早早定下的习惯"}, updated=NOW - timedelta(days=3)
    )
    w.put(MemoryKind.PREFERENCE, {"topic": "中间的主题", "content": "- 中间"}, updated=NOW - timedelta(days=2))
    w.put(MemoryKind.PREFERENCE, {"topic": "最新的主题", "content": "- 最新"}, updated=NOW - timedelta(days=1))
    overview = w.tree.read_layer(MemoryURI.parse("memory://preferences").to_directory(), MemoryLevel.OVERVIEW)
    assert "（还有 1 张卡片没有列出）" in overview and "最早的主题" not in overview

    client = ScriptedClient([pick(card_keywords=["习惯"]), judge(["c1"])])
    result = search(w.service(client))

    assert "还有 1 张卡片没有列出" in client.prompts[0][1]
    assert [str(memory.uri) for memory in result.memories] == ["memory://preferences/最早的主题.md"]


def test_a_card_written_without_refreshing_its_overview_appears_after_the_maintenance_rebuild(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.put(MemoryKind.PREFERENCE, PREFERENCE)
    later = codec().build(
        MemoryKind.PREFERENCE,
        {"topic": "通知措辞", "content": "- 通知要客气"},
        metadata=MemoryDocumentMetadata(revision=1, created_at=NOW, updated_at=NOW),
    )
    w.tree.write(later)  # 写完卡片、刷概览之前进程没了

    client = ScriptedClient([pick(), judge([])])
    search(w.service(client))
    assert "通知措辞" not in client.prompts[0][1]

    w.refresher.rebuild()
    client = ScriptedClient([pick(), judge([])])
    search(w.service(client))
    assert "preferences/通知措辞.md — 通知要客气" in client.prompts[0][1]


def test_search_config_keeps_the_accounting_reserve_inside_the_time_limit() -> None:
    with pytest.raises(ValueError, match="accounting reserve"):
        MemorySearchServiceConfig(time_limit_seconds=1.0, accounting_reserve_seconds=1.0)
    assert MemorySearchServiceConfig().time_limit_seconds < 15.0  # 插件单个请求 15 秒就断


# ── codex 复核后补的几条 ──────────────────────────────────────────────────────


class FailingSecondCall(ScriptedClient):
    """ "挑"照常答，"判"那一步模型不可用。"""

    async def complete_json_async(self, request, *, schema, name, validator, context):  # type: ignore[no-untyped-def]
        if self.prompts:
            self.prompts.append((name, request.messages[-1].content))
            raise ModelCircuitOpenError("open", opened_by=ModelErrorKind.SERVICE, retry_after_seconds=30.0)
        return await super().complete_json_async(
            request, schema=schema, name=name, validator=validator, context=context
        )


def test_a_model_failure_at_the_judge_step_also_hands_over_only_the_profile(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.put(MemoryKind.PROFILE, {"content": "- 用户是产品经理"})
    w.put(MemoryKind.PREFERENCE, PREFERENCE)
    client = FailingSecondCall([pick(read=["preferences/方案文档表述与结构.md"], records=source(None, None, ["广告"]))])

    result = search(w.service(client, records=FakeRecords((record("核对广告", 9),))))

    assert [memory.uri for memory in result.memories] == [PROFILE_URI]
    assert result.records == ()
    assert "只交出了档案" in result.notes[-1]
    assert any(item.stage is MemorySearchDegradationStage.JUDGE for item in result.degradations)


def test_reading_the_profile_after_the_model_failed_stays_within_the_time_limit(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.put(MemoryKind.PROFILE, {"content": "- 用户是产品经理"})
    config = MemorySearchServiceConfig(time_limit_seconds=1.0, accounting_reserve_seconds=0.2)
    service = w.service(ScriptedClient([], error=RuntimeError("down")), config=config)
    slow = service.snapshot_reader.read

    def stuck(uri):  # type: ignore[no-untyped-def]
        time.sleep(3)
        return slow(uri)

    service.snapshot_reader.read = stuck  # type: ignore[method-assign]

    async def timed():  # type: ignore[no-untyped-def]
        # 在协程里量：asyncio.run 收尾时还会等那个仍在睡的线程，那不是查询本身的耗时
        started = time.monotonic()
        result = await service.search("问题")
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(timed())
    assert elapsed <= 1.0 + 0.15
    assert result.memories == ()


def test_keyword_matches_among_unlisted_cards_skip_retired_ones_and_count_cold_ones_once(tmp_path: Path) -> None:
    w = world(tmp_path, semantic=MemorySemanticConfig(max_direct_entries=1))
    old = NOW - timedelta(days=500)
    w.put(MemoryKind.PREFERENCE, {"topic": "习惯甲", "content": "- 早年的习惯甲"}, updated=old)
    w.put(MemoryKind.PREFERENCE, {"topic": "习惯乙", "content": "- 早年的习惯乙"}, updated=old - timedelta(days=1))
    retired = w.put(
        MemoryKind.PREFERENCE, {"topic": "习惯丙", "content": "- 早年的习惯丙"}, updated=old - timedelta(days=2)
    )
    w.put(MemoryKind.PREFERENCE, {"topic": "最新", "content": "- 最新"}, updated=NOW - timedelta(days=1))
    target = MemoryRecallTarget(retired, 1, old - timedelta(days=2))
    w.lifecycle.store.mark_retire_candidate(target, marked_at=NOW - timedelta(days=40), expected_version=0)
    w.lifecycle.store.mark_retired(target, retired_at=NOW - timedelta(days=1), expected_version=1)
    client = ScriptedClient([pick(card_keywords=["习惯"]), judge(["c1", "c2"])])

    result = search(w.service(client))

    assert len(result.memories) == 1  # 三张都命中关键词：退休的不读，两张最冷的只读一张
    assert result.memories[0].uri != retired


def test_a_request_without_dates_or_keywords_lists_the_latest_records_and_summaries(tmp_path: Path) -> None:
    w = world(tmp_path)
    records = FakeRecords(
        tuple(record(f"事{index}", 9, day=date(2026, 3, 1) + timedelta(days=index)) for index in range(100))
    )
    config = MemorySearchServiceConfig(max_record_list=5)
    client = ScriptedClient([pick(records=source(), summaries=source()), judge([], last=False)])

    search(w.service(client, records=records, summaries=FakeSummaries((summary_match(),)), config=config))

    materials = client.prompts[1][1].split("到目前为止找到的材料")[1]
    assert "事99" in materials and "事95" in materials and "事94" not in materials
    assert "会话摘要（" in materials


def test_the_coldest_group_stays_within_the_list_character_bound(tmp_path: Path) -> None:
    w = world(tmp_path)
    old = NOW - timedelta(days=500)
    for index in range(80):
        w.put(
            MemoryKind.ENTITY,
            {"category": "人", "name": f"很久以前认识的人{index:02d}", "summary": "前同事。"},
            updated=old,
        )
    config = MemorySearchServiceConfig(max_catalog_chars=1_024)
    catalog = w.catalog(config).read(kinds=(), now=NOW, on_recall_failure=lambda _error: None)

    assert len(catalog.render()) <= config.max_catalog_chars
    assert catalog.cold and catalog.unlisted_count == 80 - len(catalog.cold)


def test_keyword_ties_in_turn_records_put_the_newer_first_and_keep_the_newer_under_the_limit() -> None:
    """用户 2026-11-23 定：命中词数相同时近的在前，超过条数上限留下的是近的；命中词更多的仍排最前。"""

    from habitus.memory.retrieval.records import search_records

    records = tuple(record(f"周报{index}", 9, day=date(2026, 3, 1) + timedelta(days=index)) for index in range(6))
    best = record("周报总结", 9, day=date(2026, 3, 1), summary="周报 总结")

    hits, total = search_records((*records, best), ("周报", "总结"), limit=3)

    assert total == 7
    assert [item.name for item in hits] == ["周报总结", "周报5", "周报4"]


def test_keyword_ties_in_summaries_put_the_newer_first(tmp_path: Path) -> None:
    from dataclasses import replace

    w = world(tmp_path)
    base = summary_match()
    matches = []
    for index, day in enumerate((5, 20, 12)):
        shift = timedelta(days=day)
        summary = replace(
            base.summary,
            segment_id=f"segment-{index}",
            started_at=base.summary.started_at + shift,
            ended_at=base.summary.ended_at + shift,
            generated_at=base.summary.generated_at + shift,
        )
        address = ConversationAddress(summary.conversation_id, summary.started_at.date())
        matches.append(ConversationSummaryMatch(summary_reference(address, summary), summary, base.content))
    config = MemorySearchServiceConfig(max_summary_hits=2)
    client = ScriptedClient([pick(summaries=source(None, None, ["转写"])), judge([], last=False)])

    search(w.service(client, summaries=FakeSummaries(tuple(matches)), config=config))

    materials = client.prompts[1][1].split("到目前为止找到的材料")[1]
    days = [f"{(base.summary.started_at + timedelta(days=day)).astimezone(CST):%Y-%m-%d}" for day in (20, 12, 5)]
    assert materials.index(days[0]) < materials.index(days[1]) and days[2] not in materials
