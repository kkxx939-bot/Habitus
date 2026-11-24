"""第二步丙一（下游认上下级、停用后不说、删候选条件；上下级分工按裁定 105）测试共用的现场。

一份真实的盘：真实行为树（记录直接发布进去，``kind_token`` 用会话初始词表的编号）、真实词表状态（``BehaviorKindStore``，在用与
自己长的节点直接写进状态）、组合根装的语义树（``runtime.scene.build_scene_nightly``）与预测层（``runtime.foresight`` 的
``build_foresight_components`` / ``build_foresight_nightly`` / ``build_foresight_days`` / ``build_foresight_tick``，词表状态经
``kind_store`` 交进去）。时钟由现场拨。

断言只落在这些口子与盘上：换代的语义树一步交出的材料（概念集、时间线上的记号、复现）、关系表、机会表、账本里的预期 / 递送 /
结算、提醒出口的可取清单、归约盖 ``reminded`` 的查法。模型一律是确定性替身：语义树与预测层不该调模型（``ModelMustNotBeCalled``），
模型队列那几条用照着提示词答的替身（``tests.unit.scene.advice_fixtures``）。

``plain`` 把这些产物换成可比的 JSON（浮点保留 10 位有效数字：跨平台的末位差不算变）；``assert_contains`` 逐项比，只认基线里
有的字段（实现新加的字段不算差别）。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import random
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import Vocabulary, Word, WordOrigin
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.conditions import LocalCalendar
from habitus.config import HabitusConfig
from habitus.foresight.edges import PastMoment
from habitus.foresight.history import Moment, UnitHistory
from habitus.foresight.ledger import AccountFact, Delivery, Expectation, account_key
from habitus.foresight.model import NoUnsealed
from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.model_client import ChatClient
from habitus.model_client.structured import StructuredChatClient
from habitus.runtime.evidence import evidence_of
from habitus.runtime.foresight import (
    build_foresight_components,
    build_foresight_days,
    build_foresight_nightly,
    build_foresight_tick,
    build_outlet,
    foresight_ledger,
)
from habitus.runtime.scene import build_scene_nightly, scene_branch_root
from habitus.runtime.scene_night import NightReport, SceneNightlyRun
from habitus.series.reader import read_series
from tests.unit.foresight.scripted_model import model_config
from tests.unit.runtime.concept_layer_site import SUBJECT, ModelMustNotBeCalled, example_mapping
from tests.unit.runtime.test_model_queue_acceptance import queue_on
from tests.unit.scene.advice_fixtures import Answering
from tests.unit.scene.fixtures import CST, Site, publish

HERE = Path(__file__).resolve().parent

WRITE_DOC = "s-k0001"  # 写文档（第 1 层）
WEEKLY = "s-k0002"  # 写周报（第 2 层，写文档的下级）
SUMMARY = "s-k0003"  # 写总结（第 2 层，写文档的下级）
PLAN = "s-k0004"  # 写方案（第 2 层，写周报的兄弟）
ABSTRACT = "s-k0019"  # 写摘要（第 1 层，没有下级）
MAIL = "s-k0043"  # 看邮件（第 1 层，没有下级）
REPLY = "s-k0046"  # 回邮件（第 1 层，没有下级）
QUERY = "s-k0098"  # 查数据（第 1 层，另一个领域）
ACCOUNT = "s-k0192"  # 记账（第 1 层，没有下级）
NAMES = {
    WRITE_DOC: "写文档",
    WEEKLY: "写周报",
    SUMMARY: "写总结",
    PLAN: "写方案",
    ABSTRACT: "写摘要",
    MAIL: "看邮件",
    REPLY: "回邮件",
    QUERY: "查数据",
    ACCOUNT: "记账",
}


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=CST)


def workdays(start: date, count: int) -> list[date]:
    found: list[date] = []
    current = start
    while len(found) < count:
        if current.weekday() < 5:
            found.append(current)
        current += timedelta(days=1)
    return found


class World:
    """一份盘：行为树、词表状态、语义树（组合根装）、预测层（组合根装，用到才建）、可拨的时钟与定稿日。"""

    def __init__(
        self, root: Path, *, structured_chat: Any | None = None, edit: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        self.root = root
        self.clock_at = at(date(2026, 4, 1), 1)
        self.site = Site(root, now=self.clock_at)
        self.tree = self.site.behavior_tree
        self.store = BehaviorKindStore(
            root / "kinds",
            lane=Lane.SESSION,
            lock_store=SQLiteLockStore(root / "locks.sqlite3"),
            lock_ttl_seconds=30,
        )
        payload = example_mapping(root)
        payload["behavior"] = {"primary_subject": SUBJECT}
        payload["scene"] = {"enabled": True}
        payload["foresight"] = {"enabled": True}
        if edit is not None:
            edit(payload)
        self.config = HabitusConfig.from_mapping(payload)
        self.client = structured_chat if structured_chat is not None else ModelMustNotBeCalled()
        self.calendar = LocalCalendar()
        run = build_scene_nightly(
            self.config,
            root=scene_branch_root(self.config, "session"),
            behavior_tree=self.tree,
            kind_store=self.store,
            structured_chat=self.client,
            calendar=self.calendar,
        )
        assert run is not None
        self.scene: SceneNightlyRun = run
        self.closed: list[date] = []
        self._foresight: Any | None = None

    # ── 时钟 ──
    def clock(self) -> datetime:
        return self.clock_at

    def set_clock(self, moment: datetime) -> None:
        self.clock_at = moment

    # ── 词表 ──
    def vocabulary(self) -> Vocabulary:
        return self.store.read()

    def use(self, *tokens: str, since: date = date(2026, 3, 1)) -> None:
        """这几个编号在用（其余在用的照旧）。"""

        def change(current: Vocabulary) -> Vocabulary:
            active = dict(current.active)
            active.update({ClassId.parse(token): since for token in tokens})
            return current.with_active(active)

        self.store.update(change)

    def stop_using(self, *tokens: str) -> None:
        """这几个编号退出在用（记退出那天）。"""

        drop = {ClassId.parse(token) for token in tokens}

        def change(current: Vocabulary) -> Vocabulary:
            active = {key: value for key, value in current.active.items() if key not in drop}
            exited = {**current.exited, **dict.fromkeys(drop, self.clock_at.date())}
            return current.with_active(active, exited)

        self.store.update(change)

    def grow(self, parent: str, name: str, *, written: datetime | None = None) -> str:
        """在 ``parent`` 下面写一个自己长的节点（候选，不在用），返回它的编号。"""

        made: list[str] = []

        def change(current: Vocabulary) -> Vocabulary:
            upper = current.get(ClassId.parse(parent))
            word = Word(
                id=current.next_id(),
                level=upper.level + 1,
                name=name,
                parent=upper.id,
                definition=f"{name}这一种",
                domain=upper.domain,
                origin=WordOrigin.GROWN,
                created_at=written or self.clock_at,
            )
            made.append(str(word.id))
            return current.with_word(word)

        self.store.update(change)
        return made[-1]

    def descendants(self, token: str) -> set[str]:
        """词表里挂在 ``token`` 下面、活着的全部节点（下级的下级也算）。"""

        vocabulary = self.vocabulary()
        found: set[str] = set()
        pending = [ClassId.parse(token)]
        while pending:
            for child in vocabulary.children(pending.pop()):
                found.add(str(child.id))
                pending.append(child.id)
        return found

    # ── 记录 ──
    def record(self, token: str, moment: datetime, *, lasts: int = 10, **fields: Any) -> str:
        """发布一条挂在 ``token`` 上的记录，返回它的 URI。"""

        return publish(
            self.tree,
            moment.date(),
            NAMES.get(token, token),
            moment.hour,
            moment.minute,
            kind=token,
            lasts_minutes=lasts,
            **fields,
        )

    # ── 换代的语义树一步 ──
    def generate(self, night: date) -> NightReport:
        """换代的语义树一步，序列截止到 ``night``（``night`` 之前的日子已封口），此刻是 ``night`` 的 01:00。"""

        self.set_clock(at(night, 1))
        return asyncio.run(
            self.scene.generate(
                series=read_series(self.tree, cutoff=night),
                words=self.scene.syncing.catalog.current(),
                now=self.clock_at,
            )
        )

    # ── 预测层 ──
    @property
    def foresight(self) -> Any:
        if self._foresight is None:
            evidence = evidence_of(self.config)
            ledger = foresight_ledger(self.config, "session")
            self._foresight = build_foresight_components(
                self.config,
                lane="session",
                ledger=ledger,
                outlet=build_outlet(self.config, {"session": ledger}, kind_stores={"session": self.store}),
                behavior_tree=self.tree,
                closed_days=lambda: tuple(self.closed),
                evidence=evidence,
                clock=self.clock,
                calendar=self.calendar,
                kind_store=self.store,
                finished=lambda _day: True,
            )
        return self._foresight

    def opportunities(self, report: NightReport) -> Any:
        """机会表那一拍（换代里语义树之后那一步）：用语义树这一代交出来的材料写这一代的表。"""

        materials = report.materials
        assert materials is not None
        nightly = build_foresight_nightly(
            self.foresight, relations=self.scene.relations, relation_config=self.scene.relation_config, clock=self.clock
        )
        return nightly.run(
            night=report.night,
            concepts=materials.concepts,
            timelines=materials.timelines,
            recurrences=materials.recurrences,
            coverage=materials.coverage,
        )

    def day_expectations(self) -> Any:
        """换代里当天的日子预期那一步。"""

        stage = build_foresight_days(
            self.foresight,
            config=self.config,
            unsealed=NoUnsealed(),
            evidence=evidence_of(self.config),
            clock=self.clock,
        )
        return asyncio.run(stage.run_once())

    def ticker(self, unsealed: Any | None = None) -> Any:
        return build_foresight_tick(
            self.foresight,
            config=self.config,
            unsealed=unsealed if unsealed is not None else NoUnsealed(),
            evidence=evidence_of(self.config),
            clock=self.clock,
        )

    def tick(self, moment: datetime, ticker: Any | None = None) -> Any:
        """``moment`` 那一拍（建预期、递送）。"""

        self.set_clock(moment)
        return asyncio.run((ticker or self.ticker()).run_once())

    def settle(self, *days: date) -> Any:
        """结算那一拍：``days`` 定稿了。"""

        self.closed = sorted(set(self.closed) | set(days))
        return self.foresight.settlement.run_once()

    # ── 账 ──
    @property
    def ledger(self) -> Any:
        return self.foresight.ledger

    def earn(
        self, kind: str, source: str = "日子", trigger: str = "interval", *, before: datetime, hits: int = 8
    ) -> str:
        """这本账前几天各一刻都来了（账上的流水 + 单元历史的快照），让它到提醒档。返回账的键。"""

        account = account_key(kind, source, trigger)
        moments = []
        for offset in range(hits, 0, -1):
            moment = before - timedelta(days=offset)
            self.ledger.append_fact(AccountFact(account, moment, True, 1.0, "准时", f"old-{kind}-{offset}"))
            moments.append(PastMoment(moment, frozenset({account}), True))
        store = self.foresight.history.store
        known = store.latest("session")
        units = dict(known.moments) if known is not None else {}
        units[kind] = tuple(Moment(item, (f"{kind}-{index}",)) for index, item in enumerate(moments))
        night = before.date() - timedelta(days=1)
        if known is not None and known.night >= night:
            night = known.night + timedelta(days=1)
        store.write(UnitHistory("session", night, units))
        return account

    def host_alive(self, moment: datetime) -> None:
        """有宿主来取过（提醒出口算活着）。"""

        outlet = self.foresight.outlet
        outlet.claim(outlet.tools[0].host, "probe", moment, 1)

    def deliveries(self, day: date) -> dict[str, list[str]]:
        """预期 → 它的递送原因（按决定先后）。"""

        found: dict[str, list[str]] = {}
        for item in sorted(self.ledger.deliveries_on(day), key=lambda made: made.decided_at):
            found.setdefault(item.expectation_id, []).append(item.reason)
        return found


def expectation(
    identity: str,
    kind: str,
    formed: datetime,
    *,
    window: tuple[int, int] = (10, 40),
    deadline_minutes: int = 120,
    source: str = "日子",
    trigger: str = "interval",
    observed_from: datetime | None = None,
) -> Expectation:
    """一条预期（窗、最晚时限按 ``formed`` 起算的分钟数；说中的范围是 (``observed_from``, 最晚时限]）。默认是「日子」触发：
    事件触发要机会表上的关系已成立才能开口（``foresight.units.voiced``），这里没有机会表；递送那几条测的是范围与上下级，不是依据。"""

    return Expectation(
        expectation_id=identity,
        kind_token=kind,
        day=formed.date(),
        formed_at=formed,
        source=source,
        trigger=trigger,
        trigger_instance=f"occ:{identity}.md",
        window=(formed + timedelta(minutes=window[0]), formed + timedelta(minutes=window[1])),
        deadline=formed + timedelta(minutes=deadline_minutes),
        basis=(),
        opportunities=8,
        hits=8,
        observed_from=observed_from or formed,
    )


def spoken(world: World, item: Expectation, moment: datetime) -> None:
    """这条预期 ``moment`` 说出口了（「说了」）。"""

    world.ledger.record_expectation(item)
    world.ledger.record_delivery(
        item.day, Delivery(item.expectation_id, moment, 1, True, "说了", 1.0, "该做了", via="codex", turn="t1")
    )


# ── 基线那一份数据（S1、S4、C30、S5、C28）─────────────────────────────────


BASELINE_START = date(2026, 3, 2)  # 周一
BASELINE_DAYS = 60
#: 第一代（发现候选）与第二代（前向验证、成立）的截止日。
BASELINE_NIGHTS = (
    workdays(BASELINE_START, 30)[-1] + timedelta(days=1),
    workdays(BASELINE_START, 60)[-1] + timedelta(days=1),
)
#: 基线里在用的词：没有一对是上下级（写周报、写方案是兄弟；其余都是没有下级的第 1 层），也没有不在用却有记录的类。
BASELINE_IN_USE = (WEEKLY, PLAN, ABSTRACT, MAIL, REPLY, ACCOUNT)


def plant_baseline(world: World, *, in_use: Iterable[str] = BASELINE_IN_USE) -> None:
    """60 个工作日：每天 09:00 看邮件、17:50 记账；中间写周报 1–2 次，一半之前 15–30 分钟先回过邮件、在书房写，只有这些之后
    九成接着写方案（另一半在办公室写、之后不接）；每天另有一次写摘要。同在的人随机多一位同事（不影响走向）。固定种子。"""

    generator = random.Random(7)
    for day in workdays(BASELINE_START, BASELINE_DAYS):
        world.record(MAIL, at(day, 9, 0))
        world.record(ACCOUNT, at(day, 17, 50))
        for _ in range(generator.randint(1, 2)):
            minute = generator.randint(9 * 60 + 20, 16 * 60)
            primed = generator.random() < 0.5
            company = ("家庭成员A", "同事B") if generator.random() < 0.4 else ("家庭成员A",)
            if primed:
                world.record(REPLY, at(day, *divmod(minute - generator.randint(15, 30), 60)))
            world.record(WEEKLY, at(day, *divmod(minute, 60)), place="书房" if primed else "办公室", subjects=company)
            if primed and generator.random() < 0.9:
                world.record(PLAN, at(day, *divmod(minute + generator.randint(15, 30), 60)))
        world.record(ABSTRACT, at(day, *divmod(generator.randint(9 * 60 + 20, 16 * 60), 60)))
    world.use(*in_use, since=BASELINE_START)


#: 基线那一天（第二代的截止日）里的记录：回邮件 → 书房写周报（预期写方案）→ 下午写摘要。
def baseline_today(world: World) -> None:
    today = BASELINE_NIGHTS[1]
    world.record(REPLY, at(today, 10, 0))
    world.record(WEEKLY, at(today, 10, 20), place="书房")
    world.record(ABSTRACT, at(today, 15, 0))


def run_baseline(world: World) -> dict[str, Any]:
    """整条链跑一遍：两代语义树 → 第二代的机会表与当天的日子预期 → 那一天 09:00–20:00 每一拍（建预期、递送）。返回 S1 的产物。"""

    first = world.generate(BASELINE_NIGHTS[0])
    second = world.generate(BASELINE_NIGHTS[1])
    today = second.night
    table_report = world.opportunities(second)
    days = world.day_expectations()
    baseline_today(world)
    ticker = world.ticker()
    for slot in range(9 * 4, 20 * 4):
        world.tick(at(today, 0) + timedelta(minutes=15 * slot), ticker)
    materials = second.materials
    assert materials is not None
    return {
        "nights": [first.night, second.night],
        "concepts": {name: plain(materials.concepts[name]) for name in materials.concepts},
        "recurrences": plain(materials.recurrences),
        "marks": {
            lane: {identity: [mark.uri for mark in marks] for identity, marks in sorted(line.marks.items())}
            for lane, line in sorted(materials.timelines.items())
        },
        "relations": {
            night.isoformat(): plain(world.scene.relations.read("session", night)) for night in BASELINE_NIGHTS
        },
        "opportunities": plain(world.foresight.opportunities.latest("session")),
        "opportunity_report": {"failed": list(table_report.failed)},
        "day_expectations": {"skipped": days.skipped},
        "expectations": sorted(
            (plain(item) for item in world.ledger.expectations_on(today)), key=lambda item: item["expectation_id"]
        ),
        "deliveries": sorted(
            (plain(item) for item in world.ledger.deliveries_on(today)),
            key=lambda item: (item["expectation_id"], item["decided_at"], item["reason"]),
        ),
    }


def sibling_readings(state: Any, pair: tuple[str, str]) -> dict[str, Any]:
    """关系表里两个节点之间（两个方向、各段、各条件）每一行当晚的读数（不含判定与权重：判定随检验族变，S4 只要读数不变）。"""

    found: dict[str, Any] = {}
    for relation in state.relations:
        key = relation.key
        if {key.antecedent, key.consequent} != set(pair) or relation.tonight is None:
            continue
        reading = plain(relation.tonight)
        for dropped in ("verdict", "weight", "key"):
            reading.pop(dropped, None)
        found[f"{key.antecedent}->{key.consequent}|{key.segment.value}|{key.condition}"] = reading
    return found


def conditioned_established(state: Any) -> list[str]:
    """成立的带条件关系（关系键的文字）。"""

    return sorted(
        f"{relation.key.antecedent}->{relation.key.consequent}|{relation.key.segment.value}|{relation.key.condition}"
        for relation in state.relations
        if relation.key.condition and relation.status.value == "established"
    )


# ── 可比的 JSON ───────────────────────────────────────────────────────────


def plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: plain(getattr(value, item.name))
            for item in dataclasses.fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Enum):
        return plain(value.value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(plain(key)) if not isinstance(key, str) else key: plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(
            (plain(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False)
        )
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return float(f"{value:.10g}")
    return repr(value)


def assert_contains(expected: Any, actual: Any, path: str = "") -> None:
    """``actual`` 逐项含着 ``expected``：字典只比 ``expected`` 里有的键（实现新加的字段不算差别），列表逐项比、长度相同。"""

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected a mapping, got {actual!r}"
        missing = sorted(set(expected) - set(actual))
        assert not missing, f"{path}: missing keys {missing}"
        for key, item in expected.items():
            assert_contains(item, actual[key], f"{path}/{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected a list, got {actual!r}"
        assert len(expected) == len(actual), f"{path}: {len(expected)} items expected, got {len(actual)}"
        for index, (item, other) in enumerate(zip(expected, actual, strict=True)):
            assert_contains(item, other, f"{path}[{index}]")
        return
    assert expected == actual, f"{path}: expected {expected!r}, got {actual!r}"


def load(name: str) -> Any:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n"


# ── 模型队列（S5、C28、D2、D3）─────────────────────────────────────────────


class RecordingAnswering(Answering):
    """照着提示词答的替身（先验都答 unsure；候选条件提名单里第一个类），另记下每一次调用的输出名、各条消息与输出形状。"""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict[str, Any]] = []

    async def complete_async(self, request: Any) -> Any:
        chat = request.request
        response_format = chat.response_format
        self.requests.append(
            {
                "name": None if response_format is None else response_format.name,
                "messages": [[str(message.role), message.content or ""] for message in chat.messages],
                "schema": None if response_format is None else plain(dict(response_format.schema)),
            }
        )
        return await super().complete_async(request)


def answering_world(root: Path) -> tuple[World, RecordingAnswering]:
    provider = RecordingAnswering()
    client = StructuredChatClient(ChatClient(model_config(), provider), validation_retries=1)
    return World(root, structured_chat=client), provider


def drain(world: World, night: date) -> Any:
    """``night`` 这一代完成之后模型队列做到没有欠账（队列由 ``tests.unit.runtime.test_model_queue_acceptance.queue_on`` 装）。"""

    moment = at(night, 2)
    queue = queue_on(world.root, world.scene, world.site, clock=lambda: moment, sealed=lambda: night)
    return asyncio.run(queue.drain())
