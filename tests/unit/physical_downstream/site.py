"""第二步丙二（物理 lane 接下游：物理一支有自己的换代、语义树、预测层；物理选词也能写新节点）测试共用的现场。

一份组合根 ``build_runtime`` 装出来的 Runtime（行为侧两支 + 语义树 + 预测层），在它上面：

- 时钟：Runtime 不收时钟，各处取 ``datetime.now``。现场把 ``habitus`` 下各模块里的 ``datetime`` 换成一个 ``now`` 可拨的子类
  （``isinstance`` 照认真的 ``datetime``），收尾时换回。只为能把十几天的日子按槽走一遍；仓库里没有 time-machine / freezegun。
- 物理交付走 ``runtime.deliver_physical``；会话的轮交给组合根装好的会话 lane（``branches.session.fusion.consume``，同第二步甲）；
  归约走各支的 ``reduction_runner.run_once``；换代与预测拍走 ``components.slot_worker.run_once``（每槽一次：先看一眼换代、再做
  预测拍）；模型队列走 ``components.model_queue.drain``；提醒出口走 ``runtime.claim_reminders``。
- 模型一律是确定性替身 ``PhysRouter``：会话那一面同第二步乙的 ``GrowRouter``；物理选词默认"按名字对词表"——【这一段】的名字
  在这一次的词表里有同名的词就答它，没有答 ``missing``；也可以排好答复（``Grow`` 写新节点、``Pick`` 原样答、异常）。

断言只落在盘上与公开入口的返回上。两支的盘上布局按设计（共享说明 3.1）：语义树 ``<scene_root>/<lane>/``，预测层
``<foresight_root>/<lane>/``；预测层的账本、单元历史直接用它们的存储类按目录读写（``ForesightLedgerStore``、``HistoryStore``），
不经组合根的部件（实现会把预测层改成两支各一套，那时只有目录是定的）。

"应当不变"那几条（S2、S4、S5）要在改动前后都过，会话一支的语义树目录用 ``session_scene``：``<scene_root>/session/`` 有换代目录
就用它，否则是 ``<scene_root>/``（改动前的布局）。
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from habitus.behavior.kinds.ids import ClassId
from habitus.behavior.model import BehaviorKind
from habitus.foresight.edges import PastMoment
from habitus.foresight.history import Moment, UnitHistory
from habitus.foresight.ledger import AccountFact, Delivery, Expectation, account_key
from habitus.pre.conversation.adapters import ConversationAdapterRegistry
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from tests.unit.word_grow.site import Grow, GrowRouter, Literal, Named
from tests.unit.word_pick.site import CST, RECORDS_HEADER, Pick, Raw, config, runtime_of, vocabulary_block, word_number

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORDS = REPOSITORY_ROOT / "habitus" / "behavior" / "kinds" / "words"


def _ids(lane: str) -> dict[str, str]:
    body = json.loads((WORDS / f"{lane}.json").read_text(encoding="utf-8"))
    return {item["name"]: item["id"] for item in body["words"]}


PHYSICAL = _ids("physical")
SESSION = _ids("session")
MEDICINE = PHYSICAL["吃药"]  # p-k0032，第 1 层
RUN = PHYSICAL["跑步"]  # p-k0136
DOG = PHYSICAL["遛狗"]  # p-k0085
WEEKLY = SESSION["写周报"]  # s-k0002
HOST = ConversationAdapterRegistry.with_builtins().question_tools()[0].host

# ── 时钟 ──────────────────────────────────────────────────────────────────────

_REAL = datetime


class _Meta(type):
    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, _REAL)

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, _REAL)


class FrozenDatetime(_REAL, metaclass=_Meta):
    """``now`` 读现场的钟；其余照 ``datetime``。"""

    moment: ClassVar[datetime] = _REAL(2026, 4, 1, tzinfo=CST)

    @classmethod
    def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
        if tz is None:
            return cls.moment.astimezone().replace(tzinfo=None)
        return cls.moment.astimezone(tz)


class Clock:
    """把 ``habitus`` 各模块的 ``datetime`` 换成 ``FrozenDatetime``；``close`` 换回。"""

    def __init__(self, moment: datetime) -> None:
        self.patched: dict[str, Any] = {}
        self.set(moment)

    def set(self, moment: datetime) -> None:
        FrozenDatetime.moment = moment
        for name, module in list(sys.modules.items()):
            if not name.startswith("habitus") or module is None or name in self.patched:
                continue
            if getattr(module, "datetime", None) is _REAL:
                module.datetime = FrozenDatetime  # type: ignore[attr-defined]
                self.patched[name] = module

    def close(self) -> None:
        for module in self.patched.values():
            module.datetime = _REAL
        self.patched.clear()


# ── 模型替身 ──────────────────────────────────────────────────────────────────

PHYSICAL_HEADER = "【这一段】"


def segment_name(text: str) -> str:
    """物理选词提示词里【这一段】的名字（材料的第一段）。"""

    tail = text.rsplit(PHYSICAL_HEADER, 1)[1].lstrip("\n")
    return tail.splitlines()[0].split("　｜")[0].strip()


class PhysRouter(GrowRouter):
    """会话那一面同 ``GrowRouter``；物理选词另排（``physical``），排空了"按名字对词表"。"""

    def __init__(self) -> None:
        super().__init__()
        self.physical: list[object] = []

    def physical_calls(self) -> list[Any]:
        return [call for call in self.calls if call.kind == "pick" and RECORDS_HEADER not in call.text]

    def _describe(self) -> str | BaseException:
        """「描述这一轮」：排好的答复里带 ``responses``、而这一次的输出形状里没有它（这一轮没带开着的提醒）时去掉它。"""

        if self.describes and isinstance(self.describes[0], dict) and "responses" in self.describes[0]:
            properties = (self.calls[-1].schema or {}).get("properties", {})
            if "responses" not in properties:
                self.describes[0] = {k: v for k, v in self.describes[0].items() if k != "responses"}
        return super()._describe()

    def _pick(self, text: str) -> str | BaseException:
        if RECORDS_HEADER in text:
            return super()._pick(text)
        scripted = self.physical.pop(0) if self.physical else None
        if isinstance(scripted, BaseException):
            return scripted
        if isinstance(scripted, Raw):
            return scripted.text
        new: object = None
        explicit = False
        if scripted is None:
            name = segment_name(text)
            known = {word.name for word in vocabulary_block(text).words}
            body: dict[str, object] = {"pick": word_number(text, name) if name in known else "missing"}
        elif isinstance(scripted, Named):
            body = {"pick": word_number(text, scripted.name)}
        elif isinstance(scripted, Grow):
            body = {"pick": word_number(text, scripted.name)}
            new, explicit = _new(scripted.first, scripted.second), True
        elif isinstance(scripted, Literal):
            pick = word_number(text, scripted.pick.name) if isinstance(scripted.pick, Named) else scripted.pick
            body = {"pick": pick}
            new, explicit = scripted.new, True
        elif isinstance(scripted, Pick):
            body = {"pick": scripted.value}
        else:
            raise AssertionError(f"unknown scripted physical pick {scripted!r}")
        if explicit or self._asks_new():
            body["new"] = new
        return json.dumps(body, ensure_ascii=False)


def _new(first: tuple[str, str] | None, second: tuple[str, str] | None) -> object:
    if first is None and second is None:
        return None
    return {
        "first": None if first is None else {"name": first[0], "definition": first[1]},
        "second": None if second is None else {"name": second[0], "definition": second[1]},
    }


# ── 物理交付 ──────────────────────────────────────────────────────────────────


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CST)


def delivery(
    delivery_id: str,
    segments: Sequence[tuple[str, datetime, datetime]],
    coverage: Sequence[tuple[datetime, datetime, str]],
    *,
    sent_at: datetime,
    device: str = "watch-01",
) -> dict[str, object]:
    """一份交付：``segments`` 是（行为, 开始, 结束），``coverage`` 是（开始, 结束, 状态），首尾相接。"""

    return {
        "source": "physical/s12",
        "device": device,
        "delivery_id": delivery_id,
        "sent_at": sent_at.isoformat(),
        "segments": [
            {"id": f"{delivery_id}-{index}", "behavior": name, "started_at": a.isoformat(), "ended_at": b.isoformat()}
            for index, (name, a, b) in enumerate(segments)
        ],
        "coverage": [{"started_at": a.isoformat(), "ended_at": b.isoformat(), "state": s} for a, b, s in coverage],
    }


def day_delivery(
    day: date, behaviors: Sequence[tuple[str, int, int, int]], *, state: str = "observing"
) -> dict[str, object]:
    """一整天一份交付：``behaviors`` 是（行为, 时, 分, 持续分钟）；覆盖当天 00:00–次日 00:00 一段 ``state``。"""

    start = at(day, 0)
    end = start + timedelta(days=1)
    segments = [(name, at(day, h, m), at(day, h, m) + timedelta(minutes=lasts)) for name, h, m, lasts in behaviors]
    return delivery(f"d-{day.isoformat()}", segments, [(start, end, state)], sent_at=end)


# ── 现场 ──────────────────────────────────────────────────────────────────────


class World:
    """装好的 Runtime（行为侧两支 + 语义树 + 预测层）、可拨的钟、模型替身。"""

    def __init__(self, tmp_path: Path, *, start: datetime, edit: Any = None) -> None:
        self.clock = Clock(start)

        def options(raw: dict[str, Any]) -> None:
            raw["foresight"] = {"enabled": True}
            if edit is not None:
                edit(raw)

        self.config = config(tmp_path, edit=options)
        self.runtime, router = runtime_of(self.config, PhysRouter())
        assert isinstance(router, PhysRouter)
        self.router = router
        self.clock.set(start)  # 组合根装完以后再换一遍（装的时候才导入的模块）
        branches = self.runtime.components.behavior
        assert branches is not None
        self.session, self.physical = branches.session, branches.physical
        self._sequences: dict[str, int] = {}
        #: 每一槽 ``run_once`` 的结果或它抛出的异常（一支失败时整槽抛不抛，是实现的事；测试只看盘）。
        self.slots: list[Any] = []

    # ── 盘 ──
    @property
    def scene_root(self) -> Path:
        return self.config.scene_root

    @property
    def foresight_root(self) -> Path:
        return self.config.foresight_root

    def scene(self, lane: str) -> Path:
        return self.scene_root / lane

    def foresight(self, lane: str) -> Path:
        return self.foresight_root / lane

    def session_scene(self) -> Path:
        """会话一支的语义树目录：改动后在 ``<scene_root>/session/``，改动前就是 ``<scene_root>/``。"""

        moved = self.scene("session")
        return moved if (moved / "generation").is_dir() else self.scene_root

    def marks(self, lane: str) -> list[str]:
        folder = (self.session_scene() if lane == "session" else self.scene(lane)) / "generation"
        return sorted(item.stem for item in folder.glob("*.json")) if folder.is_dir() else []

    def ledger(self, lane: str) -> ForesightLedgerStore:
        return ForesightLedgerStore(self.foresight(lane))

    def branch(self, lane: str) -> Any:
        return self.session if lane == "session" else self.physical

    # ── 钟 ──
    def set_clock(self, moment: datetime) -> None:
        self.clock.set(moment)

    # ── 送 ──
    def deliver(self, payload: dict[str, object], *, now: datetime | None = None) -> Any:
        if now is not None:
            self.set_clock(now)
        return asyncio.run(self.runtime.deliver_physical(payload))

    def turn(self, conversation: str, moment: datetime, said: str) -> Any:
        from habitus.behavior.fusion.lanes.session import SessionMessage, SessionMessageRole, split_turns

        start = self._sequences.get(conversation, 0)
        messages = [
            SessionMessage(start, moment, SessionMessageRole.PROMPT, said),
            SessionMessage(start + 1, moment + timedelta(minutes=1), SessionMessageRole.COMPLETION, "好的，办好了。"),
        ]
        self._sequences[conversation] = start + 2
        (turn,) = split_turns(conversation, messages)
        return asyncio.run(self.session.fusion.consume((turn,)))

    # ── 跑 ──
    def reduce(self) -> None:
        for branch in (self.session, self.physical):
            asyncio.run(branch.reduction_runner.run_once())

    def slot(self, moment: datetime, *, reduce: bool = True) -> Any:
        """``moment`` 这一槽：两支归约一次，再一槽（先看一眼换代、再做预测拍）。"""

        self.set_clock(moment)
        if reduce:
            self.reduce()
        worker = self.runtime.components.slot_worker
        assert worker is not None
        try:
            report: Any = asyncio.run(worker.run_once())
        except Exception as exc:  # noqa: BLE001 - 记下，测试看盘
            report = exc
        self.slots.append(report)
        return report

    def slots_between(self, start: datetime, end: datetime, *, reduce: bool = True) -> list[Any]:
        found = []
        moment = start
        while moment <= end:
            found.append(self.slot(moment, reduce=reduce))
            moment += timedelta(minutes=15)
        return found

    def night(self, day: date, *, until: tuple[int, int] = (1, 30)) -> list[Any]:
        """``day`` 当天零点起到 ``until`` 每一槽（这一段里够条件就换这一天的代）。"""

        return self.slots_between(at(day, 0), at(day, *until))

    def physical_days(
        self, start: date, plan: Sequence[Sequence[tuple[str, int, int, int]]], *, session: str | None = None
    ) -> None:
        """从 ``start`` 起逐天：那天的物理交付（次日 00:05 送到），``session`` 给了就在 09:00 说一轮；然后过次日零点那几槽。"""

        for offset, behaviors in enumerate(plan):
            day = start + timedelta(days=offset)
            if session is not None:
                self.set_clock(at(day, 9, 2))
                self.turn("conv", at(day, 9), session)
            self.deliver(day_delivery(day, behaviors), now=at(day + timedelta(days=1), 0, 5))
            self.night(day + timedelta(days=1))

    def drain(self, moment: datetime | None = None) -> Any:
        if moment is not None:
            self.set_clock(moment)
        queue = self.runtime.components.model_queue
        assert queue is not None
        try:
            return asyncio.run(queue.drain())
        except Exception as exc:  # noqa: BLE001 - 记下，测试看盘
            return exc

    def claim(self, moment: datetime, *, session: str = "s1", limit: int = 3) -> tuple[Any, ...]:
        self.set_clock(moment)
        return asyncio.run(self.runtime.claim_reminders(host=HOST, session=session, at=moment, limit=limit))

    # ── 词表 ──
    def kinds_state(self, lane: str) -> dict[str, Any]:
        path = Path(self.branch(lane).kind_store.root) / "kinds.state.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def in_use(self, lane: str) -> set[str]:
        return {item["id"] for item in self.kinds_state(lane).get("active", [])}

    def grown(self, lane: str) -> dict[str, dict[str, Any]]:
        return {item["id"]: item for item in self.kinds_state(lane).get("grown", [])}

    def deleted(self, lane: str) -> set[str]:
        return {item["id"] for item in self.kinds_state(lane).get("deleted", [])}

    def use(self, lane: str, *tokens: str, since: date = date(2026, 3, 1)) -> None:
        """这几个编号在这一支在用（直接写词表状态）。"""

        self.branch(lane).kind_store.update(
            lambda current: current.with_active({**current.active, **{ClassId.parse(t): since for t in tokens}})
        )

    # ── 行为树 ──
    def tokens(self, lane: str) -> dict[str, str]:
        """这一支行为树上的记录：``开始时刻 名字`` → ``kind_token``。"""

        tree = self.branch(lane).tree
        found = {}
        for address in tree.list_addresses(BehaviorKind.OCCURRENCE):
            fields = tree.read(address).fields
            found[f"{fields['started_at']} {fields['name']}"] = str(fields["kind_token"])
        return found

    def gaps(self, lane: str) -> list[dict[str, Any]]:
        tree = self.branch(lane).tree
        return [dict(tree.read(address).fields) for address in tree.list_addresses(BehaviorKind.GAP)]

    # ── 预测层（按目录读写这一支的账本与单元历史）──
    def earn(self, lane: str, kind: str, before: datetime, *, hits: int = 8, trigger: str = "interval") -> str:
        """这一支这本账前几天各一刻都来了（账上的流水 + 单元历史的快照），到提醒档。"""

        ledger, store = self.ledger(lane), HistoryStore(self.foresight(lane))
        account = account_key(kind, "日子", trigger)
        moments = []
        for offset in range(hits, 0, -1):
            moment = before - timedelta(days=offset)
            ledger.append_fact(AccountFact(account, moment, True, 1.0, "准时", f"old-{kind}-{offset}"))
            moments.append(PastMoment(moment, frozenset({account}), True))
        known = store.latest(lane)
        units = dict(known.moments) if known is not None else {}
        units[kind] = tuple(Moment(item, (f"{kind}-{index}",)) for index, item in enumerate(moments))
        night = before.date() - timedelta(days=1)
        if known is not None and known.night >= night:
            night = known.night + timedelta(days=1)
        store.write(UnitHistory(lane, night, units))
        return account

    def expect(
        self,
        lane: str,
        identity: str,
        kind: str,
        *,
        window: tuple[datetime, datetime],
        deadline: datetime,
        observed_from: datetime | None = None,
        source: str = "日子",
    ) -> Expectation:
        formed = observed_from or at(window[0].date(), 0)
        item = Expectation(
            expectation_id=identity,
            kind_token=kind,
            day=window[0].date(),
            formed_at=formed,
            source=source,
            trigger="interval",
            trigger_instance=f"occ:{identity}.md",
            window=window,
            deadline=deadline,
            basis=(),
            opportunities=8,
            hits=8,
            observed_from=formed,
        )
        self.ledger(lane).record_expectation(item)
        return item

    def spoken(self, lane: str, item: Expectation, moment: datetime) -> None:
        """这条预期 ``moment`` 说出口了（账上一条「说了」）。"""

        self.ledger(lane).record_delivery(
            item.day, Delivery(item.expectation_id, moment, 1, True, "说了", 1.0, "该做了", via="codex", turn="t1")
        )

    def reasons(self, lane: str, day: date) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for item in sorted(self.ledger(lane).deliveries_on(day), key=lambda made: made.decided_at):
            found.setdefault(item.expectation_id, []).append(item.reason)
        return found

    def close(self) -> None:
        try:
            asyncio.run(self.runtime.close())
        finally:
            self.clock.close()


def texts_under(root: Path) -> str:
    """``root`` 下全部文本文件拼起来（读不了的跳过）。"""

    parts = []
    for item in sorted(root.rglob("*")) if root.is_dir() else ():
        if item.is_file():
            try:
                parts.append(item.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, OSError):
                continue
    return "\n".join(parts)


def files_under(root: Path) -> list[str]:
    return sorted(str(item.relative_to(root)) for item in root.rglob("*") if item.is_file()) if root.is_dir() else []


__all__ = [
    "CST",
    "DOG",
    "HOST",
    "MEDICINE",
    "PHYSICAL",
    "RUN",
    "SESSION",
    "WEEKLY",
    "Clock",
    "Grow",
    "Literal",
    "Named",
    "PhysRouter",
    "Pick",
    "World",
    "at",
    "day_delivery",
    "delivery",
    "files_under",
    "texts_under",
]
