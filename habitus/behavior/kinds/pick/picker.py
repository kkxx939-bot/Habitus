"""选词：一条记录一次模型调用，答复落成 ``kind_token``（词表方案 2026-11-14 第 3 节）。

| 模型答                  | ``kind_token``                         |
|-------------------------|----------------------------------------|
| 词的编号（整数）         | 这个词                                  |
| ``none``                | 「非事件」（只有会话 lane）             |
| ``continue`` + 记录编号  | 「续话」（只有会话 lane；原来那条不改） |
| ``missing``             | 「表外」                                |
| 重问之后仍不合格         | 「待定」                                |

答复还可以带一个新节点（``new``，词表方案第 4 节；物理 lane 自丙二 2026-11-16 起也写）：代码按 ``grow.settle`` 收它，节点在
这一刻写进这条 lane 的词表状态，记录挂在最终那个节点上（``kind_token`` 仍是一个编号）。每一条按此刻的词表状态拼提示词，后面的
记录马上能选到新节点。两条 lane 收节点、生命周期是同一套代码、同一组配置；物理 lane 没有 ``none``、``continue`` 与上文。

答复结构不合格（不是 JSON、字段不对、编号不在表里、标题当词、续的记录不在列表里）由结构化客户端带着错误重问，
次数是 ``pick_retries``；用尽记「待定」，不硬归。这一份内容被拦、输入超长抛 ``KindModelOutputError``（调用方按错误处理），
模型整个用不了的原样抛出。

会话 lane 的上文（最近的记录、这场对话里之前的几轮）只在进程里记（裁定 33：材料用完不存），重启之后从空开始，
拿不到就少给。
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, TypeVar

from habitus.behavior.kinds.calls import CallFailure, KindModelCaller, KindModelOutputError, call_failure
from habitus.behavior.kinds.grow import Proposal, settle
from habitus.behavior.kinds.ids import (
    ClassId,
    Lane,
    continued_token,
    is_word,
    missing_token,
    not_event_token,
    pending_token,
)
from habitus.behavior.kinds.model import Vocabulary
from habitus.behavior.kinds.pick.prompt import (
    CONTINUE,
    NONE,
    PHYSICAL_FIXED,
    SESSION_FIXED,
    physical_system,
    physical_user,
    session_system,
    session_user,
)

T = TypeVar("T")


class VocabularyState(Protocol):
    """选词器用到的词表状态（生产里是这条 lane 的 ``BehaviorKindStore``）：此刻的词表、读改写回。"""

    @property
    def lane(self) -> Lane: ...

    def read(self) -> Vocabulary: ...

    def update(self, change: Callable[[Vocabulary], Vocabulary]) -> Vocabulary: ...


@dataclass(frozen=True)
class _Seen:
    """进程里记着的一轮：哪个会话、什么时候、给模型看的材料、选成了什么（没成记录的是 None）。"""

    conversation: str
    at: datetime
    material: str
    token: str | None


_SLOT: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "definition": {"type": "string"}},
    "required": ["definition", "name"],
    "additionalProperties": False,
}


def _schema(fixed: Sequence[str], *, continues: bool) -> dict[str, Any]:
    """``continues`` 只有会话 lane 有；``new`` 两条 lane 都有。"""

    properties: dict[str, Any] = {"pick": {"anyOf": [{"type": "integer"}, {"type": "string", "enum": list(fixed)}]}}
    if continues:
        properties["continues"] = {"type": ["integer", "null"]}
    properties["new"] = {
        "anyOf": [
            {"type": "null"},
            {
                "type": "object",
                "properties": {"first": _SLOT, "second": {"anyOf": [{"type": "null"}, _SLOT]}},
                "required": ["first", "second"],
                "additionalProperties": False,
            },
        ]
    }
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
        "additionalProperties": False,
    }


def _proposals(new: object) -> tuple[Proposal, ...]:
    """答复里的 ``new``：null，或按顺序的 first、second（只填 second 不合格）。名字、定义要是一行非空文字；长度、层数、
    重名这些数得出来的由 ``grow.settle`` 查（不过就不建，不重问）。"""

    if new is None:
        return ()
    if not isinstance(new, Mapping) or set(new) != {"first", "second"}:
        raise ValueError("new 必须是 null 或正好带 first、second 的对象")
    if new["first"] is None:
        raise ValueError("要写新节点就必须填 first；不许只填 second")
    slots = [new["first"]] if new["second"] is None else [new["first"], new["second"]]
    found = []
    for slot in slots:
        if not isinstance(slot, Mapping) or set(slot) != {"name", "definition"}:
            raise ValueError("new 的每一格必须正好是 name、definition")
        for label in ("name", "definition"):
            value = slot[label]
            if not isinstance(value, str) or not value.strip() or any(not ch.isprintable() for ch in value):
                raise ValueError(f"new 的 {label} 必须是一行非空文字")
        found.append(Proposal(slot["name"], slot["definition"]))
    return tuple(found)


def _word_token(vocabulary: Vocabulary, pick: object) -> str:
    """答复里的词编号（整数）换成这条 lane 的编号；不在表里、不是整数都不合格。"""

    if isinstance(pick, bool) or not isinstance(pick, int) or pick <= 0:
        raise ValueError(f"pick 必须是词编号整数或固定选项之一，实际 {pick!r}")
    word_id = ClassId(vocabulary.lane, pick)
    if word_id not in vocabulary:
        raise ValueError(f"pick 编号 {pick} 不在词表里")
    return str(word_id)


class WordPicker:
    """一条 lane 的选词器。``vocabulary`` 是这条 lane 的初始词表（定 lane 用）；``state`` 是词表状态的存储（每一条从它读此刻的
    词表、把新节点写进去；两条 lane 都给）。"""

    def __init__(
        self, caller: KindModelCaller, vocabulary: Vocabulary, *, state: VocabularyState | None = None
    ) -> None:
        if state is not None and state.lane is not vocabulary.lane:
            raise ValueError("the vocabulary state belongs to another lane")
        self.caller = caller
        self.vocabulary = vocabulary
        self.state = state
        self.config = caller.config
        lane = vocabulary.lane
        self._not_event, self._continued = not_event_token(lane), continued_token(lane)
        self._missing, self._pending = missing_token(lane), pending_token(lane)
        # 最近的若干轮（会话 lane 的上文）：只在进程里
        self._seen: deque[_Seen] = deque(maxlen=max(64, 4 * self.config.recent_records))
        self._by_conversation: dict[str, deque[_Seen]] = {}

    async def pick_turn(self, *, conversation: str, at: datetime, material: str) -> str:
        """会话 lane 的一轮：``material`` 是给"描述这一轮"的同一份材料，``at`` 是他开口的时刻（本地时刻；新节点"哪一刻写的"、
        "最后被选中"记它）。"""

        if self.state is None:
            raise ValueError("the session picker needs the vocabulary state")
        vocabulary = self.state.read()
        records = self._records(conversation, at)
        earlier = self._earlier(conversation, at)
        lines = [
            f"{number}. {seen.at:%m-%d %H:%M} 他说「{_said(seen.material)}」→ {_display(vocabulary, seen.token)}"
            for number, seen in enumerate(records, start=1)
        ]

        def validate(parsed: object) -> str | tuple[ClassId, tuple[Proposal, ...]]:
            if not isinstance(parsed, Mapping) or set(parsed) != {"pick", "continues", "new"}:
                raise ValueError("字段必须正好是 pick、continues、new")
            pick, continues = parsed["pick"], parsed["continues"]
            proposals = _proposals(parsed["new"])
            if isinstance(pick, str) and proposals:
                raise ValueError("pick 不是词编号时 new 必须是 null")
            if isinstance(pick, str):
                if pick not in SESSION_FIXED:
                    raise ValueError(f"pick 是字符串时只能是 {list(SESSION_FIXED)}，实际 {pick!r}")
                if pick == CONTINUE:
                    if isinstance(continues, bool) or not isinstance(continues, int):
                        raise ValueError("pick 为 continue 时 continues 必须是整数记录编号")
                    if not 1 <= continues <= len(records):
                        raise ValueError(f"continues 编号 {continues} 不在【最近的记录】里")
                    return self._continued
                if continues is not None:
                    raise ValueError("pick 不是 continue 时 continues 必须是 null")
                return self._not_event if pick == NONE else self._missing
            if continues is not None:
                raise ValueError("pick 是词编号时 continues 必须是 null")
            return ClassId.parse(_word_token(vocabulary, pick)), proposals

        answer = await self._ask(
            system=session_system(vocabulary),
            user=session_user(material, earlier, lines),
            schema=_schema(SESSION_FIXED, continues=True),
            validator=validate,
        )
        token = answer if isinstance(answer, str) else str(await self._settle(*answer, at=at))
        self._remember(_Seen(conversation, at, material, token))
        return token

    async def _settle(self, picked: ClassId, proposals: tuple[Proposal, ...], *, at: datetime) -> ClassId:
        """把选中的词与新节点收进此刻的词表状态（读、改、写回）；返回记录最终挂的节点。选中的是初始词、又没有新节点时不碰盘。
        写入要等词表状态的跨进程租约（最多一个租约时长），放到线程里做，不占着事件循环。"""

        assert self.state is not None
        final: list[ClassId] = []

        def change(vocabulary: Vocabulary) -> Vocabulary:
            settled, word_id = settle(vocabulary, picked, proposals, at=at, config=self.config)
            final.append(word_id)
            return settled

        await asyncio.to_thread(self.state.update, change)
        return final[0]

    async def pick_record(self, material: str, *, at: datetime) -> str:
        """物理 lane 的一条记录（上游送来的一段写成的判断：名字、概要、目标、步骤）。``at`` 是这一段开始的时刻（新节点"哪一刻写的"、
        "最后被选中"记它；在用计数按记录的日子算，同一个口）。按此刻的词表状态拼提示词，答复带新节点时同会话 lane 一样收进词表状态。"""

        if self.state is None:
            raise ValueError("the physical picker needs the vocabulary state")
        vocabulary = self.state.read()

        def validate(parsed: object) -> str | tuple[ClassId, tuple[Proposal, ...]]:
            if not isinstance(parsed, Mapping) or set(parsed) != {"pick", "new"}:
                raise ValueError("字段必须正好是 pick、new")
            pick = parsed["pick"]
            proposals = _proposals(parsed["new"])
            if isinstance(pick, str):
                if pick not in PHYSICAL_FIXED:
                    raise ValueError(f"pick 是字符串时只能是 {list(PHYSICAL_FIXED)}，实际 {pick!r}")
                if proposals:
                    raise ValueError("pick 不是词编号时 new 必须是 null")
                return self._missing
            return ClassId.parse(_word_token(vocabulary, pick)), proposals

        answer = await self._ask(
            system=physical_system(vocabulary),
            user=physical_user(material),
            schema=_schema(PHYSICAL_FIXED, continues=False),
            validator=validate,
        )
        return answer if isinstance(answer, str) else str(await self._settle(*answer, at=at))

    async def _ask(
        self, *, system: str, user: str, schema: Mapping[str, Any], validator: Callable[[object], T]
    ) -> T | str:
        """校验器的结果；重问用尽仍不合格是「待定」。"""

        try:
            return await self.caller.call(
                system=system, user=user, schema=schema, name="behavior_word_pick", validator=validator
            )
        except KindModelOutputError as exc:
            if call_failure(exc) is CallFailure.INVALID:
                return self._pending  # 重问用尽仍不合格：记「待定」，不硬归
            raise

    def _records(self, conversation: str, at: datetime) -> list[_Seen]:
        """这条 lane 上这一轮之前最近 ``recent_records`` 条成了记录的轮，加同一会话最近 ``recent_same_conversation`` 条，
        去重、按时间排。成了记录 = 选了词或「表外」。"""

        def became_record(seen: _Seen) -> bool:
            return seen.at < at and seen.token is not None and (is_word(seen.token) or seen.token == self._missing)

        # 按时刻取最近的（重试堆补回的一轮处理得晚、时刻却早，不能挤掉时刻更新的记录）
        recent = sorted((seen for seen in self._seen if became_record(seen)), key=lambda seen: seen.at)
        mine = sorted(
            (seen for seen in self._by_conversation.get(conversation, ()) if became_record(seen)),
            key=lambda seen: seen.at,
        )
        chosen = {id(seen): seen for seen in [*_tail(recent, self.config.recent_records)]}
        chosen.update({id(seen): seen for seen in _tail(mine, self.config.recent_same_conversation)})
        return sorted(chosen.values(), key=lambda seen: seen.at)

    def _earlier(self, conversation: str, at: datetime) -> list[str]:
        """这场对话里这一轮之前的几轮（最多 ``earlier_turns`` 轮，每轮材料截 ``earlier_chars`` 字）。"""

        before = sorted(
            (seen for seen in self._by_conversation.get(conversation, ()) if seen.at < at), key=lambda seen: seen.at
        )
        return [
            f"（{seen.at:%H:%M}）\n{seen.material[: self.config.earlier_chars]}"
            for seen in _tail(before, self.config.earlier_turns)
        ]

    def _remember(self, seen: _Seen) -> None:
        self._seen.append(seen)
        queue = self._by_conversation.pop(seen.conversation, None)
        if queue is None:
            queue = deque(maxlen=max(self.config.earlier_turns, self.config.recent_same_conversation, 1))
        queue.append(seen)
        self._by_conversation[seen.conversation] = queue  # 最近用到的会话排到最后
        while len(self._by_conversation) > _MAX_CONVERSATIONS:
            del self._by_conversation[next(iter(self._by_conversation))]


#: 进程里最多记几个会话的上文（保护闸；超出时丢最久没用到的会话）。
_MAX_CONVERSATIONS = 256


def _display(vocabulary: Vocabulary, token: str | None) -> str:
    """最近的记录归到的词：删掉的节点显示它换算后的上级。"""

    if token is not None and is_word(token):
        word_id = ClassId.parse(vocabulary.resolve_token(token))
        if word_id in vocabulary:
            return vocabulary.get(word_id).name
    return "（表里没有）"


def _tail(items: list[_Seen], count: int) -> list[_Seen]:
    return items[-count:] if count > 0 else []


def _said(material: str) -> str:
    """他当时说的第一句的前 60 字（材料第一行去掉"他说："）。"""

    return material.splitlines()[0].removeprefix("他说：")[:60] if material else ""


__all__ = ["WordPicker"]
