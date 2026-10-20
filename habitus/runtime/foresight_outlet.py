"""提醒出口（方案 32；裁定 64）：预测层决定说之后，话怎么到人、"说了"怎么回到账本。

预测层照旧只产出"递送请求 + 档 + 话"（递送那一拍记「待取」）。出口是它外面薄薄一层宿主适配：

- **取**（``claim``）：宿主在 UserPromptSubmit 时来取此刻可取的请求，带租约（租约内别的会话取不到；同一会话重取拿回自己手里的）；
  来取这件事本身就是活性（``alive``）——最近一段有宿主来取过，递送那一拍才问模型写话、记「待取」，否则记「不可达」、不花模型钱。
- **认**（``observe``）：转写照常由 Stop hook 送回服务；同一份会话先过出口、再进会话 lane。出口在取走记录对应的会话里找 agent 用
  宿主提问工具问人的那次调用：题目含类名、或与那句话相似、或选项就是我们给的三个 → 配上 → 账本追加「说了」（``decided_at`` =
  提问那一刻、``text`` = 题目原文、``via`` = 宿主、``turn`` = call_id）。他点的固定选项直接写回话（现在就做 → 正、这次不用 → 负、
  知道了 → 中），不经模型；自己填的话不写，交给会话 lane 的模型照旧判（裁定 64 第 7、8 条）。取走了、这一轮过了却没问 → 放回、
  ``attempts += 1``；到上限不再给。
- **保证**：一条预期最多一条「说了」（写前查账本已有的 spoken，重放、换会话再问都挡住）；提问出现在最晚时限之后的不算；
  租约过期只让别的会话能取（不算放回：问可能早就问了、转写只是还没回来）；历任持有者都认得出自己那次提问。

传输状态（可取清单、取走记录、问过的提问、宿主活性）在 ``runtime/foresight_outlet_store.py``，不进账本。
宿主 → 提问工具 / 题目字段 / 作答怎么回来 是这里的一张表（``QUESTION_TOOLS``，数据不是分支）。

TODO(裁定 64 第 8 条)：自由作答现在走会话 lane 的模型判正 / 中 / 负（``runtime/foresight_responses.py`` 那座桥）；推式宿主（第六节
本地通知）与别的 harness 的回复不是提问工具的作答时也要靠那条路——这里只写固定选项，模型路保留不删。
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Protocol

from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger import SPOKEN, Delivery, Response
from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_outlet_store import OutletStore

#: 三个固定选项（产品定措辞，所有宿主一样；第三个不叫"稍后再说"——不再问第二次，裁定 64 第 8 条）与它们映成的回话方向。
OPTIONS = ("现在就做", "这次不用", "知道了")
POLARITY_OF_OPTION = {"现在就做": "正", "这次不用": "负", "知道了": "中"}
#: Codex Desktop 的作答：一条用户消息，JSON 里带 questionItemId（含 call_id）、question、answer。
REPLY_TAG = "<send_user_message_question_reply>"
#: 取走时刻与转写里人那句话的时刻可能差几秒（hook 在提交时跑，转写记的是宿主自己的时间）：窗口往前留一点。
_CLAIM_TOLERANCE = timedelta(seconds=120)
_QUOTED = re.compile(r'=\s*"([^"]*)"')
_QUOTED_PAIR = re.compile(r'"([^"]+)"\s*=\s*"([^"]*)"')


@dataclass(frozen=True)
class QuestionTool:
    """一个宿主的提问工具：工具名、题目在参数 ``questions[]`` 里叫什么、作答是不是晚些才以用户消息回来。"""

    host: str
    tool: str
    title_key: str
    replies_later: bool = False


QUESTION_TOOLS: tuple[QuestionTool, ...] = (
    QuestionTool("claude-code", "AskUserQuestion", "question"),
    QuestionTool("codex", "request_user_input", "title"),
    QuestionTool("codex", "request_user_input_async", "title", replies_later=True),
)


@dataclass(frozen=True)
class OpenRequest:
    """递送那一拍写进可取清单的一行：预期身份就是请求身份（一条预期最多一条请求）。"""

    request_id: str
    kind_token: str
    text: str
    deadline: datetime
    formed_at: datetime
    level: int
    day: date

    def __post_init__(self) -> None:
        for label in ("request_id", "kind_token", "text"):
            if not isinstance(getattr(self, label), str) or not getattr(self, label):
                raise ForesightError(f"open request {label} must be non-empty text")
        for label in ("deadline", "formed_at"):
            value = getattr(self, label)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ForesightError(f"open request {label} must be a timezone-aware datetime")
        if not isinstance(self.day, date) or isinstance(self.day, datetime):
            raise ForesightError("open request day must be a date")


@dataclass(frozen=True)
class ClaimedReminder:
    """取走的一条（给宿主）：题目原文、三个选项、最晚时限、租约到何时、这次取走的时刻（宿主按它判是不是重取）。"""

    request_id: str
    kind: str
    text: str
    options: tuple[str, ...]
    deadline: datetime
    lease_until: datetime
    claimed_at: datetime


@dataclass(frozen=True)
class HostMessage:
    """出口读转写用的一条消息（会话源的消息由包装层翻成它；出口不认识会话源也不认识会话 lane）。"""

    sequence: int
    occurred_at: datetime
    role: str
    content: str
    tool_name: str | None = None
    tool_call_id: str | None = None
    failed: bool = False

    def __post_init__(self) -> None:
        if self.role not in ("prompt", "completion", "tool_call", "tool_result"):
            raise ForesightError(f"unknown host message role: {self.role!r}")
        if not isinstance(self.occurred_at, datetime) or self.occurred_at.utcoffset() is None:
            raise ForesightError("host message occurred_at must be a timezone-aware datetime")
        if not isinstance(self.content, str):
            raise ForesightError("host message content must be text")


@dataclass(frozen=True)
class _Question:
    """一次提问调用里的一题：哪次调用、第几题、题目、选项。配对与问过的记录都按 (call_id, 题序)，一次调用多题各配各的。"""

    call: HostMessage
    index: int
    title: str
    options: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.call.tool_call_id or ''}#{self.index}"


@dataclass(frozen=True)
class ObserveReport:
    """一份转写过出口的结果：认出几次提问（写了「说了」）、几条固定选项的回话、几条还等作答、放回几条、几次已说过不再写。"""

    spoken: int = 0
    responded: int = 0
    awaiting_reply: int = 0
    released: int = 0
    already_spoken: int = 0
    #: 认出了提问、但他在被问之前已经做了（账上最晚一条是「已发生」）：不算「说了」。
    already_done: int = 0


class OutletPort(Protocol):
    """递送那一拍看到的出口：有没有宿主活着、每拍重写可取清单。"""

    def alive(self, now: datetime) -> bool: ...

    def publish(self, rows: Sequence[OpenRequest], *, now: datetime) -> int: ...


@dataclass
class _Holder:
    host: str
    session: str
    claimed_at: datetime
    lease_until: datetime
    #: 进历任之后为什么不再持有：租约过期（可续租、``claimed_at`` 不变）/ 放回（转写回来了没问）/ 说出口了。
    ended: str | None = None


@dataclass
class _Claim:
    """一条请求的取走记录：现任持有者（租约内）、历任持有者（都认得出自己那次提问）、放回次数。"""

    request: OpenRequest
    kind: str
    holder: _Holder | None = None
    attempts: int = 0
    history: list[_Holder] = field(default_factory=list)
    #: 已经认出提问、写了「说了」：记录留到最晚时限（晚到的转写里别的持有者的提问认得出、但不再写第二条），不再给任何会话。
    spoken: bool = False

    def sessions(self) -> set[str]:
        found = {item.session for item in self.history}
        if self.holder is not None:
            found.add(self.holder.session)
        return found

    def claimed_at_for(self, session: str) -> datetime | None:
        moments = [
            item.claimed_at
            for item in (*self.history, *([self.holder] if self.holder else ()))
            if item.session == session
        ]
        return min(moments) if moments else None


class ReminderOutlet:
    def __init__(
        self,
        store: OutletStore,
        ledger: ForesightLedgerStore,
        *,
        liveness: timedelta,
        lease: timedelta,
        max_attempts: int,
        claim_limit: int,
        similarity: float,
        tools: Sequence[QuestionTool] = QUESTION_TOOLS,
        kind_label: Callable[[str], str | None] | None = None,
        observer: Observer | None = None,
    ) -> None:
        if not isinstance(store, OutletStore):
            raise TypeError("store must be an OutletStore")
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        for label, value in (("liveness", liveness), ("lease", lease)):
            if not isinstance(value, timedelta) or value <= timedelta(0):
                raise ValueError(f"{label} must be a positive timedelta")
        for label, count in (("max_attempts", max_attempts), ("claim_limit", claim_limit)):
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{label} must be a positive integer")
        if isinstance(similarity, bool) or not isinstance(similarity, int | float) or not 0.0 < similarity <= 1.0:
            raise ValueError("similarity must lie in (0, 1]")
        self.store = store
        self.ledger = ledger
        self.liveness = liveness
        self.lease = lease
        self.max_attempts = max_attempts
        self.claim_limit = claim_limit
        self.similarity = float(similarity)
        self.tools = tuple(tools)
        self.kind_label = kind_label
        self.observer: Observer = observer if observer is not None else NullObserver()
        #: 取（HTTP 线程）、认（消费者线程）、每拍重写清单（worker）三方都读-改-写同一批小文件：进程内一把锁串起来
        #: （单进程服务够用；与 ``ForesightLedgerStore._lock`` 同一做法）。
        self._lock = threading.RLock()

    # ── 活性与可取清单 ──────────────────────────────────────────────────────

    def alive(self, now: datetime) -> bool:
        """最近 ``liveness`` 内有宿主来取过（方案 32 第 3.3 条）。"""

        since = now - self.liveness
        return any(seen >= since for seen in self.store.hosts().values())

    def publish(self, rows: Sequence[OpenRequest], *, now: datetime) -> int:
        """递送那一拍每拍重写可取清单（按最晚时限升序），顺手清掉过了时限的取走记录与问过的提问、放掉过期的租约。"""

        with self._lock:
            ordered = sorted(rows, key=lambda row: (row.deadline, row.request_id))
            self.store.write_open([_encode_request(row) for row in ordered])
            self.sweep(now)
            return len(ordered)

    def sweep(self, now: datetime) -> int:
        """租约过期的放给别的会话（不算放回）；过了最晚时限的取走记录与问过的提问删掉（那时账上已记「宿主没取」）。"""

        with self._lock:
            changed = 0
            for request_id, raw in self.store.claims().items():
                claim = _decode_claim(raw)
                if claim is None:
                    self.store.remove_claim(request_id)
                    continue
                if claim.request.deadline <= now:
                    self.store.remove_claim(request_id)
                    changed += 1
                elif claim.holder is not None and claim.holder.lease_until <= now:
                    claim.holder.ended = "expired"
                    claim.history.append(claim.holder)
                    claim.holder = None
                    self.store.write_claim(request_id, _encode_claim(claim))
                    changed += 1
            for key, raw in self.store.asked().items():
                try:
                    deadline = datetime.fromisoformat(str(raw["deadline"]))
                except (KeyError, ValueError):
                    self.store.remove_asked(key)
                    continue
                if deadline <= now:
                    self.store.remove_asked(key)
            return changed

    # ── 取 ──────────────────────────────────────────────────────────────────

    def claim(self, host: str, session: str, at: datetime, limit: int) -> tuple[ClaimedReminder, ...]:
        """宿主来取：去掉过了时限的、别的会话租着的、放回到上限的；按时限由近到远取 ``limit`` 条；同一会话租约内重取拿回自己手里的。"""

        for label, value in (("host", host), ("session", session)):
            if not isinstance(value, str) or not value.strip():
                raise ForesightError(f"{label} must be non-empty text")
        if not isinstance(at, datetime) or at.utcoffset() is None:
            raise ForesightError("at must be a timezone-aware datetime")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.claim_limit:
            raise ForesightError(f"limit must be an integer between 1 and {self.claim_limit}")
        with self._lock:
            self.store.touch_host(host, at)
            self.sweep(at)
            claims = {key: _decode_claim(raw) for key, raw in self.store.claims().items()}
            given: list[ClaimedReminder] = []
            rows = sorted(
                (_decode_request(row) for row in self.store.read_open()), key=lambda r: (r.deadline, r.request_id)
            )
            for request in rows:
                if request.deadline <= at or len(given) >= limit:
                    continue
                claim = claims.get(request.request_id) or _Claim(request, self._label(request.kind_token))
                if claim.spoken:
                    continue
                holder = claim.holder
                if holder is not None and holder.lease_until > at:
                    if holder.session == session:
                        given.append(self._claimed(claim, holder))
                    continue
                mine = next(
                    (item for item in claim.history if item.session == session and item.ended == "expired"), None
                )
                if mine is not None and (holder is None or holder.session == session):
                    # 同一会话租约过期后再来取（一轮跑了半小时、转写还没回来）：续租、``claimed_at`` 不变——宿主按它判是不是
                    # 同一次，不再注入；换 ``claimed_at`` 只在真正放回之后（方案 32 第 3.4、4.1 条）
                    claim.history.remove(mine)
                    claim.holder = _Holder(host, session, mine.claimed_at, min(at + self.lease, request.deadline))
                    self.store.write_claim(request.request_id, _encode_claim(claim))
                    given.append(self._claimed(claim, claim.holder))
                    continue
                if claim.attempts >= self.max_attempts:
                    continue
                if holder is not None:
                    claim.history.append(holder)
                claim.holder = _Holder(host, session, at, min(at + self.lease, request.deadline))
                self.store.write_claim(request.request_id, _encode_claim(claim))
                given.append(self._claimed(claim, claim.holder))
        self._observe(ObservationStatus.SUCCESS, "claim", {"host": host, "given": len(given)})
        return tuple(given)

    @staticmethod
    def _claimed(claim: _Claim, holder: _Holder) -> ClaimedReminder:
        return ClaimedReminder(
            claim.request.request_id,
            claim.kind,
            claim.request.text,
            OPTIONS,
            claim.request.deadline,
            holder.lease_until,
            holder.claimed_at,
        )

    def _label(self, kind_token: str) -> str:
        if self.kind_label is None:
            return kind_token
        try:
            label = self.kind_label(kind_token)
        except Exception:  # noqa: BLE001 - 类名只是配对的辅助，词表读不出来不拦取
            return kind_token
        return label if isinstance(label, str) and label.strip() else kind_token

    # ── 认 ──────────────────────────────────────────────────────────────────

    def observe(self, session: str, messages: Sequence[HostMessage], *, host: str | None = None) -> ObserveReport:
        """一份会话送回来了：认提问（写「说了」、固定选项写回话、问了没答的记下）、认作答（Desktop 晚来的用户消息——同一份里
        先问后答的也认）、没问的放回。"""

        if not isinstance(session, str) or not session:
            raise ForesightError("session must be non-empty text")
        with self._lock:
            tools = {tool.tool: tool for tool in self.tools if host is None or tool.host == host}
            counts = {
                "spoken": 0,
                "responded": 0,
                "awaiting_reply": 0,
                "released": 0,
                "already_spoken": 0,
                "already_done": 0,
            }
            claims = [
                claim
                for claim in (_decode_claim(raw) for raw in self.store.claims().values())
                if claim is not None and session in claim.sessions()
            ]
            questions = [
                question
                for item in messages
                if item.role == "tool_call" and item.tool_name in tools
                for question in _questions(item, tools[item.tool_name or ""])
            ]
            results = {item.tool_call_id: item for item in messages if item.role == "tool_result" and item.tool_call_id}
            for claim, question in self._pair(session, claims, questions):
                outcome = self._settle_question(claim, question, results, tools, host, session)
                if outcome in ("spoken", "responded", "awaiting_reply"):
                    counts["spoken"] += 1
                if outcome != "spoken":
                    counts[outcome] += 1
            counts["responded"] += self._observe_replies(session, messages)
            counts["released"] += self._release_unasked(session, messages)
        report = ObserveReport(**counts)
        self._observe(ObservationStatus.SUCCESS, "observe", {"session": session, **counts})
        return report

    def _pair(
        self, session: str, claims: Sequence[_Claim], questions: Sequence[_Question]
    ) -> list[tuple[_Claim, _Question]]:
        """全部（取走的提醒 × 提问）两两打分、按分从高到低一一配对（不是按清单顺序取第一个过线的：两句模板文案只差类名时会交叉配错）。
        只看这个会话取走之后问的（取走时刻往前留一点容差）。"""

        active = sorted(claims, key=lambda c: (c.request.deadline, c.request.request_id))
        single = sum(1 for claim in active if not claim.spoken) == 1  # 兜底规则只数还没说出口的
        scored: list[tuple[float, int, int]] = []
        for c_index, claim in enumerate(active):
            since = (claim.claimed_at_for(session) or claim.request.formed_at) - _CLAIM_TOLERANCE
            for q_index, question in enumerate(questions):
                if question.call.occurred_at < since:
                    continue
                score = self._score(claim, question, single=single and not claim.spoken)
                if score is not None:
                    scored.append((score, c_index, q_index))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        used_claims: set[int] = set()
        used_questions: set[int] = set()
        pairs: list[tuple[_Claim, _Question]] = []
        for _score, c_index, q_index in scored:
            if c_index in used_claims or q_index in used_questions:
                continue
            used_claims.add(c_index)
            used_questions.add(q_index)
            pairs.append((active[c_index], questions[q_index]))
        return pairs

    def _score(self, claim: _Claim, question: _Question, *, single: bool) -> float | None:
        """一条取走的提醒与一题配不配、配得多像。先看选项：带着选项却不是我们那三个的，是 agent 自己的问题，不配。然后：
        我们的选项 + 题目含类名最强（2 + 相似度）；与那句话相似度过线次之（相似度本身；没带选项的提问只能走这条）；这个会话只剩
        一条没说出口时，带着我们三个选项的那一题兜底（0.5——题目被 agent 改写了也认得）。"""

        ours = _options_are_ours(question.options)
        if question.options and not ours:
            return None  # 带着别的选项的提问是 agent 自己的事（"写周报的模板用哪一个？"），题目再像也不是我们那句（裁定 65）
        wanted = _normalized(claim.request.text)
        similarity = _similarity(_normalized(question.title), wanted)
        if ours and claim.kind != claim.request.kind_token and claim.kind in question.title:
            return 2.0 + similarity
        if similarity >= self.similarity:
            return similarity
        if single and ours:
            return 0.5
        return None

    def _settle_question(
        self,
        claim: _Claim,
        question: _Question,
        results: Mapping[str, HostMessage],
        tools: Mapping[str, QuestionTool],
        host: str | None,
        session: str,
    ) -> str:
        """配上的一题：先写账本「说了」（写成了才标取走记录、删清单行——反过来账本写失败时记录已 spoken、谁也取不到、账上永远没「说了」），
        再认作答：同步工具的结果里他选了固定选项就写回话，晚答的记进问过的。返回计入哪一格。"""

        call = question.call
        tool = tools[call.tool_name or ""]
        if call.occurred_at > claim.request.deadline:
            self.store.remove_claim(claim.request.request_id)
            self._drop_open(claim.request.request_id)
            return (
                "already_spoken" if claim.spoken else "released"
            )  # 时限后才问的不算「说了」：那时账上已记「宿主没取」
        via = host or tool.host
        if not claim.spoken and self._ledger_latest_reason(claim) == "已发生":
            # 他在被问之前已经做了（取走到提问之间几分钟，行为识别又有滞后）：这次提问不算「说了」，账上不记、取走记录删掉——
            # 否则结算把"做了"记成提醒的功劳（裁定 65）
            self.store.remove_claim(claim.request.request_id)
            self._drop_open(claim.request.request_id)
            return "already_done"
        if claim.spoken or not self._record_spoken(claim, call, question.title, via):
            if not claim.spoken and self._ledger_has_spoken(claim):
                self._mark_spoken(claim)
            return "already_spoken"
        self._mark_spoken(claim)
        self._drop_open(claim.request.request_id)
        result = results.get(call.tool_call_id or "")
        if result is not None and not tool.replies_later and not result.failed:
            polarity = answer_in_result(result.content, question.title, single=len(_questions(call, tool)) == 1)
            if polarity is not None and self._record_response(
                claim, result.occurred_at, polarity, call.tool_call_id or ""
            ):
                return "responded"
            return "spoken"
        if call.tool_call_id:
            self.store.write_asked(
                OutletStore.asked_key(question.key), _encode_asked(claim, question, host=via, session=session)
            )
            return "awaiting_reply"
        return "spoken"

    def _mark_spoken(self, claim: _Claim) -> None:
        claim.spoken = True
        if claim.holder is not None:
            claim.holder.ended = "spoken"
            claim.history.append(claim.holder)
            claim.holder = None
        self.store.write_claim(claim.request.request_id, _encode_claim(claim))

    def _ledger_latest_reason(self, claim: _Claim) -> str | None:
        """这条预期账上最晚一条递送的原因（按决定时刻）；读不到当没有。"""

        try:
            mine = [
                d for d in self.ledger.deliveries_on(claim.request.day) if d.expectation_id == claim.request.request_id
            ]
        except (ForesightError, OSError):
            return None
        return max(mine, key=lambda d: d.decided_at).reason if mine else None

    def _ledger_has_spoken(self, claim: _Claim) -> bool:
        try:
            return any(
                d.spoken
                for d in self.ledger.deliveries_on(claim.request.day)
                if d.expectation_id == claim.request.request_id
            )
        except (ForesightError, OSError):
            return False

    def _observe_replies(self, session: str, messages: Sequence[HostMessage]) -> int:
        """Codex Desktop 的作答是一条用户消息：按 (call_id, 题序) 找到问过的那一题（本份转写里刚问的也算），固定选项直接写回话。"""

        asked = {key: raw for key, raw in self.store.asked().items() if raw.get("session") == session}
        if not asked:
            return 0
        written = 0
        for item in messages:
            if item.role != "prompt" or not item.content.lstrip().startswith(REPLY_TAG):
                continue
            for call_id, index, answer in parse_reply(item.content):
                key = OutletStore.asked_key(f"{call_id}#{index}")
                raw = asked.pop(key, None)
                if raw is None:
                    continue
                self.store.remove_asked(key)
                polarity = fixed_polarity(answer)
                if polarity is None:
                    continue  # 自己填的话：交给会话 lane 的模型判
                try:
                    day = date.fromisoformat(str(raw["day"]))
                    request_id = str(raw["request_id"])
                except (KeyError, ValueError):
                    continue
                if self._write_response(day, Response(request_id, item.occurred_at, polarity, call_id)):
                    written += 1
        return written

    def _record_spoken(self, claim: _Claim, call: HostMessage, title: str, via: str) -> bool:
        """账本追加「说了」；这条预期已经有一条说出口的就不再写（一条预期最多一条「说了」）。"""

        day = claim.request.day
        try:
            if any(d.spoken for d in self.ledger.deliveries_on(day) if d.expectation_id == claim.request.request_id):
                return False
            delivery = Delivery(
                claim.request.request_id,
                call.occurred_at,
                claim.request.level,
                True,
                SPOKEN,
                1.0,
                title,
                via=via,
                turn=call.tool_call_id or None,
            )
            self.ledger.record_delivery(day, delivery)
        except (ForesightError, OSError) as exc:
            self._observe(ObservationStatus.FAILURE, "spoken", {"error": type(exc).__name__})
            return False
        return True

    def _record_response(self, claim: _Claim, at: datetime, polarity: str, turn: str) -> bool:
        return self._write_response(
            claim.request.day, Response(claim.request.request_id, at, polarity, turn or "unknown")
        )

    def _write_response(self, day: date, response: Response) -> bool:
        try:
            self.ledger.record_response(day, response)
        except (ForesightError, OSError) as exc:
            self._observe(ObservationStatus.FAILURE, "response", {"error": type(exc).__name__})
            return False
        return True

    def _release_unasked(self, session: str, messages: Sequence[HostMessage]) -> int:
        """这个会话现持有、取走之后那一轮已经过了、却没问出口的：放回、``attempts += 1``。"""

        released = 0
        for request_id, raw in self.store.claims().items():
            claim = _decode_claim(raw)
            if claim is None or claim.holder is None or claim.holder.session != session:
                continue
            if not _turn_passed(messages, claim.holder.claimed_at):
                continue
            claim.holder.ended = "released"
            claim.history.append(claim.holder)
            claim.holder = None
            claim.attempts += 1
            self.store.write_claim(request_id, _encode_claim(claim))
            released += 1
        return released

    def _drop_open(self, request_id: str) -> None:
        rows = [row for row in self.store.read_open() if row.get("request_id") != request_id]
        self.store.write_open(rows)

    def _observe(self, status: ObservationStatus, operation: str, attributes: Mapping[str, Any]) -> None:
        try:
            self.observer.record(
                ObservationEvent(
                    category="foresight",
                    operation=f"outlet_{operation}",
                    status=status,
                    duration_seconds=0.0,
                    attributes={
                        key: value for key, value in attributes.items() if isinstance(value, str | int | float | bool)
                    },
                )
            )
        except Exception:  # noqa: BLE001 - 观测不许影响出口
            pass


# ── 认提问与作答的纯函数 ──────────────────────────────────────────────────────


def fixed_polarity(answer: str) -> str | None:
    """作答整句等于三个固定选项之一 → 方向；否则 None（自己填的话）。"""

    return POLARITY_OF_OPTION.get(" ".join(answer.split()))


def fixed_polarity_in(content: str) -> str | None:
    """同步提问工具的结果里他选了哪个固定选项：结果可能是 JSON（所有字符串叶子）、也可能是一段叙述（``="…"`` 引出的那几段、
    整句）；恰好出现一个固定选项才算，拒绝、自己填的话都给 None。"""

    found = {" ".join(item.split()) for item in _answer_candidates(content) if fixed_polarity(item) is not None}
    return POLARITY_OF_OPTION[next(iter(found))] if len(found) == 1 else None


def answer_in_result(content: str, title: str, *, single: bool) -> str | None:
    """同步提问工具的结果里这一题他选了哪个固定选项：先找 ``"<题目>"="<作答>"`` 这一题自己的那段；找不到、而这次调用只有一题时，
    整个结果里恰好出现一个固定选项也算。"""

    for match in _QUOTED_PAIR.finditer(content):
        if " ".join(match.group(1).split()) == " ".join(title.split()):
            return fixed_polarity(match.group(2))
    return fixed_polarity_in(content) if single else None


def parse_reply(content: str) -> tuple[tuple[str, int, str], ...]:
    """Codex Desktop 的作答消息 → ``(call_id, 题序, answer)``；形状不对给空（那条消息照旧当人说的话）。"""

    body = content.strip()
    if not body.startswith(REPLY_TAG):
        return ()
    body = body[len(REPLY_TAG) :]
    end = body.rfind("</")
    body = body[:end] if end >= 0 else body
    try:
        items = json.loads(body)
    except ValueError:
        return ()
    if not isinstance(items, list):
        return ()
    found: list[tuple[str, int, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        identity = _call_id_of(item.get("questionItemId"))
        answer = item.get("answer")
        if identity is not None and isinstance(answer, str):
            found.append((identity[0], identity[1], answer))
    return tuple(found)


def _call_id_of(value: object) -> tuple[str, int] | None:
    """``questionItemId`` 是一段 JSON 文本：``["request_user_input_async","call_xxx",0]`` —— 工具名、call_id、第几题。"""

    if not isinstance(value, str):
        return None
    try:
        parts = json.loads(value)
    except ValueError:
        return None
    if not (isinstance(parts, list) and len(parts) >= 2 and isinstance(parts[1], str) and parts[1]):
        return None
    index = parts[2] if len(parts) >= 3 and isinstance(parts[2], int) and not isinstance(parts[2], bool) else 0
    return parts[1], index


def _questions(call: HostMessage, tool: QuestionTool) -> tuple[_Question, ...]:
    """一次提问调用的参数 → 每一题（题序按参数里的位置，与 Desktop 作答的 questionItemId 第三项同一个数）。参数不是 JSON、
    没有 questions 的给空。"""

    try:
        arguments = json.loads(call.content)
    except ValueError:
        return ()
    questions = arguments.get("questions") if isinstance(arguments, dict) else None
    if not isinstance(questions, list):
        return ()
    found: list[_Question] = []
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        title = question.get(tool.title_key)
        if not isinstance(title, str) or not title.strip():
            continue
        options = question.get("options")
        labels: list[str] = []
        if isinstance(options, list):
            for option in options:
                if isinstance(option, str):
                    labels.append(option)
                elif isinstance(option, dict) and isinstance(option.get("label"), str):
                    labels.append(option["label"])
        found.append(_Question(call, index, title.strip(), tuple(labels)))
    return tuple(found)


def _options_are_ours(options: Sequence[str]) -> bool:
    given = {" ".join(item.split()) for item in options}
    return len(given & set(OPTIONS)) >= 2


def _answer_candidates(content: str) -> tuple[str, ...]:
    found: list[str] = [content]
    found.extend(_QUOTED.findall(content))
    try:
        parsed = json.loads(content)
    except ValueError:
        parsed = None
    found.extend(_string_leaves(parsed))
    return tuple(found)


def _string_leaves(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _string_leaves(item)]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _string_leaves(item)]
    return []


def _normalized(text: str) -> str:
    return "".join(text.split())


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _turn_passed(messages: Sequence[HostMessage], claimed_at: datetime) -> bool:
    """取走之后他那句话开的那一轮已经过了：有一条取走前后（容差）人说的话，后面跟着**取走之后**助手的答复——上一轮的转写晚到
    （他前两分钟那句话、取走前就答完了）不算这一轮过了。"""

    since = claimed_at - _CLAIM_TOLERANCE
    spoke = False
    for item in messages:
        if item.role == "prompt" and item.occurred_at >= since:
            spoke = True
        elif spoke and item.role == "completion" and item.content.strip() and item.occurred_at > claimed_at:
            return True
    return False


# ── 传输状态的编解码 ────────────────────────────────────────────────────────


def _encode_request(row: OpenRequest) -> dict[str, Any]:
    return {
        "request_id": row.request_id,
        "kind_token": row.kind_token,
        "text": row.text,
        "deadline": row.deadline.isoformat(),
        "formed_at": row.formed_at.isoformat(),
        "level": row.level,
        "day": row.day.isoformat(),
    }


def _decode_request(raw: Mapping[str, Any]) -> OpenRequest:
    try:
        return OpenRequest(
            str(raw["request_id"]),
            str(raw["kind_token"]),
            str(raw["text"]),
            datetime.fromisoformat(str(raw["deadline"])),
            datetime.fromisoformat(str(raw["formed_at"])),
            int(raw["level"]),
            date.fromisoformat(str(raw["day"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"open request row is malformed: {exc}") from exc


def _encode_holder(holder: _Holder) -> dict[str, Any]:
    return {
        "host": holder.host,
        "session": holder.session,
        "claimed_at": holder.claimed_at.isoformat(),
        "lease_until": holder.lease_until.isoformat(),
        "ended": holder.ended,
    }


def _decode_holder(raw: Mapping[str, Any]) -> _Holder:
    return _Holder(
        str(raw["host"]),
        str(raw["session"]),
        datetime.fromisoformat(str(raw["claimed_at"])),
        datetime.fromisoformat(str(raw["lease_until"])),
        None if raw.get("ended") is None else str(raw["ended"]),
    )


def _encode_claim(claim: _Claim) -> dict[str, Any]:
    return {
        "record": "claim",
        "request": _encode_request(claim.request),
        "kind": claim.kind,
        "holder": None if claim.holder is None else _encode_holder(claim.holder),
        "attempts": claim.attempts,
        "history": [_encode_holder(item) for item in claim.history],
        "spoken": claim.spoken,
    }


def _decode_claim(raw: Mapping[str, Any]) -> _Claim | None:
    try:
        holder = raw.get("holder")
        history = raw.get("history", [])
        return _Claim(
            _decode_request(raw["request"]),
            str(raw["kind"]),
            None if holder is None else _decode_holder(holder),
            int(raw["attempts"]),
            [_decode_holder(item) for item in history] if isinstance(history, list) else [],
            bool(raw.get("spoken", False)),
        )
    except (KeyError, TypeError, ValueError, ForesightError):
        return None


def _encode_asked(claim: _Claim, question: _Question, *, host: str, session: str) -> dict[str, Any]:
    """问出口了、作答还没来的一题：``session`` 是送来这份转写的会话（问的那个），不是取走记录上的持有者——租约过期后别的会话
    又取走时持有者已换人。"""

    return {
        "record": "asked",
        "request_id": claim.request.request_id,
        "day": claim.request.day.isoformat(),
        "deadline": claim.request.deadline.isoformat(),
        "host": host,
        "session": session,
        "call_id": question.call.tool_call_id,
        "question": question.index,
        "title": question.title,
        "asked_at": question.call.occurred_at.isoformat(),
    }


__all__ = [
    "OPTIONS",
    "POLARITY_OF_OPTION",
    "QUESTION_TOOLS",
    "REPLY_TAG",
    "ClaimedReminder",
    "HostMessage",
    "ObserveReport",
    "OpenRequest",
    "OutletPort",
    "QuestionTool",
    "ReminderOutlet",
    "answer_in_result",
    "fixed_polarity",
    "fixed_polarity_in",
    "parse_reply",
]
