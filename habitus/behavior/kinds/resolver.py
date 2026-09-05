"""把融合层的自由文本行为名归属到词表：模型判"是不是同一件事"，代码只备候选、只守结构。

## 分工（BHV-KINDS-002，用户裁定）

- **确定性快路径**：名字按规范身份（NFC + casefold）精确查词表（token、别名、label），命中即返回，
  不调模型——名字和某个 kind 的可读名一模一样，就是那个 kind。
- **候选准备只管召回**：未知名字的 embedding 对全部 kind 取最近 30 个 ∪ 命中天数最多的 20 个；
  embedding 不可用时退回字面重合并留信号。候选只决定模型看得到谁，不决定归属。
- **语义判断只由模型做**：一批（默认 10 个）名字一次调用，每个名字附证据（所在判断的 summary），
  模型对每个名字回答"和哪个候选是同一件事"（或同批里另一个名字、或都不是）；都不是时给这一类
  的可读 label。代码不用子串、字面、泛/具体之类的规则替它判——规则覆盖不了日常场景。
- **同 label 即同类**：模型说"这是一次交谈"，而词表里已有（或同批刚建了）label 为「交谈」的 kind，
  就归入它——这是对模型自己结论的一致性执行，不是代码替它判。
- **校验只守结构**：match 必须在给它的候选里或同批名字里、每个名字恰好一项、null 时 label 非空、
  同批互指不成环。违约只重问违约的名字；重问耗尽当 null 新建并留信号——不让一轮归约失败。

## 归一不记账

命中账在**发布时**记（归约把链写进树、账本记下 chain_digest 的那一步），与树上的 occurrence 一一
对应、重放幂等。这里只回答"名字 → token"。

## 模型不造 token

新 kind 的 token 是**原始名字本身**（稳定 id，创建后不改）；模型给的只是 label（可读名，可改）。
词表里的每个 token 都逐字来自融合层的原话。

## 宁分勿并

拿不准就 null。错误合并会把两类行为的统计搅在一起；错误分裂只是暂时多一个 kind，离线整理再并，
原始名一直在 occurrence 上。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.model import BehaviorKindError, BehaviorKindRegistry
from habitus.behavior.kinds.vectors import (
    BehaviorKindVectorIndex,
    literal_kinds,
    names_missing_vectors,
    nearest_kinds,
)
from habitus.behavior.model import semantic_name
from habitus.foundation.ids import canonical_path_identity
from habitus.model_client import ChatMessage, ChatRequest, StructuredChatClient
from habitus.model_client.contracts import ModelResponseError, ModelStructuredOutputError, ModelTransportError
from habitus.model_client.embedding import Embedder

KIND_PROMPT_VERSION = "behavior_kind_prompt_v3"

KIND_SYSTEM_PROMPT = """\
你在为一套行为记忆系统维护「行为类型词表」：让同一件事的不同说法落到同一个类型上，按类的统计
（多久做一次、通常什么时候做）才不会被措辞的随机波动切碎。

会给你一批**待归类的名字**（来自行为观测的原话，可能附一句它所在判断的摘要作证据），以及
每个名字各自的**候选类型**（编号 K…，每行是这一类的可读名和几个例子）。对每个名字判断：

- 它说的事和某个候选**是不是同一件事**。判据是**提醒句测试**：如果要提醒他做这件事，提醒的
  话是不是同一句。"该和人聊聊了"——「与Shure交谈」「参与讨论」「闲聊」是同一件事；
  "该吃饭了"——「吃披萨」「吃夜宵」是同一件事；对方是谁、在哪里、聊什么、朝哪个方向，通常
  不改变提醒句。是 → match 填那个候选的编号，逐字复制。
- 对象**决定了这件事是什么**时，就不是同一件：「吃药」不是「吃饭」，「查看手机」不是
  「查看后备箱」，「找笔」不是「找手套」，「洗手」不是「洗碗」，「上楼」不是「下楼」，
  「写白板」不是「擦白板」。只是动词相同、发生在相近场合，不算同一类。
- 候选里已经有一类的可读名和你要给这个名字的类型名相同（比如候选 K3 的可读名是「交谈」，
  而你判断这个名字就是一次交谈），那就是同一类：填 K3，不要再新建。
- 同一批里若两个名字其实是一件事而候选里没有，后一个的 match 填前一个名字的编号（N…）。
- 都不是，或拿不准 → match 填 null，并给 label：这一类的可读名，写**那件事本身**，不带对方、
  地点、话题、方向（「交谈」而不是「与Shure交谈」，「吃饭」而不是「吃披萨」）。

【重要】宁可 null，不要勉强合并。错误的合并会把两类行为的统计搅在一起，比暂时多出一个类型
更糟——多出的类型以后还能并回去，搅在一起的统计分不开。
"""


@dataclass(frozen=True)
class BehaviorKindRequest:
    """一个待归属的名字；``evidence`` 是给模型看的证据（它所在判断的 summary）。"""

    name: str
    evidence: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", semantic_name(self.name, "behavior kind name"))
        if self.evidence is not None and not isinstance(self.evidence, str):
            raise TypeError("evidence must be text")

    @property
    def identity(self) -> str:
        return canonical_path_identity(self.name, "behavior kind name")


@dataclass(frozen=True)
class BehaviorKindBatchResolution:
    """一批名字的归属结果：``tokens`` 按名字（每种拼写）给 token；词表与向量索引是应用了本批变化的新实例。"""

    tokens: Mapping[str, str]
    registry: BehaviorKindRegistry
    vectors: BehaviorKindVectorIndex | None
    created: tuple[str, ...]
    model_calls: int
    signals: tuple[str, ...]


def kind_match_schema(name_ids: Sequence[str], candidate_ids: Sequence[str]) -> dict[str, Any]:
    """一批名字的结构化输出：每个名字一项；match 只能是候选编号、同批名字编号或 null。

    "match 必须在**这个名字自己的**候选里"由校验器守（schema 做不到按项区分 enum）。
    """

    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"enum": list(name_ids)},
                        "match": {
                            "anyOf": [
                                {"enum": [*candidate_ids, *name_ids]},
                                {"type": "null"},
                            ],
                            "description": "同一件事的候选编号（K…）或同批名字编号（N…）；都不是或拿不准填 null。",
                        },
                        "label": {
                            "anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}],
                            "description": "match 为 null 时必填：这一类的可读名，写那件事本身，不带对象/对方/方向/话题。",
                        },
                    },
                    "required": ["name", "match", "label"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class _BatchView:
    """一批名字在模型眼里的编号表：N… ↔ 名字、K… ↔ 候选 token。渲染、校验、落账共用一份。"""

    chunk: tuple[BehaviorKindRequest, ...]
    candidates: Mapping[str, tuple[str, ...]]  # 名字 → 它自己的候选 token
    name_ids: Mapping[str, str] = field(init=False)  # N1 → 名字
    candidate_ids: Mapping[str, str] = field(init=False)  # K1 → token

    def __post_init__(self) -> None:
        name_ids = {f"N{i}": request.name for i, request in enumerate(self.chunk, start=1)}
        tokens_in_view: list[str] = []
        for request in self.chunk:
            for token in self.candidates[request.name]:
                if token not in tokens_in_view:
                    tokens_in_view.append(token)
        candidate_ids = {f"K{i}": token for i, token in enumerate(tokens_in_view, start=1)}
        object.__setattr__(self, "name_ids", name_ids)
        object.__setattr__(self, "candidate_ids", candidate_ids)

    @property
    def by_name(self) -> Mapping[str, BehaviorKindRequest]:
        return {request.name: request for request in self.chunk}

    def candidate_id_of(self, token: str) -> str:
        return next(cid for cid, value in self.candidate_ids.items() if value == token)


@dataclass(frozen=True)
class _Verdict:
    """一批的判定：名字 → (候选 token | 同批另一名字 | None, label)。"""

    matches: Mapping[str, str | None]
    labels: Mapping[str, str | None]
    model_calls: int


class BehaviorKindResolver:
    """给一批行为名找到（或建立）它们的类型归属。"""

    def __init__(
        self,
        client: StructuredChatClient,
        *,
        config: BehaviorKindConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be StructuredChatClient")
        resolved = config or BehaviorKindConfig()
        if not isinstance(resolved, BehaviorKindConfig):
            raise TypeError("config must be BehaviorKindConfig")
        if embedder is not None and not callable(getattr(embedder, "embed_documents", None)):
            raise TypeError("embedder must implement the Embedder contract")
        self.client = client
        self.config = resolved
        self.embedder = embedder

    # ── 入口 ────────────────────────────────────────────────────────────────────

    async def resolve_many(
        self,
        requests: Sequence[BehaviorKindRequest],
        registry: BehaviorKindRegistry,
        *,
        vectors: BehaviorKindVectorIndex | None = None,
    ) -> BehaviorKindBatchResolution:
        """全部名字一次归属完（内部仍按批调模型）；调用方不需要逐批落盘时用它。"""

        tokens: dict[str, str] = {}
        created: list[str] = []
        signals: list[str] = []
        calls = 0
        async for batch in self.resolve_batches(requests, registry, vectors=vectors):
            tokens.update(batch.tokens)
            created.extend(batch.created)
            signals.extend(batch.signals)
            calls += batch.model_calls
            registry, vectors = batch.registry, batch.vectors
        return BehaviorKindBatchResolution(tokens, registry, vectors, tuple(created), calls, tuple(signals))

    async def resolve_batches(
        self,
        requests: Sequence[BehaviorKindRequest],
        registry: BehaviorKindRegistry,
        *,
        vectors: BehaviorKindVectorIndex | None = None,
    ) -> AsyncIterator[BehaviorKindBatchResolution]:
        """按批产出归属结果：第一批是快路径（零调用），之后每 ``batch_size`` 个未知名字一批。

        每批的 ``registry``/``vectors`` 是累计到该批的新实例，调用方可在批间落盘、续租。
        """

        if not isinstance(registry, BehaviorKindRegistry):
            raise TypeError("registry must be BehaviorKindRegistry")
        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            raise TypeError("requests must be a sequence")

        # 按规范身份去重（「Wash」「wash」是同一个名字）：证据取先给出的，各拼写都拿到同一 token。
        merged: dict[str, BehaviorKindRequest] = {}
        spellings: dict[str, list[str]] = {}
        for request in requests:
            if not isinstance(request, BehaviorKindRequest):
                raise TypeError("requests must contain BehaviorKindRequest values")
            key = request.identity
            if key not in merged:
                merged[key] = request
            elif merged[key].evidence is None and request.evidence:
                merged[key] = BehaviorKindRequest(name=merged[key].name, evidence=request.evidence)
            spellings.setdefault(key, [])
            if request.name not in spellings[key]:
                spellings[key].append(request.name)

        def settled(names: Mapping[str, str]) -> dict[str, str]:
            out: dict[str, str] = {}
            for name, token in names.items():
                for spelling in spellings[canonical_path_identity(name, "behavior kind name")]:
                    out[spelling] = token
            return out

        # 快路径：token / 别名 / label 精确命中，零调用。旁册缺的向量顺带补齐（派生物自愈；
        # 没有缺就不调 embedding）。
        known: dict[str, str] = {}
        unknown: list[BehaviorKindRequest] = []
        for request in merged.values():
            token = registry.token_for(request.name)
            if token is not None:
                known[request.name] = token
            else:
                unknown.append(request)
        signals: list[str] = []
        vectors = await self._fill_missing_vectors(registry, vectors, signals)
        yield BehaviorKindBatchResolution(settled(known), registry, vectors, (), 0, tuple(signals))
        if not unknown:
            return

        # 词表为空且只有一个名字：没有任何东西可比，直接建。
        if registry.kind_count == 0 and len(unknown) == 1:
            request = unknown[0]
            registry = registry.with_new_kind(request.name)
            yield BehaviorKindBatchResolution(
                settled({request.name: request.name}), registry, vectors, (request.name,), 0, ()
            )
            return

        signals = []
        if self.embedder is None or vectors is None:
            signals.append(f"kind_candidates_literal {len(unknown)} names (no embedding configured)")
        query_vectors, vectors = await self._query_vectors(unknown, registry, vectors, signals)
        for start in range(0, len(unknown), self.config.batch_size):
            chunk = tuple(unknown[start : start + self.config.batch_size])
            view = _BatchView(
                chunk, {r.name: self._candidates(r.name, registry, vectors, query_vectors) for r in chunk}
            )
            verdict = await self._judge(view, registry, signals)
            tokens, registry, created = self._settle(view, verdict, registry, signals)
            vectors = await self._index_created(created, registry, vectors, query_vectors, signals)
            yield BehaviorKindBatchResolution(
                settled(tokens), registry, vectors, tuple(created), verdict.model_calls, tuple(signals)
            )
            signals = []

    # ── 向量 ────────────────────────────────────────────────────────────────────

    async def _fill_missing_vectors(
        self,
        registry: BehaviorKindRegistry,
        vectors: BehaviorKindVectorIndex | None,
        signals: list[str],
    ) -> BehaviorKindVectorIndex | None:
        """词表里缺向量的 token/label 补齐（旁册丢失/损坏/换模型后的自愈）；失败留信号。"""

        if self.embedder is None or vectors is None:
            return vectors
        missing = names_missing_vectors(registry, vectors)
        if not missing:
            return vectors
        try:
            return vectors.with_vectors(await self._embed(missing))
        except (ModelTransportError, ModelResponseError) as exc:
            signals.append(f"kind_embedding_fallback fill: {exc}")
            return vectors

    async def _query_vectors(
        self,
        unknown: Sequence[BehaviorKindRequest],
        registry: BehaviorKindRegistry,
        vectors: BehaviorKindVectorIndex | None,
        signals: list[str],
    ) -> tuple[dict[str, tuple[float, ...]], BehaviorKindVectorIndex | None]:
        """给未知名字算向量；失败退字面并留信号。"""

        if self.embedder is None or vectors is None:
            return {}, vectors
        try:
            embedded = await self._embed([r.name for r in unknown])
        except (ModelTransportError, ModelResponseError) as exc:
            signals.append(f"kind_embedding_fallback literal: {exc}")
            return {}, vectors
        return {r.name: embedded[r.name] for r in unknown}, vectors

    async def _index_created(
        self,
        created: Sequence[str],
        registry: BehaviorKindRegistry,
        vectors: BehaviorKindVectorIndex | None,
        query_vectors: Mapping[str, tuple[float, ...]],
        signals: list[str],
    ) -> BehaviorKindVectorIndex | None:
        """新 kind 的 token（已算）与 label（补算）进向量索引。"""

        if vectors is None or not created:
            return vectors
        fresh = {
            name: query_vectors[name] for name in created if name in query_vectors and not vectors.has(name)
        }
        if fresh:
            vectors = vectors.with_vectors(fresh)
        if self.embedder is None:
            return vectors
        pending = [label for label in (registry.label_of(name) for name in created) if not vectors.has(label)]
        if pending:
            try:
                vectors = vectors.with_vectors(await self._embed(pending))
            except (ModelTransportError, ModelResponseError) as exc:
                signals.append(f"kind_embedding_fallback labels: {exc}")
        return vectors

    async def _embed(self, names: Sequence[str]) -> dict[str, tuple[float, ...]]:
        assert self.embedder is not None
        unique = list(dict.fromkeys(names))
        if not unique:
            return {}
        embedded = await self.embedder.embed_documents(unique)
        return {name: tuple(vector.values) for name, vector in zip(unique, embedded, strict=True)}

    def _candidates(
        self,
        name: str,
        registry: BehaviorKindRegistry,
        vectors: BehaviorKindVectorIndex | None,
        query_vectors: Mapping[str, tuple[float, ...]],
    ) -> tuple[str, ...]:
        """向量最近邻 ∪ 高频；没有向量时字面重合 ∪ 高频。顺序稳定：先相近、后高频。"""

        if registry.kind_count == 0:
            return ()
        query = query_vectors.get(name)
        if query is not None and vectors is not None:
            near = nearest_kinds(query, registry, vectors, limit=self.config.vector_candidates)
        else:
            near = literal_kinds(name, registry, limit=self.config.literal_candidates)
        ordered = list(near)
        for token in registry.most_hit(self.config.frequent_candidates):
            if token not in ordered:
                ordered.append(token)
        return tuple(ordered)

    # ── 判定 ────────────────────────────────────────────────────────────────────

    async def _judge(self, view: _BatchView, registry: BehaviorKindRegistry, signals: list[str]) -> _Verdict:
        """按项校验、只重问违约项；轮次耗尽当 null 新建并留信号。"""

        matches: dict[str, str | None] = {}
        labels: dict[str, str | None] = {}
        remaining = view
        calls = 0
        for _ in range(self.config.validation_rounds + 1):
            if not remaining.chunk:
                break
            calls += 1
            try:
                items = await self._call(remaining, registry)
            except ModelStructuredOutputError as exc:
                # 整批级违约（结构层重试也没吐出合法形状）：算作本轮全部违约，进入下一轮重问。
                signals.append(f"kind_validation_rejected {sorted(r.name for r in remaining.chunk)}: {exc}")
                continue
            accepted, rejected = self._split(items, remaining)
            for name, (match, label) in accepted.items():
                matches[name] = match
                labels[name] = label
            if rejected:
                signals.append(f"kind_validation_rejected {sorted(rejected)}")
            remaining = _BatchView(
                tuple(r for r in remaining.chunk if r.name in rejected), remaining.candidates
            )
        for request in remaining.chunk:
            matches[request.name] = None
            labels[request.name] = None
            signals.append(f"kind_validation_exhausted {request.name!r} created as its own kind")
        return _Verdict(matches, labels, calls)

    async def _call(self, view: _BatchView, registry: BehaviorKindRegistry) -> list[dict[str, Any]]:
        """一次结构化调用；瞬态错误有界重试。返回已通过结构校验的 items。"""

        request = self._request(view, registry)
        schema = kind_match_schema(tuple(view.name_ids), tuple(view.candidate_ids))
        response = None
        for attempt in range(self.config.transient_retries + 1):
            try:
                response = await self.client.complete_json_async(
                    request,
                    schema=schema,
                    name="behavior_kind_match",
                    validator=lambda parsed: self._validated(parsed, view),
                )
                break
            except (ModelTransportError, ModelResponseError):
                if attempt >= self.config.transient_retries:
                    raise
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
        assert response is not None
        return cast("list[dict[str, Any]]", response.value)  # 校验器已保证形状

    @staticmethod
    def _validated(parsed: object, view: _BatchView) -> list[dict[str, Any]]:
        """结构校验：items 是列表、每项形状合法、名字编号已知且不重复。

        "match 在本名字自己的候选里""互指不成环""缺项"在 ``_split`` 里按项判，只重问那一项。
        """

        if not isinstance(parsed, Mapping) or set(parsed) != {"items"}:
            raise BehaviorKindError("kind match output must be an object with exactly `items`")
        items = parsed["items"]
        if not isinstance(items, list) or not items:
            raise BehaviorKindError("kind match items must be a non-empty list")
        seen: set[str] = set()
        cleaned: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping) or set(item) != {"name", "match", "label"}:
                raise BehaviorKindError("kind match item must have exactly name/match/label")
            name_id = item["name"]
            if name_id not in view.name_ids or name_id in seen:
                raise BehaviorKindError("kind match item names must be known and unique")
            seen.add(name_id)
            match = item["match"]
            if match is not None and match not in view.candidate_ids and match not in view.name_ids:
                raise BehaviorKindError("kind match must be a listed candidate, a batch name, or null")
            label = item["label"]
            if label is not None and (not isinstance(label, str) or not label.strip()):
                raise BehaviorKindError("kind label must be non-empty text or null")
            cleaned.append({"name": name_id, "match": match, "label": label})
        return cleaned

    @staticmethod
    def _split(
        items: Sequence[Mapping[str, Any]], view: _BatchView
    ) -> tuple[dict[str, tuple[str | None, str | None]], set[str]]:
        """按项判：缺项、候选越界、null 无 label、互指成环 → 该名字重问；其余接收。"""

        by_id = {view.name_ids[item["name"]]: item for item in items}
        accepted: dict[str, tuple[str | None, str | None]] = {}
        rejected: set[str] = set()
        for request in view.chunk:
            item = by_id.get(request.name)
            if item is None:
                rejected.add(request.name)
                continue
            match, label = item["match"], item["label"]
            if match is None:
                if label is None:
                    rejected.add(request.name)
                else:
                    accepted[request.name] = (None, semantic_name(label.strip(), "behavior kind label"))
            elif match in view.candidate_ids:
                token = view.candidate_ids[match]
                if token in view.candidates[request.name]:
                    accepted[request.name] = (token, None)
                else:
                    rejected.add(request.name)  # 候选是别的名字的，不是它自己的
            else:
                other = view.name_ids[match]
                if other == request.name:
                    rejected.add(request.name)
                else:
                    accepted[request.name] = (other, None)
        # 互指成环（N1→N2→N1）：环上以及走进环的名字都定不下来，一起重问。
        for name in list(accepted):
            seen = {name}
            cursor = accepted[name][0]
            while cursor in accepted and cursor in by_id:
                if cursor in seen:
                    rejected.update(seen)
                    break
                seen.add(cursor)
                cursor = accepted[cursor][0]
        for name in rejected:
            accepted.pop(name, None)
        return accepted, rejected

    # ── 落账 ────────────────────────────────────────────────────────────────────

    def _settle(
        self,
        view: _BatchView,
        verdict: _Verdict,
        registry: BehaviorKindRegistry,
        signals: list[str],
    ) -> tuple[dict[str, str], BehaviorKindRegistry, list[str]]:
        """把判定落进词表：先落 null/候选命中的，再落指向同批名字的（目标已有 token）。

        同 label 即同类；容量撞顶只降级撞顶的那个名字（原始名作 token、留信号），同批其它判定照落。
        """

        by_name = view.by_name
        tokens: dict[str, str] = {}
        created: list[str] = []
        state = {"registry": registry}

        def alias(token: str, request: BehaviorKindRequest) -> None:
            current = state["registry"]
            if current.token_for(token) is None:
                signals.append(f"kind_alias_unrecorded {request.name!r} -> {token!r} (token not registered)")
                return
            if len(current.aliases_of(token)) >= self.config.max_aliases_per_kind:
                signals.append(f"kind_alias_capacity_full {request.name!r} -> {token!r} kept unrecorded")
                return
            state["registry"] = current.with_alias(token, request.name)

        def settle(name: str) -> str:
            if name in tokens:
                return tokens[name]
            request = by_name[name]
            target = verdict.matches[name]
            if target is None:
                label = verdict.labels[name]
                owner = state["registry"].token_for_label(label) if label is not None else None
                if owner is not None:
                    alias(owner, request)  # 模型说"这是一次 X"，词表里已有 X：归入，不另建
                    token = owner
                elif state["registry"].kind_count >= self.config.max_kinds:
                    signals.append(f"kind_registry_full {request.name!r} kept as its own token")
                    token = request.name
                else:
                    # label 为 None 只出现在重问耗尽的降级路径：留记号，离线整理先复核它。
                    state["registry"] = state["registry"].with_new_kind(
                        request.name,
                        label=label,
                        review_reason=None if label is not None else "validation_exhausted",
                    )
                    created.append(request.name)
                    token = request.name
            else:
                token = settle(target) if target in by_name else target
                alias(token, request)
            tokens[name] = token
            return token

        for request in view.chunk:
            settle(request.name)
        return tokens, state["registry"], created

    # ── 渲染 ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _request(view: _BatchView, registry: BehaviorKindRegistry) -> ChatRequest:
        """候选行 = 编号、可读名、几种说法；名字行 = 编号、名字、它自己的候选编号、证据。"""

        candidate_lines = ["【候选类型】（编号  可读名  例：这一类的几种说法）"]
        if not view.candidate_ids:
            candidate_lines.append("（无）")
        for cid, token in view.candidate_ids.items():
            entry = registry.entry_of(token)
            examples = [name for name in (entry.token, *entry.aliases) if name != entry.label][:3]
            shown = f"  例：{'、'.join(examples)}" if examples else ""
            candidate_lines.append(f"{cid}  {entry.label}{shown}")
        name_lines = ["【待归类的名字】（编号  名字  它的候选编号  证据）"]
        for nid, request in zip(view.name_ids, view.chunk, strict=True):
            ids = "、".join(view.candidate_id_of(token) for token in view.candidates[request.name]) or "（无候选）"
            evidence = f"  证据：{request.evidence}" if request.evidence else ""
            name_lines.append(f"{nid}  {request.name}  候选：{ids}{evidence}")
        return ChatRequest(
            messages=(
                ChatMessage(role="system", content=KIND_SYSTEM_PROMPT),
                ChatMessage(role="user", content="\n".join((*candidate_lines, "", *name_lines))),
            )
        )


__all__ = [
    "KIND_PROMPT_VERSION",
    "KIND_SYSTEM_PROMPT",
    "BehaviorKindBatchResolution",
    "BehaviorKindRequest",
    "BehaviorKindResolver",
    "kind_match_schema",
]
