"""把融合层的自由文本行为名归属到词表：模型判"是不是同一件事"，代码只备候选、只守结构。

## 分工（BHV-KINDS-002，用户裁定）

- **确定性快路径**：名字按规范身份（NFC + casefold）精确查词表，命中即返回，不调模型。
- **候选准备只管召回**：未知名字的 embedding 对全部 kind 取最近 30 个 ∪ 命中天数最多的 20 个；
  embedding 不可用时退回字面重合并留信号。候选只决定模型看得到谁，不决定归属。
- **语义判断只由模型做**：一批（默认 10 个）名字一次调用，每个名字附证据（所在判断的 summary），
  模型对每个名字回答"和哪个候选是同一件事"（或同批里另一个名字、或都不是）；都不是时给这一类
  的可读 label。代码不用子串、字面、泛/具体之类的规则替它判——规则覆盖不了日常场景。
- **校验只守结构**：match 必须在给它的候选里或同批名字里、每个名字恰好一项、null 时 label 非空、
  同批互指不成环。违约只重问违约的名字；重问耗尽当 null 新建并留信号——不让一轮归约失败。

## 模型不造 token

新 kind 的 token 是**原始名字本身**（稳定 id，创建后不改）；模型给的只是 label（可读名，可改）。
词表里的每个 token 都逐字来自融合层的原话。

## 宁分勿并

拿不准就 null。错误合并会把两类行为的统计搅在一起；错误分裂只是暂时多一个 kind，离线整理
（成对交模型判"是否同一件事"）再并，原始名一直在 occurrence 上。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from behavior.kinds.config import BehaviorKindConfig
from behavior.kinds.model import (
    BehaviorKindError,
    BehaviorKindLimitError,
    BehaviorKindRegistry,
)
from behavior.kinds.vectors import (
    BehaviorKindVectorIndex,
    literal_kinds,
    names_missing_vectors,
    nearest_kinds,
)
from behavior.model import semantic_name
from ModelClient import ChatMessage, ChatRequest, StructuredChatClient
from ModelClient.contracts import ModelResponseError, ModelTransportError
from ModelClient.embedding import Embedder

KIND_PROMPT_VERSION = "behavior_kind_prompt_v2"

KIND_SYSTEM_PROMPT = """\
你在为一套行为记忆系统维护「行为类型词表」：让同一件事的不同说法落到同一个类型上，按类的统计
（多久做一次、通常什么时候做）才不会被措辞的随机波动切碎。

会给你一批**待归类的名字**（来自行为观测的原话，可能附一句它所在判断的摘要作证据），以及
每个名字各自的**候选类型**（编号 K…，每行是这一类的可读名和几个例子）。对每个名字判断：

- 它说的事和某个候选**是不是同一件事**——换了措辞、说得更具体或更笼统、带上了对象、对方、
  地点、话题、方向，都还是同一件事：「与Shure交谈」「参与讨论」「闲聊」都是「交谈」；
  「吃披萨」「吃夜宵」都是「吃饭」；「给Katrina充电」「拔下充电宝」都是「给设备充电」。
  是 → match 填那个候选的编号，逐字复制。
- 同一批里若两个名字其实是一件事而候选里没有，后一个的 match 填前一个名字的编号（N…）。
- 都不是，或拿不准 → match 填 null，并给 label：这一类的可读名，写**那件事本身**，不带对象、
  对方、方向、话题（「交谈」而不是「与Shure交谈」，「吃饭」而不是「吃披萨」）。

只是发生在相近场合、用了相近动词的不同行为不算同一类：「洗手」与「洗碗」是两类，「上楼」与
「下楼」是两类，「拿起」与「放下」是两类。

【重要】宁可 null，不要勉强合并。错误的合并会把两类行为的统计搅在一起，比暂时多出一个类型
更糟——多出的类型以后还能并回去，搅在一起的统计分不开。
"""


@dataclass(frozen=True)
class BehaviorKindRequest:
    """一个待归属的名字：``day`` 是它发生的行为日（记命中用），``evidence`` 是给模型看的证据。"""

    name: str
    day: date | None = None
    evidence: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", semantic_name(self.name, "behavior kind name"))
        if self.day is not None and not isinstance(self.day, date):
            raise TypeError("day must be a date")
        if self.evidence is not None and not isinstance(self.evidence, str):
            raise TypeError("evidence must be text")


@dataclass(frozen=True)
class BehaviorKindResolution:
    """单个名字的归属结果（``resolve`` 的返回；保留给只处理一个名字的调用方）。"""

    token: str
    registry: BehaviorKindRegistry
    created: bool
    model_called: bool
    validation_attempts: int


@dataclass(frozen=True)
class BehaviorKindBatchResolution:
    """一批名字的归属结果：``tokens`` 按名字给 token；词表与向量索引是应用了本批变化的新实例。"""

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

    # ── 单名入口（兼容只处理一个名字的调用方）─────────────────────────────────────

    async def resolve(self, name: object, registry: BehaviorKindRegistry) -> BehaviorKindResolution:
        """归属一个名字（不记命中日）；已知名字零模型调用。"""

        request = BehaviorKindRequest(name=semantic_name(name, "behavior kind name"))
        batch = await self.resolve_many((request,), registry)
        return BehaviorKindResolution(
            token=batch.tokens[request.name],
            registry=batch.registry,
            created=request.name in batch.created,
            model_called=batch.model_calls > 0,
            validation_attempts=batch.model_calls,
        )

    # ── 批量入口 ─────────────────────────────────────────────────────────────────

    async def resolve_many(
        self,
        requests: Sequence[BehaviorKindRequest],
        registry: BehaviorKindRegistry,
        *,
        vectors: BehaviorKindVectorIndex | None = None,
    ) -> BehaviorKindBatchResolution:
        if not isinstance(registry, BehaviorKindRegistry):
            raise TypeError("registry must be BehaviorKindRegistry")
        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            raise TypeError("requests must be a sequence")
        tokens: dict[str, str] = {}
        created: list[str] = []
        signals: list[str] = []
        model_calls = 0

        # 同名去重（保留最早给出的证据），已知名字走快路径并记命中。
        unknown: dict[str, BehaviorKindRequest] = {}
        for request in requests:
            if not isinstance(request, BehaviorKindRequest):
                raise TypeError("requests must contain BehaviorKindRequest values")
            if request.name in tokens or request.name in unknown:
                continue
            known = registry.token_for(request.name)
            if known is not None:
                tokens[request.name] = known
                if request.day is not None:
                    registry = registry.with_hit(known, request.day)
                continue
            unknown[request.name] = request
        if not unknown:
            return BehaviorKindBatchResolution(tokens, registry, vectors, (), 0, ())

        # 词表为空且只有一个名字：没有任何东西可比，直接建（也免得空候选把模型逼出编造）。
        if registry.kind_count == 0 and len(unknown) == 1:
            request = next(iter(unknown.values()))
            registry = self._create(registry, request, label=None)
            created.append(request.name)
            tokens[request.name] = request.name
            return BehaviorKindBatchResolution(tokens, registry, vectors, tuple(created), 0, ())

        # 向量：给未知名字算，顺带补齐词表里缺向量的 token/label（旁册是派生物）。
        query_vectors: dict[str, tuple[float, ...]] = {}
        if self.embedder is not None and vectors is not None:
            try:
                missing = names_missing_vectors(registry, vectors)
                embedded = await self._embed((*unknown, *missing))
                query_vectors = {name: embedded[name] for name in unknown}
                if missing:
                    vectors = vectors.with_vectors({name: embedded[name] for name in missing})
            except (ModelTransportError, ModelResponseError) as exc:
                signals.append(f"embedding_fallback literal: {exc}")
                query_vectors = {}

        pending = list(unknown.values())
        for start in range(0, len(pending), self.config.batch_size):
            chunk = pending[start : start + self.config.batch_size]
            outcome = await self._judge_chunk(chunk, registry, vectors, query_vectors, signals)
            model_calls += outcome.model_calls
            registry, vectors = self._apply(
                chunk, outcome.matches, outcome.labels, registry, vectors, query_vectors, tokens, created
            )
        # 新 kind 的 label 也进向量索引（候选检索按 token/label 取最大相似度）。
        if self.embedder is not None and vectors is not None and created:
            labels = [registry.label_of(name) for name in created]
            fresh = [label for label in labels if not vectors.has(label)]
            if fresh:
                try:
                    vectors = vectors.with_vectors(await self._embed(fresh))
                except (ModelTransportError, ModelResponseError) as exc:
                    signals.append(f"embedding_fallback labels: {exc}")
        return BehaviorKindBatchResolution(
            tokens, registry, vectors, tuple(created), model_calls, tuple(signals)
        )

    # ── 一批的判定 ───────────────────────────────────────────────────────────────

    @dataclass(frozen=True)
    class _Outcome:
        matches: Mapping[str, str | None]  # 名字 → 候选 token / 同批名字 / None
        labels: Mapping[str, str | None]
        model_calls: int

    async def _judge_chunk(
        self,
        chunk: Sequence[BehaviorKindRequest],
        registry: BehaviorKindRegistry,
        vectors: BehaviorKindVectorIndex | None,
        query_vectors: Mapping[str, tuple[float, ...]],
        signals: list[str],
    ) -> _Outcome:
        candidates = {request.name: self._candidates(request.name, registry, vectors, query_vectors) for request in chunk}
        matches: dict[str, str | None] = {}
        labels: dict[str, str | None] = {}
        remaining = list(chunk)
        calls = 0
        for _ in range(self.config.validation_rounds + 1):
            if not remaining:
                break
            if registry.kind_count == 0 and len(remaining) == 1:
                # 只剩一个名字且没有候选：无可比对，直接新建（同批其余已判完）。
                matches[remaining[0].name] = None
                labels[remaining[0].name] = None
                remaining = []
                break
            calls += 1
            parsed = await self._call(remaining, candidates, registry)
            accepted, rejected = self._split(parsed, remaining, candidates)
            for name, (match, label) in accepted.items():
                matches[name] = match
                labels[name] = label
            remaining = [request for request in remaining if request.name in rejected]
            if rejected:
                signals.append(f"kind_validation_rejected {sorted(rejected)}")
        for request in remaining:
            # 重问耗尽：当 null 新建，label 暂用名字本身——不让一轮归约失败。
            matches[request.name] = None
            labels[request.name] = None
            signals.append(f"kind_validation_exhausted {request.name!r} created as its own kind")
        return self._Outcome(matches, labels, calls)

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

    async def _call(
        self,
        chunk: Sequence[BehaviorKindRequest],
        candidates: Mapping[str, tuple[str, ...]],
        registry: BehaviorKindRegistry,
    ) -> list[dict[str, Any]]:
        """一次结构化调用；瞬态错误有界重试。返回已通过结构校验的 items。"""

        name_ids = {f"N{i}": request.name for i, request in enumerate(chunk, start=1)}
        tokens_in_view: list[str] = []
        for request in chunk:
            for token in candidates[request.name]:
                if token not in tokens_in_view:
                    tokens_in_view.append(token)
        candidate_ids = {f"K{i}": token for i, token in enumerate(tokens_in_view, start=1)}
        request_message = self._request(chunk, name_ids, candidate_ids, candidates, registry)
        schema = kind_match_schema(tuple(name_ids), tuple(candidate_ids))
        response = None
        for attempt in range(self.config.transient_retries + 1):
            try:
                response = await self.client.complete_json_async(
                    request_message,
                    schema=schema,
                    name="behavior_kind_match",
                    validator=lambda parsed: self._validated(parsed, name_ids, candidate_ids),
                )
                break
            except (ModelTransportError, ModelResponseError):
                if attempt >= self.config.transient_retries:
                    raise
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
        assert response is not None
        return list(response.value)  # type: ignore[arg-type]  # 校验器已保证形状

    @staticmethod
    def _validated(
        parsed: object, name_ids: Mapping[str, str], candidate_ids: Mapping[str, str]
    ) -> list[dict[str, Any]]:
        """结构校验：items 是列表、每项形状合法、名字编号已知且不重复。

        "match 在本名字自己的候选里"与"互指不成环"在 ``_split`` 里按项判，违约只重问那一项。
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
            if name_id not in name_ids or name_id in seen:
                raise BehaviorKindError("kind match item names must be known and unique")
            seen.add(name_id)
            match = item["match"]
            if match is not None and match not in candidate_ids and match not in name_ids:
                raise BehaviorKindError("kind match must be a listed candidate, a batch name, or null")
            label = item["label"]
            if label is not None and (not isinstance(label, str) or not label.strip()):
                raise BehaviorKindError("kind label must be non-empty text or null")
            cleaned.append({"name": name_id, "match": match, "label": label})
        return cleaned

    def _split(
        self,
        items: Sequence[Mapping[str, Any]],
        chunk: Sequence[BehaviorKindRequest],
        candidates: Mapping[str, tuple[str, ...]],
    ) -> tuple[dict[str, tuple[str | None, str | None]], set[str]]:
        """按项判：候选越界、null 无 label、互指成环 → 该名字重问；其余接收。"""

        name_ids = {f"N{i}": request.name for i, request in enumerate(chunk, start=1)}
        by_name = {name_ids[item["name"]]: item for item in items}
        tokens_in_view: list[str] = []
        for request in chunk:
            for token in candidates[request.name]:
                if token not in tokens_in_view:
                    tokens_in_view.append(token)
        candidate_ids = {f"K{i}": token for i, token in enumerate(tokens_in_view, start=1)}
        accepted: dict[str, tuple[str | None, str | None]] = {}
        rejected: set[str] = set()
        for request in chunk:
            item = by_name.get(request.name)
            if item is None:
                rejected.add(request.name)
                continue
            match, label = item["match"], item["label"]
            if match is None:
                if label is None:
                    rejected.add(request.name)
                else:
                    accepted[request.name] = (None, semantic_name(label.strip(), "behavior kind label"))
                continue
            if match in candidate_ids:
                token = candidate_ids[match]
                if token not in candidates[request.name]:
                    rejected.add(request.name)  # 候选是别的名字的，不是它自己的
                    continue
                accepted[request.name] = (token, None)
                continue
            other = name_ids[match]
            if other == request.name:
                rejected.add(request.name)
                continue
            accepted[request.name] = (other, None)
        # 互指成环（N1→N2→N1）：环上的名字全部重问。
        for name in list(accepted):
            seen = {name}
            cursor = accepted[name][0]
            while cursor in accepted and accepted[cursor][0] is not None and cursor in by_name:
                if cursor in seen:
                    rejected.update(seen)
                    break
                seen.add(cursor)
                cursor = accepted[cursor][0]
        for name in rejected:
            accepted.pop(name, None)
        return accepted, rejected

    def _apply(
        self,
        chunk: Sequence[BehaviorKindRequest],
        matches: Mapping[str, str | None],
        labels: Mapping[str, str | None],
        registry: BehaviorKindRegistry,
        vectors: BehaviorKindVectorIndex | None,
        query_vectors: Mapping[str, tuple[float, ...]],
        tokens: dict[str, str],
        created: list[str],
    ) -> tuple[BehaviorKindRegistry, BehaviorKindVectorIndex | None]:
        """把判定落进词表：先落 null/候选命中的，再落指向同批名字的（目标已有 token）。"""

        by_name = {request.name: request for request in chunk}
        settled: dict[str, str] = {}

        def settle(name: str) -> str:
            if name in settled:
                return settled[name]
            request = by_name[name]
            target = matches[name]
            if target is None:
                token = self._create_token(registry_box, request, labels[name], created)
            elif target in by_name:
                token = settle(target)
                registry_box[0] = self._alias(registry_box[0], token, request)
            else:
                token = target
                registry_box[0] = self._alias(registry_box[0], token, request)
            settled[name] = token
            tokens[name] = token
            return token

        registry_box = [registry]
        for request in chunk:
            settle(request.name)
        registry = registry_box[0]
        if vectors is not None:
            fresh = {name: query_vectors[name] for name in created if name in query_vectors and not vectors.has(name)}
            if fresh:
                vectors = vectors.with_vectors(fresh)
        return registry, vectors

    def _create_token(
        self,
        registry_box: list[BehaviorKindRegistry],
        request: BehaviorKindRequest,
        label: str | None,
        created: list[str],
    ) -> str:
        registry_box[0] = self._create(registry_box[0], request, label=label)
        created.append(request.name)
        return request.name

    def _create(
        self, registry: BehaviorKindRegistry, request: BehaviorKindRequest, *, label: str | None
    ) -> BehaviorKindRegistry:
        if registry.kind_count >= self.config.max_kinds:
            raise BehaviorKindLimitError("behavior kind registry has no remaining kind capacity")
        return registry.with_new_kind(request.name, label=label, day=request.day)

    def _alias(
        self, registry: BehaviorKindRegistry, token: str, request: BehaviorKindRequest
    ) -> BehaviorKindRegistry:
        if len(registry.aliases_of(token)) >= self.config.max_aliases_per_kind:
            raise BehaviorKindLimitError(f"behavior kind has no remaining alias capacity: {token}")
        return registry.with_alias(token, request.name, day=request.day)

    async def _embed(self, names: Sequence[str]) -> dict[str, tuple[float, ...]]:
        assert self.embedder is not None
        unique = list(dict.fromkeys(names))
        if not unique:
            return {}
        vectors = await self.embedder.embed_documents(unique)
        return {name: tuple(vector.values) for name, vector in zip(unique, vectors, strict=True)}

    @staticmethod
    def _request(
        chunk: Sequence[BehaviorKindRequest],
        name_ids: Mapping[str, str],
        candidate_ids: Mapping[str, str],
        candidates: Mapping[str, tuple[str, ...]],
        registry: BehaviorKindRegistry,
    ) -> ChatRequest:
        """渲染一批：候选行 = 编号、可读名、几种说法；名字行 = 编号、名字、它自己的候选编号、证据。"""

        candidate_lines = ["【候选类型】（编号  可读名  例：这一类的几种说法）"]
        if not candidate_ids:
            candidate_lines.append("（无）")
        for cid, token in candidate_ids.items():
            entry = registry.entry_of(token)
            examples = [name for name in (entry.token, *entry.aliases) if name != entry.label][:3]
            shown = f"  例：{'、'.join(examples)}" if examples else ""
            candidate_lines.append(f"{cid}  {entry.label}{shown}")
        id_of = {token: cid for cid, token in candidate_ids.items()}
        name_lines = ["【待归类的名字】（编号  名字  它的候选编号  证据）"]
        for nid, request in zip(name_ids, chunk, strict=True):
            ids = "、".join(id_of[token] for token in candidates[request.name]) or "（无候选）"
            evidence = f"  证据：{request.evidence}" if request.evidence else ""
            name_lines.append(f"{nid}  {request.name}  候选：{ids}{evidence}")
        content = "\n".join((*candidate_lines, "", *name_lines))
        return ChatRequest(
            messages=(
                ChatMessage(role="system", content=KIND_SYSTEM_PROMPT),
                ChatMessage(role="user", content=content),
            )
        )


__all__ = [
    "KIND_PROMPT_VERSION",
    "KIND_SYSTEM_PROMPT",
    "BehaviorKindBatchResolution",
    "BehaviorKindRequest",
    "BehaviorKindResolution",
    "BehaviorKindResolver",
    "kind_match_schema",
]
