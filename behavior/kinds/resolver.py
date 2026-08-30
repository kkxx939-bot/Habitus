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
from foundation.ids import canonical_path_identity
from ModelClient import ChatMessage, ChatRequest, StructuredChatClient
from ModelClient.contracts import ModelResponseError, ModelStructuredOutputError, ModelTransportError
from ModelClient.embedding import Embedder

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
    """一个待归属的名字：``days`` 是它发生过的行为日（每一天各记一次命中），``evidence`` 是给模型看的证据。

    同一轮归约常含多天数据（积压、重放）；同名多天的请求由调用方聚合成一条，天数不能丢。
    """

    name: str
    days: tuple[date, ...] = ()
    evidence: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", semantic_name(self.name, "behavior kind name"))
        if isinstance(self.days, str) or not isinstance(self.days, Sequence) or any(
            not isinstance(day, date) for day in self.days
        ):
            raise TypeError("days must be a sequence of dates")
        object.__setattr__(self, "days", tuple(sorted(set(self.days))))
        if self.evidence is not None and not isinstance(self.evidence, str):
            raise TypeError("evidence must be text")

    @property
    def identity(self) -> str:
        return canonical_path_identity(self.name, "behavior kind name")

    def merged_with(self, other: BehaviorKindRequest) -> BehaviorKindRequest:
        """同身份的两条请求合并：天数并集，证据取先给出的。"""

        return BehaviorKindRequest(
            name=self.name, days=(*self.days, *other.days), evidence=self.evidence or other.evidence
        )


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

        # 按规范身份去重（「Wash」「wash」是同一个名字）：同身份的请求合并天数，各拼写都拿到同一 token。
        merged: dict[str, BehaviorKindRequest] = {}
        spellings: dict[str, list[str]] = {}
        for request in requests:
            if not isinstance(request, BehaviorKindRequest):
                raise TypeError("requests must contain BehaviorKindRequest values")
            key = request.identity
            merged[key] = merged[key].merged_with(request) if key in merged else request
            spellings.setdefault(key, [])
            if request.name not in spellings[key]:
                spellings[key].append(request.name)

        def settle_all(key: str, token: str) -> None:
            for spelling in spellings[key]:
                tokens[spelling] = token

        # 已知名字走快路径，只记命中。
        unknown: dict[str, BehaviorKindRequest] = {}
        for key, request in merged.items():
            known = registry.token_for(request.name)
            if known is not None:
                settle_all(key, known)
                for day in request.days:
                    registry = registry.with_hit(known, day)
                continue
            unknown[request.name] = request
        if not unknown:
            return BehaviorKindBatchResolution(tokens, registry, vectors, (), 0, ())

        # 词表为空且只有一个名字：没有任何东西可比，直接建（也免得空候选把模型逼出编造）。
        if registry.kind_count == 0 and len(unknown) == 1:
            request = next(iter(unknown.values()))
            registry = self._create(registry, request, label=None)
            created.append(request.name)
            settle_all(request.identity, request.name)
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
                signals.append(f"kind_embedding_fallback literal: {exc}")
                query_vectors = {}

        pending = list(unknown.values())
        for start in range(0, len(pending), self.config.batch_size):
            chunk = pending[start : start + self.config.batch_size]
            outcome = await self._judge_chunk(chunk, registry, vectors, query_vectors, signals)
            model_calls += outcome.model_calls
            registry, vectors = self._apply(
                chunk, outcome.matches, outcome.labels, registry, vectors, query_vectors, tokens, created, signals
            )
            for request in chunk:
                settle_all(request.identity, tokens[request.name])
        # 新 kind 的 label 也进向量索引（候选检索按 token/label 取最大相似度）。
        if self.embedder is not None and vectors is not None and created:
            labels = [registry.label_of(name) for name in created]
            fresh = [label for label in labels if not vectors.has(label)]
            if fresh:
                try:
                    vectors = vectors.with_vectors(await self._embed(fresh))
                except (ModelTransportError, ModelResponseError) as exc:
                    signals.append(f"kind_embedding_fallback labels: {exc}")
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
            calls += 1
            try:
                parsed = await self._call(remaining, candidates, registry)
            except ModelStructuredOutputError as exc:
                # 整批级违约（结构层重试也没吐出合法形状）：算作本轮全部违约，进入下一轮重问；
                # 轮次耗尽后在下面当新建——不让一轮归约失败。
                signals.append(f"kind_validation_rejected {sorted(r.name for r in remaining)}: {exc}")
                continue
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
        signals: list[str],
    ) -> tuple[BehaviorKindRegistry, BehaviorKindVectorIndex | None]:
        """把判定落进词表：先落 null/候选命中的，再落指向同批名字的（目标已有 token）。

        **同 label 即同类**：模型对一个名字给出 label「交谈」，就是它的语义结论——词表里已有 label
        为「交谈」的 kind（或同批里刚建的）就归入它，不另建。这不是代码替模型判语义，是对模型自己
        结论的一致性执行；否则「与Tasha交谈」「与Lucia交谈」会各自成 kind 而 label 全是「交谈」
        （DAY1 v18 实测）。容量撞顶只降级撞顶的那个名字（原始名作 token、留信号），同批其它判定照落。
        """

        by_name = {request.name: request for request in chunk}
        settled: dict[str, str] = {}
        registry_box = [registry]

        def settle(name: str) -> str:
            if name in settled:
                return settled[name]
            request = by_name[name]
            target = matches[name]
            if target is None:
                token = self._create_or_join(registry_box, request, labels[name], created, signals)
            elif target in by_name:
                token = settle(target)
                registry_box[0] = self._alias_or_keep(registry_box[0], token, request, signals)
            else:
                token = target
                registry_box[0] = self._alias_or_keep(registry_box[0], token, request, signals)
            settled[name] = token
            tokens[name] = token
            return token

        for request in chunk:
            settle(request.name)
        registry = registry_box[0]
        if vectors is not None:
            fresh = {name: query_vectors[name] for name in created if name in query_vectors and not vectors.has(name)}
            if fresh:
                vectors = vectors.with_vectors(fresh)
        return registry, vectors

    def _create_or_join(
        self,
        registry_box: list[BehaviorKindRegistry],
        request: BehaviorKindRequest,
        label: str | None,
        created: list[str],
        signals: list[str],
    ) -> str:
        registry = registry_box[0]
        if label is not None:
            owner = registry.token_for_label(label)
            if owner is not None:
                # 模型说"这是一次 X"，而词表里已有 X：归入，不另建。
                registry_box[0] = self._alias_or_keep(registry, owner, request, signals)
                return owner
        if registry.kind_count >= self.config.max_kinds:
            signals.append(f"kind_registry_full {request.name!r} kept as its own token")
            return request.name
        registry_box[0] = registry.with_new_kind(request.name, label=label)
        for day in request.days:
            registry_box[0] = registry_box[0].with_hit(request.name, day)
        created.append(request.name)
        return request.name

    def _alias_or_keep(
        self,
        registry: BehaviorKindRegistry,
        token: str,
        request: BehaviorKindRequest,
        signals: list[str],
    ) -> BehaviorKindRegistry:
        """把名字并入 token 作别名并记命中；token 已不在词表（同批撞顶降级）或别名撞顶时只记信号。"""

        if registry.token_for(token) is None:
            signals.append(f"kind_alias_unrecorded {request.name!r} -> {token!r} (token not registered)")
            return registry
        if len(registry.aliases_of(token)) >= self.config.max_aliases_per_kind:
            signals.append(f"kind_alias_capacity_full {request.name!r} -> {token!r} kept unrecorded")
            return registry
        registry = registry.with_alias(token, request.name)
        for day in request.days:
            registry = registry.with_hit(token, day)
        return registry

    def _create(
        self, registry: BehaviorKindRegistry, request: BehaviorKindRequest, *, label: str | None
    ) -> BehaviorKindRegistry:
        if registry.kind_count >= self.config.max_kinds:
            raise BehaviorKindLimitError("behavior kind registry has no remaining kind capacity")
        registry = registry.with_new_kind(request.name, label=label)
        for day in request.days:
            registry = registry.with_hit(request.name, day)
        return registry

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
    "BehaviorKindResolver",
    "kind_match_schema",
]
