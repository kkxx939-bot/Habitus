"""把一条行为投影成上下文视图（读时计算，不落盘），以及候选行为在钟面邻域内的历史视图列表。

全部机械：所在的事来自情景文档的成员表；此前步骤是同一件事里更早的必要/顺带成员；前提两路
同收（事的 needs 目标、事里更早成员留下的待用前提）；起因是事的 results_from 与行为树自带的
results_from；上一次是回看窗内同 kind 的上一条；同时在做与和谁来自行为树。角色为 irrelevant 的
成员只保留"所在的事"（"发生在这件事期间"是事实），不继承事的前提与起因（模型自己说它与事无关）。
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from habitus.behavior.uri import BehaviorURI
from habitus.scene.model import SceneLinkType, SceneRole
from habitus.scene.uri import SceneURI
from habitus.scene.views.index import DayIndex, DayIndexCache
from habitus.scene.views.model import ActionRef, ContextView, LastTime, Precondition, SceneRef


def context_view(occurrence_uri: str, cache: DayIndexCache, *, window_days: int) -> ContextView:
    """一条已落树行为的上下文；``window_days`` 是"上一次"的回看边界。"""

    parsed = BehaviorURI.parse(occurrence_uri)
    address = parsed.to_address()
    index = cache.day(address.occurred_on)
    uri = str(parsed)
    if uri not in index.occurrences:
        raise KeyError(f"occurrence is not on the behaviour tree (or is a disambiguated duplicate): {uri}")
    return _project(uri, index, cache, window_days=window_days)


def history_contexts(
    kind_token: str,
    cache: DayIndexCache,
    *,
    days: tuple[date, ...],
    window_days: int,
    slot_minutes: int | None = None,
    slot_index: int | None = None,
    slot_half_width: int = 0,
) -> tuple[ContextView, ...]:
    """候选 kind 在给定日子里的历史视图，按时间升序。只取情景树覆盖过的日子（没覆盖的那天不是历史
    视图，是"语义层还没处理"）。给了 ``slot_minutes`` 与 ``slot_index`` 就只取钟面邻域内的：槽的口径与
    预测树同一公式（``minute_of_day // slot_minutes``，环形距离 ≤ ``slot_half_width``），由调用方把
    预测树的配置值原样传进来，两侧不会各算各的。"""

    if not isinstance(kind_token, str) or not kind_token:
        raise ValueError("kind_token must be non-empty text")
    if (slot_minutes is None) != (slot_index is None):
        raise ValueError("slot_minutes and slot_index must be given together")
    if slot_minutes is not None and (isinstance(slot_minutes, bool) or not isinstance(slot_minutes, int) or not 1 <= slot_minutes <= 1440):
        raise ValueError("slot_minutes must be an integer between 1 and 1440")
    if isinstance(slot_half_width, bool) or not isinstance(slot_half_width, int) or slot_half_width < 0:
        raise ValueError("slot_half_width must be a non-negative integer")
    views: list[ContextView] = []
    for day in sorted(set(days)):
        index = cache.day(day)
        if not index.covered:
            continue
        for uri in index.ordered:
            document = index.occurrences[uri]
            if str(document.fields["kind_token"]) != kind_token:
                continue
            if slot_minutes is not None and slot_index is not None:
                started = document.address.started_at
                slots_per_day = -(-1440 // slot_minutes)
                own = (started.hour * 60 + started.minute) // slot_minutes
                distance = abs(own - slot_index)
                if min(distance, slots_per_day - distance) > slot_half_width:
                    continue
            views.append(_project(uri, index, cache, window_days=window_days))
    return tuple(views)


def _project(uri: str, index: DayIndex, cache: DayIndexCache, *, window_days: int) -> ContextView:
    document = index.occurrences[uri]
    fields = document.fields
    kind_token = str(fields["kind_token"])
    started = document.address.started_at
    scene_uri = index.scene_of.get(uri)
    scene: SceneRef | None = None
    role: str | None = None
    prior: list[ActionRef] = []
    preconditions: list[Precondition] = []
    causes: list[ActionRef | SceneRef] = []
    if scene_uri is not None:
        scene_document = index.scenes[scene_uri]
        scene = SceneRef(uri=scene_uri, label=scene_document.address.label)
        members = list(scene_document.fields["members"])
        role = next((str(member["role"]) for member in members if member["uri"] == uri), None)
    if scene_uri is not None and role != SceneRole.IRRELEVANT.value:
        scene_document = index.scenes[scene_uri]
        pending_by_producer: dict[str, list[str]] = {}
        for item in scene_document.fields["pending_effects"]:
            pending_by_producer.setdefault(str(item["producer_uri"]), []).append(str(item["text"]))
        for member in scene_document.fields["members"]:
            member_uri = str(member["uri"])
            if member_uri == uri or member_uri not in index.occurrences:
                continue
            if index.occurrences[member_uri].address.started_at.astimezone(UTC) >= started.astimezone(UTC):
                continue
            if member["role"] in (SceneRole.ESSENTIAL.value, SceneRole.OPTIONAL.value):
                prior.append(index.ref(member_uri))
            for text in pending_by_producer.get(member_uri, ()):
                preconditions.append(Precondition("pending", text, member_uri, (index.ref(member_uri).kind_token,)))
        for link in scene_document.links:
            target = resolve_target(str(link.to_uri), cache)
            if target is None:
                continue
            if link.link_type is SceneLinkType.NEEDS:
                preconditions.append(_needs_precondition(target))
            elif link.link_type is SceneLinkType.RESULTS_FROM:
                causes.append(target)
    for cause_uri in index.results_from_targets.get(uri, ()):
        resolved = resolve_target(cause_uri, cache)
        if isinstance(resolved, ActionRef):
            causes.append(resolved)
    return ContextView(
        kind_token=kind_token,
        at=started,
        occurrence_uri=uri,
        name=str(fields["name"]),
        covered=index.covered,
        scene=scene,
        role=role,
        prior_steps=tuple(prior),
        preconditions=tuple(preconditions),
        causes=tuple(causes),
        last_time=last_time(kind_token, before=started, cache=cache, window_days=window_days),
        concurrent=concurrent_refs(uri, index, cache),
        subjects=index.others(uri),
        summary=str(fields["summary"]),
    )


def concurrent_refs(uri: str, index: DayIndex, cache: DayIndexCache) -> tuple[ActionRef, ...]:
    """concurrent_with 语义对称、存储单向（后封口指向先封口）：自己指出去的 + 同日与次日指回来的。"""

    found: dict[str, ActionRef] = {}
    for target in index.concurrent_targets.get(uri, ()):
        resolved = resolve_target(target, cache)
        if isinstance(resolved, ActionRef):
            found[resolved.uri] = resolved
    for other_index in (index, cache.day(index.day + timedelta(days=1))):
        for other, targets in other_index.concurrent_targets.items():
            if uri in targets and other != uri:
                found[other] = other_index.ref(other)
    return tuple(found[key] for key in sorted(found))


def last_time(kind_token: str, *, before: datetime, cache: DayIndexCache, window_days: int) -> LastTime | None:
    """``before`` 之前、回看窗内同 kind 的上一条：距今天数、当时所在的事。"""

    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days <= 0:
        raise ValueError("window_days must be a positive integer")
    today = before.date()
    for offset in range(window_days + 1):
        day = today - timedelta(days=offset)
        index = cache.day(day)
        for uri in reversed(index.ordered):
            document = index.occurrences[uri]
            if str(document.fields["kind_token"]) != kind_token:
                continue
            if document.address.started_at.astimezone(UTC) >= before.astimezone(UTC):
                continue
            scene_uri = index.scene_of.get(uri)
            scene = None if scene_uri is None else SceneRef(uri=scene_uri, label=index.scenes[scene_uri].address.label)
            return LastTime(uri=uri, days_ago=(today - day).days, scene=scene)
    return None


def _needs_precondition(target: ActionRef | SceneRef) -> Precondition:
    if isinstance(target, ActionRef):
        return Precondition("needs", target.name, target.uri, (target.kind_token,))
    return Precondition("needs", target.label, target.uri, target.kinds)


def resolve_target(target_uri: str, cache: DayIndexCache) -> ActionRef | SceneRef | None:
    """边的目标：行为（当前树上）或情景（当前一代，落到 kind 上）；解析不到即悬空，少说一点。"""

    if target_uri.startswith("scene://"):
        day = SceneURI.parse(target_uri).to_address().occurred_on
        index = cache.day(day)
        scene = index.scenes.get(target_uri)
        if scene is None:
            return None
        producers = [str(item["producer_uri"]) for item in scene.fields["pending_effects"]]
        kinds = tuple(dict.fromkeys(index.ref(uri).kind_token for uri in producers if uri in index.occurrences))
        return SceneRef(uri=target_uri, label=scene.address.label, kinds=kinds)
    day = BehaviorURI.parse(target_uri).to_address().occurred_on
    index = cache.day(day)
    return index.ref(target_uri) if target_uri in index.occurrences else None


__all__ = ["concurrent_refs", "context_view", "history_contexts", "last_time", "resolve_target"]
