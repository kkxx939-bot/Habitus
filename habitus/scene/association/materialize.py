"""把草稿变成可落盘的记录：编号在这里还原成 URI，两类边在这里建出来。

装配层只判断"这个编号在输入里有没有、是不是更早"，**换成 URI 是这一层的事**。还原的唯一入口是
``AssociationInput``：当天的流用 ``row(no)``，前因候选先经 ``cause_no`` 回到它自己的编号。

两类边都复用 ``SceneStoredLink``（方向固定晚指早、``lag_seconds`` 必须等于两端时刻差）：
``causes`` → ``results_from`` 指向那条行为，``consumed`` → ``needs`` 指向建立那条前提的行为。
边建不出来就丢那一条并留信号——权威在字段上，边只是给读侧走图用的。
"""

from __future__ import annotations

from datetime import datetime

from habitus.scene.association.model import AssociationDraft, AssociationInput
from habitus.scene.model import AssociationAddress, SceneLinkType
from habitus.scene.regularity.document import AssociationDocument
from habitus.scene.regularity.link import SceneStoredLink
from habitus.scene.regularity.overview import Overview
from habitus.scene.uri import SceneURI


def materialize(
    payload: AssociationInput, draft: AssociationDraft, overview: Overview, *, created_at: datetime
) -> tuple[AssociationDocument | None, tuple[str, ...]]:
    """一条草稿 → 一份记录。整条建不出来时返回 None 并说明原因，不抛。"""

    signals: list[str] = []
    row = payload.row(draft.occurrence_no)
    address = AssociationAddress(payload.kind_token, payload.day, row.name, row.started_at)
    from_uri = SceneURI.from_association(address)
    consumed = _consumed(payload, draft, signals)
    links = [*_cause_links(payload, draft, from_uri, signals), *_needs_links(consumed, from_uri, signals)]
    try:
        document = AssociationDocument(
            address=address,
            created_at=created_at,
            occurrence_uri=row.uri,
            context=draft.context,
            situation=_situation(draft, overview, signals),
            left=draft.left,
            consumed=consumed,
            links=tuple(links),
        )
    except (TypeError, ValueError) as exc:
        signals.append(f"record_dropped: #{draft.occurrence_no} cannot be materialized ({exc})")
        return None, tuple(signals)
    return document, tuple(signals)


def _situation(draft: AssociationDraft, overview: Overview, signals: list[str]) -> str | None:
    """编号还原成那一种情形的原话。指向一个不存在的编号只降级，不丢整条——上下文才是主产物。"""

    if draft.new_situation is not None:
        return draft.new_situation
    if draft.situation_no is None:
        return None
    if 1 <= draft.situation_no <= len(overview.situations):
        return overview.situations[draft.situation_no - 1].text
    signals.append(f"situation_degraded: #{draft.occurrence_no} names situation {draft.situation_no}, which is gone")
    return None


def _consumed(payload: AssociationInput, draft: AssociationDraft, signals: list[str]) -> tuple[tuple[str, str], ...]:
    resolved: list[tuple[str, str]] = []
    for number in draft.consumed:
        if not 1 <= number <= len(payload.pending):
            signals.append(f"premise_dropped: #{draft.occurrence_no} consumed P{number}, which is gone")
            continue
        premise = payload.pending[number - 1]
        resolved.append((premise.producer_uri, premise.text))
    return tuple(resolved)


def _cause_links(
    payload: AssociationInput, draft: AssociationDraft, from_uri: SceneURI, signals: list[str]
) -> list[SceneStoredLink]:
    links: list[SceneStoredLink] = []
    for cited in draft.causes:
        offset = payload.cause_no(cited)
        target = payload.causes[offset - 1].uri if offset is not None else payload.row(cited).uri
        link = _link(from_uri, target, SceneLinkType.RESULTS_FROM, draft, signals)
        if link is not None:
            links.append(link)
    return links


def _needs_links(
    consumed: tuple[tuple[str, str], ...], from_uri: SceneURI, signals: list[str]
) -> list[SceneStoredLink]:
    links: list[SceneStoredLink] = []
    seen: set[str] = set()
    for producer_uri, _text in consumed:
        # 同一条行为留下的两条前提都被用掉时，边只需要一条——边的粒度是行为，字段才是前提。
        if producer_uri in seen:
            continue
        seen.add(producer_uri)
        link = _link(from_uri, producer_uri, SceneLinkType.NEEDS, None, signals)
        if link is not None:
            links.append(link)
    return links


def _link(
    from_uri: SceneURI, target: str, kind: SceneLinkType, draft: AssociationDraft | None, signals: list[str]
) -> SceneStoredLink | None:
    try:
        return SceneStoredLink.between(from_uri, target, kind)
    except (TypeError, ValueError) as exc:
        where = "" if draft is None else f"#{draft.occurrence_no} "
        signals.append(f"link_dropped: {where}{kind.value} edge cannot be built ({exc})")
        return None


__all__ = ["materialize"]
