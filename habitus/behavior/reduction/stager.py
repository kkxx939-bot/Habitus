"""stage：本轮全部文档在此物化定格——命名与撞车消歧、空白段合并、干跑校验、检查点字节上界缩批。

全部易漂移输入（归类结果、是否被提醒过、时间戳）都在这里定成最终 payload；之后 runner 只做确定性落盘。
归类本身（写入层唯一的 LLM 触点）由 runner 先算好交进来。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from habitus.behavior.document import BehaviorDocumentMetadata
from habitus.behavior.document.config import BehaviorDocumentLimitError
from habitus.behavior.kinds.calls import CallFailure
from habitus.behavior.model import BehaviorAddress, BehaviorKind
from habitus.behavior.observation import BehaviorObservation, BehaviorObservationStore
from habitus.behavior.reduction.checkpoint import CheckpointStore, document_days
from habitus.behavior.reduction.errors import BehaviorReductionError
from habitus.behavior.reduction.kinds_step import SET_ASIDE, KindStamps, judgement_content
from habitus.behavior.reduction.ledger import BehaviorReductionEntry
from habitus.behavior.reduction.payloads import (
    REDUCTION_VERSION,
    gap_kind_of,
    gap_payload,
    occurrence_address,
    occurrence_payload,
)
from habitus.behavior.reduction.pending import PendingJudgements
from habitus.behavior.reduction.record import ReducibleJudgement
from habitus.behavior.retry import RetryCause
from habitus.behavior.schema.model import BehaviorSchemaError
from habitus.behavior.tree import BehaviorTree, BehaviorTreeIntegrityError
from habitus.behavior.uri import BehaviorURI
from habitus.foundation.integrity import canonical_digest
from habitus.infrastructure.store.contracts.path_lock import LeaseGuard

#: stage 里出了错误的一条：判断、原因、进夜里待处理堆的原因（定夜里试不试，裁定 94 错误分类）。
Failed = tuple[ReducibleJudgement, str, RetryCause]
_TRUNCATED_HEAD = 20
_TRUNCATED_TAIL = 5


@dataclass(frozen=True)
class StagedBatch:
    """缩批收敛后的一批：检查点内容与编码、说明、出了错误的判断，以及最终留下的链与空白段。"""

    documents: list[dict[str, Any]]
    dropped: tuple[str, ...]
    failed: tuple[Failed, ...]
    checkpoint: dict[str, Any]
    encoded: bytes
    active_indexes: tuple[int, ...]
    active_gaps: tuple[ReducibleJudgement, ...]
    shrink_rounds: int


@dataclass
class _Naming:
    """命名阶段的产物：地址、payload、跳过的链、出了错误的判断。"""

    uris: dict[int, str]
    payloads: dict[int, dict[str, Any]]
    skipped: set[int]
    failed: list[Failed]


class ReductionStager:
    """物化本轮文档；``reminded`` 与 ``signals`` 每轮由 runner 交进来（查法会在装配后接上，信号按轮收集）。"""

    def __init__(
        self, *, tree: BehaviorTree, observations: BehaviorObservationStore, checkpoints: CheckpointStore
    ) -> None:
        self.tree = tree
        self.observations = observations
        self.checkpoints = checkpoints

    async def stage_bounded(
        self,
        pending: PendingJudgements,
        ready: tuple[tuple[int, ...], tuple[ReducibleJudgement, ...]],
        now: datetime,
        stamps: KindStamps,
        *,
        entries: Sequence[BehaviorReductionEntry],
        refresh_days: set[date],
        reminded: Callable[[str, datetime], bool],
        signals: list[str],
        guard: LeaseGuard,
    ) -> StagedBatch:
        """检查点字节上界的确定性缩批：超限时按封口顺序留前一半重来（留下的下一轮自然处理）。"""

        shrink_rounds = 0
        active_indexes, active_gaps = ready
        while True:
            documents, dropped, failed = await self.stage(
                pending,
                active_indexes,
                active_gaps,
                now,
                stamps,
                entries=entries,
                reminded=reminded,
                signals=signals,
                guard=guard,
            )
            checkpoint = {
                "reduction_version": REDUCTION_VERSION,
                "staged_at": now.isoformat(timespec="microseconds"),
                "refresh_days": sorted(day.isoformat() for day in (refresh_days | document_days(documents))),
                "documents": documents,
            }
            encoded = self.checkpoints.encode(checkpoint)
            # 写入与重放共用同一上界：写得进读不回的检查点 = 永久卡死（对齐融合层
            # "作业放得下而回执放不下"的既有教训）。超限发生在 stage 前，缩批可自愈。
            if self.checkpoints.fits(encoded):
                return StagedBatch(
                    documents, dropped, failed, checkpoint, encoded, active_indexes, active_gaps, shrink_rounds
                )
            shrink_rounds += 1
            total = len(active_indexes) + len(active_gaps)
            if total <= 1:
                raise BehaviorReductionError("a single reduction document exceeds the checkpoint byte bound")
            dropped_count = total - max(total // 2, 1)
            if len(active_gaps) >= dropped_count:
                active_gaps = active_gaps[: len(active_gaps) - dropped_count]
            else:
                keep_chains = len(active_indexes) - (dropped_count - len(active_gaps))
                active_gaps = ()
                active_indexes = active_indexes[: max(keep_chains, 0)]

    async def stage(
        self,
        pending: PendingJudgements,
        ready_indexes: tuple[int, ...],
        ready_gaps: tuple[ReducibleJudgement, ...],
        now: datetime,
        stamps: KindStamps,
        *,
        entries: Sequence[BehaviorReductionEntry],
        reminded: Callable[[str, datetime], bool],
        signals: list[str],
        guard: LeaseGuard | None = None,
    ) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[Failed, ...]]:
        """物化本轮全部文档；返回 (检查点内容, 说明, 出了错误的判断、原因与进堆的原因)。检查点内容 JSON 可序列化、逐字节确定。"""

        staged_at = now.isoformat(timespec="microseconds")
        dropped: list[str] = []
        ready = [(index, pending.occurrences[index]) for index in ready_indexes]

        ledger_uris = frozenset(entry.uri for entry in entries)
        # 占用集合 = 账本地址（可能已按窗口过期）∪ **树上已有地址** ∪ 本批。树是最终真相；但"树上有没有"
        # 必须按天整体问一次——逐条 ``tree.exists`` 每次都重新枚举整个日目录（周尺度实测 9,270 次
        # ≈ 133 秒，而整天列举全树只要 0.33 秒），那是周尺度踩满 sweep 锁 TTL 的主因之一。
        occupied = set(ledger_uris) | self._addresses_on_tree(
            {record.started_at.date() for _, record in ready} | {record.started_at.date() for record in ready_gaps}
        )

        # 归类结果由调用方先算好（按节点身份给，不依赖地址），这里只取用。
        sources = {ref for _, record in ready for ref in record.source_refs}
        observation_index = self._observation_index(sources, signals) if sources else {}
        naming = _name_occurrences(
            ready, stamps, occupied, observation_index, reminded=reminded, dropped=dropped, guard=guard
        )

        documents: list[dict[str, Any]] = _gap_documents(ready_gaps, occupied, staged_at, dropped)
        documents += _occurrence_documents(ready, naming, stamps, staged_at, guard)
        documents, rejected = _publishable(self.tree, documents, now, dropped, guard)
        # 干跑校验不过的行为：同一份判断再归约也一样，走我们自己处理不了的那条路（换成「错误」、进堆不白试，那天等它落成空白
        # 才定稿），不留在判断存储里每轮重新归类
        by_id = {record.judgement_id: record for _index, record in ready}
        failed = naming.failed + [
            (by_id[judgement_id], reason, RetryCause.DETERMINISTIC)
            for judgement_id, reason in rejected
            if judgement_id in by_id
        ]
        return documents, tuple(dropped), tuple(failed)

    def _addresses_on_tree(self, days: set[date]) -> set[str]:
        """本批涉及的那些天，树上已有的全部 occurrence 与 gap 地址（每天每类只枚举一次目录）。"""

        found: set[str] = set()
        for day in sorted(days):
            for kind in (BehaviorKind.OCCURRENCE, BehaviorKind.GAP):
                for address in self.tree.list_day_addresses(kind, day):
                    found.add(str(BehaviorURI.from_address(address)))
        return found

    def _observation_index(self, source_refs: Iterable[str], signals: list[str]) -> dict[str, BehaviorObservation]:
        index: dict[str, BehaviorObservation] = {}
        for source_ref in sorted(set(source_refs)):
            envelope = self.observations.read(source_ref)
            if envelope is None:
                # 交付已不在存储里（运维删除、隔离窗口）：不整轮失败。带 basis 的链在物化时会因
                # "步骤引用的观测已不存在"按链跳过；**没有 basis 的链照常发布**（判断本体自带
                # summary/goal，不依赖观测），所以这里留信号是唯一的可见痕迹（审计 NEW-8）。
                signals.append(f"source_delivery_missing {source_ref}")
                continue
            for observation in envelope.batch.observations:
                index.setdefault(observation.observation_id, observation)
        return index


def _name_occurrences(
    ready: list[tuple[int, ReducibleJudgement]],
    stamps: KindStamps,
    occupied: set[str],
    observations: dict[str, BehaviorObservation],
    *,
    reminded: Callable[[str, datetime], bool],
    dropped: list[str],
    guard: LeaseGuard | None,
) -> _Naming:
    """撞车消歧（死规则②）：按 (evidence_ready_at, 身份) 定序，后序加确定性序号后缀，原始名照存语义面。

    **先物化再占位**：payload 物化失败（坏名字、basis 引用已丢观测）的只跳过、不占地址——否则它占了基名，幸存者却带着
    -2/original_name 落盘，被预测源当"已知重复"整条丢掉。占用集合 = 账本地址 ∪ 树上已存在的地址（账本按窗口过期，树不会）∪ 本批。
    """

    naming = _Naming({}, {}, set(), [])
    for position, (index, record) in enumerate(
        sorted(ready, key=lambda pair: (pair[1].evidence_ready_at, pair[1].judgement_id))
    ):
        if guard is not None and position % 50 == 0:
            # 周尺度实测：9,270 条的命名+物化是一个单阶段长循环，只在阶段边界续约会踩满
            # 600s 租约 TTL（另一条线的冻结周重放实测）。长循环内按条数周期性续约。
            guard.checkpoint()
        token = stamps.tokens.get(record.chain_digest)
        if token is None:
            naming.skipped.add(index)
            _set_aside(record, stamps, naming.failed, dropped)
            continue
        was_reminded = bool(reminded(token, record.started_at))
        try:
            uri, payload = _materialize(record, token, occupied, observations, reminded=was_reminded)
        except BehaviorTreeIntegrityError:
            raise  # 树本身不可安全检视（别名冲突、符号链接）：整轮失败、可见告警，不当坏名字吞
        except (TypeError, ValueError, BehaviorReductionError) as exc:
            naming.skipped.add(index)
            naming.failed.append((record, str(exc), RetryCause.DETERMINISTIC))
            dropped.append(f"chain {record.chain_digest} isolated: {exc}")
            continue
        occupied.add(uri)
        naming.uris[index] = uri
        naming.payloads[index] = payload
    return naming


def _set_aside(record: ReducibleJudgement, stamps: KindStamps, failed: list[Failed], dropped: list[str]) -> None:
    """归类没给出 token 的链：换成「错误」、进夜里待处理的堆。"""

    if judgement_content(record) is None:
        # 没有可给归类看的内容：我们自己处理不了，换成「错误」、进夜里待处理的堆但不白试（2.4.2；裁定 67、94）
        failed.append((record, "behavior name is not addressable", RetryCause.DETERMINISTIC))
        dropped.append(f"chain {record.chain_digest} isolated: behavior name is not addressable")
    else:
        # 有内容、归类这一轮没给出结论：不再"下轮再归"拖住那天——换成「错误」、放一边（裁定 94 第 3 条）。
        # 进堆的原因按错误分类定（``kinds_step.SET_ASIDE``）：模型整个用不了的夜里放回去再归一次，内容安全与
        # 输入超长不重试，bug 当场报
        failure = stamps.set_aside.get(record.chain_digest, CallFailure.BUG)
        failed.append((record, f"classification {failure.value}", SET_ASIDE[failure]))
        dropped.append(f"chain {record.chain_digest} set aside: classification {failure.value}")


def _materialize(
    record: ReducibleJudgement,
    token: str,
    occupied: set[str],
    observations: dict[str, BehaviorObservation],
    *,
    reminded: bool,
) -> tuple[str, dict[str, Any]]:
    """先以基名物化一次（坏名字、死引用在这里抛），再挑第一个没被占的后缀地址物化最终 payload。"""

    base = str(record.behavior)
    # TODO(BHV-PROMPTED-BY-001)：谁发起的（裁定 54）——宿主助手先发了通知（包裹到了、账单到期）他才去做的，要由融合从会话里
    # 认出来交到这里（``occurrence_payload(prompted_by="host_agent")``）；事件融合按 lane 重做之前不改上游，这里只传系统提醒
    # （reminded → system），其余不写这个字段。详见 ``待办.md``「行为树"谁发起的"标识由上游产出」。
    occurrence_payload(
        record,
        name=base,
        original_name=None,
        kind_token=token,
        observations=observations,
        reminded=reminded,
    )
    candidate, ordinal = base, 1
    while str(BehaviorURI.from_address(occurrence_address(record, candidate))) in occupied:
        ordinal += 1
        candidate = f"{base}-{ordinal}"
    uri = str(BehaviorURI.from_address(occurrence_address(record, candidate)))
    payload = occurrence_payload(
        record,
        name=candidate,
        original_name=base if ordinal > 1 else None,
        kind_token=token,
        observations=observations,
        reminded=reminded,
    )
    return uri, payload


def _gap_documents(
    ready_gaps: tuple[ReducibleJudgement, ...], occupied: set[str], staged_at: str, dropped: list[str]
) -> list[dict[str, Any]]:
    """gap 先落。gap 地址的叶名是枚举类型、无法加后缀：批内**同一微秒同偏移**起点的空白机械合并成
    一个节点（溯源并集、终点取最大）；撞上账本里既有空白时，本段记账指向既有节点——
    空白事实已在树上有代表，不重复、不丢账、不卡队列。"""

    documents: list[dict[str, Any]] = []
    merged_gaps: dict[str, list[ReducibleJudgement]] = {}
    for record in ready_gaps:
        address = BehaviorAddress.gap(record.started_at.date(), gap_kind_of(record), record.started_at)
        merged_gaps.setdefault(str(BehaviorURI.from_address(address)), []).append(record)
    for uri in sorted(merged_gaps):
        group = merged_gaps[uri]
        judgement_ids = sorted(item.judgement_id for item in group)
        entry = {
            "chain_digest": canonical_digest({"judgement_ids": judgement_ids}),
            "kind": BehaviorKind.GAP.value,
            "uri": uri,
            "judgement_ids": judgement_ids,
            "staged_at": staged_at,
            "reduction_version": REDUCTION_VERSION,
        }
        if uri in occupied:
            dropped.append(f"gap at {uri} already published; consuming {judgement_ids} by reference")
            documents.append({"kind": "ledger-only", "payload": None, "ledger": entry})
            continue
        payload = _merged_gap_payload(group, chain_digest=str(entry["chain_digest"]))
        documents.append({"kind": BehaviorKind.GAP.value, "payload": payload, "ledger": entry})
    return documents


def _occurrence_documents(
    ready: list[tuple[int, ReducibleJudgement]],
    naming: _Naming,
    stamps: KindStamps,
    staged_at: str,
    guard: LeaseGuard | None,
) -> list[dict[str, Any]]:
    """occurrence 按 order_key 升序落盘。"""

    documents: list[dict[str, Any]] = []
    for position, (index, record) in enumerate(sorted(ready, key=lambda pair: pair[1].order_key)):
        if guard is not None and position % 50 == 0:
            guard.checkpoint()
        if index in naming.skipped:
            continue
        entry = {
            "chain_digest": record.chain_digest,
            "kind": BehaviorKind.OCCURRENCE.value,
            "uri": naming.uris[index],
            "judgement_ids": [record.judgement_id],
            "staged_at": staged_at,
            "reduction_version": REDUCTION_VERSION,
        }
        item: dict[str, Any] = {
            "kind": BehaviorKind.OCCURRENCE.value,
            "payload": naming.payloads[index],
            "ledger": entry,
        }
        note = stamps.pending.get(record.chain_digest)
        if note is not None:
            # 待定条目发布后进待定池要用这两样；随文档一起定格，重放时原样再用一次（按地址幂等）。
            item["kind_pending"] = note.to_mapping()
        documents.append(item)
    return documents


def _merged_gap_payload(group: list[ReducibleJudgement], *, chain_digest: str) -> dict[str, Any]:
    """同地址空白段的机械合并：起点同刻（合并前提）、终点取最大；溯源只留账本条目身份。"""

    first = min(group, key=lambda item: item.judgement_id)
    payload = gap_payload(first, chain_digest=chain_digest)
    if len(group) > 1:
        ended = max((item.last_observed_at for item in group), key=_as_instant)
        payload["ended_at"] = ended.isoformat(timespec="microseconds")
    return payload


def _publishable(
    tree: BehaviorTree,
    documents: list[dict[str, Any]],
    now: datetime,
    dropped: list[str],
    guard: LeaseGuard | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """stage 末端的干跑校验：落盘期只允许**已验证会成功**的确定性动作。

    任何会在 publish 期抛错的文档（schema 校验），若放进检查点就会变成"检查点已写、重放永败"的
    永久卡死；在这里按文档剔除并留信号，其余照常落。返回（留下的文档，剔掉的行为文档消费的判断与原因）。
    """

    metadata = BehaviorDocumentMetadata.initial(now)

    def dry_run(kind: str, payload: Mapping[str, Any]) -> None:
        # 与写盘同一道检查：schema、正文字数、编码后字节（只 build 不 encode 的话，字节超限要到发布时才抛）
        tree.storable_bytes(tree.document_codec.build(BehaviorKind(kind), payload, metadata=metadata))

    valid: list[dict[str, Any]] = []
    rejected: list[tuple[str, str]] = []
    for position, item in enumerate(documents):
        if guard is not None and position % 50 == 0:
            guard.checkpoint()
        if item["kind"] == "ledger-only":
            valid.append(item)
            continue
        uri = str(item["ledger"]["uri"])
        try:
            try:
                dry_run(item["kind"], item["payload"])
            except BehaviorDocumentLimitError:
                # 最后一道保险：去掉死引用之后仍超限（要两千步以上的链）——截断 basis 中段、
                # 留信号继续发布，不让一个文档打掉整轮。goal/summary/起止/状态不动，
                # 它们才是下游真正读的东西；丢的只是步骤明细的中段，首尾与总步数保留。
                truncated = _truncate_basis(item["payload"])
                if truncated is None:
                    raise
                dropped.append(
                    f"document {uri} truncated: basis {len(item['payload']['basis'])} steps "
                    f"-> {len(truncated['basis'])} (document exceeded its byte bound)"
                )
                item["payload"] = truncated
                dry_run(item["kind"], truncated)
        except (
            BehaviorReductionError,
            BehaviorSchemaError,
            BehaviorDocumentLimitError,
            TypeError,
            ValueError,
        ) as exc:
            dropped.append(f"document {uri} skipped: fails validation before stage ({exc})")
            if item["kind"] == BehaviorKind.OCCURRENCE.value:
                reason = f"document fails validation before stage ({exc})"
                rejected += [(str(judgement_id), reason) for judgement_id in item["ledger"]["judgement_ids"]]
            continue
        valid.append(item)
    return valid, rejected


def _as_instant(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _truncate_basis(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """保留 basis 首尾各若干步；已经不可再截时返回 None。"""

    basis = list(payload.get("basis", ()))
    if len(basis) <= _TRUNCATED_HEAD + _TRUNCATED_TAIL:
        return None
    trimmed = dict(payload)
    trimmed["basis"] = basis[:_TRUNCATED_HEAD] + basis[-_TRUNCATED_TAIL:]
    return trimmed
