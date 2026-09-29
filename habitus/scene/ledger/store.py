"""``scene/ledger/`` 这一支的存储：按假设分目录，承诺、结算、提醒、回应四支。

::

    ledger/<后件身份>/<假设叶名>/claims/YYYY/MM/DD/<触发 occurrence 叶名>.json
    ledger/<后件身份>/<假设叶名>/settlements/YYYY/MM/DD/<同名>.json
    ledger/<后件身份>/<假设叶名>/interventions/YYYY/MM/DD/<同名>--<提醒时刻>.json
    ledger/<后件身份>/<假设叶名>/responses/YYYY/MM/DD/<同名>--<提醒时刻>--<回应时刻>.json

四支都是 **add-only**：写一次不改（同内容重放是空操作，不同内容拒绝）。承诺没有完成标记——开承诺按
"这条承诺已经在盘上"跳过，重跑同一天是幂等的；机会快照事后树重建了也不改，所以已存在的不比对内容。
日目录按触发 occurrence 的本地日，与 ``occurrences/`` 同一口径。

**作废是按天整批撤**（``retain_only`` / ``void_day``），与 ``occurrences.retain_only`` 对称：改了概念定义重映射某天之后，
那天开的承诺与结算是按旧命中记的，要撤了重开——单条 add-only 不动，撤的是"这一天这条假设的账"。提醒与回应**不撤**：
它们是这个系统里唯一的干预数据，重开的承诺若还是同一条触发就自然接上。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import ClassVar

from habitus.behavior.model import is_ascii_digits
from habitus.infrastructure.store.filesystem import ImmutableArtifactConflictError, atomic_create_bytes
from habitus.scene.concepts.model import ConceptError, concept_identity
from habitus.scene.ledger.codec import (
    LedgerCodecError,
    LedgerSchemaError,
    decode_claim,
    decode_intervention,
    decode_response,
    decode_settlement,
    encode_claim,
    encode_intervention,
    encode_response,
    encode_settlement,
)
from habitus.scene.ledger.model import (
    OUTCOMES_BY_ASPECT,
    Claim,
    ClaimRef,
    Intervention,
    InterventionResponse,
    Settlement,
)
from habitus.scene.storage import SceneStore

LEDGER_SEGMENT = "ledger"
CLAIMS_SEGMENT = "claims"
SETTLEMENTS_SEGMENT = "settlements"
INTERVENTIONS_SEGMENT = "interventions"
RESPONSES_SEGMENT = "responses"
MAX_RECORD_BYTES = 64 * 1024


class LedgerStoreError(ValueError):
    """账本存储的路径逃逸、损坏记录，或对 add-only 记录的改写。"""


class LedgerStore(SceneStore):
    error_type: ClassVar[type[ValueError]] = LedgerStoreError

    def __init__(self, scene_root: str | Path, **options: int) -> None:
        super().__init__(scene_root, **options)
        self.directory = self._inside(Path(LEDGER_SEGMENT))

    # ── 承诺 ────────────────────────────────────────────────────────────────

    def write_claim(self, claim: Claim) -> Path:
        """写一条承诺。盘上已有同地址的**结算**而没有承诺 → 拒："这条承诺一写进去就带着一条按旧命中结出来的结算"，
        ``open_claims`` 会把它当已结算、再也不碰（评审 B1）。撤干净（``retain_only``）之后再重开。"""

        if not isinstance(claim, Claim):
            raise TypeError("claim must be a Claim")
        path = self.claim_path(claim.ref)
        if not self._file_exists(path) and self.settlement_exists(claim.ref):
            raise LedgerStoreError(
                "a settlement already exists at this address without its claim (a voiding was interrupted); clear it before reopening the claim"
            )
        self._create(path, encode_claim(claim))
        return path

    def claim_exists(self, ref: ClaimRef) -> bool:
        return self._file_exists(self.claim_path(ref))

    def read_claim(self, ref: ClaimRef) -> Claim:
        payload = self._read_bytes(self.claim_path(ref), MAX_RECORD_BYTES)
        if payload is None:
            raise LedgerStoreError(f"claim {ref} does not exist")
        try:
            return decode_claim(payload, expected=ref)
        except LedgerSchemaError:
            raise  # 旧口径不是损坏：原样往上抛，让重算路径认得出（B9）
        except LedgerCodecError as exc:
            raise LedgerStoreError(f"claim {ref.leaf!r} is corrupt: {exc}") from exc

    def claims_for(self, hypothesis_identity: str) -> tuple[Claim, ...]:
        """这条假设的全部承诺，按锚点时刻升序。"""

        claims = [self.read_claim(ref) for ref in self._refs(hypothesis_identity, CLAIMS_SEGMENT)]
        return tuple(sorted(claims, key=lambda item: (item.anchor, item.trigger_uri)))

    def open_claims(self, hypothesis_identity: str) -> tuple[Claim, ...]:
        """还没结算的承诺（就是"还立着的前提"），按锚点升序——无节律型先开先结靠这个序。

        **只解码开着的那些**：开没开按文件名比 ``claims/`` 与 ``settlements/`` 两棵日目录的差集就知道，不用把整本账
        解码一遍再逐条 stat（一年 40,440 条承诺时那是 4 万次读 + 4 万次 stat，实测 0.25 s/条假设 × 320 条）。
        """

        settled = {(ref.day, ref.leaf) for ref in self._refs(hypothesis_identity, SETTLEMENTS_SEGMENT)}
        claims = [
            self.read_claim(ref) for ref in self._refs(hypothesis_identity, CLAIMS_SEGMENT) if (ref.day, ref.leaf) not in settled
        ]
        return tuple(sorted(claims, key=lambda item: (item.anchor, item.trigger_uri)))

    # ── 结算 ────────────────────────────────────────────────────────────────

    def write_settlement(self, settlement: Settlement) -> Path:
        """核对承诺在盘上、且这个结果是那条承诺的方面允许的。

        后半条只有这一层做得到：``Settlement`` 自己不知道 aspect。不核的话，一条写给概率承诺的 COUNTED
        会被收下，然后在读侧既不算验证/落空也不算删失——那次机会静默蒸发。反正 ``claim_exists`` 已经走了
        一趟文件系统，这里顺手把承诺读出来。
        """

        if not isinstance(settlement, Settlement):
            raise TypeError("settlement must be a Settlement")
        if not self.claim_exists(settlement.ref):
            raise LedgerStoreError("a settlement needs its claim on disk first")
        claim = self.read_claim(settlement.ref)
        allowed = OUTCOMES_BY_ASPECT[claim.aspect]
        if settlement.outcome not in allowed:
            raise LedgerStoreError(
                f"a {claim.aspect.value} claim cannot settle as {settlement.outcome.value}; "
                f"allowed: {sorted(item.value for item in allowed)}"
            )
        path = self.settlement_path(settlement.ref)
        self._create(path, encode_settlement(settlement))
        return path

    def settlement_exists(self, ref: ClaimRef) -> bool:
        return self._file_exists(self.settlement_path(ref))

    def read_settlement(self, ref: ClaimRef) -> Settlement | None:
        payload = self._read_bytes(self.settlement_path(ref), MAX_RECORD_BYTES)
        if payload is None:
            return None
        try:
            return decode_settlement(payload, expected=ref)
        except LedgerSchemaError:
            raise
        except LedgerCodecError as exc:
            raise LedgerStoreError(f"settlement {ref.leaf!r} is corrupt: {exc}") from exc

    def settlements_for(self, hypothesis_identity: str) -> tuple[Settlement, ...]:
        found = []
        for ref in self._refs(hypothesis_identity, SETTLEMENTS_SEGMENT):
            settlement = self.read_settlement(ref)
            if settlement is not None:
                found.append(settlement)
        return tuple(sorted(found, key=lambda item: (item.ref.day, item.ref.leaf)))

    # ── 作废 ────────────────────────────────────────────────────────────────

    def retain_only(self, hypothesis_identity: str, day: date, keep: frozenset[str]) -> tuple[str, ...]:
        """撤掉这条假设在 ``day`` 开的、叶名不在 ``keep`` 里的承诺连同它们的结算；返回撤掉的叶名。提醒与回应不动。

        **先删结算、再删承诺**：崩在两步之间剩下的是"有承诺没结算"（就是"还开着"，下一夜自愈）；反过来剩下的是
        孤儿结算，而 ``open_claims`` 按 ``settlement_exists`` 判开闭——重开的同触发承诺会被那条按**旧命中**结出来的
        结算静默盖章，永不再结（评审 B1 实测）。
        """

        removed: list[str] = []
        for segment in (SETTLEMENTS_SEGMENT, CLAIMS_SEGMENT):
            directory = self._hypothesis_dir(hypothesis_identity) / segment / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
            records, _noise = self._content_files(directory)
            for entry in records:
                leaf = entry.name.removesuffix(".json")
                if leaf in keep:
                    continue
                self._discard(directory / entry.name)
                if segment == CLAIMS_SEGMENT:
                    removed.append(leaf)
            self._discard_noise(directory)
        return tuple(sorted(removed))

    def orphan_settlements(self, hypothesis_identity: str) -> tuple[ClaimRef, ...]:
        """有结算、没承诺的地址：作废撤到一半崩了留下的毒（旧结算会盖住重开的承诺）。夜批开头查一遍。"""

        claims = {ref for ref in self._refs(hypothesis_identity, CLAIMS_SEGMENT)}
        return tuple(sorted((ref for ref in self._refs(hypothesis_identity, SETTLEMENTS_SEGMENT) if ref not in claims), key=lambda ref: (ref.day, ref.leaf)))

    def void_settlements_touching(self, hypothesis_identity: str, day: date) -> tuple[ClaimRef, ...]:
        """撤掉这条假设里**依据跨过 ``day``** 的结算；**承诺不动**，下一夜按重映射之后的记录重结。返回撤掉的地址。

        重映射一天之后，靠那天的记录结出来的账是脏的：那天可能多出一条后件（原来结成了"没来"），也可能少一条
        （原来结成了"来了"）。结算 add-only 改不了，只能撤。``void_day`` 撤的是**那天开的**承诺与结算，而这里撤的是
        **别的天开、却读了那天记录**的结算——两件事，一个重映射之后都要做。

        依据区间取 ``[锚, 快照最后一个机会结束]``（没有快照 = 锚之后全都算）。比"它到底看了哪几个机会"宽：
        缺席与计数那两种结算没把看过的机会时刻记进去（只记了第几次机会与计数），从盘上推不出来。宽的代价只是
        多重结一次——承诺还在，重结是幂等的。
        """

        voided: list[ClaimRef] = []
        for ref in self._refs(hypothesis_identity, SETTLEMENTS_SEGMENT):
            if not self.claim_exists(ref):
                continue  # 孤儿结算不在这条路上处理（见 ``orphan_settlements``）
            claim = self.read_claim(ref)
            if day < claim.anchor.date():
                continue
            if claim.control is not None and day > claim.control.last_end.date():
                continue
            self._discard(self.settlement_path(ref))
            voided.append(ref)
        return tuple(sorted(voided, key=lambda item: (item.day, item.leaf)))

    def void_day(self, hypothesis_identity: str, day: date) -> tuple[str, ...]:
        """撤掉这条假设在 ``day`` 的全部承诺与结算（重映射那一天之后调）。"""

        return self.retain_only(hypothesis_identity, day, frozenset())

    # ── 提醒与回应 ──────────────────────────────────────────────────────────

    def record_intervention(self, intervention: Intervention) -> Path:
        """跨层写入口：预测层决定发的提醒经组合根写进来。承诺必须已在盘上。"""

        if not isinstance(intervention, Intervention):
            raise TypeError("intervention must be an Intervention")
        if not self.claim_exists(intervention.ref):
            raise LedgerStoreError("an intervention needs its claim on disk first")
        path = self._day_dir(intervention.ref, INTERVENTIONS_SEGMENT) / f"{intervention.leaf}.json"
        self._create(path, encode_intervention(intervention))
        return path

    def record_response(self, response: InterventionResponse) -> Path:
        """他对某次提醒的回应：那次提醒必须已在盘上。"""

        if not isinstance(response, InterventionResponse):
            raise TypeError("response must be an InterventionResponse")
        reminder = self._day_dir(response.ref, INTERVENTIONS_SEGMENT) / f"{response.intervention_leaf}.json"
        if not self._file_exists(reminder):
            raise LedgerStoreError("a response needs its intervention on disk first")
        path = self._day_dir(response.ref, RESPONSES_SEGMENT) / f"{response.leaf}.json"
        self._create(path, encode_response(response))
        return path

    def interventions_for(self, ref: ClaimRef) -> tuple[Intervention, ...]:
        found = [self._decode(decode_intervention, payload, ref, name) for name, payload in self._prefixed(ref, INTERVENTIONS_SEGMENT)]
        return tuple(sorted(found, key=lambda item: item.reminded_at))

    def responses_for(self, ref: ClaimRef) -> tuple[InterventionResponse, ...]:
        found = [self._decode(decode_response, payload, ref, name) for name, payload in self._prefixed(ref, RESPONSES_SEGMENT)]
        return tuple(sorted(found, key=lambda item: (item.reminded_at, item.responded_at)))

    def _prefixed(self, ref: ClaimRef, segment: str) -> list[tuple[str, bytes]]:
        """``<承诺叶名>--…`` 的记录 (文件名, 字节)；顺手清掉噪音。"""

        directory = self._day_dir(ref, segment)
        records, _noise = self._content_files(directory)
        prefix = f"{ref.leaf}--"
        found: list[tuple[str, bytes]] = []
        for entry in records:
            if not entry.name.endswith(".json"):
                raise LedgerStoreError(f"unexpected entry in a ledger {segment} directory: {entry.name!r}")
            if not entry.name.startswith(prefix):
                continue
            payload = self._read_bytes(directory / entry.name, MAX_RECORD_BYTES)
            if payload is not None:
                found.append((entry.name, payload))
        return found

    @staticmethod
    def _decode[T](decoder: Callable[..., T], payload: bytes, ref: ClaimRef, name: str) -> T:
        try:
            return decoder(payload, expected=ref)
        except LedgerSchemaError:
            raise
        except LedgerCodecError as exc:
            raise LedgerStoreError(f"ledger record {name!r} is corrupt: {exc}") from exc

    # ── 目录 ────────────────────────────────────────────────────────────────

    def hypotheses(self) -> tuple[str, ...]:
        """盘上有账的假设身份。"""

        found: list[str] = []
        for consequent in self._directories(self.directory):
            if self._identity(consequent.name) != consequent.name:
                raise LedgerStoreError(f"ledger directory {consequent.name!r} does not carry a canonical identity")
            for leaf in self._directories(consequent):
                found.append(f"{consequent.name}/{leaf.name}")
        return tuple(sorted(found))

    def claim_path(self, ref: ClaimRef) -> Path:
        return self._day_dir(ref, CLAIMS_SEGMENT) / f"{ref.leaf}.json"

    def settlement_path(self, ref: ClaimRef) -> Path:
        return self._day_dir(ref, SETTLEMENTS_SEGMENT) / f"{ref.leaf}.json"

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _hypothesis_dir(self, hypothesis_identity: str) -> Path:
        consequent, separator, leaf = hypothesis_identity.partition("/")
        if not separator or not leaf or "/" in leaf:
            raise LedgerStoreError("hypothesis identity is '<consequent>/<leaf>'")
        return self._inside(Path(LEDGER_SEGMENT, self._identity(consequent), leaf))

    def _day_dir(self, ref: ClaimRef, segment: str) -> Path:
        if not isinstance(ref, ClaimRef):
            raise TypeError("ref must be a ClaimRef")
        return self._inside(
            self._hypothesis_dir(ref.hypothesis_identity).relative_to(self.root)
            / segment
            / f"{ref.day.year:04d}"
            / f"{ref.day.month:02d}"
            / f"{ref.day.day:02d}"
        )

    def _interventions_dir(self, ref: ClaimRef) -> Path:
        return self._day_dir(ref, INTERVENTIONS_SEGMENT)

    def _refs(self, hypothesis_identity: str, segment: str) -> Iterator[ClaimRef]:
        base = self._hypothesis_dir(hypothesis_identity) / segment
        for year in self._directories(base):
            if not _is_digits(year.name, 4):
                continue
            for month in self._directories(year):
                if not _is_digits(month.name, 2):
                    continue
                for day_dir in self._directories(month):
                    if not _is_digits(day_dir.name, 2):
                        continue
                    try:
                        day = date(int(year.name), int(month.name), int(day_dir.name))
                    except ValueError:
                        continue
                    # 只列一次目录：``_content_files`` 已经把噪音分出来了，不再为清理多列一遍（读路径也不删——
                    # 会杀掉并发写者正在 link 的 .tmp，评审 B7；清理放写路径）。
                    records, _noise = self._content_files(day_dir)
                    for entry in sorted(records, key=lambda item: item.name):
                        if not entry.name.endswith(".json"):
                            raise LedgerStoreError(f"unexpected entry in a ledger day: {entry.name!r}")
                        try:
                            yield ClaimRef(hypothesis_identity, day, entry.name[:-5])
                        except ValueError as exc:
                            raise LedgerStoreError(f"ledger record {entry.name!r} has a non-canonical leaf name") from exc

    def _create(self, path: Path, payload: bytes) -> None:
        """add-only：同内容重放是空操作，不同内容拒绝。

        ``atomic_create_bytes`` 把父目录的路径问题也折成"冲突"报出来；这里先把父目录建好、核过，剩下的冲突才真是
        "同地址不同内容"，排障不会被指到错误的方向。
        """

        if len(payload) > MAX_RECORD_BYTES:
            raise LedgerStoreError("ledger record exceeds its byte budget")
        self._ensure_directory(path.parent)
        self._discard_noise(path.parent)  # 清噪音只在写路径：读路径清会删掉并发写者正在 link 的 .tmp（评审 B7）
        self._require_capacity(path.parent, path.name)
        try:
            atomic_create_bytes(path, payload, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            raise LedgerStoreError("ledger records are add-only; this address already holds different content") from exc
        if self._read_bytes(path, MAX_RECORD_BYTES) != payload:
            raise LedgerStoreError("ledger record failed read-back verification")

    @staticmethod
    def _identity(name: object) -> str:
        try:
            return concept_identity(name)
        except ConceptError as exc:
            raise LedgerStoreError(str(exc)) from exc


def _is_digits(value: str, width: int) -> bool:
    return len(value) == width and is_ascii_digits(value)


__all__ = [
    "CLAIMS_SEGMENT",
    "INTERVENTIONS_SEGMENT",
    "LEDGER_SEGMENT",
    "MAX_RECORD_BYTES",
    "RESPONSES_SEGMENT",
    "SETTLEMENTS_SEGMENT",
    "LedgerStore",
    "LedgerStoreError",
]
