"""此刻场景里"还没封口"的那一截：把判断存储里尚未归约的判断读成 ``UnsealedRow``。

行为树上只有归约发布过的 occurrence；最近一段（融合静默期 + 归约回看窗）里的判断还在判断存储
里等封口。此刻场景要把它们补上，否则"到此刻已发生"永远缺最新的几条。

这是一座桥，所以住在组合根：behavior 对 foresight 零知识（不能产 ``UnsealedRow``），foresight 不读
行为侧的任何存储（架构测试钉死）。桥上**一个判断都不发明**，三样都直接调归约自己的东西：

- 哪些判断算"没封口"：``reduction.pending_judgements``——归约每轮第一步用的同一个函数
  （存储里有且账本里没有；坏记录隔离）。
- 一条的起点与最后看到：判断的 ``started_at`` 与 ``last_observed_at``，occurrence 上写的就是这两个。
- 选的哪个词（裁定 101）：判断上带着选词结果的（会话 lane）直接用；没带的（物理 lane）用归约的同一个选词器
  提前选、按节点身份缓存。归约封口时照常再选一次，这里的结果只给此刻场景用，不写回任何地方。

选词要调模型，所以分两步：``prepare``（异步，每拍装配之前调一次）选新出现的没带结果的判断；``rows``（同步，装配时调）只读
判断与缓存。模型此刻用不了就留着下一拍再选（实时化方案第 2 节：模型调不通不停一天）。这一份被内容安全拦下、输入超长
（``KindModelOutputError``）同一份再问也一样：按错误处理，记进缓存为"选不出"、不再问、不算欠，这一行不带词（裁定 102；
原来每拍重问）。缓存只在进程里，重启后会再问一次。换代里没有模型调用（实时化阶段 2）：
换代里的日子预期读 ``cache_only`` 那一份——只读缓存，不问模型；换代之前由模型队列把还没选的选好（``unclassified`` 是它认
欠账的数），没选上的按未归类进场景。
判断上的编号与行为树上的一样过同一个换算口（``series.model.KindResolution``：删掉的节点换成最近的活着的上级）。
还没选上的、选成「表外」「待定」的带原名进场景、标未归类，不猜；「非事件」「续话」不进场景（与序列同一口径）。
没读懂、出错的判断（``behavior`` 为 None）是一段看不到的空白，按空白行给出；依据写着"没在看"（``unobserved``）的空判断是
「未观测」，不是看不到——照常说话（裁定 68 第 2 条、94 第 3 条），不给出。隔离的坏记录归约每轮都会报，这里不报第二遍。
"""

from __future__ import annotations

from datetime import UTC, date, datetime, tzinfo

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.kinds.calls import KindModelOutputError
from habitus.behavior.kinds.ids import continued_token, is_word, not_event_token
from habitus.behavior.reduction.kinds_step import KindStamping, record_material
from habitus.behavior.reduction.ledger import BehaviorReductionLedger
from habitus.behavior.reduction.payloads import UNOBSERVED_GAP_KIND, gap_kind_of
from habitus.behavior.reduction.pending import pending_judgements
from habitus.behavior.reduction.record import ReducibleJudgement
from habitus.foresight import UnsealedRow
from habitus.model_client import MODEL_UNAVAILABLE
from habitus.series.model import KindResolution

#: 缓存里"选不出"（被拦、超长）的记法：不是任何 lane 的编号。
_UNPICKABLE = ""


class UnsealedFromJudgements:
    """``UnsealedReader`` 的生产实现：判断存储 + 消费账本 + 归约的选词器。"""

    def __init__(
        self,
        judgements: BehaviorJudgementStore,
        ledger: BehaviorReductionLedger,
        stamping: KindStamping,
        *,
        kinds: KindResolution | None = None,
    ) -> None:
        self.judgements = judgements
        self.kinds = kinds
        self.ledger = ledger
        self.stamping = stamping
        #: 节点身份 → 选词结果（只缓存判断上没带结果的；只留还没封口的）。被拦、超长的记 ``_UNPICKABLE``。
        self._picked: dict[str, str] = {}
        #: 最近一次 ``prepare`` 因模型此刻用不了没选成时的那个异常（选成了、或没有要选的是 None）：模型队列读它，
        #: 好分清是整体性的故障还是这一项自己没做成。
        self.unavailable: BaseException | None = None

    async def prepare(self, *, until: datetime) -> None:
        """把判断上没带选词结果、还没选过的未封口判断交选词器。模型这一拍不可用（``MODEL_UNAVAILABLE``）就留着不选（下一拍
        再试），不让此刻场景因此失败；这一份被拦、输入超长的这一拍跳过；配置错误照样抛出来——吞掉它会让每一拍都悄悄选不上。"""

        if until.utcoffset() is None:
            raise ValueError("until must be a timezone-aware datetime")
        self.unavailable = None
        picker = self.stamping.picker
        if picker is None:
            return
        for record in self._unpicked():
            material = record_material(record, max_steps=picker.config.prompt_max_steps)
            if material is None:
                continue
            try:
                self._picked[record.chain_digest] = await picker.pick_record(material, at=record.started_at)
            except MODEL_UNAVAILABLE as exc:
                self.unavailable = exc
                return
            except KindModelOutputError:
                # 这一份被拦、输入超长：按错误处理，不再问（同一份再问也一样）；此刻场景里这一行不带词，归约封口时照常处理
                self._picked[record.chain_digest] = _UNPICKABLE

    def _unpicked(self) -> list[ReducibleJudgement]:
        """还没选过词的未封口判断（判断上没带结果、缓存里也没有）；已封口的从缓存里清掉。不问模型。"""

        records = pending_judgements(self.judgements, self.ledger).occurrences
        alive = {record.chain_digest for record in records}
        self._picked = {key: value for key, value in self._picked.items() if key in alive}
        return [record for record in records if record.kind_token is None and record.chain_digest not in self._picked]

    def unclassified(self) -> int:
        """还没选过词的未封口判断条数（会话 lane 的判断都带着结果，不算）。"""

        return 0 if self.stamping.picker is None else len(self._unpicked())

    def cache_only(self) -> CachedUnsealed:
        """只读缓存的那一份（换代里的日子预期用）：``prepare`` 不问模型。"""

        return CachedUnsealed(self)

    def rows(self, *, since: datetime, until: datetime) -> tuple[UnsealedRow, ...]:
        """与 ``[since, until]`` 有交集的未封口行为与空白，按开始时刻排。"""

        for label, value in (("since", since), ("until", until)):
            if value.utcoffset() is None:
                raise ValueError(f"{label} must be a timezone-aware datetime")
        begin, finish = since.astimezone(UTC), until.astimezone(UTC)
        pending = pending_judgements(self.judgements, self.ledger)
        lane = self.stamping.lane
        uncounted = {not_event_token(lane), continued_token(lane)}
        resolve = None if self.kinds is None else self.kinds()
        rows: list[UnsealedRow] = []
        for record in pending.occurrences:
            token = record.kind_token if record.kind_token is not None else self._picked.get(record.chain_digest)
            if token == _UNPICKABLE:
                token = None
            if token in uncounted:
                continue
            if token is not None and resolve is not None:
                token = resolve(token)
            rows.append(
                UnsealedRow(
                    name=str(record.behavior),
                    kind_token=token if token is not None and is_word(token) else None,
                    started_at=record.started_at,
                    last_observed_at=record.last_observed_at,
                    summary=record.summary,
                )
            )
        rows.extend(
            UnsealedRow(
                name=None,
                kind_token=None,
                started_at=gap.started_at,
                last_observed_at=gap.last_observed_at,
                summary=None,
            )
            for gap in pending.gaps
            if gap_kind_of(gap) != UNOBSERVED_GAP_KIND
        )
        inside = [row for row in rows if _instant(row.last_observed_at) >= begin and _instant(row.started_at) <= finish]
        return tuple(sorted(inside, key=_row_order))


class CachedUnsealed:
    """``UnsealedFromJudgements`` 只读缓存的那一份：同一座桥、同一份缓存，``prepare`` 不调模型。"""

    def __init__(self, bridge: UnsealedFromJudgements) -> None:
        self.bridge = bridge

    async def prepare(self, *, until: datetime) -> None:
        if until.utcoffset() is None:
            raise ValueError("until must be a timezone-aware datetime")
        self.bridge.unclassified()

    def rows(self, *, since: datetime, until: datetime) -> tuple[UnsealedRow, ...]:
        return self.bridge.rows(since=since, until=until)


def unreduced_gaps(
    judgements: BehaviorJudgementStore, ledger: BehaviorReductionLedger, *, cutoff: date, zone: tzinfo
) -> int:
    """截止日之前（本地日）还没归约的空判断条数：换代按槽留余量，等这几槽里归约把没在看的空白落树——这个数正常应为 0
    （用户 11-10 已知局限的读数）。"""

    return sum(
        1 for gap in pending_judgements(judgements, ledger).gaps if gap.started_at.astimezone(zone).date() < cutoff
    )


def _instant(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _row_order(row: UnsealedRow) -> tuple[datetime, str]:
    return _instant(row.started_at), row.name or ""


__all__ = ["CachedUnsealed", "UnsealedFromJudgements", "unreduced_gaps"]
