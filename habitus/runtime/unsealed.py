"""此刻场景里"还没封口"的那一截：把判断存储里尚未归约的判断读成 ``UnsealedRow``。

行为树上只有归约发布过的 occurrence；最近一段（融合静默期 + 归约回看窗）里的判断还在判断存储
里等封口。此刻场景要把它们补上，否则"到此刻已发生"永远缺最新的几条。

这是一座桥，所以住在组合根：behavior 对 foresight 零知识（不能产 ``UnsealedRow``），foresight 不读
行为侧的任何存储（架构测试钉死）。桥上**一个判断都不发明**，三样都直接调归约自己的东西：

- 哪些判断算"没封口"：``reduction.pending_judgements``——归约每轮第一步用的同一个函数
  （存储里有且账本里没有；坏记录隔离）。
- 一条的起点与最后看到：判断的 ``started_at`` 与 ``last_observed_at``，occurrence 上写的就是这两个。
- 归到哪一类：归约用的**同一个白天归类**（``KindStamping``，看原话 + 概要 + 目标 + 步骤），只是提前到
  封口之前做；按节点身份缓存。归约封口时照常再归一次，这里的结果只给此刻场景用，不写回任何地方。
  （原来按"近 30 天原话 → 编号"查：新口径下原话是每一次自己的名字，几乎不逐字重复——54 天真实数据 560 条只命中
  9 条，1.6%，最近一小时等于全是未归类。）

归类要调模型，所以分两步：``prepare``（异步，每拍装配之前调一次）归新出现的判断；``rows``（同步，装配时调）只读缓存。
模型此刻用不了就留着下一拍再归——预测拍与换代里的日子预期同一口径（实时化方案第 2 节：模型调不通不停一天）。
还没归上的（模型这一拍不可用、或归为「待定」）带原名进场景、标未归类，不猜；归为「非事件」的不进场景（与序列同一口径）。
没读懂、出错的判断（``behavior`` 为 None）是一段看不到的空白，按空白行给出；依据写着"没在看"（``unobserved``）的空判断是
「未观测」，不是看不到——照常说话（裁定 68 第 2 条、94 第 3 条），不给出。隔离的坏记录归约每轮都会报，这里不报第二遍。
"""

from __future__ import annotations

from datetime import UTC, date, datetime, tzinfo

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.kinds.classify import EXHAUSTED, ClassifyRequest, Outcome
from habitus.behavior.reduction.kinds_step import KindStamping, judgement_content
from habitus.behavior.reduction.ledger import BehaviorReductionLedger
from habitus.behavior.reduction.payloads import UNOBSERVED_GAP_KIND, gap_kind_of
from habitus.behavior.reduction.pending import pending_judgements
from habitus.foresight import UnsealedRow
from habitus.model_client import MODEL_UNAVAILABLE


class UnsealedFromJudgements:
    """``UnsealedReader`` 的生产实现：判断存储 + 消费账本 + 归约的白天归类。"""

    def __init__(
        self,
        judgements: BehaviorJudgementStore,
        ledger: BehaviorReductionLedger,
        stamping: KindStamping,
    ) -> None:
        self.judgements = judgements
        self.ledger = ledger
        self.stamping = stamping
        #: （节点身份, 词表版本）→ （结局, 编号）。只留还没封口的；词表拆改之后版本变了，自动重归（不留停用的编号）。
        self._classified: dict[tuple[str, int], tuple[Outcome, str]] = {}
        self._version = 0

    async def prepare(self, *, until: datetime) -> None:
        """把还没归过的未封口判断交白天归类。模型这一拍不可用（``MODEL_UNAVAILABLE``）就留着不归（下一拍再试），不让此刻场景
        因此失败；配置错误照样抛出来——吞掉它会让每一拍都悄悄归不上。"""

        if until.utcoffset() is None:
            raise ValueError("until must be a timezone-aware datetime")
        records = pending_judgements(self.judgements, self.ledger).occurrences
        vocabulary = self.stamping.store.read()
        self._version = vocabulary.version
        alive = {(record.chain_digest, vocabulary.version) for record in records}
        self._classified = {key: value for key, value in self._classified.items() if key in alive}
        if vocabulary.version == 0:
            return  # 词表还是空的（还没有一类从待定池长出来）：没有类可归
        requests = []
        for record in records:
            content = judgement_content(record)
            if (record.chain_digest, vocabulary.version) not in self._classified and content is not None:
                requests.append(ClassifyRequest(record.chain_digest, self.stamping.lane, content))
        if not requests:
            return
        try:
            result = await self.stamping.classify_requests(requests, vocabulary, checkpoint=lambda: None)
        except MODEL_UNAVAILABLE:
            return
        for key, verdict in result.verdicts.items():
            if verdict.outcome is Outcome.PENDING and verdict.reason == EXHAUSTED:
                continue  # 这次没问成（结构化输出重问用尽），不是模型说"都不是"：不缓存，下一拍再试
            self._classified[(key, vocabulary.version)] = (verdict.outcome, verdict.token)

    def rows(self, *, since: datetime, until: datetime) -> tuple[UnsealedRow, ...]:
        """与 ``[since, until]`` 有交集的未封口行为与空白，按开始时刻排。"""

        for label, value in (("since", since), ("until", until)):
            if value.utcoffset() is None:
                raise ValueError(f"{label} must be a timezone-aware datetime")
        begin, finish = since.astimezone(UTC), until.astimezone(UTC)
        pending = pending_judgements(self.judgements, self.ledger)
        rows: list[UnsealedRow] = []
        for record in pending.occurrences:
            outcome, token = self._classified.get((record.chain_digest, self._version), (Outcome.PENDING, ""))
            if outcome is Outcome.NOT_EVENT:
                continue
            rows.append(
                UnsealedRow(
                    name=str(record.behavior),
                    kind_token=token if outcome is Outcome.CLASS else None,
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


__all__ = ["UnsealedFromJudgements", "unreduced_gaps"]
