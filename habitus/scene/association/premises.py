"""还立着的待用前提：一轮夜批扫一次规律级，在内存里投影成一张表。

**为什么是投影而不是索引。** 前提是**跨候选**的——买菜留下的被做饭消费——而规律级按候选分目录，
所以取回要看所有候选的记录。每个任务扫一次全树不可行；一轮扫一次、整轮共用，本轮新产生与新兑现
增量更新，跑完就扔。这也正是既有裁定的形状：权威只能是派生树（目录 + 文档），"表"只允许是读时
在内存里投影出来的东西。树大到扫不动了再谈持久索引，而那个索引必须能从记录全量重建。

**没有窗口，没有过期。** 旧实现（已删）按 ``expiry_days`` 画一个日历窗，窗内收集产生与兑现——
于是三件事同时错：一个月前建立的前提永远看不见；消费者掉出窗口后前提会**复活**；而"到期没到期"
本身就是预测层的判断，不该长在这一层。这里只回答事实：**产生了、且没有任何兑现记录指过它**。
要不要因为等太久而不再当回事，是判断者读的时候自己决定的，它手里有"建立于何时"这个事实。

**粒度是前提，不是行为。** 旧实现把兑现记在"整条行为被某条 ``needs`` 边指过"上，于是去超市一趟
留下的"家里有菜"和"买到了灯泡"，用掉其中一条另一条跟着消失。这里按 ``premise_identity``
（产生方 + 那句话的规范形式）记账。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime

from habitus.foundation.ids import canonical_path_identity
from habitus.scene.regularity.document import AssociationDocument, premise_identity
from habitus.scene.regularity.store import RegularityTree, RegularityTreeError


@dataclass(frozen=True)
class Premise:
    """一条已经建立的前提。``consumptions`` 是它被用掉过几次——**事实，不是状态**。"""

    producer_uri: str
    text: str
    waits_for: str
    created_on: date
    created_at: datetime
    consumptions: int = 0

    @property
    def identity(self) -> tuple[str, str]:
        return premise_identity(self.producer_uri, self.text)

    @property
    def standing(self) -> bool:
        """还没被用掉过。默认只把这些摆进提示词——要改成"一条前提可以服务多次"，改的是取回口径，
        不是已经存下来的事实。"""

        return self.consumptions == 0


class PremiseTable:
    """一轮之内共用的前提投影；本轮新产生与新兑现就地更新。"""

    def __init__(self, premises: Iterable[Premise] = ()) -> None:
        self._by_identity: dict[tuple[str, str], Premise] = {}
        for premise in premises:
            self.add(premise)

    @classmethod
    def scan(cls, tree: RegularityTree) -> tuple[PremiseTable, tuple[str, ...]]:
        """扫遍规律级：先收全部产生，再按兑现记账。两趟，因为兑现可以出现在产生之前被读到。

        取材按**日目录里读得出的记录**，不按完成标记：记录先落盘、标记最后写，只认标记的话，
        一个还没完成的日子里已经写下的 ``left`` 永远看不见，而它 ``consumed`` 掉的前提会一直算
        "还立着"，下一轮被另一条发生第二次用掉。重做那一天时 ``retain_only`` 会先清掉旧记录，
        所以按记录取材不会重复计。

        **坏的一天只丢那一天。**这个方法是编排整轮的第一条语句，让树上任意一个杂文件掀翻整轮
        是不可接受的——返回信号，由调用方决定怎么说。
        """

        if not isinstance(tree, RegularityTree):
            raise TypeError("tree must be a RegularityTree")
        table = cls()
        signals: list[str] = []
        consumed: list[tuple[str, str]] = []
        for kind in tree.list_kinds():
            for day in tree.day_directories(kind):
                try:
                    documents = tree.read_day(kind, day)
                except (RegularityTreeError, ValueError) as exc:
                    signals.append(f"premises_skipped: {kind}/{day.isoformat()} is not readable ({exc})")
                    continue
                for document in documents:
                    table._absorb(document, day)
                    consumed.extend(document.consumed)
        for producer_uri, text in consumed:
            if not table.consume(producer_uri, text):
                signals.append(f"premises_unmatched: a record consumed «{text}» that no record produced")
        return table, tuple(signals)

    def _absorb(self, document: AssociationDocument, day: date) -> None:
        for text, waits_for in document.left:
            self.add(
                Premise(
                    producer_uri=document.occurrence_uri,
                    text=text,
                    waits_for=waits_for,
                    created_on=day,
                    created_at=document.address.started_at,
                )
            )

    def add(self, premise: Premise) -> None:
        if not isinstance(premise, Premise):
            raise TypeError("premise must be a Premise")
        # 同一身份重复出现：保留第一条。前提一次写下之后文本不再变，所以这只可能是重跑的产物。
        self._by_identity.setdefault(premise.identity, premise)

    def consume(self, producer_uri: str, text: str) -> bool:
        """记一次兑现。指向一条没见过的前提时返回 False——那是记账疏漏，由调用方留信号。"""

        key = premise_identity(producer_uri, text)
        premise = self._by_identity.get(key)
        if premise is None:
            return False
        self._by_identity[key] = Premise(
            producer_uri=premise.producer_uri,
            text=premise.text,
            waits_for=premise.waits_for,
            created_on=premise.created_on,
            created_at=premise.created_at,
            consumptions=premise.consumptions + 1,
        )
        return True

    def waiting_for(
        self, kind_token: str, *, before: datetime, limit: int
    ) -> tuple[tuple[Premise, ...], tuple[str, ...]]:
        """在等这一类行为、且**建立于此刻之前**的那些前提，连同"这次少说了多少"的信号。

        截断是**渲染预算**，不是失效判断，所以两件事必须成立：

        **一、少说要说出来。**单纯按时间截断等于一条没写进配置里的过期——被挤出去的前提从此不再
        进任何一次提示词，也就永远不可能被兑现，而用户完全看不见自己放弃了什么。这比旧实现那个
        写在 yaml 上的 90 天更糟：那个至少是明说的。

        **二、取样要分层。**只留最近的几条，会让低频、长跨度的前提被系统性挤掉——恰好是最值得
        关联的那一类（挂号等就诊、买了胶皮等下一次打球）。所以最早的一半与最近的一半各留。
        这也关系到将来那套统一的生命周期机制：它要按"产生到兑现隔了多久"的分布定档，而单调偏向
        最近的取样会往那份分布里注入一个与行为无关的偏置。
        """

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        wanted = canonical_path_identity(kind_token, "premise consumed_by")
        matched = sorted(
            (
                premise
                for premise in self._by_identity.values()
                if premise.standing and premise.created_at < before and _matches(premise.waits_for, wanted)
            ),
            key=lambda premise: (premise.created_at, premise.producer_uri, premise.text),
        )
        if len(matched) <= limit:
            return tuple(reversed(matched)), ()
        oldest = limit // 2
        kept = [*matched[:oldest], *matched[len(matched) - (limit - oldest) :]]
        dropped = len(matched) - len(kept)
        return tuple(reversed(kept)), (
            f"premises_truncated: {dropped} more premises are waiting for «{kind_token}»",
        )

    def all_premises(self) -> tuple[Premise, ...]:
        return tuple(sorted(self._by_identity.values(), key=lambda premise: (premise.created_at, premise.text)))


def _matches(waits_for: str, wanted: str) -> bool:
    """``consumed_by`` 按规范身份比。它**不是检索键的权威**——对不上的前提只是不会被摆出来，
    不会丢失，也不会因此被判掉：一个还没在树上出现过的种类是完全可能的。"""

    try:
        return canonical_path_identity(waits_for, "premise consumed_by") == wanted
    except (TypeError, ValueError):
        return False


__all__ = ["Premise", "PremiseTable"]
