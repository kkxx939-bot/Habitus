"""从行为树读取一次重建所需的全部输入。

这是本层**唯一**碰行为树的读取入口，也是 ``prediction → behavior`` 这条依赖的唯一落点。
产物类型 ``BehaviorSnapshot`` 刻意**不**住在这里而住在 ``model.py``：它一个 behavior 类型都不沾，
放在本模块会让所有想引用它的模块（builder、evaluation）连坐 import 本模块，从而把整张行为树的
导入图拖进来——边界就只剩字面意义了。架构测试查的是传递闭包，不是谁写了 import。

读取纪律：

- 只读不写；
- ``original_name`` 非空的 occurrence 是撞车消歧的已知重复，**机械跳过**——标记在写入时
  由归约层打好，本层只认标记、自己不做判重（判重只在融合层解决）；
- 两类 gap（没读懂 / 未观测）都进曝光扣减，但**证伪能力不同**：本层把 ``gap_kind`` 翻译成
  ``ObservedGap.watched``（没读懂=在看、未观测=没在看），翻译只在这里做一次，域内不再出现
  上游的中文词表；两者的分工见 ``nodes.reconcile_gaps``；
- ``concurrent_with`` 链接原样读出交给配对层分流，不在这里解释它的含义。行为树只存前向一条，
  本层交上去的是**无序的下标对**，配对层再按动作身份序规范成一个键（``edges._parallel_key``），
  方向由查询层给出——所以单向足够。
- ``reminded`` 为真的记录**硬拒**：被提醒之后的发生不属于自然率，而干预账本还没建。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime

from behavior.document import BehaviorDocument, BehaviorLinkType
from behavior.tree import BehaviorAddress, BehaviorKind, BehaviorTree
from behavior.uri import BehaviorURI
from prediction.errors import PredictionTreeError
from prediction.model import BehaviorSnapshot, ObservedAction, ObservedGap

_PAGE_LIMIT = 10_000

# TODO(BHV-LIFECYCLE-001·预测读取): 预测夜批读行为树是全量扫描，日集合枚举在文档数超过分页
# 上限后二次增长。
# - 现状：``_documents`` 为了拿到"有哪些天"，用 ``list_addresses(kind, limit=10_000, after=…)``
#   分页枚举**全部叶子**；``BehaviorTree.list_addresses`` 每页都从 ``_iter_kind`` 从头重走、
#   按游标跳过，总成本随 页数 × 文档数 增长。树上没有"只列日期"的读接口
#   （``list_day_addresses`` 要求先知道是哪一天）。
# - 具体场景：实测 9,270 条 occurrence，页大小 10,000（1 页）0.21s、2,000（5 页）1.67s、
#   1,000（10 页）3.20s。按 WP4 折叠粒度约 400 条/天推算，一年约 146,000 条 = 15 页 →
#   仅"有哪些天"这一步就约 50 秒，而它的答案只是 365 个日期。``read_day`` 的解码成本是
#   线性的（9,270 篇 4.59s），不在此列。
# - 影响大小：中。夜批是离线的，失败会被 worker 吞掉并在健康面报 stale，不影响正确性；
#   但成本随历史增长，与观测存储 ``list()`` 的枚举上限是同一类欠账。
# - 改造方案（取其一或都做）：① 行为树增加只读的 ``list_days(kind) -> tuple[date, ...]``，
#   只枚举 YYYY/MM/DD 目录、不构造叶子地址，``_documents`` 改用它；② 生命周期给行为树定下
#   保留期之后，夜批只读保留窗内的日期（衰减半衰期 60 天，超过约 2τ 的历史对数字的贡献已
#   可忽略），全量扫描随之消失。
# - 时机：**等预测算法定稿之后，与 BHV-LIFECYCLE-001 统一批次做**（用户裁定 2026-09-01）。
#   生命周期必须三棵树统一设计，不在预测层单独开一个保留期。在此之前夜批一律全量读——
#   没有统一门槛之前少读任何一天都会静默改变曝光分母。


# 上游 ``behavior.schema.vocabulary.GAP_KINDS`` 的两个取值。**刻意不 import 那个 frozenset**：
# 本层要的是"当时在不在看"这个布尔，不是词表；把翻译钉在这一处，上游改名时这里会以未知取值
# 硬失败，而不是让一个静默的 False 一路流进曝光分母。
_GAP_WATCHED = "没读懂"
_GAP_UNOBSERVED = "未观测"


def _watched(value: object) -> bool:
    """``gap_kind`` → "这段空白期间我们在不在看"。"""

    if value == _GAP_WATCHED:
        return True
    if value == _GAP_UNOBSERVED:
        return False
    raise PredictionTreeError(f"unknown gap kind: {value!r}")


def read(tree: BehaviorTree) -> BehaviorSnapshot:
    """读出全部 occurrence 与 gap。

    全量扫描：单人一年的量级（万条上下）在夜批里可以接受，分片重建留给
    ``TODO(BHV-LIFECYCLE-001·预测读取)`` 的生命周期方案一起做。
    """

    if not isinstance(tree, BehaviorTree):
        raise PredictionTreeError("tree must be a BehaviorTree")

    unsorted: list[ObservedAction] = []
    index_by_uri: dict[str, int] = {}
    links: list[tuple[str, str]] = []
    skipped = 0
    for document in _documents(tree, BehaviorKind.OCCURRENCE):
        if document.fields.get("original_name") is not None:
            skipped += 1
            continue
        if document.fields.get("reminded") is True:
            # TODO(PRED-TREE-001) 的待定项：被提醒之后的发生不得进入自然率的估计，而干预
            # 账本还没建，本层也就没有过滤它的依据。事后无法把两种数据分开，所以宁可硬拒
            # 也不能静默把它数进去——这条护栏必须在提醒功能上线**之前**就在这里挡着。
            raise PredictionTreeError(
                "occurrence is marked as reminded but the intervention ledger does not exist yet; "
                "counting it would silently pollute the natural rate (see TODO(PRED-TREE-001))"
            )
        unsorted.append(
            ObservedAction(
                action=_text(document.fields.get("kind_token"), "kind_token"),
                started_at=_local(document.fields.get("started_at"), "started_at"),
                day=_day(document.fields.get("occurred_on")),
            )
        )
        uri = str(BehaviorURI.from_address(document.address))
        index_by_uri[uri] = len(unsorted) - 1
        links.extend(
            (uri, str(link.to_uri))
            for link in document.links
            if link.link_type is BehaviorLinkType.CONCURRENT_WITH
        )

    gaps = tuple(
        ObservedGap(
            started_at=_local(document.fields.get("started_at"), "started_at"),
            ended_at=_local(document.fields.get("ended_at"), "ended_at"),
            watched=_watched(document.fields.get("gap_kind")),
        )
        for document in _documents(tree, BehaviorKind.GAP)
    )

    order = sorted(range(len(unsorted)), key=lambda index: unsorted[index].started_at)
    rank = {index: position for position, index in enumerate(order)}
    # 指向被跳过（消歧重复）的一端时整条丢弃：那条并行关系没有两个可用的端点。
    concurrent = {
        _unordered(rank[index_by_uri[source]], rank[index_by_uri[target]])
        for source, target in links
        if source in index_by_uri and target in index_by_uri
    }
    return BehaviorSnapshot(
        actions=tuple(unsorted[index] for index in order),
        gaps=gaps,
        concurrent=tuple(sorted(concurrent)),
        skipped_duplicates=skipped,
    )


def _documents(tree: BehaviorTree, kind: BehaviorKind) -> Iterator[BehaviorDocument]:
    """按天整块读：逐篇 ``read`` 会让每天的目录被重复枚举、成本随篇数平方增长（实测一周 90 秒）。"""

    days: list[date] = []
    cursor: BehaviorAddress | None = None
    while True:
        page = tree.list_addresses(kind, limit=_PAGE_LIMIT, after=cursor)
        for address in page:
            if not days or days[-1] != address.occurred_on:
                if address.occurred_on not in days:
                    days.append(address.occurred_on)
        if len(page) < _PAGE_LIMIT:
            break
        cursor = page[-1]
    for day in sorted(set(days)):
        yield from tree.read_day(kind, day)


def _unordered(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def _text(raw: object, field: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise PredictionTreeError(f"{field} must be non-empty text")
    return raw


def _day(raw: object) -> date:
    try:
        return date.fromisoformat(_text(raw, "occurred_on"))
    except ValueError as exc:
        raise PredictionTreeError("occurred_on is not a calendar date") from exc


def _local(raw: object, field: str) -> datetime:
    """行为树上的时间是本地时刻加显式偏移；本层原样保留，槽位按本地时分映射。"""

    try:
        parsed = datetime.fromisoformat(_text(raw, field))
    except ValueError as exc:
        raise PredictionTreeError(f"{field} is not a timestamp") from exc
    if parsed.utcoffset() is None:
        raise PredictionTreeError(f"{field} must carry its local offset")
    return parsed


__all__ = ["read"]
