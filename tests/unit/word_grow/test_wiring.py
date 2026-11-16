"""组合根接线（阶段 C、D 复核补的测试）：模型队列那份序列读者也过换算口；``kinds_max_level`` 真的进了选词；选词等不到词表状态
的租约时走哪条路（跨进程写修补）。"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from habitus.behavior.kinds.model import Vocabulary
from habitus.behavior.kinds.store import BehaviorKindStore
from tests.unit.word_grow.site import WRITE_DOC, Grow, GrowSite, at, grown, names, node

D13 = date(2026, 4, 13)


def _max_level(to: int) -> Any:
    def edit(raw: dict[str, Any]) -> None:
        raw.setdefault("behavior", {})["kinds_max_level"] = to

    return edit


def test_the_model_queue_series_reader_reads_deleted_nodes_as_their_parent(tmp_path: Path) -> None:
    site = GrowSite(tmp_path)
    try:
        site.record("conv", at(D13, 9), "帮我写一份产品需求书", Grow("写文档", node("写需求书")))
        site.session.kind_store.update(lambda vocabulary: vocabulary.without(vocabulary.named("写需求书").id, on=D13))  # type: ignore[union-attr]
        queue = site.runtime.components.model_queue
        assert queue is not None
        reader = queue.queues[0].sealed.reader
        series = reader.read(cutoff=date(2026, 4, 14))
        assert [record.kind_token for record in series.records] == [WRITE_DOC]
    finally:
        site.close()


def test_the_level_limit_comes_from_the_config(tmp_path: Path) -> None:
    site = GrowSite(tmp_path, edit=_max_level(2))
    try:
        site.record(
            "conv", at(D13, 9), "帮我写一份产品需求书的验收段", Grow("写文档", node("写需求书"), node("写验收段"))
        )
        assert site.token_at(at(D13, 9)) == grown(1)
        block = names(site.probe(D13))
        assert "写需求书" in block and "写验收段" not in block
    finally:
        site.close()


def test_a_turn_that_cannot_get_the_vocabulary_lease_is_not_written_and_is_redone_later(tmp_path: Path) -> None:
    """选词这一侧等不到词表状态的租约（另一个写者持着、超过等待上限）：``TimeoutError`` 照 ``PathLock`` 抛给会话源（还没准备好，
    与回执锁等不到同一条路），这一轮什么都不落、也不写成「错误」；放手以后同一轮重做，照常建节点、落判断。"""

    def short_lease(raw: dict[str, Any]) -> None:
        raw.setdefault("behavior", {})["write_lock_ttl_seconds"] = 1

    site = GrowSite(tmp_path, edit=short_lease)
    try:
        store = site.session.kind_store
        other = BehaviorKindStore(
            store.root, lane=store.lane, lock_store=store.lock_store, lock_ttl_seconds=30, config=store.config
        )
        holding, release = threading.Event(), threading.Event()

        def hold(vocabulary: Vocabulary) -> Vocabulary:
            holding.set()
            release.wait(30)
            return vocabulary

        holder = threading.Thread(target=other.update, args=(hold,))
        holder.start()
        assert holding.wait(10)
        site.router.picks.append(Grow("写文档", node("写需求书")))
        turn = site.turn("conv", at(D13, 9), "帮我写一份产品需求书")
        try:
            with pytest.raises(TimeoutError):
                site.consume(turn)
            assert not site.session.judgements.list() and store.read().named("写需求书") is None
        finally:
            release.set()
            holder.join(30)
        site.router.picks.append(Grow("写文档", node("写需求书")))
        site.consume(turn)
        site.reduce()
        assert site.token_at(at(D13, 9)) == grown(1)
    finally:
        site.close()
