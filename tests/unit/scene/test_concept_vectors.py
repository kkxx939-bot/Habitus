"""概念定义的向量旁册：只管召回；定义改了要重算并按名字报出来；换模型整表作废；根与数值都要过校验。"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from habitus.scene.concepts import (
    ConceptSet,
    ConceptVectorError,
    ConceptVectorIndex,
    ConceptVectorStore,
    embedding_text,
    refresh_concept_vectors,
)
from tests.unit.scene.concept_fixtures import (
    BREAKFAST,
    DIMENSION,
    EXERCISE,
    LATE_GRADES,
    LATE_RULE,
    SLEEP_LATE,
    TableEmbedder,
    concept,
    concept_set,
)

TABLE = {
    embedding_text(SLEEP_LATE): (1.0, 0.0, 0.0, 0.0),
    embedding_text(BREAKFAST): (0.0, 1.0, 0.0, 0.0),
    embedding_text(EXERCISE): (0.0, 0.0, 1.0, 0.0),
}


def test_nearest_ranks_by_cosine_within_the_allowed_pool() -> None:
    index = ConceptVectorIndex("fake-embed", DIMENSION)
    for definition in (SLEEP_LATE, BREAKFAST, EXERCISE):
        index = index.with_vector(definition, TABLE[embedding_text(definition)])
    ranked = index.nearest((0.9, 0.4, 0.0, 0.0), limit=2)
    assert [identity for identity, _ in ranked] == ["晚睡", "早餐"]
    assert ranked[0][1] > ranked[1][1] > 0
    assert [identity for identity, _ in index.nearest((0.9, 0.4, 0.0, 0.0), limit=5, among=("运动", "早餐"))] == ["早餐", "运动"]
    with pytest.raises(ConceptVectorError):
        index.nearest((1.0, 0.0), limit=1)
    with pytest.raises(ConceptVectorError):
        index.nearest((float("nan"), 0.0, 0.0, 0.0), limit=1)
    with pytest.raises(ConceptVectorError):
        index.with_vector(SLEEP_LATE, (1.0, 0.0))
    with pytest.raises(ConceptVectorError, match="finite"):
        index.with_vector(SLEEP_LATE, (float("nan"), 1.0, 0.0, 0.0))


def test_a_changed_definition_makes_its_vector_stale() -> None:
    index = ConceptVectorIndex("fake-embed", DIMENSION).with_vector(SLEEP_LATE, TABLE[embedding_text(SLEEP_LATE)])
    assert index.is_current(SLEEP_LATE)
    assert not index.is_current(concept("晚睡", "入睡晚于常态三小时", rule=LATE_RULE, grades=LATE_GRADES))
    assert not index.is_current(BREAKFAST)
    assert index.stale(concept_set(), among=("晚睡", "早餐")) == ("早餐",)
    assert set(index.retain(("早餐",)).vectors) == set()


def test_store_round_trips_in_float16_and_treats_a_foreign_model_as_empty(tmp_path) -> None:
    store = ConceptVectorStore(tmp_path / "scene", model="fake-embed", dimension=DIMENSION)
    index = store.empty().with_vector(SLEEP_LATE, (0.3, 0.2, 0.1, 0.9))
    store.replace(index)
    assert store.read() == index  # 写入时已按 float16 截断，读回逐值相等
    other = ConceptVectorStore(tmp_path / "scene", model="other-embed", dimension=DIMENSION)
    assert other.read().vectors == {}
    with pytest.raises(ConceptVectorError):
        other.replace(index)
    store.path.write_text(
        json.dumps({"schema_version": "scene_concept_vectors_v1", "model": "fake-embed", "dimension": 4, "vectors": {"晚睡": {"digest": "x", "vector": "!!"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ConceptVectorError, match="decodable"):
        store.read()
    store.path.write_text("not json", encoding="utf-8")
    with pytest.raises(ConceptVectorError, match="corrupt"):
        store.read()


def test_the_vector_store_refuses_a_symlinked_root_like_every_other_scene_store(tmp_path) -> None:
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path / "real", tmp_path / "scene")
    with pytest.raises(ConceptVectorError, match="symbolic link"):
        ConceptVectorStore(tmp_path / "scene", model="fake-embed", dimension=DIMENSION)


def test_refresh_embeds_only_what_is_missing_or_changed_and_names_what_it_touched(tmp_path) -> None:
    store = ConceptVectorStore(tmp_path / "scene", model="fake-embed", dimension=DIMENSION)
    embedder = TableEmbedder(TABLE)
    concepts = concept_set()

    index, report = asyncio.run(refresh_concept_vectors(store, concepts, embedder))
    assert report.embedded == tuple(sorted(concepts)) and report.dropped == ()
    assert set(index.vectors) == set(concepts)
    assert embedder.documents == [tuple(embedding_text(concepts[identity]) for identity in sorted(concepts))]

    # 第二次：只有改了定义的那一个重算，被删掉的概念从旁册里收走——都按名字报，夜批据此定重算范围。
    smaller = ConceptSet([concept("晚睡", "入睡晚于常态三小时", rule=LATE_RULE, grades=LATE_GRADES), BREAKFAST])
    index, report = asyncio.run(refresh_concept_vectors(store, smaller, embedder))
    assert report.embedded == ("晚睡",)
    assert set(report.dropped) == set(concepts) - {"晚睡", "早餐"}
    assert set(index.vectors) == {"晚睡", "早餐"}
    assert embedder.documents[-1] == (embedding_text(smaller["晚睡"]),)
    assert store.read() == index
