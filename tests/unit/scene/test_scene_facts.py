"""事实门：键与值的形状由源自己声明；只答那一刻已知的；一个键只能有一个源；矛盾硬拒。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from habitus.scene import CompositeFacts, Conditions, FactKey, FactProvider, NoFacts, conditions_of, require_local

CST = timezone(timedelta(hours=8))
WHEN = datetime(2026, 8, 11, 10, 30, tzinfo=CST)


class Weather:
    version = "weather_v1"

    def keys(self) -> tuple[FactKey, ...]:
        return (FactKey("天气.温度", "数值", "℃"), FactKey("天气.天象", "类别"))

    def at(self, when: datetime) -> Conditions:
        require_local(when)
        return (("天气.天象", "晴"), ("天气.温度", "31.5"))


class Broken:
    version = "broken_v1"

    def keys(self) -> tuple[FactKey, ...]:
        return (FactKey("坏.键", "类别"),)

    def at(self, when: datetime) -> Conditions:
        raise RuntimeError("source down")


def test_a_key_declares_how_it_is_compared_and_in_what_unit() -> None:
    assert FactKey("天气.温度", "数值", "℃").unit == "℃"
    with pytest.raises(ValueError, match="kind"):
        FactKey("天气.温度", "连续")
    with pytest.raises(ValueError, match="no unit"):
        FactKey("天气.天象", "类别", "℃")
    with pytest.raises(ValueError, match="name"):
        FactKey(" 天气 ", "类别")


def test_no_facts_is_an_explicit_nothing_and_every_provider_refuses_a_naive_time() -> None:
    assert NoFacts().at(WHEN) == () and NoFacts().keys() == () and NoFacts().version == "none"
    naive = datetime(2026, 8, 11, 10, 30)
    for provider in (NoFacts(), Weather(), CompositeFacts(), CompositeFacts(Weather())):
        with pytest.raises(TypeError, match="timezone-aware"):
            provider.at(naive)
    with pytest.raises(TypeError, match="timezone-aware"):
        require_local(None)


def test_a_composite_merges_versions_and_refuses_a_key_from_two_sources() -> None:
    composite = CompositeFacts(Weather(), NoFacts())
    assert isinstance(composite, FactProvider)
    assert [key.name for key in composite.keys()] == ["天气.天象", "天气.温度"]
    assert composite.at(WHEN) == (("天气.天象", "晴"), ("天气.温度", "31.5"))
    assert composite.version == "none+weather_v1" and CompositeFacts().version == "none"
    # 嵌套也是提供者；撞键在构造时就拒（哪个源赢都是错的）。
    assert CompositeFacts(CompositeFacts(Weather()), NoFacts()).at(WHEN) == composite.at(WHEN)
    with pytest.raises(ValueError, match="two providers"):
        CompositeFacts(Weather(), Weather())
    with pytest.raises(TypeError, match="FactProvider"):
        CompositeFacts(object())  # type: ignore[arg-type]


def test_merging_conditions_sorts_dedups_and_refuses_a_contradiction() -> None:
    # 按键的码位序排，不是按谁先来——读历史时同一组条件要长成同一个样子。
    assert conditions_of((("日型", "工作日"), ("天气.天象", "晴")), (("天气.天象", "晴"),)) == (
        ("天气.天象", "晴"),
        ("日型", "工作日"),
    )
    with pytest.raises(ValueError, match="two values"):
        conditions_of((("日型", "工作日"),), (("日型", "周末"),))
    with pytest.raises(ValueError, match="non-empty text"):
        conditions_of((("日型", ""),))


def test_a_broken_source_raises_here_and_is_handled_by_the_caller() -> None:
    """事实门自己不吞异常——谁用它谁决定怎么降级（预测层记成"没问到"，见 runtime/foresight.py）。"""

    with pytest.raises(RuntimeError, match="source down"):
        CompositeFacts(Broken()).at(WHEN)
