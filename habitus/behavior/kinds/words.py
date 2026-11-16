"""两条 lane 的初始词表：随代码发布的只读数据文件（``words/<lane>.json``），编号写死在文件里。

源文件是词表方案（裁定 101）的 ``初始词表-会话.md`` / ``初始词表-物理.md``（裁定 101 第 1 条），转换脚本不进仓库。
编号是下游记账的键，不能随文件顺序漂，所以写在数据里、不由读取时的顺序派生。读的时候整份严格校验：未知字段、缺上级、
层数和上级对不上、编号或名字重复都明确报错并点出文件。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from importlib import resources

from habitus.behavior.kinds.ids import ClassId, KindIdError, Lane
from habitus.behavior.kinds.model import BehaviorKindError, Vocabulary, Word, WordOrigin

WORDS_SCHEMA_VERSION = "behavior_initial_words_v1"
_TOP_KEYS = frozenset({"schema_version", "lane", "words"})
_WORD_KEYS = frozenset({"id", "level", "name", "parent", "definition", "domain"})


def initial_words(lane: Lane) -> Vocabulary:
    """这条 lane 的初始词表（在用的词 0 个，修订号 0）。读一次、进程里缓存。"""

    return _packaged(Lane(lane))


@cache
def _packaged(lane: Lane) -> Vocabulary:
    source = resources.files("habitus.behavior.kinds").joinpath("words").joinpath(f"{lane.value}.json")
    return parse_words(source.read_text(encoding="utf-8"), lane=lane, source=f"words/{lane.value}.json")


def parse_words(text: str, *, lane: Lane, source: str) -> Vocabulary:
    """``source`` 只用于报错时点出文件。"""

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BehaviorKindError(f"{source}: initial words are not valid JSON: {exc}") from exc
    try:
        if not isinstance(raw, Mapping) or set(raw) != _TOP_KEYS:
            raise BehaviorKindError(f"top-level keys must be {sorted(_TOP_KEYS)}")
        if raw["schema_version"] != WORDS_SCHEMA_VERSION:
            raise BehaviorKindError(f"schema_version must be {WORDS_SCHEMA_VERSION}")
        if raw["lane"] != Lane(lane).value:
            raise BehaviorKindError(f"lane must be {Lane(lane).value}")
        if not isinstance(raw["words"], list) or not raw["words"]:
            raise BehaviorKindError("words must be a non-empty list")
        return Vocabulary(Lane(lane), tuple(_word(item, index) for index, item in enumerate(raw["words"], start=1)))
    except (BehaviorKindError, KindIdError) as exc:
        raise BehaviorKindError(f"{source}: {exc}") from exc


def _word(raw: object, index: int) -> Word:
    if not isinstance(raw, Mapping) or set(raw) != _WORD_KEYS:
        raise BehaviorKindError(f"word #{index} keys must be {sorted(_WORD_KEYS)}")
    level = raw["level"]
    if isinstance(level, bool) or not isinstance(level, int):
        raise BehaviorKindError(f"word #{index} level must be an integer")
    parent = raw["parent"]
    return Word(
        id=ClassId.parse(raw["id"]),
        level=level,
        name=raw["name"],
        parent=None if parent is None else ClassId.parse(parent),
        definition=raw["definition"],
        domain=raw["domain"],
        origin=WordOrigin.INITIAL,
    )


__all__ = ["WORDS_SCHEMA_VERSION", "initial_words", "parse_words"]
