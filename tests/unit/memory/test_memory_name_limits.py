"""W1 第 4 项 B1：模型写的记忆名字（拼进文件名的字段）有 UTF-8 字节上限。

上限只卡新写入（模型给出的候选）；磁盘上已有的超长旧名照常列出、读出，不让一个旧文件拖垮整棵树。
超长名字在候选校验这一步就被拒，走"域校验失败 → 把错误发回模型改"的回路，不会落到写盘才 ``OSError``。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from habitus.memory.editor.candidate import MemoryCandidateBatch, MemoryCandidateError
from habitus.memory.model import MAX_MEMORY_NAME_UTF8_BYTES, MemoryAddress, MemoryKind
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from habitus.memory.workflow.failure import MemoryJobFailureKind, memory_job_failure_kind
from habitus.model_client import (
    ChatClient,
    ChatModelConfig,
    ModelResponse,
    ModelStreamEvent,
    ModelStructuredOutputError,
    PreparedChatRequest,
    ProviderCapabilities,
    ProviderConfig,
)
from habitus.model_client.structured import StructuredChatClient
from tests.helpers import codec, document, memory_fields
from tests.model_helpers import prepare_chat_request

LIMIT = MAX_MEMORY_NAME_UTF8_BYTES

# (输出数组, Schema 字段) —— 每个拼进路径的记忆名字段
NAME_FIELDS = [
    ("preferences", "topic"),
    ("entities", "category"),
    ("entities", "name"),
    ("tools", "tool_name"),
    ("events", "event_name"),
    ("intentions", "intent_name"),
]
KIND_OF = {
    "preferences": MemoryKind.PREFERENCE,
    "entities": MemoryKind.ENTITY,
    "tools": MemoryKind.TOOL,
    "events": MemoryKind.EVENT,
    "intentions": MemoryKind.INTENTION,
}


def test_the_limit_leaves_room_for_the_md_suffix_and_the_atomic_temporary_name() -> None:
    """最长派生单段名是原子写的临时名 ``.{name}.md.{32 hex}.tmp``：比名字多 41 字节，必须放得进 255。"""

    assert LIMIT == 200
    assert LIMIT + len(".md") + len(".") + len(".") + 32 + len(".tmp") <= 255


def _item(output: str, name_field: str, value: str) -> dict[str, Any]:
    fields = dict(memory_fields(KIND_OF[output]))
    fields[name_field] = value
    event_date = fields.get("event_date")
    if isinstance(event_date, date):
        fields["event_date"] = event_date.isoformat()
    item: dict[str, Any] = {"page_id": 100, **fields}
    if output == "intentions":
        item["confirmed"] = True
    return item


def _payload(output: str, item: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "profile": [],
        "preferences": [],
        "entities": [],
        "tools": [],
        "events": [],
        "intentions": [],
        "identity_proposals": [],
        "relations": [],
    }
    payload[output] = [item]
    return payload


def _name(size: int, unit: str = "a") -> str:
    width = len(unit.encode("utf-8"))
    count, rest = divmod(size, width)
    return unit * count + "a" * rest


@pytest.mark.parametrize(("output", "name_field"), NAME_FIELDS)
def test_a_name_exactly_at_the_limit_is_written_and_read_back(tmp_path: Path, output: str, name_field: str) -> None:
    name = _name(LIMIT)
    batch = MemoryCandidateBatch.model_validate(_payload(output, _item(output, name_field, name)))
    (candidate,) = getattr(batch, output)
    tree = MemoryTree(tmp_path / "memory")
    written = tree.write(document(KIND_OF[output], fields=dict(candidate.fields)))
    assert tree.read(written.address).fields[name_field] == name
    assert written.address in tree.list_addresses(KIND_OF[output], limit=10)


@pytest.mark.parametrize(("output", "name_field"), NAME_FIELDS)
def test_one_byte_over_the_limit_is_rejected_with_field_size_and_limit(output: str, name_field: str) -> None:
    with pytest.raises(MemoryCandidateError) as rejected:
        MemoryCandidateBatch.model_validate(_payload(output, _item(output, name_field, _name(LIMIT + 1))))
    message = str(rejected.value)
    assert name_field in message
    assert str(LIMIT + 1) in message
    assert str(LIMIT) in message


@pytest.mark.parametrize(
    ("unit", "accepted", "rejected"),
    [
        # 中文 3 字节：66 个字 + 2 个 a = 200；67 个字 = 201
        ("会", "会" * 66 + "aa", "会" * 67),
        # emoji 4 字节：50 个 = 200；50 个 + a = 201
        ("😀", "😀" * 50, "😀" * 50 + "a"),
    ],
)
def test_multibyte_boundaries_count_bytes_not_characters(unit: str, accepted: str, rejected: str) -> None:
    assert unit in accepted
    assert len(accepted.encode("utf-8")) == LIMIT
    assert len(rejected.encode("utf-8")) == LIMIT + 1
    MemoryCandidateBatch.model_validate(_payload("preferences", _item("preferences", "topic", accepted)))
    with pytest.raises(MemoryCandidateError, match=f"topic must be at most {LIMIT} UTF-8 bytes, got {LIMIT + 1}"):
        MemoryCandidateBatch.model_validate(_payload("preferences", _item("preferences", "topic", rejected)))


def test_casefold_growth_is_counted_on_the_on_disk_name() -> None:
    """落盘文件名是归一后的名字：``İ``（2 字节）casefold 成 ``i̇``（3 字节）。原文没超、归一后超了也拒。"""

    name = "İ" * 100
    assert len(name.encode("utf-8")) == LIMIT
    with pytest.raises(MemoryCandidateError, match=f"got {LIMIT + 100}"):
        MemoryCandidateBatch.model_validate(_payload("preferences", _item("preferences", "topic", name)))


def test_an_old_overlong_memory_on_disk_does_not_break_reading_the_tree(tmp_path: Path) -> None:
    """读路径宽容：上限只卡模型新给的候选。改动前的代码能写出的最长名字是 214 字节（临时名 214 + 41 = 255），
    磁盘上已有这样一个旧记忆时，整棵树照样能列出、读出，它自己也读得出。"""

    tree = MemoryTree(tmp_path / "memory")
    long_name = _name(214)
    assert len(long_name.encode("utf-8")) > LIMIT
    old = tree.write(document(MemoryKind.PREFERENCE, fields={"topic": long_name, "content": "- 旧记忆"}))
    other = tree.write(document(MemoryKind.PREFERENCE))
    listed = tree.list_addresses(MemoryKind.PREFERENCE, limit=10)
    assert set(listed) == {old.address, other.address}
    assert tree.read(other.address).fields["topic"] == "回答风格"
    assert tree.read(old.address).fields["topic"] == long_name
    assert MemoryURI.parse(str(MemoryURI.from_address(old.address))).to_address() == old.address
    assert codec().decode(codec().encode(old), expected_address=old.address) == old


@dataclass
class _ScriptedChat:
    """按次回放模型输出的确定性替身。"""

    bodies: list[dict[str, Any]]
    provider_name: str = "test-provider"
    model: str = "test-model"
    is_remote: bool = False
    capabilities: ProviderCapabilities = ProviderCapabilities()
    requests: list[PreparedChatRequest] = field(default_factory=list)

    prepare = staticmethod(prepare_chat_request)

    def complete(self, request: PreparedChatRequest) -> ModelResponse:
        self.requests.append(request)
        body = self.bodies[min(len(self.requests) - 1, len(self.bodies) - 1)]
        return ModelResponse(json.dumps(body, ensure_ascii=False), self.model, self.provider_name, finish_reason="stop")

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:  # pragma: no cover
        return self.complete(request)

    def stream(self, request: PreparedChatRequest) -> Iterator[ModelStreamEvent]:  # pragma: no cover
        raise NotImplementedError(request)

    async def stream_async(self, request: PreparedChatRequest) -> AsyncIterator[ModelStreamEvent]:  # pragma: no cover
        raise NotImplementedError(request)
        yield

    def health_check(self) -> dict[str, object]:  # pragma: no cover
        return {"ok": True}

    async def aclose(self) -> None:  # pragma: no cover
        return None


def _structured(provider: _ScriptedChat, retries: int) -> StructuredChatClient:
    route = ProviderConfig(
        provider="test-provider",
        adapter="test-adapter",
        model="test-model",
        base_url="https://example.com/v1",
        max_retries=0,
        retry_base_delay_seconds=0.01,
        retry_max_delay_seconds=0.1,
        max_concurrent=1,
    )
    return StructuredChatClient(ChatClient(ChatModelConfig(route), provider), validation_retries=retries)


def test_an_overlong_model_name_goes_back_to_the_model_and_the_correction_succeeds() -> None:
    overlong = _payload("entities", _item("entities", "name", "很长的实体名" * 12))  # 216 字节
    fixed = _payload("entities", _item("entities", "name", "实体名"))
    provider = _ScriptedChat([overlong, fixed])
    result = _structured(provider, retries=1).complete_model("抽取记忆", model_class=MemoryCandidateBatch)

    assert result.validation_attempts == 2
    (candidate,) = result.value.entities
    assert candidate.address == MemoryAddress.entity("项目", "实体名")
    correction = provider.requests[1].request.messages[-1].content
    assert isinstance(correction, str)
    assert f"name must be at most {LIMIT} UTF-8 bytes, got 216" in correction


def test_a_model_that_never_shortens_the_name_fails_as_retryable_structured_output_not_oserror() -> None:
    overlong = _payload("events", _item("events", "event_name", _name(300)))
    provider = _ScriptedChat([overlong])
    with pytest.raises(ModelStructuredOutputError) as failed:
        _structured(provider, retries=1).complete_model("抽取记忆", model_class=MemoryCandidateBatch)
    assert len(provider.requests) == 2
    assert not isinstance(failed.value, OSError)
    assert f"event_name must be at most {LIMIT} UTF-8 bytes, got 300" in str(failed.value.__cause__)
    assert memory_job_failure_kind(failed.value) is MemoryJobFailureKind.RETRYABLE


def test_event_directory_dates_are_not_part_of_the_name_budget() -> None:
    item = _item("events", "event_name", _name(LIMIT))
    item["event_date"] = date(2026, 12, 31).isoformat()
    MemoryCandidateBatch.model_validate(_payload("events", item))
