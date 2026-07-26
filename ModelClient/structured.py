"""支持分层 JSON 修复和严格校验的结构化模型调用。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Generic, Literal, TypeVar, cast

from ModelClient.client import ChatClient
from ModelClient.contracts import (
    ChatMessage,
    ChatRequest,
    ModelResponse,
    ModelStructuredOutputError,
    ResponseFormat,
)
from ModelClient.json_parser import JSONParseMode, parse_json_response
from ModelClient.schema_validation import JSONSchemaValidationError, validate_json_schema

T = TypeVar("T")
_ValidationPhase = Literal["response", "json_parse", "json_schema", "domain"]


class _StructuredValidationFailure(ValueError):
    """标记模型结果在哪一层失败，避免把开发者 Schema 错误当成可重试输出错误。"""

    def __init__(self, phase: _ValidationPhase, error: Exception) -> None:
        detail = str(error).replace("\n", " ")
        super().__init__(f"{phase}: {detail}")
        self.phase = phase


@dataclass(frozen=True)
class StructuredResponse(Generic[T]):
    """通过校验的值，以及原始响应和解析审计元数据。"""

    value: T
    response: ModelResponse
    raw_text: str
    parse_mode: JSONParseMode
    validation_attempts: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.validation_attempts, bool)
            or not isinstance(self.validation_attempts, int)
            or self.validation_attempts <= 0
        ):
            raise ValueError("validation_attempts must be a positive integer")

    @property
    def repaired(self) -> bool:
        return self.parse_mode != "strict"


class StructuredChatClient:
    """将 Schema 提示和校验逻辑与传输供应商分离。"""

    def __init__(
        self,
        client: ChatClient,
        *,
        allow_json_repair: bool = True,
        validation_retries: int = 1,
    ) -> None:
        if not isinstance(client, ChatClient):
            raise TypeError("client must be ChatClient")
        if not isinstance(allow_json_repair, bool):
            raise TypeError("allow_json_repair must be boolean")
        if (
            not isinstance(validation_retries, int)
            or isinstance(validation_retries, bool)
            or not 0 <= validation_retries <= 5
        ):
            raise ValueError("validation_retries must be between zero and five")
        self.client = client
        self.allow_json_repair = allow_json_repair
        self.validation_retries = validation_retries

    def complete_json(
        self,
        request: ChatRequest | str,
        *,
        schema: Mapping[str, object],
        name: str = "structured_response",
        validator: Callable[[object], T] | None = None,
    ) -> StructuredResponse[T | object]:
        prepared = self._prepare(request, schema=schema, name=name)
        last_error: _StructuredValidationFailure | None = None
        for attempt in range(self.validation_retries + 1):
            response = self.client.complete(prepared)
            try:
                return self._validate_response(
                    response,
                    schema=schema,
                    validator=validator,
                    validation_attempts=attempt + 1,
                )
            except _StructuredValidationFailure as exc:
                last_error = exc
                if attempt < self.validation_retries:
                    prepared = self._correction_request(prepared, response, exc)
        assert last_error is not None
        raise ModelStructuredOutputError(
            f"model failed {last_error.phase} validation after "
            f"{self.validation_retries + 1} attempt(s)"
        ) from last_error

    async def complete_json_async(
        self,
        request: ChatRequest | str,
        *,
        schema: Mapping[str, object],
        name: str = "structured_response",
        validator: Callable[[object], T] | None = None,
    ) -> StructuredResponse[T | object]:
        prepared = self._prepare(request, schema=schema, name=name)
        last_error: _StructuredValidationFailure | None = None
        for attempt in range(self.validation_retries + 1):
            response = await self.client.complete_async(prepared)
            try:
                return self._validate_response(
                    response,
                    schema=schema,
                    validator=validator,
                    validation_attempts=attempt + 1,
                )
            except _StructuredValidationFailure as exc:
                last_error = exc
                if attempt < self.validation_retries:
                    prepared = self._correction_request(prepared, response, exc)
        assert last_error is not None
        raise ModelStructuredOutputError(
            f"model failed {last_error.phase} validation after "
            f"{self.validation_retries + 1} attempt(s)"
        ) from last_error

    def complete_model(
        self,
        request: ChatRequest | str,
        *,
        model_class: type[T],
        name: str | None = None,
    ) -> StructuredResponse[T]:
        schema, validator = _model_contract(model_class)
        result = self.complete_json(
            request,
            schema=schema,
            name=name or model_class.__name__,
            validator=validator,
        )
        return cast(StructuredResponse[T], result)

    async def complete_model_async(
        self,
        request: ChatRequest | str,
        *,
        model_class: type[T],
        name: str | None = None,
    ) -> StructuredResponse[T]:
        schema, validator = _model_contract(model_class)
        result = await self.complete_json_async(
            request,
            schema=schema,
            name=name or model_class.__name__,
            validator=validator,
        )
        return cast(StructuredResponse[T], result)

    def _prepare(
        self,
        request: ChatRequest | str,
        *,
        schema: Mapping[str, object],
        name: str,
    ) -> ChatRequest:
        if isinstance(request, str):
            if not request.strip():
                raise ValueError("structured model prompt cannot be empty")
            request = ChatRequest(messages=(ChatMessage(role="user", content=request),))
        if not isinstance(request, ChatRequest):
            raise TypeError("structured request must be ChatRequest or non-empty text")
        if not isinstance(schema, Mapping) or not schema:
            raise ValueError("structured output schema must be a non-empty object")
        try:
            schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True, indent=2)
        except (TypeError, ValueError) as exc:
            raise ValueError("structured output schema must be JSON serializable") from exc
        instruction = ChatMessage(
            role="system",
            content=(
                "Return exactly one JSON value that satisfies the following JSON Schema. "
                "Do not add Markdown fences, commentary, or fields not declared by the schema.\n"
                f"{schema_text}"
            ),
        )
        response_format = ResponseFormat(name=name, schema=schema, strict=True)
        return replace(
            request,
            messages=(instruction, *request.messages),
            response_format=response_format,
        )

    @staticmethod
    def _correction_request(
        request: ChatRequest,
        response: ModelResponse,
        error: _StructuredValidationFailure,
    ) -> ChatRequest:
        messages = list(request.messages)
        if response.content:
            messages.append(ChatMessage(role="assistant", content=response.content))
        detail = str(error)[:768]
        messages.append(
            ChatMessage(
                role="user",
                content=(
                    f"The previous JSON response failed validation ({detail}). "
                    "Return exactly one corrected JSON value only. Preserve every already-valid value unless "
                    "the reported constraint requires changing it. Do not stringify arrays or objects, wrap "
                    "values in arrays, substitute defaults for invalid enums, ignore unknown fields, or invent "
                    "missing facts to bypass validation. Use only information from the original request and fix "
                    "the reported syntax or schema problem."
                ),
            )
        )
        return replace(request, messages=tuple(messages))

    def _validate_response(
        self,
        response: ModelResponse,
        *,
        schema: Mapping[str, object],
        validator: Callable[[object], T] | None,
        validation_attempts: int,
    ) -> StructuredResponse[T | object]:
        if response.finish_reason == "length":
            raise _StructuredValidationFailure(
                "response",
                ValueError("structured model response was truncated"),
            )
        if response.finish_reason in {"content_filter", "safety"}:
            raise _StructuredValidationFailure(
                "response",
                ValueError("structured model response was blocked by content safety"),
            )
        if not response.content:
            raise _StructuredValidationFailure(
                "response",
                ValueError("structured model response has no text content"),
            )
        try:
            parsed = parse_json_response(response.content, allow_repair=self.allow_json_repair)
        except ValueError as exc:
            raise _StructuredValidationFailure("json_parse", exc) from exc
        try:
            validate_json_schema(parsed.value, schema)
        except JSONSchemaValidationError as exc:
            raise _StructuredValidationFailure("json_schema", exc) from exc
        value: T | object = parsed.value
        if validator is not None:
            try:
                value = validator(parsed.value)
            except (TypeError, ValueError) as exc:
                raise _StructuredValidationFailure("domain", exc) from exc
        return StructuredResponse(
            value=value,
            response=response,
            raw_text=response.content,
            parse_mode=parsed.mode,
            validation_attempts=validation_attempts,
        )


def _model_contract(model_class: type[T]) -> tuple[Mapping[str, object], Callable[[object], T]]:
    schema_builder = getattr(model_class, "model_json_schema", None)
    model_validator = getattr(model_class, "model_validate", None)
    if not callable(schema_builder) or not callable(model_validator):
        raise TypeError("model_class must provide model_json_schema() and model_validate()")
    schema = schema_builder()
    if not isinstance(schema, Mapping):
        raise TypeError("model_json_schema() must return an object")

    def validate(value: object) -> T:
        return cast(T, model_validator(value))

    return schema, validate


__all__ = ["StructuredChatClient", "StructuredResponse"]
