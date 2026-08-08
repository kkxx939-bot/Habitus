"""Adapter 输出进入 Evidence 领域时唯一执行的业务策略。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from behavior._validation import json_value_snapshot, strict_utc
from behavior.config import BehaviorEvidenceConfig
from behavior.errors import (
    BehaviorAdapterCapabilityError,
    BehaviorEvidenceClockError,
    BehaviorEvidenceSchemaError,
)
from behavior.evidence.adapter import BehaviorSemanticInput
from behavior.evidence.content import BehaviorSemanticContent
from behavior.evidence.provenance import BehaviorOriginKind
from behavior.evidence.specs import record_spec
from behavior.evidence.trust import (
    AdapterOutputContract,
    BehaviorAdapterCapability,
    BehaviorTimeMode,
)


class EvidenceBatchCapacityRejection(Exception):
    def __init__(self, indexes: tuple[int, ...]) -> None:
        self.indexes = indexes


class EvidenceBatchClockRejection(BehaviorEvidenceClockError):
    def __init__(self, indexes: tuple[int, ...]) -> None:
        super().__init__("one or more Evidence items failed the clock policy")
        self.indexes = indexes


@dataclass(frozen=True, slots=True)
class ValidatedEvidenceInput:
    value: BehaviorSemanticInput
    output_contract: AdapterOutputContract


class EvidencePolicy:
    def __init__(self, config: BehaviorEvidenceConfig) -> None:
        self.config = config

    def validate_batch(
        self,
        items: tuple[BehaviorSemanticInput, ...],
        capability: BehaviorAdapterCapability,
        *,
        producer_digest: str,
        now: datetime,
    ) -> tuple[ValidatedEvidenceInput, ...]:
        if not items or any(not isinstance(item, BehaviorSemanticInput) for item in items):
            raise BehaviorEvidenceSchemaError("Adapter returned an invalid semantic batch")
        if len(items) > capability.maximum_batch_size:
            raise BehaviorAdapterCapabilityError("Adapter exceeded its declared batch capacity")
        if len(items) > self.config.max_batch_size:
            raise EvidenceBatchCapacityRejection(tuple(range(len(items))))
        self._validate_batch_identity(items, producer_digest)
        moment = strict_utc(now, "clock.now")
        validated: list[ValidatedEvidenceInput] = []
        clock_rejections: list[int] = []
        for index, item in enumerate(items):
            contract = self._validate_item(item, capability)
            try:
                self._validate_time(item.content, capability.time_mode, moment)
            except BehaviorEvidenceClockError:
                clock_rejections.append(index)
            validated.append(ValidatedEvidenceInput(item, contract))
        if clock_rejections:
            raise EvidenceBatchClockRejection(tuple(clock_rejections))
        return tuple(validated)

    def _validate_item(
        self,
        item: BehaviorSemanticInput,
        capability: BehaviorAdapterCapability,
    ) -> AdapterOutputContract:
        content = item.content
        source = item.source
        if source.origin_kind is BehaviorOriginKind.CONVERSATION_PROJECTION:
            raise BehaviorAdapterCapabilityError(
                "external Adapter cannot emit conversation projection origin"
            )
        contract = capability.match(
            origin_kind=source.origin_kind,
            record_kind=content.record_kind,
            modality=content.modality,
            subject_role=content.subject_role,
            actor_role=content.actor_role,
        )
        if contract is None:
            raise BehaviorAdapterCapabilityError(
                "Adapter output has no exact AdapterOutputContract"
            )
        spec = record_spec(content.record_kind)
        if not isinstance(content.payload, spec.payload_codec.payload_type):
            raise BehaviorEvidenceSchemaError("RecordKind payload does not match RecordSpec")
        try:
            spec.role_policy.validate(content)
            self._validate_capacities(content, item)
        except (TypeError, ValueError) as exc:
            raise BehaviorEvidenceSchemaError("Evidence output violates its domain policy") from exc
        return contract

    def _validate_capacities(
        self,
        content: BehaviorSemanticContent,
        item: BehaviorSemanticInput,
    ) -> None:
        source = item.source
        limits = self.config
        counts = (
            (content.evidence_refs, limits.max_evidence_refs),
            (content.object_refs, limits.max_object_refs),
            (content.entity_refs, limits.max_entity_refs),
            (source.parent_source_event_refs, limits.max_parent_source_refs),
            (source.correlation_refs, limits.max_correlation_refs),
            (source.causal_refs, limits.max_causal_refs),
        )
        if any(len(values) > maximum for values, maximum in counts):
            raise ValueError("Evidence collection exceeds its configured capacity")
        if content.event_time_uncertainty_ms > limits.max_event_time_uncertainty_ms:
            raise ValueError("event uncertainty exceeds its configured capacity")
        identifiers = (
            content.clock_domain,
            content.scene_ref,
            content.location_ref,
            *content.object_refs,
            *content.entity_refs,
            source.source_event_ref.namespace,
            source.source_event_ref.value,
            source.stream_ref.namespace,
            source.stream_ref.value,
        )
        if any(value is not None and len(value) > limits.max_identifier_chars for value in identifiers):
            raise ValueError("Evidence identifier exceeds its configured capacity")
        references = [source.source_ref]
        references.extend(reference.reference for reference in content.evidence_refs)
        references.extend(reference.source_system_ref for reference in content.evidence_refs)
        references.extend(reference.reference for reference in source.causal_refs)
        if any(value is not None and len(value) > limits.max_reference_chars for value in references):
            raise ValueError("Evidence reference exceeds its configured capacity")
        spec = record_spec(content.record_kind)
        payload = spec.payload_codec.encode(content.payload)
        json_value_snapshot(
            payload,
            "payload",
            maximum_chars=limits.max_payload_chars,
            maximum_items=limits.max_payload_items,
            maximum_depth=limits.max_payload_depth,
        )
        for name in spec.payload_codec.identifier_fields:
            value = payload[name]
            values = value if isinstance(value, tuple) else (value,)
            if any(item is not None and len(item) > limits.max_identifier_chars for item in values):
                raise ValueError("Payload identifier exceeds its configured capacity")
        for name in spec.payload_codec.reference_fields:
            value = payload[name]
            if value is not None and len(value) > limits.max_reference_chars:
                raise ValueError("Payload reference exceeds its configured capacity")
        for name in spec.payload_codec.text_fields:
            value = payload[name]
            if value is not None and len(value) > limits.max_text_chars:
                raise ValueError("Payload text exceeds its configured capacity")

    def _validate_time(
        self,
        content: BehaviorSemanticContent,
        mode: BehaviorTimeMode,
        now: datetime,
    ) -> None:
        start = strict_utc(content.event_time_start, "event_time_start")
        end = strict_utc(content.event_time_end, "event_time_end")
        if end < start:
            raise BehaviorEvidenceClockError("event end precedes start")
        if (end - start).total_seconds() > self.config.max_record_duration_seconds:
            raise BehaviorEvidenceClockError("event duration exceeds the configured boundary")
        uncertainty = timedelta(milliseconds=content.event_time_uncertainty_ms)
        earliest = start - uncertainty
        latest = end + uncertainty
        for reference in content.evidence_refs:
            if reference.event_time_end < earliest or reference.event_time_start > latest:
                raise BehaviorEvidenceClockError(
                    "EvidenceReference does not overlap the event interval"
                )
        future_limit = now + timedelta(seconds=self.config.max_future_event_skew_seconds)
        if latest > future_limit:
            raise BehaviorEvidenceClockError("event interval exceeds the allowed future skew")
        if mode is BehaviorTimeMode.LIVE:
            past_limit = now - timedelta(seconds=self.config.max_live_event_age_seconds)
            if earliest < past_limit:
                raise BehaviorEvidenceClockError("LIVE event interval exceeds the allowed age")

    @staticmethod
    def _validate_batch_identity(
        items: tuple[BehaviorSemanticInput, ...],
        producer_digest: str,
    ) -> None:
        source_keys: set[tuple[str, str, str, int]] = set()
        stream_keys: set[tuple[str, str, str, int, int, int]] = set()
        for item in items:
            source = item.source
            source_key = (
                producer_digest,
                source.source_event_ref.namespace,
                source.source_event_ref.value,
                source.source_item_index,
            )
            stream_key = (
                producer_digest,
                source.stream_ref.namespace,
                source.stream_ref.value,
                source.stream_ref.generation,
                source.source_sequence,
                source.source_item_index,
            )
            if source_key in source_keys or stream_key in stream_keys:
                raise BehaviorEvidenceSchemaError("batch contains duplicate Evidence identities")
            source_keys.add(source_key)
            stream_keys.add(stream_key)
