"""Behavior 第一层 SQLite Schema 与机械漂移校验。"""

from __future__ import annotations

import re
import sqlite3

from behavior.errors import BehaviorStoreError

BEHAVIOR_SCHEMA_VERSION = "behavior_first_layer_v1"

TABLE_SQL = {
    "behavior_metadata": """CREATE TABLE behavior_metadata (
        metadata_key TEXT PRIMARY KEY NOT NULL,
        metadata_value TEXT NOT NULL
    )""",
    "behavior_evidence_records": """CREATE TABLE behavior_evidence_records (
        evidence_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        evidence_record_id TEXT UNIQUE NOT NULL,
        producer_fingerprint TEXT NOT NULL,
        capability_digest TEXT NOT NULL,
        source_trust TEXT NOT NULL,
        origin_kind TEXT NOT NULL,
        source_event_namespace TEXT NOT NULL,
        source_event_value TEXT NOT NULL,
        source_item_index INTEGER NOT NULL CHECK (source_item_index >= 0),
        stream_namespace TEXT NOT NULL,
        stream_value TEXT NOT NULL,
        stream_generation INTEGER NOT NULL CHECK (stream_generation >= 0),
        source_sequence INTEGER NOT NULL CHECK (source_sequence >= 0),
        record_kind TEXT NOT NULL,
        subject_role TEXT NOT NULL,
        actor_role TEXT,
        event_time_start TEXT NOT NULL,
        event_time_end TEXT NOT NULL,
        ingested_at TEXT NOT NULL,
        semantic_digest TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        encoded_digest TEXT NOT NULL,
        content_json TEXT NOT NULL
    )""",
    "behavior_evidence_correlations": """CREATE TABLE behavior_evidence_correlations (
        evidence_record_id TEXT NOT NULL,
        member_order INTEGER NOT NULL CHECK (member_order >= 0),
        namespace TEXT NOT NULL,
        value TEXT NOT NULL,
        root_value TEXT,
        PRIMARY KEY (evidence_record_id, member_order),
        UNIQUE (evidence_record_id, namespace, value, root_value),
        FOREIGN KEY (evidence_record_id) REFERENCES behavior_evidence_records(evidence_record_id)
    )""",
    "behavior_evidence_causal_refs": """CREATE TABLE behavior_evidence_causal_refs (
        evidence_record_id TEXT NOT NULL,
        member_order INTEGER NOT NULL CHECK (member_order >= 0),
        kind TEXT NOT NULL,
        reference TEXT NOT NULL,
        reference_digest TEXT NOT NULL,
        PRIMARY KEY (evidence_record_id, member_order),
        UNIQUE (evidence_record_id, kind, reference_digest),
        FOREIGN KEY (evidence_record_id) REFERENCES behavior_evidence_records(evidence_record_id)
    )""",
    "behavior_evidence_parent_sources": """CREATE TABLE behavior_evidence_parent_sources (
        evidence_record_id TEXT NOT NULL,
        member_order INTEGER NOT NULL CHECK (member_order >= 0),
        namespace TEXT NOT NULL,
        value TEXT NOT NULL,
        identity_digest TEXT NOT NULL,
        PRIMARY KEY (evidence_record_id, member_order),
        UNIQUE (evidence_record_id, identity_digest),
        FOREIGN KEY (evidence_record_id) REFERENCES behavior_evidence_records(evidence_record_id)
    )""",
    "behavior_evidence_ingress_receipts": """CREATE TABLE behavior_evidence_ingress_receipts (
        delivery_id TEXT PRIMARY KEY NOT NULL,
        request_digest TEXT NOT NULL,
        adapter_name TEXT NOT NULL,
        adapter_fingerprint TEXT NOT NULL,
        capability_digest TEXT NOT NULL,
        status TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        encoded_digest TEXT NOT NULL,
        content_json TEXT NOT NULL
    )""",
    "behavior_claims": """CREATE TABLE behavior_claims (
        claim_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id TEXT UNIQUE NOT NULL,
        evidence_record_id TEXT NOT NULL,
        claim_kind TEXT NOT NULL,
        subject_role TEXT NOT NULL,
        actor_role TEXT,
        semantic_family TEXT,
        predicate TEXT NOT NULL,
        time_start TEXT NOT NULL,
        time_end TEXT NOT NULL,
        derivation_class TEXT NOT NULL,
        source_epistemic_class TEXT NOT NULL,
        effective_confidence REAL NOT NULL CHECK (effective_confidence >= 0.0 AND effective_confidence <= 1.0),
        semantic_fingerprint TEXT NOT NULL,
        created_at TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        encoded_digest TEXT NOT NULL,
        content_json TEXT NOT NULL,
        FOREIGN KEY (evidence_record_id) REFERENCES behavior_evidence_records(evidence_record_id)
    )""",
    "claim_normalization_attempts": """CREATE TABLE claim_normalization_attempts (
        attempt_id TEXT PRIMARY KEY NOT NULL,
        processing_identity TEXT NOT NULL,
        evidence_record_id TEXT NOT NULL,
        normalizer_name TEXT NOT NULL,
        normalizer_fingerprint TEXT NOT NULL,
        lane TEXT NOT NULL,
        attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
        status TEXT NOT NULL,
        retryable INTEGER NOT NULL CHECK (retryable IN (0, 1)),
        completed_at TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        encoded_digest TEXT NOT NULL,
        content_json TEXT NOT NULL,
        UNIQUE (processing_identity, attempt_number),
        FOREIGN KEY (evidence_record_id) REFERENCES behavior_evidence_records(evidence_record_id)
    )""",
    "claim_normalization_receipts": """CREATE TABLE claim_normalization_receipts (
        processing_identity TEXT PRIMARY KEY NOT NULL,
        evidence_record_id TEXT NOT NULL,
        lane TEXT NOT NULL,
        normalizer_fingerprint TEXT NOT NULL,
        planner_policy_digest TEXT NOT NULL,
        compatibility_policy_digest TEXT NOT NULL,
        binding_policy_digest TEXT NOT NULL,
        confidence_policy_digest TEXT NOT NULL,
        status TEXT NOT NULL,
        completed_at TEXT NOT NULL,
        publication_recorded_at TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        encoded_digest TEXT NOT NULL,
        content_json TEXT NOT NULL,
        FOREIGN KEY (evidence_record_id) REFERENCES behavior_evidence_records(evidence_record_id)
    )""",
    "claim_receipt_members": """CREATE TABLE claim_receipt_members (
        processing_identity TEXT NOT NULL,
        member_kind TEXT NOT NULL CHECK (member_kind IN ('ATTEMPT', 'CLAIM')),
        member_order INTEGER NOT NULL CHECK (member_order >= 0),
        attempt_id TEXT,
        claim_id TEXT,
        PRIMARY KEY (processing_identity, member_kind, member_order),
        UNIQUE (processing_identity, attempt_id),
        UNIQUE (processing_identity, claim_id),
        CHECK ((member_kind = 'ATTEMPT' AND attempt_id IS NOT NULL AND claim_id IS NULL) OR
               (member_kind = 'CLAIM' AND claim_id IS NOT NULL AND attempt_id IS NULL)),
        FOREIGN KEY (processing_identity) REFERENCES claim_normalization_receipts(processing_identity),
        FOREIGN KEY (attempt_id) REFERENCES claim_normalization_attempts(attempt_id),
        FOREIGN KEY (claim_id) REFERENCES behavior_claims(claim_id)
    )""",
}

INDEX_SQL = {
    "idx_evidence_sequence": "CREATE INDEX idx_evidence_sequence ON behavior_evidence_records(evidence_sequence)",
    "idx_evidence_source_identity": """CREATE UNIQUE INDEX idx_evidence_source_identity
        ON behavior_evidence_records(producer_fingerprint, source_event_namespace, source_event_value, source_item_index)""",
    "idx_evidence_stream_identity": """CREATE UNIQUE INDEX idx_evidence_stream_identity
        ON behavior_evidence_records(producer_fingerprint, stream_namespace, stream_value, stream_generation, source_sequence, source_item_index)""",
    "idx_evidence_event_time": "CREATE INDEX idx_evidence_event_time ON behavior_evidence_records(event_time_start, evidence_sequence)",
    "idx_evidence_record_kind_time": "CREATE INDEX idx_evidence_record_kind_time ON behavior_evidence_records(record_kind, event_time_start, evidence_sequence)",
    "idx_evidence_source_event": "CREATE INDEX idx_evidence_source_event ON behavior_evidence_records(source_event_namespace, source_event_value, evidence_sequence)",
    "idx_evidence_correlation": "CREATE INDEX idx_evidence_correlation ON behavior_evidence_correlations(namespace, value, root_value, evidence_record_id)",
    "idx_evidence_causal": "CREATE INDEX idx_evidence_causal ON behavior_evidence_causal_refs(kind, reference_digest, evidence_record_id)",
    "idx_claim_sequence": "CREATE INDEX idx_claim_sequence ON behavior_claims(claim_sequence)",
    "idx_claim_evidence": "CREATE INDEX idx_claim_evidence ON behavior_claims(evidence_record_id, claim_sequence)",
    "idx_claim_event_time": "CREATE INDEX idx_claim_event_time ON behavior_claims(time_start, claim_sequence)",
    "idx_claim_kind_time": "CREATE INDEX idx_claim_kind_time ON behavior_claims(claim_kind, time_start, claim_sequence)",
    "idx_claim_semantic_time": "CREATE INDEX idx_claim_semantic_time ON behavior_claims(semantic_fingerprint, time_start, claim_sequence)",
    "idx_claim_derivation_time": "CREATE INDEX idx_claim_derivation_time ON behavior_claims(derivation_class, time_start, claim_sequence)",
    "idx_attempt_processing_latest": "CREATE INDEX idx_attempt_processing_latest ON claim_normalization_attempts(processing_identity, attempt_number DESC)",
    "idx_attempt_route_latest": "CREATE INDEX idx_attempt_route_latest ON claim_normalization_attempts(evidence_record_id, lane, normalizer_name, attempt_number DESC)",
    "idx_receipt_evidence": "CREATE INDEX idx_receipt_evidence ON claim_normalization_receipts(evidence_record_id, lane)",
}

def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        existing = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if not existing:
            for statement in TABLE_SQL.values():
                connection.execute(statement)
            for statement in INDEX_SQL.values():
                connection.execute(statement)
            connection.execute(
                "INSERT INTO behavior_metadata(metadata_key, metadata_value) VALUES (?, ?)",
                ("schema_version", BEHAVIOR_SCHEMA_VERSION),
            )
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    validate_schema(connection)


def validate_schema(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if tables != set(TABLE_SQL):
        raise BehaviorStoreError("Behavior database table set has drifted")
    metadata = connection.execute(
        "SELECT metadata_value FROM behavior_metadata WHERE metadata_key=?",
        ("schema_version",),
    ).fetchone()
    if metadata is None or metadata[0] != BEHAVIOR_SCHEMA_VERSION:
        raise BehaviorStoreError("Behavior database schema version is incompatible")
    for table, columns in EXPECTED_COLUMNS.items():
        actual = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
        if actual != columns:
            raise BehaviorStoreError(f"Behavior database columns drifted for {table}")
        stored = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if stored is None or _normalize_sql(stored[0]) != _normalize_sql(TABLE_SQL[table]):
            raise BehaviorStoreError(f"Behavior database table SQL drifted for {table}")
    indexes = {
        row[0]: row[1]
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        )
    }
    if set(indexes) != set(INDEX_SQL):
        raise BehaviorStoreError("Behavior database index set has drifted")
    for name, sql in INDEX_SQL.items():
        if _normalize_sql(indexes[name]) != _normalize_sql(sql):
            raise BehaviorStoreError(f"Behavior database index SQL drifted for {name}")
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise BehaviorStoreError("Behavior database foreign key check failed")


def _normalize_sql(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _column_names(sql: str) -> list[str]:
    body = sql[sql.index("(") + 1 : sql.rindex(")")]
    result: list[str] = []
    for raw in body.splitlines():
        line = raw.strip().rstrip(",")
        if not line or line.startswith(("(", "PRIMARY ", "UNIQUE ", "FOREIGN ", "CHECK ")):
            continue
        name = line.split()[0]
        if name not in result:
            result.append(name)
    return result


EXPECTED_COLUMNS = {
    name: tuple(_column_names(sql)) for name, sql in TABLE_SQL.items()
}
