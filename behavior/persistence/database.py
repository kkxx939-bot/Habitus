"""Behavior 第一层数据库生命周期与只读健康快照。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from behavior.config import BehaviorConfig
from behavior.errors import BehaviorStoreError
from behavior.persistence.connection import BehaviorSQLiteConnection
from behavior.persistence.schema import BEHAVIOR_SCHEMA_VERSION, initialize_schema, validate_schema


class BehaviorDatabase:
    def __init__(
        self,
        root: Path,
        *,
        config: BehaviorConfig,
        initialize: bool = False,
    ) -> None:
        if not isinstance(config, BehaviorConfig):
            raise TypeError("config must be BehaviorConfig")
        self.root = root
        self.config = config
        self.connection = BehaviorSQLiteConnection(root, config=config.store)
        if initialize:
            self.initialize()

    @property
    def initialized(self) -> bool:
        return self.connection.initialized

    def initialize(self) -> Path:
        path = self.connection.initialize_path()
        try:
            with self.connection.read() as connection:
                initialize_schema(connection)
                if self.connection.database_size(connection) > self.config.store.max_database_bytes:
                    raise BehaviorStoreError("Behavior schema exceeds configured database capacity")
        except sqlite3.Error as exc:
            raise BehaviorStoreError("Behavior schema initialization failed") from exc
        return path

    def readiness(self) -> tuple[bool, str]:
        if not self.initialized:
            return False, "not_initialized"
        try:
            with self.connection.read() as connection:
                validate_schema(connection)
        except Exception as exc:
            return False, type(exc).__name__
        return True, BEHAVIOR_SCHEMA_VERSION

    def health_snapshot(self) -> dict[str, str | int | bool]:
        with self.connection.read() as connection:
            validate_schema(connection)
            counts = {
                "evidence_count": connection.execute(
                    "SELECT COUNT(*) FROM behavior_evidence_records"
                ).fetchone()[0],
                "claim_count": connection.execute("SELECT COUNT(*) FROM behavior_claims").fetchone()[0],
                "attempt_count": connection.execute(
                    "SELECT COUNT(*) FROM claim_normalization_attempts"
                ).fetchone()[0],
                "receipt_count": connection.execute(
                    "SELECT COUNT(*) FROM claim_normalization_receipts"
                ).fetchone()[0],
            }
            size = self.connection.database_size(connection)
        return {
            "schema_version": BEHAVIOR_SCHEMA_VERSION,
            **counts,
            "database_size_warning": size >= int(self.config.store.max_database_bytes * 0.9),
        }

    def assert_ready(self) -> None:
        ready, detail = self.readiness()
        if not ready:
            raise BehaviorStoreError(f"Behavior database is not ready: {detail}")

    def close(self) -> None:
        self.connection.close()
