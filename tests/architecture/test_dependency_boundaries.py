"""已经确认的新架构边界不得被兼容层或反向依赖破坏。"""

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOTS = (
    "Config",
    "Runtime",
    "ModelClient",
    "pre",
    "memory",
    "behavior",
    "infrastructure",
    "foundation",
)
RETIRED_NAMES = (
    "EvidenceSlice",
    "MemoryEditSource",
    "MemoryEditBatch",
    "SessionArchive",
    "ActionPolicy",
    "ContextURI",
    "memoryos://",
    "viking://",
)


def production_files() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for root in PRODUCTION_ROOTS
            for path in (REPOSITORY_ROOT / root).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    )


def imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_retired_memory_contracts_and_uri_schemes_do_not_reappear() -> None:
    violations = []
    for path in production_files():
        source = path.read_text(encoding="utf-8")
        for retired in RETIRED_NAMES:
            if retired in source:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {retired}")
    assert violations == []


@pytest.mark.parametrize(
    "package",
    ("ModelClient", "pre", "memory", "behavior", "infrastructure", "foundation"),
)
def test_domain_packages_never_import_top_level_runtime(package: str) -> None:
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / package).rglob("*.py")
        if "Runtime" in imported_roots(path)
    ]
    assert violations == []


def test_pre_contains_only_conversation_schema_and_no_storage_dependency() -> None:
    python_files = tuple((REPOSITORY_ROOT / "pre").rglob("*.py"))
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in python_files
        if imported_roots(path) & {"memory", "infrastructure", "Runtime", "Config"}
    ]
    assert violations == []
    assert not (REPOSITORY_ROOT / "pre" / "session").exists()


def test_config_root_does_not_branch_on_specific_model_or_vector_adapters() -> None:
    source = (REPOSITORY_ROOT / "Config" / "root.py").read_text(encoding="utf-8")
    for adapter_name in ("vikingdb", "ark_multimodal", "openai_compatible_chat"):
        assert adapter_name not in source


def test_local_setup_cli_renders_registry_without_vendor_branches() -> None:
    source = (REPOSITORY_ROOT / "integrations" / "local_service" / "cli.py").read_text(
        encoding="utf-8"
    )
    for vendor_identifier in (
        "deepseek",
        "volcengine",
        "aliyun",
        "qwen3-rerank",
        "vikingdb",
        "openai_compatible_chat",
        "ark_multimodal",
    ):
        assert vendor_identifier not in source.casefold()


def test_memory_kernel_never_imports_the_local_product_shell() -> None:
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "memory").rglob("*.py")
        if "integrations" in imported_roots(path)
    ]
    assert violations == []


def test_memory_schema_contains_exactly_the_six_confirmed_l2_kinds() -> None:
    definitions = REPOSITORY_ROOT / "memory" / "schema" / "definitions"
    assert {path.stem for path in definitions.glob("*.yaml")} == {
        "profile",
        "preferences",
        "entities",
        "tools",
        "events",
        "intentions",
    }


def test_behavior_and_memory_keep_strict_peer_dependency_boundaries() -> None:
    behavior_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "behavior").rglob("*.py")
        if imported_roots(path) & {"memory", "integrations", "Runtime", "Config"}
    ]
    memory_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "memory").rglob("*.py")
        if "behavior" in imported_roots(path)
    ]
    integration_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "integrations").rglob("*.py")
        if "behavior" in imported_roots(path)
    ]
    assert behavior_violations == []
    assert memory_violations == []
    assert integration_violations == []


def test_behavior_has_no_identity_partition_model() -> None:
    disallowed_partition_terms = (
        "owner_ref",
        "owner_binding",
        "owner_identity",
        "user_id",
        "tenant_id",
        "account_id",
    )
    violations = []
    for path in (REPOSITORY_ROOT / "behavior").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for term in disallowed_partition_terms:
            if term in source:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {term}")
    assert violations == []


def test_behavior_does_not_extend_memory_kind() -> None:
    source = (REPOSITORY_ROOT / "memory" / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    memory_kind = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MemoryKind"
    )
    member_names = {
        target.id
        for node in memory_kind.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "BEHAVIOR" not in member_names
    memory_tree_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPOSITORY_ROOT / "memory" / "tree").rglob("*.py")
    )
    assert "behaviors" not in memory_tree_source
    assert not (REPOSITORY_ROOT / "memory" / "schema" / "definitions" / "behavior.yaml").exists()


def test_behavior_old_first_layer_contracts_are_removed() -> None:
    retired = (
        "ConfirmedOwnerBinding",
        "OwnerScopedSemanticRecord",
        "SemanticRecordInput",
        "SemanticEvidenceBundle",
        "SemanticEvidenceBundleAssembler",
        "EvidenceBundleState",
        "EvidenceManifest",
        "EvidenceWatermark",
        "allowed_lateness",
        "LATE_REJECTED",
        "ClaimAdmissionPolicy",
        "ClaimAdmissionDecision",
        "ClaimBatch",
        "ClaimProcessingLane",
        "ClaimProcessingReceipt",
        "ClaimPipelineService",
        "ClaimNormalizationRouter",
        "EvidenceService",
        "SQLiteBehaviorEvidenceClaimStore",
        "EvidenceWindow",
    )
    violations = []
    for path in (REPOSITORY_ROOT / "behavior").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for symbol in retired:
            if symbol in source:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {symbol}")
    assert violations == []
    assert not (REPOSITORY_ROOT / "behavior" / "owner.py").exists()
    assert not (REPOSITORY_ROOT / "behavior" / "ingress").exists()
    behavior_source = "\n".join(
        path.read_text(encoding="utf-8") for path in (REPOSITORY_ROOT / "behavior").rglob("*.py")
    )
    assert "evidence_claims.sqlite3" not in behavior_source


def test_behavior_has_no_raw_media_or_later_layer_implementation() -> None:
    forbidden = (
        "CAMERA_FRAME",
        "AUDIO_CLIP",
        "EventCandidate",
        "CanonicalEvent",
        "EventResolver",
        "EventRelationGraph",
        "EventTree",
        "Episode",
        "BehaviorHypothesis",
        "Opportunity",
        "BehaviorCase",
        "BehaviorPattern",
        "Prediction",
        "ActionPolicy",
        "PolicyGate",
        "AgentActionDirective",
        "MemoryBehaviorRouter",
        "ConversationBehaviorProjector",
        "ProjectionWorker",
        "ProjectionJob",
    )
    violations = []
    for path in (REPOSITORY_ROOT / "behavior").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for symbol in forbidden:
            if symbol in source:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {symbol}")
    assert violations == []


def test_behavior_production_has_no_placeholder_or_suppression_escape_hatches() -> None:
    forbidden = ("NotImplementedError", "# noqa", "# type: ignore")
    violations = []
    for path in (REPOSITORY_ROOT / "behavior").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        if any(isinstance(node, ast.Pass) for node in ast.walk(tree)):
            violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: pass")
        for marker in forbidden:
            if marker in source:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {marker}")
    assert violations == []


def test_behavior_claim_proposal_keeps_system_fields_out_of_model_contract() -> None:
    proposal_source = (REPOSITORY_ROOT / "behavior" / "claim" / "proposal.py").read_text(
        encoding="utf-8"
    )
    proposal_tree = ast.parse(proposal_source)
    system_fields = {
        "subject_role",
        "actor_role",
        "time_start",
        "time_end",
        "source_epistemic_class",
        "derivation_class",
        "evidence_record_id",
        "source_trust",
        "compatibility_policy_digest",
        "claim_id",
        "created_at",
    }
    proposal = next(
        node for node in proposal_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ClaimSemanticProposal"
    )
    names = {
        node.target.id
        for node in proposal.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert names.isdisjoint(system_fields)
    assert "ClaimNormalizationConfig()" not in proposal_source
    assert '"additionalProperties": False' in proposal_source


def test_behavior_sqlite_has_two_sequences_foreign_keys_and_atomic_receipt_members() -> None:
    model_source = (REPOSITORY_ROOT / "behavior" / "claim" / "model.py").read_text(encoding="utf-8")
    schema_source = (REPOSITORY_ROOT / "behavior" / "persistence" / "schema.py").read_text(
        encoding="utf-8"
    )
    model_tree = ast.parse(model_source)
    claim = next(node for node in model_tree.body if isinstance(node, ast.ClassDef) and node.name == "BehaviorClaim")
    claim_fields = {
        target.id
        for node in claim.body
        if isinstance(node, ast.AnnAssign) and isinstance((target := node.target), ast.Name)
    }
    assert "event_id" not in claim_fields
    assert "claim_sequence INTEGER PRIMARY KEY AUTOINCREMENT" in schema_source
    assert "evidence_sequence INTEGER PRIMARY KEY AUTOINCREMENT" in schema_source
    assert "CREATE TABLE claim_receipt_members" in schema_source
    assert "FOREIGN KEY (evidence_record_id) REFERENCES behavior_evidence_records" in schema_source
    assert "behavior_first_layer_v1" in schema_source


def test_behavior_model_client_and_processing_lock_boundaries_are_mechanical() -> None:
    normalizer_source = (REPOSITORY_ROOT / "behavior" / "claim" / "normalizer.py").read_text(
        encoding="utf-8"
    )
    runtime_source = (REPOSITORY_ROOT / "Runtime" / "components.py").read_text(encoding="utf-8")
    service_source = (REPOSITORY_ROOT / "behavior" / "claim" / "service.py").read_text(
        encoding="utf-8"
    )
    assert "model_client: StructuredChatClient" in normalizer_source
    assert 'getattr(normalizer, "model_client", None)' in runtime_source
    assert "isinstance(normalizer, StructuredModelClaimNormalizer)" not in runtime_source
    assert "async with self.processing_lock.acquire(identity)" in service_source
    assert "await normalizer.normalize(record)" in service_source
    assert "publish_route(attempt, claims, receipt)" in service_source


def test_behavior_content_safety_and_full_digest_code_paths_remain_distinct() -> None:
    structured_source = (REPOSITORY_ROOT / "ModelClient" / "structured.py").read_text(encoding="utf-8")
    behavior_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (REPOSITORY_ROOT / "behavior").rglob("*.py")
    )
    assert 'finish_reason in {"content_filter", "safety"}' in structured_source
    assert "raise ModelContentSafetyError" in structured_source
    assert "content_digest=semantic_digest" not in behavior_sources
    assert "content_digest = semantic_digest" not in behavior_sources


def test_behavior_public_api_does_not_export_sqlite_or_private_policies() -> None:
    namespace: dict[str, object] = {}
    source = (REPOSITORY_ROOT / "behavior" / "__init__.py").read_text(encoding="utf-8")
    exec(compile(source, "behavior/__init__.py", "exec"), namespace)
    exported = set(namespace["__all__"])
    assert not {
        "BehaviorDatabase",
        "BehaviorSQLiteConnection",
        "SQLiteBehaviorEvidenceLedger",
        "SQLiteBehaviorClaimLedger",
        "ClaimCompatibilityPolicy",
        "ClaimBinder",
    } & exported


def test_behavior_ledgers_expose_no_business_update_or_delete() -> None:
    evidence_source = (REPOSITORY_ROOT / "behavior" / "evidence" / "ledger.py").read_text(encoding="utf-8")
    claim_source = (REPOSITORY_ROOT / "behavior" / "claim" / "ledger.py").read_text(encoding="utf-8")
    for source in (evidence_source, claim_source):
        tree = ast.parse(source)
        methods = {
            node.name
            for item in tree.body
            if isinstance(item, ast.ClassDef)
            for node in item.body
            if isinstance(node, ast.FunctionDef)
        }
        assert "update" not in methods
        assert "delete" not in methods
