"""关联：输入渲染、提示词、schema、装配校验与模型服务（规律级语义树的模型触点）。"""

from habitus.scene.association.assembly import AssociationAssemblyError, assemble_association
from habitus.scene.association.input import build_association_input
from habitus.scene.association.materialize import materialize
from habitus.scene.association.model import (
    AssociationAssembly,
    AssociationDraft,
    AssociationInput,
    CauseRow,
    DayFacts,
    OccurrenceRow,
    PendingRow,
    SituationRow,
)
from habitus.scene.association.premises import Premise, PremiseTable
from habitus.scene.association.progress import AssociationProgress, Checkpoint, TaskKey
from habitus.scene.association.prompt import (
    ASSOCIATION_PROMPT_VERSION,
    ASSOCIATION_SYSTEM_PROMPT,
    build_request,
    render_causes,
    render_facts,
    render_occurrences,
    render_pending,
    render_situations,
)
from habitus.scene.association.refresher import (
    AssociationBusyError,
    AssociationRefreshConfig,
    AssociationRefresher,
    AssociationRefreshReport,
)
from habitus.scene.association.schema import ASSOCIATION_JSON_SCHEMA, SCHEMA_FINGERPRINT, association_json_schema
from habitus.scene.association.service import (
    ASSOCIATION_VERSION,
    AssociationConfig,
    AssociationLimitError,
    Associator,
    LLMAssociator,
)

__all__ = [
    "ASSOCIATION_JSON_SCHEMA",
    "ASSOCIATION_PROMPT_VERSION",
    "ASSOCIATION_SYSTEM_PROMPT",
    "ASSOCIATION_VERSION",
    "SCHEMA_FINGERPRINT",
    "AssociationAssembly",
    "AssociationAssemblyError",
    "AssociationBusyError",
    "AssociationConfig",
    "AssociationProgress",
    "AssociationRefreshConfig",
    "AssociationRefreshReport",
    "AssociationRefresher",
    "AssociationDraft",
    "AssociationInput",
    "AssociationLimitError",
    "Associator",
    "CauseRow",
    "DayFacts",
    "LLMAssociator",
    "OccurrenceRow",
    "Checkpoint",
    "Premise",
    "PremiseTable",
    "PendingRow",
    "SituationRow",
    "TaskKey",
    "assemble_association",
    "build_association_input",
    "materialize",
    "association_json_schema",
    "build_request",
    "render_causes",
    "render_facts",
    "render_occurrences",
    "render_pending",
    "render_situations",
]
