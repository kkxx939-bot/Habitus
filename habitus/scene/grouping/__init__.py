"""情景归组：输入渲染、提示词、schema、装配校验与模型服务。"""

from habitus.scene.grouping.assembly import SceneGroupingAssemblyError, assemble_grouping
from habitus.scene.grouping.model import (
    DraftRelation,
    GapRow,
    GroupingAssembly,
    LastOccurrenceReference,
    OccurrenceRow,
    PendingReference,
    SceneDraft,
    SceneGroupingInput,
    SceneReference,
)
from habitus.scene.grouping.prompt import (
    SCENE_GROUPING_PROMPT_VERSION,
    SCENE_GROUPING_SYSTEM_PROMPT,
    build_request,
    render_occurrences,
    render_references,
)
from habitus.scene.grouping.schema import SCENE_GROUPING_JSON_SCHEMA, grouping_json_schema
from habitus.scene.grouping.service import (
    SCENE_VERSION,
    LLMSceneGrouper,
    SceneGrouper,
    SceneGroupingConfig,
    SceneGroupingLimitError,
)

__all__ = [
    "SCENE_GROUPING_JSON_SCHEMA",
    "SCENE_GROUPING_PROMPT_VERSION",
    "SCENE_GROUPING_SYSTEM_PROMPT",
    "SCENE_VERSION",
    "DraftRelation",
    "GapRow",
    "GroupingAssembly",
    "LLMSceneGrouper",
    "LastOccurrenceReference",
    "OccurrenceRow",
    "PendingReference",
    "SceneDraft",
    "SceneGrouper",
    "SceneGroupingAssemblyError",
    "SceneGroupingConfig",
    "SceneGroupingInput",
    "SceneGroupingLimitError",
    "SceneReference",
    "assemble_grouping",
    "build_request",
    "grouping_json_schema",
    "render_occurrences",
    "render_references",
]
