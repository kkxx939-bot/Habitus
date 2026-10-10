"""定期拆改：输入事实（``facts``）、提示词（``prompt``）、提议解析（``proposals``）、封闭集合重判（``rejudge``）、
证据规则（``evidence``）、锚点自检（``anchors``）、编排（``reviser``）；定时与合并证据在 ``kinds/schedule.py``。"""

from habitus.behavior.kinds.revision.anchors import AnchorCheck, AnchorGate
from habitus.behavior.kinds.revision.evidence import split_shortfall
from habitus.behavior.kinds.revision.facts import Member
from habitus.behavior.kinds.revision.prompt import REVISION_PROMPT_VERSION, REVISION_SYSTEM_PROMPT
from habitus.behavior.kinds.revision.reviser import Reviser, RevisionResult, cooling_classes

__all__ = [
    "REVISION_PROMPT_VERSION",
    "REVISION_SYSTEM_PROMPT",
    "AnchorCheck",
    "AnchorGate",
    "Member",
    "Reviser",
    "RevisionResult",
    "cooling_classes",
    "split_shortfall",
]
