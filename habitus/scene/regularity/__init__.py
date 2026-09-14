"""规律级：语义关联按**候选行为**归档、不断叠加的那一层的文档与存储。

地址住在 ``scene.model``（与按天情景的地址放在一起）——``uri`` 要同时认两种文档形态，而这里的
文档层反过来要用 ``uri``；地址与地址放在一起，这条环就不存在。

与现有的按天情景树**并存**：新形状先建起来、读侧接过去，再整块删旧的——一次性重写会把所有读
旧形状的代码同时打断，而这个仓库的验收是每步整链绿。
"""

from habitus.scene.regularity.document import (
    AssociationDocument,
    AssociationDocumentError,
    decode,
    encode,
)
from habitus.scene.regularity.link import (
    SceneStoredLink,
    link_lag_seconds,
    normalize_stored_links,
    parse_link_target,
    parse_stored_links,
)
from habitus.scene.regularity.store import (
    MAX_DIRECTORY_ENTRIES,
    MAX_LAYER_BYTES,
    MAX_RECORD_BYTES,
    RegularityTree,
    RegularityTreeError,
)

__all__ = [
    "SceneStoredLink",
    "link_lag_seconds",
    "normalize_stored_links",
    "parse_link_target",
    "parse_stored_links",
    "MAX_DIRECTORY_ENTRIES",
    "MAX_LAYER_BYTES",
    "MAX_RECORD_BYTES",
    "AssociationDocument",
    "AssociationDocumentError",
    "RegularityTree",
    "RegularityTreeError",
    "decode",
    "encode",
]
