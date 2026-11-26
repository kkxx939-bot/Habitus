"""大模型给关系检验的建议：先验（``prior``）。

大模型只提议，统计握否决权。包根只导出纯值（``model``）；模型触点不从包根转手——包根一 import 它，
任何读先验表的地方都会把 ``model_client`` 拖进来。组合根按模块路径直接 import 它与存储（``store``）。
"""

from habitus.scene.advice.model import PriorLevel, PriorTable, settle

__all__ = ["PriorLevel", "PriorTable", "settle"]
