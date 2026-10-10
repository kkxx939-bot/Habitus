"""⑥ ``scene/advice/``：大模型给关系检验的两样建议——先验（``prior``）与候选调节条件（``conditions``）（语义树新方案 ``13`` ②）。

大模型只提议，统计握否决权。包根只导出纯值（``model``）；两个模型触点不从包根转手——包根一 import 它们，
任何读先验表的地方都会把 ``model_client`` 拖进来。组合根按模块路径直接 import 它们与存储（``store``）。
"""

from habitus.scene.advice.model import PriorLevel, PriorTable, Proposal, ProposalBook, settle

__all__ = ["PriorLevel", "PriorTable", "Proposal", "ProposalBook", "settle"]
