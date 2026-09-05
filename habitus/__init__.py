"""Habitus：单主体的长期记忆与行为模型。

子包按层排列：foundation / infrastructure 是无领域语义的底座；model_client 是模型能力契约；
pre 是 Conversation 数据契约；conversation、memory、behavior、prediction 是四个领域；
config 是唯一外部配置边界；runtime 是唯一组合根；integrations 是本地服务、HTTP 与 SDK 外壳；
benchmark 是评测工具。依赖方向由 tests/architecture 守住。
"""
