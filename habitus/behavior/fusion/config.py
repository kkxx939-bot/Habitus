"""判断与回执存储的容量边界，以及归约的封口视界。

这些是纯运维参数：只回答"最多存多少"，不回答任何语义问题。
"""

from __future__ import annotations

from dataclasses import dataclass

# 归约的封口视界：一条判断的证据齐备之后过了这么久，才封口落树（``behavior/reduction``）。
# 这个数原是逐帧融合"往回看多久的先前判断"的窗口——出了窗口就不会再有后续指向它，链才能机械闭合；
# 逐帧融合退役后两条 lane 都是一个单位一条、互不指向，它只剩"落树之前等多久"这一个作用。
# 归约只准从这里取，组合根把 ``Config.behavior`` 解析出的同一份数值喂进去。会话 lane 的记录一送到就封口、不等它
# （裁定 31），它只管物理 lane 的落树与两条 lane 的定稿日（``behavior/reduction/sealing``）。
FUSION_CONTEXT_LOOKBACK_SECONDS = 3_600.0


@dataclass(frozen=True)
class BehaviorFusionConfig:
    """判断存储与回执存储的上限。回执只记处置不复制正文，所以很小；判断携带完整语义，单独给一档。"""

    max_receipt_file_bytes: int = 4_194_304
    max_receipt_files: int = 100_000
    max_judgement_file_bytes: int = 4_194_304
    max_judgement_files: int = 1_000_000

    def __post_init__(self) -> None:
        for name, value in {
            "max_receipt_file_bytes": self.max_receipt_file_bytes,
            "max_receipt_files": self.max_receipt_files,
            "max_judgement_file_bytes": self.max_judgement_file_bytes,
            "max_judgement_files": self.max_judgement_files,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")


__all__ = ["FUSION_CONTEXT_LOOKBACK_SECONDS", "BehaviorFusionConfig"]
