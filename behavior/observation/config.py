"""行为观测清洗层的统一容量边界。

这些是纯运维参数：它们只回答"一次交付最多接受多少"，不回答"这条观测有没有价值"。
清洗层不做任何语义判断，因此这里没有、也不应该出现置信度阈值之类的过滤开关——
低置信度观测照常收下，是否采信由后续融合层在语义里裁量。

TODO(BHV-FUSION-003): 这些参数当前只有构造器默认值，尚未接入唯一的 ``Config/`` 边界。
- 现状：``BehaviorObservationConfig`` 与实现同处一个领域模块（符合仓库对领域 Config 的
  规定），但 ``Config/`` 下没有 behavior 段，示例 YAML 也没有对应字段，因此运维无法在不
  改代码的情况下调整。同域的 ``behavior/tree/config.py`` 与 ``behavior/document/config.py``
  处境相同——behavior 整体尚未进入 Runtime 组合根。
- 影响大小：中。单机默认值可用，但"可调运维参数必须经由 ``Config/`` 进入主链、不得把可调
  默认值藏在 Python 构造器里"是仓库明文约束，现状是暂时偏离而非豁免。
- 改造方案：behavior 接入 Runtime 组合根（``TODO(BHV-RUNTIME-001)``）时，把三个 behavior
  域 Config 一并纳入 ``Config/`` 聚合，并在示例 YAML 中完整列出；届时 Store 的 ``config``
  参数应改为必填，与 ``ConversationSourceStore`` 刻意不给默认值的做法对齐。
- 时机：与 behavior 接入主链同批，不单独提前。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorObservationConfig:
    max_observations_per_batch: int = 512
    max_semantics_chars: int = 2_000
    max_participants: int = 32
    max_evidence_refs: int = 64
    max_reference_chars: int = 512
    max_identifier_chars: int = 256
    max_file_bytes: int = 2_000_000
    max_files: int = 100_000
    # 上游用自己的时钟标注 available_at，m2bOS 用自己的时钟标注 recorded_at；两者之间允许
    # 的最大反向偏差。超过它说明交付携带了"尚未可用"的语义，属于结构性矛盾而非时钟抖动。
    max_clock_skew_seconds: int = 300

    def __post_init__(self) -> None:
        for name, value in {
            "max_observations_per_batch": self.max_observations_per_batch,
            "max_semantics_chars": self.max_semantics_chars,
            "max_participants": self.max_participants,
            "max_evidence_refs": self.max_evidence_refs,
            "max_reference_chars": self.max_reference_chars,
            "max_identifier_chars": self.max_identifier_chars,
            "max_file_bytes": self.max_file_bytes,
            "max_files": self.max_files,
            "max_clock_skew_seconds": self.max_clock_skew_seconds,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


__all__ = ["BehaviorObservationConfig"]
