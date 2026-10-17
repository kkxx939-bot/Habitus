"""预测层实现的指纹（裁定 54）：代码改了指纹就变，每条预期记下形成它的那份代码。"""

from __future__ import annotations

from habitus.runtime.foresight_implementation import IMPLEMENTATION, implementation_fingerprint


def test_the_fingerprint_follows_the_source(tmp_path) -> None:
    (tmp_path / "foresight" / "triggers").mkdir(parents=True)
    (tmp_path / "runtime").mkdir()
    (tmp_path / "foresight" / "ladder.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "foresight" / "triggers" / "days.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "runtime" / "foresight_tick.py").write_text("z = 3\n", encoding="utf-8")
    (tmp_path / "runtime" / "memory.py").write_text("unrelated\n", encoding="utf-8")
    first = implementation_fingerprint(tmp_path)
    assert first == implementation_fingerprint(tmp_path) and len(first) == 12
    (tmp_path / "runtime" / "memory.py").write_text("changed\n", encoding="utf-8")
    assert implementation_fingerprint(tmp_path) == first  # 别的模块改了不算
    (tmp_path / "foresight" / "triggers" / "days.py").write_text("y = 3\n", encoding="utf-8")
    assert implementation_fingerprint(tmp_path) != first
    assert len(IMPLEMENTATION) == 12
