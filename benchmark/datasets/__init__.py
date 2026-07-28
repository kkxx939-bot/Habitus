"""正式记忆基准数据集 Adapter 注册入口。"""

from dataclasses import replace

from benchmark.datasets.locomo import load_locomo
from benchmark.datasets.longmemeval import load_longmemeval
from benchmark.datasets.native import load_native
from benchmark.model import BenchmarkDataset, BenchmarkDatasetName


def load_dataset(
    name: BenchmarkDatasetName | str,
    path: str,
    *,
    sample_indices: tuple[int, ...] = (),
    question_limit: int | None = None,
    include_adversarial: bool = False,
) -> BenchmarkDataset:
    """按显式协议读取原始数据，不把数据转换成测试步骤。"""

    selected = BenchmarkDatasetName(name)
    if selected is BenchmarkDatasetName.LOCOMO:
        dataset = load_locomo(
            path,
            sample_indices=sample_indices,
            include_adversarial=include_adversarial,
        )
    elif selected is BenchmarkDatasetName.LONGMEMEVAL:
        dataset = load_longmemeval(
            path,
            sample_indices=sample_indices,
        )
    else:
        dataset = load_native(path, sample_indices=sample_indices)
    return _limit_questions(dataset, question_limit)


def _limit_questions(
    dataset: BenchmarkDataset,
    question_limit: int | None,
) -> BenchmarkDataset:
    if question_limit is None:
        return dataset
    if question_limit <= 0:
        raise ValueError("question_limit must be positive")
    remaining = question_limit
    samples = []
    for sample in dataset.samples:
        if remaining <= 0:
            break
        selected = sample.questions[:remaining]
        if selected:
            samples.append(replace(sample, questions=selected))
            remaining -= len(selected)
    return replace(dataset, samples=tuple(samples))


__all__ = ["load_dataset", "load_locomo", "load_longmemeval", "load_native"]
