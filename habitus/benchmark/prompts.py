"""记忆问答和独立 Judge 使用的稳定、数据集感知提示词。"""

from __future__ import annotations

from habitus.benchmark.model import BenchmarkAnswerRecord, BenchmarkDatasetName, BenchmarkQuestion
from habitus.benchmark.protocol import BenchmarkJudgePolicy


def answer_prompt(dataset: str | BenchmarkDatasetName, question: BenchmarkQuestion, context: str) -> str:
    """按数据集语义构造回答提示词，同时保持 Habitus 上下文来源不变。"""

    dataset_name = BenchmarkDatasetName(dataset)
    reference_time = question.question_time.date().isoformat() if question.question_time else "unknown"
    dataset_rules = _answer_rules(dataset_name)
    return f"""You are answering a long-term-memory benchmark question from {dataset_name.value}.
Use only the retrieved memory context below. Read every selected memory before answering and combine facts across memories when necessary.
Verify that each fact belongs to the exact person, entity, role, item, and event asked about.
Resolve updates and direct contradictions by chronology: the later confirmed information supersedes the older value.
Compute dates, durations, counts, and comparisons from the supplied facts instead of guessing.
{dataset_rules}
If the context cannot support the requested fact, explicitly state that the information is unavailable.
Return only a concise final answer; do not discuss retrieval, benchmark mechanics, or hidden reasoning.

Reference date: {reference_time}
Question type: {question.question_type}

Retrieved memory context:
{context or "[no memory context retrieved]"}

Question: {question.question}
Answer:"""


def judge_prompt(
    record: BenchmarkAnswerRecord,
    *,
    evidence_texts: tuple[str, ...] = (),
    policy: BenchmarkJudgePolicy = BenchmarkJudgePolicy.STRICT,
) -> str:
    """按已选评测口径比较模型回答和数据集参考答案。"""

    if not isinstance(policy, BenchmarkJudgePolicy):
        policy = BenchmarkJudgePolicy(policy)
    if policy is BenchmarkJudgePolicy.OPENVIKING_DEFAULT and record.dataset == BenchmarkDatasetName.HABITUS.value:
        raise ValueError("openviking-default judge policy is not defined for the Habitus native dataset")
    reference_answer = _reference_answer(record)
    evidence = ""
    if evidence_texts:
        evidence = "\nSupporting source evidence:\n" + "\n".join(f"- {item}" for item in evidence_texts)
    rules = (
        _strict_judge_rules(record.dataset)
        if policy is BenchmarkJudgePolicy.STRICT
        else _default_judge_rules(
            record.dataset,
            has_evidence=bool(evidence_texts),
        )
    )
    return f"""You are an independent evaluator for a long-term-memory benchmark.
Judge whether the model response correctly answers the question according to the reference answer.

Rules:
{rules}

Dataset: {record.dataset}
Judge policy: {policy.value}
Question type: {record.question_type}
Question time: {record.question_time or "unknown"}
Question: {record.question}
Reference answer: {reference_answer}
Model response: {record.response}
{evidence}

Return the strict JSON verdict requested by the response schema."""


def _answer_rules(dataset: BenchmarkDatasetName) -> str:
    if dataset is BenchmarkDatasetName.LOCOMO:
        return (
            "For multi-hop, list, and count questions, scan all selected memories and include every distinct supported item. "
            "Use the reference date for temporal disambiguation and prefer the most specific supported fact."
        )
    if dataset is BenchmarkDatasetName.LONGMEMEVAL:
        return (
            "Preserve user anti-preferences, distinguish user facts from assistant advice, distinguish completed actions from "
            "plans, and abstain when the question names a different role, entity, or item variant than the memories."
        )
    return (
        "Respect explicit forgetting, completed-intention scope, relation evidence, and the lower authority of Conversation "
        "Summary fallback compared with durable Memory."
    )


def _reference_answer(record: BenchmarkAnswerRecord) -> str:
    # OpenViking 的 LoCoMo Judge 会在 category 3 的分号答案上只采用第一个规范答案。
    if record.dataset == BenchmarkDatasetName.LOCOMO.value and record.question_type == "category_3":
        return record.reference_answer.split(";", 1)[0].strip()
    return record.reference_answer


def _strict_judge_rules(dataset: str) -> str:
    common = (
        "1. The response must contain every essential fact, constraint, list item, update, date, duration, count, and "
        "negation required by the reference answer.\n"
        "2. Semantically equivalent wording is correct, but a merely related or partial answer is wrong.\n"
        "3. Chronology, latest valid facts, exact entities, requested roles, and temporal relations must be correct.\n"
        "4. A reference abstention requires a clear abstention; a concrete reference fact makes abstention wrong.\n"
        "5. Extra detail is allowed only after the core answer is correct and only when it does not contradict it.\n"
        "6. Evaluate answer quality only; retrieval of related material alone earns no credit."
    )
    if dataset == BenchmarkDatasetName.LONGMEMEVAL.value:
        return (
            common
            + "\n7. Preserve positive and negative preferences and use the current value for knowledge-update questions."
        )
    return common


def _default_judge_rules(dataset: str, *, has_evidence: bool) -> str:
    evidence_rule = (
        " Source evidence may accept a better-supported answer when the gold answer is incomplete; never use evidence to "
        "reject an otherwise correct answer more strictly."
        if has_evidence
        else ""
    )
    if dataset == BenchmarkDatasetName.LOCOMO.value:
        return (
            "1. Follow the OpenViking default LoCoMo lenient policy: one correct item from a multi-item gold answer is enough.\n"
            "2. Paraphrases and added non-contradictory detail are correct; judge semantic knowledge, not wording.\n"
            "3. Dates within 14 days and durations within roughly 50 percent are acceptable.\n"
            "4. Mark wrong only when no gold fact is present and source evidence does not support the response, or when the "
            f"response addresses a different topic.{evidence_rule}"
        )
    if dataset == BenchmarkDatasetName.LONGMEMEVAL.value:
        return (
            "1. Follow the OpenViking default LongMemEval lenient policy: judge semantic equivalence and accept a correct "
            "superset unless an added claim is demonstrably wrong.\n"
            "2. Accept reasonable numeric approximations, equivalent date expressions, and clear abstention equivalents.\n"
            "3. For personalization, require awareness of the main user context rather than every rubric bullet.\n"
            "4. Before marking wrong, verify that a core concept is actually absent or contradicted."
        )
    return _strict_judge_rules(dataset)


__all__ = ["answer_prompt", "judge_prompt"]
