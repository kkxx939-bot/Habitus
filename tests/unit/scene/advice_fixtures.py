"""关系建议测试共用的假模型：照着提示词答（先验把列出的前因都答 unsure；候选条件提名单里第一个行为类）。"""

from __future__ import annotations

import json

from habitus.model_client import ChatClient, ModelResponse
from habitus.model_client.structured import StructuredChatClient
from tests.unit.foresight.scripted_model import ScriptedProvider, model_config


def ratings(level_of: dict[str, str]) -> dict[str, object]:
    return {"ratings": [{"concept": name, "level": level} for name, level in level_of.items()]}


class Answering(ScriptedProvider):
    """照着提示词答的假模型：先验把列出的前因都答 unsure；候选条件提"之前出现过名单里第一个行为类"。"""

    def __init__(self) -> None:
        super().__init__([])

    async def complete_async(self, request):  # type: ignore[no-untyped-def, override]
        prompt = request.request.messages[-1].content or ""
        self.prompts.append(prompt)
        self.calls += 1
        if "## 前因" in prompt:
            names = [
                line[2:].split("：")[0] for line in prompt.split("## 前因")[1].splitlines() if line.startswith("- ")
            ]
            body: dict[str, object] = ratings(dict.fromkeys(names, "unsure"))
        else:
            first = (
                prompt.split("## 可以用的行为类（timeline 模板的 concept 只能从这里选）\n")[1]
                .splitlines()[0]
                .split("、")[0]
            )
            body = {
                "conditions": [
                    {"template": "timeline", "field": None, "value": None, "concept": first, "why": "之前做过"}
                ]
            }
        return ModelResponse(
            content=json.dumps(body, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            finish_reason="stop",
        )


def answering_client() -> tuple[StructuredChatClient, Answering]:
    provider = Answering()
    return StructuredChatClient(ChatClient(model_config(), provider), validation_retries=1), provider
