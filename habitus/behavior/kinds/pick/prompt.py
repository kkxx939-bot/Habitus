"""选词的提示词（裁定 101）：一条记录一次调用，在词表里选一个词。

词表是此刻的整张表：初始词加这个用户自己长的、还活着的节点，每轮按此刻的状态拼。物理 lane 同一份去掉 ``none``、``continue``、
【最近的记录】、【之前的几轮】，开头改成说明这是摄像头、手表看到的现实活动；【新节点】一节同一份文字，"这一轮"说成"这一段"
（裁定 102）。

排法：不变的部分（说明、规则、整张词表、输出格式）在系统消息，这一条的材料在用户消息，便于做提示词缓存。
词在提示词里用编号的序号（整数）：``s-k0012`` 写成 ``12``，答复里的整数按这条 lane 换回编号。
"""

from __future__ import annotations

from collections.abc import Sequence

from habitus.behavior.kinds.model import Vocabulary
from habitus.behavior.kinds.rules import SAMENESS_RULE

#: 会话 lane 选词提示词的版本，改一次升一次（会话 lane 把它拼进 ``fusion_version``，进判断与回执的身份）；物理记录上不记提示词版本。
PICK_PROMPT_VERSION = "pick-v4"

NONE, CONTINUE, MISSING = "none", "continue", "missing"
SESSION_FIXED = (NONE, CONTINUE, MISSING)
PHYSICAL_FIXED = (MISSING,)


def vocabulary_block(vocabulary: Vocabulary) -> str:
    """整张词表（``Vocabulary.ordered`` 的顺序：初始词按源文件，自己长的紧跟在上级的初始下级后面）；第 1 层的词前面在领域
    变了时加一行 "# 领域"；下级缩进、带"（上级 的下级）"。删掉的节点不出现。"""

    by_id = {word.id: word for word in vocabulary.words}
    lines: list[str] = []
    domain = None
    for word in vocabulary.ordered():
        if word.level == 1 and word.domain != domain:
            domain = word.domain
            lines.append(f"# {domain}")
        if word.parent is None:
            lines.append(f"{word.id.number} {word.name}：{word.definition}")
        else:
            parent = by_id[word.parent]
            lines.append(
                f"{'    ' * (word.level - 1)}{word.id.number} {word.name}"
                f"（{parent.id.number} {parent.name} 的下级）：{word.definition}"
            )
    return "\n".join(lines)


_PICK_RULES = """选词的规矩：
- 词表带上下级：缩进的词是它前面那个上级词的下级，说的是上级那件事里更具体的一种。选最贴切的那一层：下级里有正好说的是这件事的，选下级；下级都不合适，选上级。
- 按目标选，不按做法、工具选；时间不算：同一件事早上做和晚上做是同一个词。
- "# " 开头的行只是分组标题，不是词，不能选。"""


def _new_section(unit: str) -> str:
    """【新节点：new】一节（裁定 101）；两条 lane 同一份文字，只有说这一条的那个词不同（会话"这一轮"、物理"这一段"）。"""

    return f"""【新节点：new】（可不填，默认填 null）
- 只有同时满足这几条才填：pick 是词表里一个词的编号；这个词下面还有层数可长（见下）；{unit}的事是这个词里更具体的一种；这个词下面已有的下级都不是这件事；而且你拿得准。拿不准就填 null，{unit}记在选中的词上。
- 新节点挂在你选中的那个词下面，位置由系统放，你不能指定挂到别处。它说的必须是选中的词那件事里更窄的一种，不能比选中的词宽，也不能和它无关。
- 两个按顺序排的格子：first 是选中的词紧挨着的下一层，要写新节点就必须填 first；second 是 first 下面再细一层，可不填。不许只填 second。
- 词表一共最多 3 层：不缩进的是第 1 层，缩进一级的是第 2 层，缩进两级的是第 3 层。选中的词在第 1 层，可以填 first 和 second；在第 2 层，只能填 first；在第 3 层，不能写新节点。
- 名字和词表里的词同一个写法：一个动作 + 一种东西，四个字左右，不超过 6 个字；不带替谁、在哪、哪个行业，不带具体是哪一个（哪个人、哪个项目、哪一天），不按做法、工具命名，时间不进名字。每一格配一句定义，只写一句，只说这一种是什么。

"""


def session_system(vocabulary: Vocabulary) -> str:
    return f"""你在给一个人和助手的对话编码。下面有一张固定的词表，每个词是一件事。请判断【这一轮】是什么，在下面选一项，并可选地写一个新节点。

{SAMENESS_RULE}

【选一项：pick】
- "none"：不是一件事。打招呼、道谢、闲聊、一句感受或抱怨、没有要办的事，都选这个。
  只是问一个问题、要一个解释或答案，听完就完了、没有让助手替他办什么事：先看词表里有没有一个词说的正是问这一种东西（提醒他"该<那个词>了"，他这一轮问的就算做到了），有就选那个词；问的是【最近的记录】里某一条那件事的进展、结果或要求，是接着那一条，选 "continue"；两样都不是，就是没有要办的事，选 "none"。
- "continue"：接着【最近的记录】里的某一条，同时在 continues 填那条记录的编号。先看这一轮是不是还在办"这场对话里的"某一件事：是，就接那一条（不一定是最后一条）；不是了，再往下选。
  一件事看的是他要办成什么，不是这一句让助手做哪个动作。为同一件事往下做的每一步——问进展、补要求、改口、叫停、换个做法、改其中一处、再检查一遍——要办成的东西没变，都是接着那一条，说没说要办什么都算。要办成的东西变了才是新的一件事，哪怕和前面那件有关：拿上一步的结果去办另一件事，是新的；同一种事换个场合再做一次（另一场对话、另一天、"跟上次一样再来一份"），也是新的。
  只能填【最近的记录】里给出的编号。看不出接的是哪一条、列表里也没有它接的那件事时，不要硬配：它说了要办什么事就按新的一件事选词，否则选 "none"。
  拿不准是接着还是新的一件事时：可能接的是"这场对话里的"某一条，就当接着；不在这场对话里，就当新的一件事。
- 词表里一个词的编号（整数）：他开始了一件事（新的，或者同一种事又做了一次），而词表里有说这件事的词。
- "missing"：是一件事，但词表里没有说这件事的词。宁可答 "missing"，也不要把它硬塞进一个说的不是这件事的词——选错了，系统会对他说错话。

{_PICK_RULES}

{_new_section("这一轮")}【词表】
{vocabulary_block(vocabulary)}

只输出一个 JSON 对象，不要别的文字：
{{"pick": 词编号（整数） | "none" | "continue" | "missing", "continues": 记录编号（整数，只在 pick 为 "continue" 时填，其余填 null）, "new": null | {{"first": {{"name": 名字, "definition": 一句定义}}, "second": null | {{"name": 名字, "definition": 一句定义}}}}}}"""


def session_user(material: str, earlier: Sequence[str], mine: Sequence[str], others: Sequence[str]) -> str:
    """``mine`` 是这场对话里的记录、``others`` 是别的对话里的，各自已按时间排、带好编号（编号连着：先 ``mine`` 后 ``others``）。"""

    earlier_text = "\n\n".join(earlier) if earlier else "（这场对话里之前没有别的轮）"
    mine_text = "\n".join(mine) if mine else "（还没有）"
    others_text = "\n".join(others) if others else "（还没有）"
    return f"""【最近的记录】（编号、时刻、他当时说的、归到的词）
这场对话里的：
{mine_text}
别的对话里的：
{others_text}

【这场对话里之前的几轮】（只作上文参考）
{earlier_text}

【这一轮】
{material}"""


def physical_system(vocabulary: Vocabulary) -> str:
    return f"""你在给一个人的现实活动编码：下面【这一段】是摄像头、手表看到的他做的一段活动，已经写成了一条记录。下面有一张固定的词表，每个词是一件事。请判断【这一段】是什么，在下面选一项，并可选地写一个新节点。

{SAMENESS_RULE}

【选一项：pick】
- 词表里一个词的编号（整数）：词表里有说这件事的词。
- "missing"：是一件事，但词表里没有说这件事的词。宁可答 "missing"，也不要把它硬塞进一个说的不是这件事的词——选错了，系统会对他说错话。

{_PICK_RULES}

{_new_section("这一段")}【词表】
{vocabulary_block(vocabulary)}

只输出一个 JSON 对象，不要别的文字：
{{"pick": 词编号（整数） | "missing", "new": null | {{"first": {{"name": 名字, "definition": 一句定义}}, "second": null | {{"name": 名字, "definition": 一句定义}}}}}}"""


def physical_user(material: str) -> str:
    return f"【这一段】\n{material}"


__all__ = [
    "CONTINUE",
    "MISSING",
    "NONE",
    "PHYSICAL_FIXED",
    "PICK_PROMPT_VERSION",
    "SESSION_FIXED",
    "physical_system",
    "physical_user",
    "session_system",
    "session_user",
    "vocabulary_block",
]
