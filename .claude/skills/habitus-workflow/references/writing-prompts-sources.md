# 出处与原文摘录（2026-11-06 读）

## Prompting best practices（Claude Platform Docs）
https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices

- Be clear and direct："Think of Claude as a brilliant but new employee who lacks context on your norms and workflows." "Key test: Show your prompt to a colleague with minimal context on the task and ask them to follow it. If they'd be confused, Claude will be too."
- Add context："Providing context or motivation behind your instructions, such as explaining to Claude why such behavior is important, can give Claude more information about your goals." 例子：说明"文字会被语音朗读"，比直接写 "NEVER use ellipses" 效果好。"Claude can generalize from the explanation."
- Use examples：3–5 个例子，要相关、要多样，用 `<example>` 标签包起来。
- Structure prompts with XML tags：指令、上下文、例子、输入分别用各自的标签。
- Long-horizon / multiwindow：首个上下文窗口先搭框架（测试、脚本）；用结构化文件记状态（tests.json）、用自由文本记进度（progress.txt）；"It is unacceptable to remove or edit tests"。
- Overeagerness："Only make changes that are directly requested or clearly necessary"；不为不会发生的情况加错误处理和回退；只在系统边界校验；只用一次的操作不建 helper。
- Avoid focusing on passing tests："Tests are there to verify correctness, not to define the solution." "If … any of the tests are incorrect, please inform me rather than working around them."
- Minimizing hallucinations："Never speculate about code you have not opened."
- Subagent orchestration：子代理适合能并行的、需要隔离上下文的、互不共享状态的任务；简单任务、顺序操作、单文件修改直接做。

## Best practices for Claude Code
https://code.claude.com/docs/en/best-practices

- "Only write down what Claude can't find by reading the repo, or what it would otherwise spend time rediscovering every session." 判断方法：它读代码能找到吗？它会去找吗？两个都是"会"就不写。
- 写具体的提示词："name the file, describe the symptom, and say what 'fixed' looks like up front"；对两种都合理的做法有偏好时，直接说。
- "Give Claude something it can run that comes back pass or fail." "ask Claude for evidence: the command it ran and its output" "'All tests pass' on its own tells you very little."
- 按需求审查："report gaps that affect correctness and skip style" —— "A reviewer that's asked to find problems will always find something."
- 不能有例外的规矩："If something has to happen every time, make it a hook, a deny rule, or a managed setting." "If you keep adding emphasis … that's a sign the line belongs in a hook."
- 已过时的老建议："Add IMPORTANT or ALL CAPS …"，现在 "piling on emphasis mostly makes Claude over-apply the rule"；`think hard` / `be thorough` 只是普通的提示词文字；"Delegate to a subagent for a better answer" 并不会让答案更好，新的审查者有用是因为它没看过实现时的推理。
- 先写需求再开新 session："Interview me until you could write the spec … then start a fresh session to build from it."
- 纠偏："after two failed corrections run /clear and write a better prompt instead of a third."
- CLAUDE.md 要短：每一行都会在每个任务里占用注意力；建议不超过 200 行。

## Skill authoring best practices
https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices

- Concise is key："Only add context Claude doesn't already have." 写每一段之前问自己："Does this paragraph justify its token cost?"
- Degrees of freedom：两边是悬崖的窄桥，要给确切的护栏；没有危险的开阔地，给方向就行。
- 引用只给一层；超过 100 行的参考文件，开头放目录。
- 前后用词一致；不写会过时的信息；复杂流程给清单让对方勾选；用"校验 → 修正 → 再校验"的循环。
- 先拿真实任务评估、再写说明；Claude A 写、Claude B 用，观察 B 怎么用，再回来改。

## How we built our multi-agent research system（Anthropic Engineering）
https://www.anthropic.com/engineering/multi-agent-research-system

- "Each subagent needs an objective, an output format, guidance on the tools and sources to use, and clear task boundaries." "Without detailed task descriptions, agents duplicate work, leave gaps, or fail to find necessary information."
- 在提示词里写明投入多少力气要跟任务的复杂度匹配；"Think like your agents"：拿完全相同的提示词和工具一步步跑一遍，看它会在哪里出错。

## Prompting Claude Opus 5（模型专页）
https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5

- 完整需求一次给："performs best when given the complete task specification up front and left to run."
- 审查："If your review prompt says 'only report high-severity issues' or 'be conservative,' the model might follow that instruction literally and report less; ask it to report everything and filter in a separate pass instead."
- 过度验证："Claude Opus 5 verifies its own work without being told to. If your prompt contains explicit verification instructions ('include a final verification step for any non-trivial task,' 'use a subagent to verify'), remove them."
- 范围："Deliver what was asked, at the scope intended … If the request seems mistaken or a better approach exists, say so in a sentence and continue with the task as asked rather than quietly narrowing, widening, or transforming it."
- 子代理："Do not delegate work you can finish yourself in a handful of tool calls, and do not use subagents to verify or double-check your own work. If one subagent can complete the task, use one rather than several."
- 自我纠正：不要写"double-check your answer""re-verify before responding"这类指令。
- 写出来的文件偏长："Match the length of written documents to what the task needs: cover the substance, but do not pad with filler sections, redundant summaries, or boilerplate."
- 正面例子："Positive examples of the communication style you want tend to be more effective than instructions about what not to do."
- 不要求把推理原样写进回复（可能触发 `reasoning_extraction` 拒绝）。

## Prompting Claude Opus 5.5（模型专页）
https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5

- Opus 5 的写法照样适用；effort 是控制思考深度的主要手段（默认 medium）；chat 场景里"想仔细再答"之类的指令可以删掉。
- 无人值守的长任务：模型会用一段进度汇报结束这一轮。把任务拆成清单，让它维护在文件里，并事先写明完成条件；还有未完成项时，点名提醒它继续。
- 跨多个应用的任务，让它先广泛查看相关资料再动手。

## Claude Code：Extend Claude with skills
https://code.claude.com/docs/en/skills

- description 和 when_to_use 合起来，在列表里会被截到 1,536 字，所以关键用途写在最前面。
- "Keep the body itself concise. Once a skill loads, its content stays in context across turns … State what to do rather than narrating how or why."
- SKILL.md 控制在 500 行以内；压缩之后，每个 skill 只重新附上前 5,000 个 token。
- 个人 skill 放在 `~/.claude/skills/<name>/SKILL.md`。`/doctor prompt-audit` 会标出为旧模型写的、已经过时的指令。
