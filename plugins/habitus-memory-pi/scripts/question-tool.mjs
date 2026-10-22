// pi 没有内置"停下来问人、给选项、等作答"的工具（它的 README 说跳过了这类功能）。提醒出口押在 agent 用提问工具问他上
// （方案 32 第 4.2 条），这里注册一个：一次问一题或几题，每题给选项、可以自己填；作答按 `"<题目>"="<作答>"` 一行一题写进结果，
// 与 Claude Code、OpenCode 的作答同一个形状，服务端的提醒出口与会话 lane 按这个认。他把提问关掉（Esc）当工具失败，
// 提醒出口据此记「说了」、不等晚到的作答。没有交互界面（-p / json 模式）时问不了，也当失败。

export const QUESTION_TOOL = "ask_user";
const OTHER = "自己填…";

// 普通 JSON Schema：pi 用 TypeBox 1.x 的 Compile 校验参数，它直接认 JSON Schema，插件不必带 typebox
const PARAMETERS = {
  type: "object",
  properties: {
    questions: {
      type: "array",
      minItems: 1,
      maxItems: 4,
      items: {
        type: "object",
        properties: {
          question: { type: "string", description: "The question to show the user, complete and self-contained" },
          options: {
            type: "array",
            minItems: 1,
            items: {
              type: "object",
              properties: {
                label: { type: "string", description: "Short option label" },
                description: { type: "string" },
              },
              required: ["label"],
            },
          },
        },
        required: ["question", "options"],
      },
    },
  },
  required: ["questions"],
};

export function questionTool() {
  return {
    name: QUESTION_TOOL,
    label: "Ask user",
    description: "Stop and ask the user one or more multiple-choice questions, then wait for the answers. "
      + "Use it when you need the user's decision; the user may also type a free answer.",
    parameters: PARAMETERS,
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      if (!ctx?.hasUI || typeof ctx.ui?.select !== "function") {
        throw new Error("No interactive UI to ask the user in this mode");
      }
      const answers = [];
      for (const item of params.questions) {
        const labels = item.options.map((option) => option.label);
        let answer = await ctx.ui.select(item.question, [...labels, OTHER]);
        if (answer === OTHER) answer = await ctx.ui.input(item.question, "");
        if (answer == null || String(answer).trim() === "") throw new Error("The user dismissed this question");
        answers.push(String(answer).trim());
      }
      const lines = params.questions.map((item, index) => `"${item.question}"="${answers[index]}"`);
      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: { questions: params.questions.map((item) => item.question), answers },
      };
    },
  };
}
