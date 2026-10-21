import { createHash } from "node:crypto";
import { appendFile, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";

// 把 OpenCode 会话里已完成的消息按顺序镜像成 JSONL（一条 = client.session.messages 的一项 {info, parts}）。只追加不改写：
// 共享库的游标按"文件前缀不变"认增量。question 工具的作答从 question.replied 事件来，挂在 state.habitus_answers 上，
// 服务端据此写成 "<题目>"="<作答>"（方案 36 第四节）。

export function transcriptPath(stateRoot, sessionId) {
  const key = createHash("sha256").update(`opencode\n${sessionId}`).digest("hex");
  return join(stateRoot, "transcripts", "opencode", `${key}.jsonl`);
}

export class TranscriptMirror {
  constructor(stateRoot) {
    this.stateRoot = stateRoot;
    this.mirrored = new Map();
    this.questions = new Map();
    this.answers = new Map();
  }

  // question.asked：记下这次提问的 requestID → (callID, 题目)
  asked(properties) {
    const callID = properties?.tool?.callID;
    if (typeof properties?.id !== "string" || typeof callID !== "string" || !Array.isArray(properties?.questions)) return;
    this.questions.set(properties.id, { callID, titles: properties.questions.map((item) => String(item?.question ?? "")) });
  }

  // question.replied：answers 是每题一组选中的标签（或他自己填的话）
  replied(properties) {
    const asked = this.questions.get(properties?.requestID);
    if (!asked || !Array.isArray(properties?.answers)) return;
    this.questions.delete(properties.requestID);
    this.answers.set(asked.callID, asked.titles.map((question, index) => ({
      question,
      answers: Array.isArray(properties.answers[index]) ? properties.answers[index].map(String) : [],
    })));
  }

  async mirror(sessionId, messages) {
    const path = transcriptPath(this.stateRoot, sessionId);
    const seen = await this.#seen(sessionId, path);
    const lines = [];
    for (const message of Array.isArray(messages) ? messages : []) {
      const id = message?.info?.id;
      if (typeof id !== "string" || seen.has(id)) continue;
      if (message.info.role === "assistant" && !Number.isFinite(message.info?.time?.completed)) break;
      lines.push(JSON.stringify(this.#withAnswers(message)));
      seen.add(id);
    }
    if (lines.length > 0) {
      await mkdir(join(this.stateRoot, "transcripts", "opencode"), { recursive: true, mode: 0o700 });
      await appendFile(path, `${lines.join("\n")}\n`, { mode: 0o600 });
    }
    return { path, appended: lines.length };
  }

  #withAnswers(message) {
    const parts = (message.parts || []).map((part) => {
      const answers = part?.type === "tool" && part.tool === "question" ? this.answers.get(part.callID) : undefined;
      if (!answers) return part;
      this.answers.delete(part.callID);
      return { ...part, state: { ...part.state, habitus_answers: answers } };
    });
    return { info: message.info, parts };
  }

  async #seen(sessionId, path) {
    if (this.mirrored.has(sessionId)) return this.mirrored.get(sessionId);
    const seen = new Set();
    try {
      for (const line of (await readFile(path, "utf8")).split("\n")) {
        if (!line.trim()) continue;
        try { const id = JSON.parse(line)?.info?.id; if (typeof id === "string") seen.add(id); } catch {}
      }
    } catch (error) { if (error?.code !== "ENOENT") throw error; }
    this.mirrored.set(sessionId, seen);
    return seen;
  }
}
