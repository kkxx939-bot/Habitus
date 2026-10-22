/**
 * Agent Harness 的声明式注册边界。
 *
 * 生命周期协调器只消费这些描述，不包含 Codex、Claude Code 或未来
 * Harness 的条件分支。新增 Harness 时增加一个描述及其插件目录即可。
 */

export const MARKETPLACE_NAME = "habitus-local";
export const PLUGIN_NAME = "habitus-memory";
export const PLUGIN_ID = `${PLUGIN_NAME}@${MARKETPLACE_NAME}`;

const DEFINITIONS = [
  {
    id: "codex",
    aliases: ["codex"],
    displayName: "OpenAI Codex",
    command: "codex",
    protocol: "codex_rollout",
    pluginDirectory: "habitus-memory",
    pluginManifest: ".codex-plugin/plugin.json",
    hooksManifest: "hooks/hooks.json",
    requiredHooks: ["SessionStart", "UserPromptSubmit", "Stop", "PreCompact", "SessionEnd"],
    requiredAssets: [
      ".codex-plugin/plugin.json",
      "hooks/hooks.json",
      "scripts/shared/plugin-core.mjs",
    ],
    marketplaceManifest: {
      path: ".agents/plugins/marketplace.json",
      document: {
        name: MARKETPLACE_NAME,
        interface: { displayName: "Habitus Local" },
        plugins: [{
          name: PLUGIN_NAME,
          source: { source: "local", path: "./plugins/habitus-memory" },
          policy: { installation: "AVAILABLE", authentication: "ON_INSTALL" },
          category: "Productivity",
        }],
      },
    },
    marketplaceInventory: ["plugin", "marketplace", "list", "--json"],
    pluginInventory: ["plugin", "list", "--json", "--available"],
    marketplaceAdd: ["plugin", "marketplace", "add", "{root}"],
    marketplaceRemove: ["plugin", "marketplace", "remove", MARKETPLACE_NAME],
    pluginAdd: ["plugin", "add", PLUGIN_ID],
    pluginRemove: ["plugin", "remove", PLUGIN_ID, "--json"],
    pluginEnable: null,
    // 插件 hook 要宿主的特性开着才跑（OpenViking 的 Codex 安装写 `plugin_hooks = true`；Codex 0.147 起这个开关已标 removed、
    // 并入常开的 `hooks`）。doctor 按这张表查：列出的特性必须是开着的
    featureInventory: { args: ["features", "list"], required: ["hooks"] },
  },
  {
    id: "claude-code",
    aliases: ["claude-code", "claude", "cc"],
    displayName: "Claude Code",
    command: "claude",
    protocol: "claude_code",
    pluginDirectory: "habitus-memory-claude-code",
    pluginManifest: ".claude-plugin/plugin.json",
    hooksManifest: "hooks/hooks.json",
    requiredHooks: [
      "SessionStart",
      "UserPromptSubmit",
      "Stop",
      "PreCompact",
      "SessionEnd",
      "SubagentStart",
      "SubagentStop",
    ],
    requiredAssets: [
      ".claude-plugin/plugin.json",
      "hooks/hooks.json",
      "scripts/shared/plugin-core.mjs",
      "scripts/subagent-start.mjs",
      "scripts/subagent-stop.mjs",
    ],
    marketplaceManifest: {
      path: ".claude-plugin/marketplace.json",
      document: {
        name: MARKETPLACE_NAME,
        description: "Local Habitus plugins for Claude Code.",
        owner: { name: "Habitus" },
        plugins: [{
          name: PLUGIN_NAME,
          description: "Single-user local semantic memory for Claude Code.",
          source: "./plugins/habitus-memory-claude-code",
          category: "productivity",
        }],
      },
    },
    marketplaceInventory: ["plugin", "marketplace", "list", "--json"],
    pluginInventory: ["plugin", "list", "--json"],
    marketplaceAdd: ["plugin", "marketplace", "add", "{root}"],
    marketplaceRemove: ["plugin", "marketplace", "remove", MARKETPLACE_NAME],
    pluginAdd: ["plugin", "install", PLUGIN_ID],
    pluginRemove: ["plugin", "uninstall", PLUGIN_ID],
    pluginEnable: ["plugin", "enable", PLUGIN_ID],
  },
  {
    // ZCode（zai-org/ZCode，Apache-2.0）认 Claude Code 的插件格式与 `.claude-plugin/marketplace.json`，第三方插件的 hook 默认执行：
    // 装的就是 Claude Code 那份插件（同一个 marketplace 清单），插件在 hook 进程里看到 ZCODE_SESSION_ID 就换 ZCode 的适配器
    // （转写从 ZCode 的会话库读）。它只有 SessionStart / UserPromptSubmit / Stop 等七个事件，其余几个 hook 它跳过。
    id: "zcode",
    aliases: ["zcode", "zc"],
    displayName: "ZCode",
    command: "zcode",
    protocol: "zcode",
    pluginDirectory: "habitus-memory-claude-code",
    pluginManifest: ".claude-plugin/plugin.json",
    hooksManifest: "hooks/hooks.json",
    requiredHooks: ["SessionStart", "UserPromptSubmit", "Stop"],
    requiredAssets: [".claude-plugin/plugin.json", "hooks/hooks.json", "scripts/zcode-adapter.mjs", "scripts/shared/parts-transcript.mjs"],
    // 清单由 claude-code 那条写，这里只在宿主里登记
    manifestOf: "claude-code",
    marketplaceInventory: ["plugins", "marketplace", "list", "--json"],
    pluginInventory: ["plugins", "list", "--json"],
    marketplaceAdd: ["plugins", "marketplace", "add", "{root}"],
    marketplaceRemove: ["plugins", "marketplace", "remove", MARKETPLACE_NAME],
    pluginAdd: ["plugins", "install", PLUGIN_ID],
    pluginRemove: ["plugins", "uninstall", PLUGIN_ID],
    pluginEnable: ["plugins", "enable", PLUGIN_ID],
  },
  {
    // OpenCode 没有插件 marketplace：它启动时加载 `<配置目录>/opencode/plugins/` 下的 .js。注册 = 在那里放一个一行的入口文件，
    // 转手到 marketplace 根里暂存的插件目录（与 OpenViking 源码安装的 wrapper 同一个做法）
    id: "opencode",
    aliases: ["opencode", "oc"],
    displayName: "OpenCode",
    command: "opencode",
    protocol: "opencode",
    pluginDirectory: "habitus-memory-opencode",
    pluginManifest: "package.json",
    hooksManifest: null,
    requiredHooks: [],
    requiredAssets: [
      "package.json",
      "index.mjs",
      "scripts/host-adapter.mjs",
      "scripts/shared/parts-transcript.mjs",
      "scripts/shared/plugin-core.mjs",
    ],
    registration: "file",
    fileEntry: { base: "config", configPath: ["opencode", "plugins"], name: "habitus-memory.js", module: "index.mjs" },
  },
  {
    // pi（earendil-works/pi，MIT）启动时自动加载 `~/.pi/agent/extensions/` 下的扩展（默认导出一个工厂函数）：同 OpenCode，
    // 在那里放一个一行的入口文件。pi 没有内置提问工具，插件自己注册一个 `ask_user`
    id: "pi",
    aliases: ["pi", "pi-coding-agent"],
    displayName: "pi",
    command: "pi",
    protocol: "pi",
    pluginDirectory: "habitus-memory-pi",
    pluginManifest: "package.json",
    hooksManifest: null,
    requiredHooks: [],
    requiredAssets: [
      "package.json",
      "index.mjs",
      "scripts/host-adapter.mjs",
      "scripts/question-tool.mjs",
      "scripts/shared/plugin-core.mjs",
    ],
    registration: "file",
    fileEntry: { base: "home", configPath: [".pi", "agent", "extensions"], name: "habitus-memory.js", module: "index.mjs" },
  },
  {
    // DeepSeek Harness（deepseek-ai/deepseek-harness，MIT）：经它官方的 Claude Code hook 桥跑我们的 hooks.json。注册 = 用
    // `dsh plugin --profile web add` 把桥装进 web profile，再在 $DSH_HOME/cordis.patch.yml 插一条桥的配置（受管标记之间）。
    // 桥给的 transcript_path 是空串：插件自己读 $DSH_HOME/sessions 下的会话日志。提问工具是它自带的 ask_user_question。
    id: "dsh",
    aliases: ["dsh", "deepseek-harness", "deepseek"],
    displayName: "DeepSeek Harness",
    command: "dsh",
    protocol: "dsh",
    pluginDirectory: "habitus-memory-dsh",
    pluginManifest: "package.json",
    hooksManifest: "hooks/hooks.json",
    requiredHooks: ["SessionStart", "UserPromptSubmit", "Stop"],
    requiredAssets: ["package.json", "hooks/hooks.json", "scripts/host-adapter.mjs", "scripts/session-log.mjs", "scripts/shared/plugin-core.mjs"],
    registration: "cordis-patch",
    patchEntry: { package: "@deepseek-ai/dsh-hooks-claude-code@next", profile: "web" },
  },
];

function validateDefinition(value) {
  if (!value || !/^[a-z][a-z0-9-]{0,63}$/.test(value.id || "")) {
    throw new Error("harness id is invalid");
  }
  if (!Array.isArray(value.aliases) || !value.aliases.includes(value.id)) {
    throw new Error(`harness aliases must include its id: ${value.id}`);
  }
  if (!value.command || !value.protocol || !value.pluginDirectory) {
    throw new Error(`harness execution metadata is incomplete: ${value.id}`);
  }
  const registration = value.registration ?? "marketplace";
  if (registration === "marketplace" && !value.manifestOf && (!value.marketplaceManifest?.path || !value.marketplaceManifest?.document)) {
    throw new Error(`harness marketplace metadata is incomplete: ${value.id}`);
  }
  if (registration === "file" && (
    !Array.isArray(value.fileEntry?.configPath) || !/^[a-z0-9-]+\.js$/.test(value.fileEntry?.name || "") || !value.fileEntry?.module
    || !["config", "home"].includes(value.fileEntry?.base ?? "config")
  )) {
    throw new Error(`harness file entry metadata is incomplete: ${value.id}`);
  }
  if (registration === "cordis-patch" && (!value.patchEntry?.package || !value.patchEntry?.profile || !value.hooksManifest)) {
    throw new Error(`harness cordis patch metadata is incomplete: ${value.id}`);
  }
  if (!["marketplace", "file", "cordis-patch"].includes(registration)) throw new Error(`unknown harness registration: ${value.id}`);
  if (value.featureInventory != null && (
    !Array.isArray(value.featureInventory.args) || !Array.isArray(value.featureInventory.required)
  )) {
    throw new Error(`harness feature inventory is invalid: ${value.id}`);
  }
  return Object.freeze({
    ...value,
    registration,
    marketplaceManifest: value.marketplaceManifest ?? null,
    manifestOf: value.manifestOf ?? null,
    patchEntry: value.patchEntry ? Object.freeze({ ...value.patchEntry }) : null,
    fileEntry: value.fileEntry
      ? Object.freeze({ ...value.fileEntry, base: value.fileEntry.base ?? "config", configPath: Object.freeze([...value.fileEntry.configPath]) })
      : null,
    featureInventory: value.featureInventory ?? null,
    aliases: Object.freeze([...value.aliases]),
    requiredHooks: Object.freeze([...value.requiredHooks]),
    requiredAssets: Object.freeze([...value.requiredAssets]),
  });
}

export function createHarnessRegistry(definitions) {
  if (!Array.isArray(definitions) || definitions.length === 0) {
    throw new Error("at least one Agent Harness must be registered");
  }
  const entries = definitions.map(validateDefinition);
  const aliases = new Map();
  for (const entry of entries) {
    for (const alias of entry.aliases) {
      if (!/^[a-z][a-z0-9-]{0,63}$/.test(alias) || aliases.has(alias)) {
        throw new Error(`duplicate or invalid harness alias: ${alias}`);
      }
      aliases.set(alias, entry);
    }
  }
  // 借别的宿主清单的（ZCode 借 Claude Code 的）：那条必须在、写清单、插件目录相同
  for (const entry of entries.filter((item) => item.manifestOf)) {
    const owner = entries.find((item) => item.id === entry.manifestOf);
    if (!owner?.marketplaceManifest || owner.pluginDirectory !== entry.pluginDirectory) {
      throw new Error(`harness ${entry.id} borrows the manifest of ${entry.manifestOf}, which does not write one for the same plugin`);
    }
  }
  // 两个宿主写同一份清单 / 同一个入口文件，后写的会静默盖掉先写的：注册时就拒
  const targets = new Map();
  for (const entry of entries.filter((item) => !item.manifestOf)) {
    const target = entry.marketplaceManifest
      ? `marketplace:${entry.marketplaceManifest.path}`
      : entry.fileEntry
        ? `file:${entry.fileEntry.base}:${[...entry.fileEntry.configPath, entry.fileEntry.name].join("/")}`
        : `patch:${entry.command}`;
    if (targets.has(target)) {
      throw new Error(`harnesses ${targets.get(target)} and ${entry.id} would write the same ${target}`);
    }
    targets.set(target, entry.id);
  }
  return Object.freeze({
    list: () => entries,
    resolve(value) {
      const normalized = String(value || "").trim().toLowerCase();
      const entry = aliases.get(normalized);
      if (!entry) throw new Error(`unknown Agent Harness: ${value} (supported: ${entries.map((item) => item.id).join(", ")})`);
      return entry;
    },
  });
}

export const HARNESS_REGISTRY = createHarnessRegistry(DEFINITIONS);

export function publicHarnessDescriptor(definition, { available }) {
  return {
    id: definition.id,
    aliases: [...definition.aliases],
    display_name: definition.displayName,
    command: definition.command,
    protocol: definition.protocol,
    available: Boolean(available),
  };
}
