/**
 * Agent Harness 的声明式注册边界。
 *
 * 生命周期协调器只消费这些描述，不包含 Codex、Claude Code 或未来
 * Harness 的条件分支。新增 Harness 时增加一个描述及其插件目录即可。
 */

export const MARKETPLACE_NAME = "m2bos-local";
export const PLUGIN_NAME = "m2bos-memory";
export const PLUGIN_ID = `${PLUGIN_NAME}@${MARKETPLACE_NAME}`;

const DEFINITIONS = [
  {
    id: "codex",
    aliases: ["codex"],
    displayName: "OpenAI Codex",
    command: "codex",
    protocol: "codex_rollout",
    pluginDirectory: "m2bos-memory",
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
        interface: { displayName: "m2bOS Local" },
        plugins: [{
          name: PLUGIN_NAME,
          source: { source: "local", path: "./plugins/m2bos-memory" },
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
  },
  {
    id: "claude-code",
    aliases: ["claude-code", "claude", "cc"],
    displayName: "Claude Code",
    command: "claude",
    protocol: "claude_code",
    pluginDirectory: "m2bos-memory-claude-code",
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
        description: "Local m2bOS plugins for Claude Code.",
        owner: { name: "m2bOS" },
        plugins: [{
          name: PLUGIN_NAME,
          description: "Single-user local semantic memory for Claude Code.",
          source: "./plugins/m2bos-memory-claude-code",
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
  if (!value.marketplaceManifest?.path || !value.marketplaceManifest?.document) {
    throw new Error(`harness marketplace metadata is incomplete: ${value.id}`);
  }
  return Object.freeze({
    ...value,
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
  return Object.freeze({
    list: () => entries,
    resolve(value) {
      const normalized = String(value || "").trim().toLowerCase();
      const entry = aliases.get(normalized);
      if (!entry) throw new Error(`unknown Agent Harness: ${value}`);
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
