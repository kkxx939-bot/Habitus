#!/usr/bin/env node

import { createHash, randomUUID } from "node:crypto";
import { spawnSync } from "node:child_process";
import { cp, mkdir, open, readFile, readdir, realpath, rename, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, join, parse, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPOSITORY_PLUGINS = dirname(fileURLToPath(import.meta.url));
const MARKETPLACE = "m2bos-local";
const PLUGIN = "m2bos-memory";
const PLUGIN_ID = `${PLUGIN}@${MARKETPLACE}`;
const INSTALL_SCHEMA = 1;
const ACTIONS = new Set(["install", "status", "update", "remove"]);

function marketplaceManifests() {
  return {
    codex: {
      name: MARKETPLACE,
      interface: { displayName: "m2bOS Local" },
      plugins: [{
        name: PLUGIN,
        source: { source: "local", path: "./plugins/m2bos-memory" },
        policy: { installation: "AVAILABLE", authentication: "ON_INSTALL" },
        category: "Productivity",
      }],
    },
    claude: {
      name: MARKETPLACE,
      description: "Local m2bOS plugins for Claude Code.",
      owner: { name: "m2bOS" },
      plugins: [{
        name: PLUGIN,
        description: "Single-user local semantic memory for Claude Code.",
        source: "./plugins/m2bos-memory-claude-code",
        category: "productivity",
      }],
    },
  };
}

function dedicatedRoot(root, sourceRoot) {
  const resolvedRoot = resolve(root);
  const resolvedSource = resolve(sourceRoot);
  if (resolvedRoot === parse(resolvedRoot).root || resolvedRoot === resolve(homedir())) {
    throw new Error("marketplace root must be a dedicated child directory");
  }
  const relation = relative(resolvedSource, resolvedRoot);
  if (relation === "" || (!relation.startsWith("..") && !parse(relation).isAbsolute)) {
    throw new Error("marketplace root must be outside the source plugin directory");
  }
  return resolvedRoot;
}

async function syncDirectory(path) {
  const handle = await open(path, "r");
  try { await handle.sync(); }
  catch (error) { if (!["EINVAL", "ENOTSUP", "EBADF"].includes(error?.code)) throw error; }
  finally { await handle.close(); }
}

async function sourceDigest(sourceRoot) {
  const hash = createHash("sha256");
  for (const name of ["m2bos-memory", "m2bos-memory-claude-code", "memory-plugin-shared"]) {
    await digestTree(join(sourceRoot, name), name, hash);
  }
  return hash.digest("hex");
}

async function digestTree(path, logicalPath, hash) {
  const metadata = await stat(path);
  if (metadata.isDirectory()) {
    const entries = (await readdir(path, { withFileTypes: true })).sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) await digestTree(join(path, entry.name), `${logicalPath}/${entry.name}`, hash);
    return;
  }
  if (!metadata.isFile()) throw new Error(`plugin source contains a non-regular entry: ${logicalPath}`);
  hash.update(logicalPath).update("\0").update(await readFile(path)).update("\0");
}

async function stageMarketplace(root, sourceRoot) {
  const parent = dirname(root);
  await mkdir(parent, { recursive: true, mode: 0o700 });
  const staging = join(parent, `.${basename(root)}.staging-${randomUUID()}`);
  await mkdir(join(staging, "plugins"), { recursive: true, mode: 0o700 });
  try {
    for (const name of ["m2bos-memory", "m2bos-memory-claude-code"]) {
      await cp(join(sourceRoot, name), join(staging, "plugins", name), {
        recursive: true,
        force: false,
        errorOnExist: true,
      });
    }
    await mkdir(join(staging, ".agents", "plugins"), { recursive: true, mode: 0o700 });
    await mkdir(join(staging, ".claude-plugin"), { recursive: true, mode: 0o700 });
    const manifests = marketplaceManifests();
    await writeFile(
      join(staging, ".agents", "plugins", "marketplace.json"),
      `${JSON.stringify(manifests.codex, null, 2)}\n`,
      { mode: 0o600 },
    );
    await writeFile(
      join(staging, ".claude-plugin", "marketplace.json"),
      `${JSON.stringify(manifests.claude, null, 2)}\n`,
      { mode: 0o600 },
    );
    await writeFile(
      join(staging, ".m2bos-install.json"),
      `${JSON.stringify({
        schemaVersion: INSTALL_SCHEMA,
        marketplace: MARKETPLACE,
        sourceDigest: await sourceDigest(sourceRoot),
        preparedAt: new Date().toISOString(),
      }, null, 2)}\n`,
      { mode: 0o600 },
    );
    return staging;
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}

async function swapMarketplace(rootValue, { sourceRoot = REPOSITORY_PLUGINS } = {}) {
  const root = dedicatedRoot(rootValue, sourceRoot);
  const staging = await stageMarketplace(root, sourceRoot);
  const backup = `${root}.backup-${randomUUID()}`;
  let hadPrevious = false;
  try {
    try { await rename(root, backup); hadPrevious = true; }
    catch (error) { if (error?.code !== "ENOENT") throw error; }
    try {
      await rename(staging, root);
      await syncDirectory(dirname(root));
    } catch (error) {
      if (hadPrevious) await rename(backup, root);
      throw error;
    }
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
  let finished = false;
  return {
    root,
    async commit() {
      if (finished) return;
      if (hadPrevious) await rm(backup, { recursive: true, force: true });
      await syncDirectory(dirname(root));
      finished = true;
    },
    async rollback() {
      if (finished) return;
      await rm(root, { recursive: true, force: true });
      if (hadPrevious) await rename(backup, root);
      await syncDirectory(dirname(root));
      finished = true;
    },
  };
}

export async function prepareMarketplace(root, options = {}) {
  const transaction = await swapMarketplace(root, options);
  await transaction.commit();
  return transaction.root;
}

class CommandRunner {
  available(command) {
    const result = spawnSync(command, ["--version"], { encoding: "utf8", timeout: 3000 });
    return result.status === 0 && !result.error;
  }

  json(command, args) {
    const result = spawnSync(command, args, { encoding: "utf8", timeout: 10_000, maxBuffer: 16 * 1024 * 1024 });
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed with status ${result.status}`);
    try { return JSON.parse(result.stdout); }
    catch (error) { throw new Error(`${command} returned invalid JSON`, { cause: error }); }
  }

  run(command, args) {
    const result = spawnSync(command, args, { stdio: "inherit", timeout: 30_000 });
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed with status ${result.status}`);
  }
}

function hostDefinition(host) {
  if (host === "codex") return {
    command: "codex",
    marketplaceInventory: ["plugin", "marketplace", "list", "--json"],
    pluginInventory: ["plugin", "list", "--json", "--available"],
    marketplaceAdd: (root) => ["plugin", "marketplace", "add", root],
    marketplaceRemove: ["plugin", "marketplace", "remove", MARKETPLACE],
    pluginAdd: ["plugin", "add", PLUGIN_ID],
    pluginRemove: ["plugin", "remove", PLUGIN_ID, "--json"],
    pluginEnable: null,
  };
  return {
    command: "claude",
    marketplaceInventory: ["plugin", "marketplace", "list", "--json"],
    pluginInventory: ["plugin", "list", "--json"],
    marketplaceAdd: (root) => ["plugin", "marketplace", "add", root],
    marketplaceRemove: ["plugin", "marketplace", "remove", MARKETPLACE],
    pluginAdd: ["plugin", "install", PLUGIN_ID],
    pluginRemove: ["plugin", "uninstall", PLUGIN_ID],
    pluginEnable: ["plugin", "enable", PLUGIN_ID],
  };
}

function marketplaceEntries(value) {
  if (Array.isArray(value?.marketplaces)) return value.marketplaces;
  if (Array.isArray(value)) return value;
  return [];
}

function pluginEntries(value) {
  if (Array.isArray(value?.installed)) return value.installed;
  if (Array.isArray(value?.plugins)) return value.plugins;
  if (Array.isArray(value)) return value;
  return [];
}

function marketplaceSource(entry) {
  return entry?.root || entry?.path || entry?.source?.path || entry?.marketplaceSource?.source || null;
}

function pluginIsInstalled(entry) {
  return entry?.installed !== false;
}

async function inspectHost(host, runner) {
  const definition = hostDefinition(host);
  if (!runner.available(definition.command)) {
    return { host, available: false, marketplaceSource: null, installed: false, enabled: false };
  }
  const marketplaces = marketplaceEntries(runner.json(definition.command, definition.marketplaceInventory));
  const plugins = pluginEntries(runner.json(definition.command, definition.pluginInventory));
  const marketplace = marketplaces.find((entry) => entry?.name === MARKETPLACE);
  const plugin = plugins.find((entry) => (
    entry?.pluginId === PLUGIN_ID
    || (entry?.name === PLUGIN && (entry?.marketplaceName === MARKETPLACE || entry?.marketplace === MARKETPLACE))
  ));
  return {
    host,
    available: true,
    marketplaceSource: marketplaceSource(marketplace),
    installed: Boolean(plugin && pluginIsInstalled(plugin)),
    enabled: Boolean(plugin && plugin?.enabled !== false),
  };
}

function removeHostRegistration(snapshot, runner) {
  const definition = hostDefinition(snapshot.host);
  if (snapshot.installed) runner.run(definition.command, definition.pluginRemove);
  if (snapshot.marketplaceSource) runner.run(definition.command, definition.marketplaceRemove);
}

function installHostRegistration(snapshot, root, runner, { refresh }) {
  const definition = hostDefinition(snapshot.host);
  const sourceChanged = snapshot.marketplaceSource != null
    && resolve(snapshot.marketplaceSource) !== resolve(root);
  const mustRefresh = refresh || sourceChanged || (snapshot.installed && !snapshot.enabled);
  if (mustRefresh) removeHostRegistration(snapshot, runner);
  const marketplacePresent = snapshot.marketplaceSource && !mustRefresh;
  const pluginPresent = snapshot.installed && snapshot.enabled && !mustRefresh;
  if (!marketplacePresent) runner.run(definition.command, definition.marketplaceAdd(root));
  if (!pluginPresent) runner.run(definition.command, definition.pluginAdd);
  if (definition.pluginEnable) runner.run(definition.command, definition.pluginEnable);
}

async function restoreHost(snapshot, runner) {
  if (!snapshot.available) return;
  const current = await inspectHost(snapshot.host, runner);
  try { removeHostRegistration(current, runner); } catch {}
  if (snapshot.marketplaceSource) {
    const definition = hostDefinition(snapshot.host);
    runner.run(definition.command, definition.marketplaceAdd(snapshot.marketplaceSource));
    if (snapshot.installed) {
      runner.run(definition.command, definition.pluginAdd);
      if (definition.pluginEnable) runner.run(definition.command, definition.pluginEnable);
    }
  }
}

function selectedHosts(host) {
  return host === "all" ? ["codex", "claude-code"] : [host];
}

async function marketplaceStatus(root) {
  try {
    const marker = JSON.parse(await readFile(join(root, ".m2bos-install.json"), "utf8"));
    if (marker.schemaVersion !== INSTALL_SCHEMA || marker.marketplace !== MARKETPLACE) throw new Error("invalid marker");
    return { prepared: true, sourceDigest: marker.sourceDigest, preparedAt: marker.preparedAt };
  } catch (error) {
    if (error?.code === "ENOENT") return { prepared: false, sourceDigest: null, preparedAt: null };
    return { prepared: false, sourceDigest: null, preparedAt: null, error: error?.message || String(error) };
  }
}

async function removeMarketplaceRoot(rootValue) {
  const root = dedicatedRoot(rootValue, REPOSITORY_PLUGINS);
  const status = await marketplaceStatus(root);
  if (!status.prepared) return false;
  await rm(root, { recursive: true, force: true });
  await syncDirectory(dirname(root));
  return true;
}

export async function executePluginLifecycle(options, { runner = new CommandRunner(), sourceRoot = REPOSITORY_PLUGINS } = {}) {
  const root = dedicatedRoot(options.root, sourceRoot);
  const hosts = selectedHosts(options.host);
  const snapshots = [];
  for (const host of hosts) {
    const snapshot = await inspectHost(host, runner);
    if (!snapshot.available && options.host !== "all" && options.action !== "status") {
      throw new Error(`${host} CLI is not installed`);
    }
    snapshots.push(snapshot);
  }
  if (options.action === "status") {
    return { action: "status", root, marketplace: await marketplaceStatus(root), hosts: snapshots };
  }
  if (options.action === "remove") {
    const processed = [];
    try {
      for (const snapshot of snapshots.filter((value) => value.available)) {
        processed.push(snapshot);
        removeHostRegistration(snapshot, runner);
      }
    } catch (error) {
      for (const snapshot of processed.reverse()) await restoreHost(snapshot, runner).catch(() => {});
      throw error;
    }
    return { action: "remove", root, removed: await removeMarketplaceRoot(root), hosts: snapshots };
  }

  const transaction = await swapMarketplace(root, { sourceRoot });
  if (options.prepareOnly) {
    await transaction.commit();
    return { action: options.action, root, preparedOnly: true, hosts: snapshots };
  }
  const processed = [];
  try {
    for (const snapshot of snapshots.filter((value) => value.available)) {
      processed.push(snapshot);
      installHostRegistration(snapshot, root, runner, { refresh: options.action === "update" });
    }
    await transaction.commit();
  } catch (error) {
    await transaction.rollback();
    for (const snapshot of processed.reverse()) await restoreHost(snapshot, runner).catch(() => {});
    throw error;
  }
  return { action: options.action, root, preparedOnly: false, hosts: snapshots };
}

export function parseArgs(argv) {
  const values = [...argv];
  const action = ACTIONS.has(values[0]) ? values.shift() : "install";
  const options = {
    action,
    host: "all",
    root: join(homedir(), ".m2bos", "plugin-marketplace"),
    prepareOnly: false,
    json: false,
  };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === "--host") options.host = values[++index];
    else if (value === "--root") options.root = values[++index];
    else if (value === "--prepare-only") options.prepareOnly = true;
    else if (value === "--json") options.json = true;
    else throw new Error(`unknown argument: ${value}`);
  }
  if (!["all", "codex", "claude-code"].includes(options.host)) {
    throw new Error("--host must be all, codex, or claude-code");
  }
  if (options.prepareOnly && !["install", "update"].includes(options.action)) {
    throw new Error("--prepare-only is valid only for install or update");
  }
  return options;
}

async function isMainModule() {
  if (!process.argv[1]) return false;
  try { return await realpath(fileURLToPath(import.meta.url)) === await realpath(process.argv[1]); }
  catch { return false; }
}

if (await isMainModule()) {
  try {
    const options = parseArgs(process.argv.slice(2));
    const result = await executePluginLifecycle(options);
    if (options.json || options.action === "status") process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    else process.stdout.write(`m2bOS plugin ${options.action} completed at ${result.root}\n`);
  } catch (error) {
    process.stderr.write(`m2bOS plugin lifecycle failed: ${error?.message || error}\n`);
    process.exitCode = 1;
  }
}
