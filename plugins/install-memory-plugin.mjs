#!/usr/bin/env node

import { createHash, randomUUID } from "node:crypto";
import { spawnSync } from "node:child_process";
import { cp, mkdir, open, readFile, readdir, realpath, rename, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, join, parse, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  HARNESS_REGISTRY,
  MARKETPLACE_NAME,
  PLUGIN_ID,
  PLUGIN_NAME,
  publicHarnessDescriptor,
} from "./harnesses.mjs";

const REPOSITORY_PLUGINS = dirname(fileURLToPath(import.meta.url));
const INSTALL_SCHEMA = 1;
const ACTIONS = new Set(["install", "status", "update", "remove", "harnesses"]);

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

async function sourceDigest(sourceRoot, registry) {
  const hash = createHash("sha256");
  const names = [
    ...new Set(registry.list().map((definition) => definition.pluginDirectory)),
    "memory-plugin-shared",
    "harnesses.mjs",
  ];
  for (const name of names.sort()) {
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

async function stageMarketplace(root, sourceRoot, registry) {
  const parent = dirname(root);
  await mkdir(parent, { recursive: true, mode: 0o700 });
  const staging = join(parent, `.${basename(root)}.staging-${randomUUID()}`);
  await mkdir(join(staging, "plugins"), { recursive: true, mode: 0o700 });
  try {
    const definitions = registry.list();
    for (const name of [...new Set(definitions.map((definition) => definition.pluginDirectory))]) {
      await cp(join(sourceRoot, name), join(staging, "plugins", name), {
        recursive: true,
        force: false,
        errorOnExist: true,
      });
    }
    for (const definition of definitions) {
      const destination = join(staging, definition.marketplaceManifest.path);
      await mkdir(dirname(destination), { recursive: true, mode: 0o700 });
      await writeFile(
        destination,
        `${JSON.stringify(definition.marketplaceManifest.document, null, 2)}\n`,
        { mode: 0o600 },
      );
    }
    await writeFile(
      join(staging, ".m2bos-install.json"),
      `${JSON.stringify({
        schemaVersion: INSTALL_SCHEMA,
        marketplace: MARKETPLACE_NAME,
        sourceDigest: await sourceDigest(sourceRoot, registry),
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

async function swapMarketplace(
  rootValue,
  { sourceRoot = REPOSITORY_PLUGINS, registry = HARNESS_REGISTRY } = {},
) {
  const root = dedicatedRoot(rootValue, sourceRoot);
  const staging = await stageMarketplace(root, sourceRoot, registry);
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

function commandArguments(values, root) {
  return values.map((value) => value === "{root}" ? root : value);
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

async function inspectHarness(definition, runner) {
  if (!runner.available(definition.command)) {
    return {
      harness: definition.id,
      available: false,
      marketplaceSource: null,
      installed: false,
      enabled: false,
    };
  }
  const marketplaces = marketplaceEntries(runner.json(definition.command, definition.marketplaceInventory));
  const plugins = pluginEntries(runner.json(definition.command, definition.pluginInventory));
  const marketplace = marketplaces.find((entry) => entry?.name === MARKETPLACE_NAME);
  const plugin = plugins.find((entry) => (
    entry?.pluginId === PLUGIN_ID
    || (entry?.name === PLUGIN_NAME
      && (entry?.marketplaceName === MARKETPLACE_NAME || entry?.marketplace === MARKETPLACE_NAME))
  ));
  return {
    harness: definition.id,
    available: true,
    marketplaceSource: marketplaceSource(marketplace),
    installed: Boolean(plugin && pluginIsInstalled(plugin)),
    enabled: Boolean(plugin && plugin?.enabled !== false),
  };
}

function removeHarnessRegistration(snapshot, definition, runner) {
  if (snapshot.installed) runner.run(definition.command, definition.pluginRemove);
  if (snapshot.marketplaceSource) runner.run(definition.command, definition.marketplaceRemove);
}

function installHarnessRegistration(snapshot, definition, root, runner, { refresh }) {
  const sourceChanged = snapshot.marketplaceSource != null
    && resolve(snapshot.marketplaceSource) !== resolve(root);
  const mustRefresh = refresh || sourceChanged || (snapshot.installed && !snapshot.enabled);
  if (mustRefresh) removeHarnessRegistration(snapshot, definition, runner);
  const marketplacePresent = snapshot.marketplaceSource && !mustRefresh;
  const pluginPresent = snapshot.installed && snapshot.enabled && !mustRefresh;
  if (!marketplacePresent) {
    runner.run(definition.command, commandArguments(definition.marketplaceAdd, root));
  }
  if (!pluginPresent) runner.run(definition.command, definition.pluginAdd);
  if (definition.pluginEnable) runner.run(definition.command, definition.pluginEnable);
}

async function restoreHarness(snapshot, definition, runner) {
  if (!snapshot.available) return;
  const current = await inspectHarness(definition, runner);
  try { removeHarnessRegistration(current, definition, runner); } catch {}
  if (snapshot.marketplaceSource) {
    runner.run(
      definition.command,
      commandArguments(definition.marketplaceAdd, snapshot.marketplaceSource),
    );
    if (snapshot.installed) {
      runner.run(definition.command, definition.pluginAdd);
      if (definition.pluginEnable) runner.run(definition.command, definition.pluginEnable);
    }
  }
}

function selectedHarnesses(values, registry) {
  const requested = values.length === 0 ? ["all"] : values;
  if (requested.includes("all")) return registry.list();
  const unique = new Map();
  for (const value of requested) {
    const definition = registry.resolve(value);
    unique.set(definition.id, definition);
  }
  return [...unique.values()];
}

async function marketplaceStatus(root) {
  try {
    const marker = JSON.parse(await readFile(join(root, ".m2bos-install.json"), "utf8"));
    if (marker.schemaVersion !== INSTALL_SCHEMA || marker.marketplace !== MARKETPLACE_NAME) throw new Error("invalid marker");
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

export async function executePluginLifecycle(
  options,
  {
    runner = new CommandRunner(),
    sourceRoot = REPOSITORY_PLUGINS,
    registry = HARNESS_REGISTRY,
  } = {},
) {
  if (options.action === "harnesses") {
    return {
      action: "harnesses",
      harnesses: registry.list().map((definition) => publicHarnessDescriptor(
        definition,
        { available: runner.available(definition.command) },
      )),
    };
  }
  const root = dedicatedRoot(options.root, sourceRoot);
  const requestedHarnesses = options.harnesses
    ?? (options.host ? [options.host] : []);
  const definitions = selectedHarnesses(requestedHarnesses, registry);
  const allowUnavailable = requestedHarnesses.length === 0 || requestedHarnesses.includes("all");
  const snapshots = [];
  for (const definition of definitions) {
    const snapshot = await inspectHarness(definition, runner);
    if (
      !snapshot.available
      && !allowUnavailable
      && options.action !== "status"
    ) {
      throw new Error(`${definition.id} CLI is not installed`);
    }
    snapshots.push({ snapshot, definition });
  }
  if (options.action === "status") {
    return {
      action: "status",
      root,
      marketplace: await marketplaceStatus(root),
      harnesses: snapshots.map((value) => value.snapshot),
    };
  }
  if (options.action === "remove") {
    const processed = [];
    try {
      for (const value of snapshots.filter((item) => item.snapshot.available)) {
        processed.push(value);
        removeHarnessRegistration(value.snapshot, value.definition, runner);
      }
    } catch (error) {
      for (const value of processed.reverse()) {
        await restoreHarness(value.snapshot, value.definition, runner).catch(() => {});
      }
      throw error;
    }
    return {
      action: "remove",
      root,
      removed: await removeMarketplaceRoot(root),
      harnesses: snapshots.map((value) => value.snapshot),
    };
  }

  const transaction = await swapMarketplace(root, { sourceRoot, registry });
  if (options.prepareOnly) {
    await transaction.commit();
    return {
      action: options.action,
      root,
      preparedOnly: true,
      harnesses: snapshots.map((value) => value.snapshot),
    };
  }
  const processed = [];
  try {
    for (const value of snapshots.filter((item) => item.snapshot.available)) {
      processed.push(value);
      installHarnessRegistration(
        value.snapshot,
        value.definition,
        root,
        runner,
        { refresh: options.action === "update" },
      );
    }
    await transaction.commit();
  } catch (error) {
    await transaction.rollback();
    for (const value of processed.reverse()) {
      await restoreHarness(value.snapshot, value.definition, runner).catch(() => {});
    }
    throw error;
  }
  return {
    action: options.action,
    root,
    preparedOnly: false,
    harnesses: snapshots.map((value) => value.snapshot),
  };
}

export function parseArgs(argv) {
  const values = [...argv];
  const action = ACTIONS.has(values[0]) ? values.shift() : "install";
  const options = {
    action,
    harnesses: [],
    root: join(homedir(), ".m2bos", "plugin-marketplace"),
    prepareOnly: false,
    json: false,
  };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === "--host" || value === "--harness") options.harnesses.push(values[++index]);
    else if (value === "--root") options.root = values[++index];
    else if (value === "--prepare-only") options.prepareOnly = true;
    else if (value === "--json") options.json = true;
    else if (value === "-h" || value === "--help") options.help = true;
    else throw new Error(`unknown argument: ${value}`);
  }
  if (options.harnesses.some((value) => value == null)) {
    throw new Error("--harness requires an Agent Harness id");
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
    if (options.help) {
      process.stdout.write(
        "usage: m2bos-plugin [install|status|update|remove|harnesses] "
        + "[--harness ID|--host ID] [--root PATH] [--prepare-only] [--json]\n",
      );
      process.exitCode = 0;
    } else {
    const result = await executePluginLifecycle(options);
    if (options.json || ["status", "harnesses"].includes(options.action)) {
      process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    }
    else process.stdout.write(`m2bOS plugin ${options.action} completed at ${result.root}\n`);
    }
  } catch (error) {
    process.stderr.write(`m2bOS plugin lifecycle failed: ${error?.message || error}\n`);
    process.exitCode = 1;
  }
}
