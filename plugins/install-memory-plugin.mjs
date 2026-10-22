import { createHash, randomUUID } from "node:crypto";
import { cp, mkdir, open, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, join, parse, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { HARNESS_REGISTRY, MARKETPLACE_NAME, publicHarnessDescriptor } from "./harnesses.mjs";
import { CommandRunner, registrationOf } from "./registrations.mjs";

function inspectHarness(definition, runner, env) {
  return registrationOf(definition).inspect(definition, runner, env);
}

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

export async function sourceDigest(sourceRoot = REPOSITORY_PLUGINS, registry = HARNESS_REGISTRY) {
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
    for (const definition of definitions.filter((item) => item.marketplaceManifest)) {
      const destination = join(staging, definition.marketplaceManifest.path);
      await mkdir(dirname(destination), { recursive: true, mode: 0o700 });
      await writeFile(
        destination,
        `${JSON.stringify(definition.marketplaceManifest.document, null, 2)}\n`,
        { mode: 0o600 },
      );
    }
    await writeFile(
      join(staging, ".habitus-install.json"),
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

// stale：宿主跑的是 marketplace 根里的暂存副本，npm 包升级后两边摘要对不上，要 `habitus plugin update` 才换新
export async function marketplaceStatus(root, { sourceRoot = REPOSITORY_PLUGINS, registry = HARNESS_REGISTRY } = {}) {
  let marker;
  try {
    marker = JSON.parse(await readFile(join(root, ".habitus-install.json"), "utf8"));
    if (marker.schemaVersion !== INSTALL_SCHEMA || marker.marketplace !== MARKETPLACE_NAME) throw new Error("invalid marker");
  } catch (error) {
    if (error?.code === "ENOENT") return { prepared: false, stale: false, sourceDigest: null, preparedAt: null };
    return { prepared: false, stale: false, sourceDigest: null, preparedAt: null, error: error?.message || String(error) };
  }
  const current = await sourceDigest(sourceRoot, registry);
  return { prepared: true, stale: marker.sourceDigest !== current, sourceDigest: marker.sourceDigest, preparedAt: marker.preparedAt };
}

export function defaultMarketplaceRoot(env = process.env) {
  return join(env.HOME || homedir(), ".habitus", "plugin-marketplace");
}

async function removeMarketplaceRoot(rootValue, sourceRoot, registry) {
  const root = dedicatedRoot(rootValue, sourceRoot);
  const status = await marketplaceStatus(root, { sourceRoot, registry });
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
    env = process.env,
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
    const snapshot = await inspectHarness(definition, runner, env);
    if (
      !snapshot.available
      && !allowUnavailable
      && options.action !== "status"
    ) {
      throw new Error(`${definition.id} CLI is not installed`);
    }
    snapshots.push({ snapshot, definition });
  }
  // 报给人看的是动手之后的状态
  const inspectSelected = () => Promise.all(definitions.map((definition) => inspectHarness(definition, runner, env)));
  if (options.action === "status") {
    return {
      action: "status",
      root,
      marketplace: await marketplaceStatus(root, { sourceRoot, registry }),
      harnesses: snapshots.map((value) => value.snapshot),
    };
  }
  if (options.action === "remove") {
    const processed = [];
    try {
      // 不靠宿主 CLI 的注册方式（文件入口）CLI 不在也照样删
      const removable = snapshots.filter((item) => item.snapshot.available || !registrationOf(item.definition).needsCommand);
      for (const value of removable) {
        processed.push(value);
        await registrationOf(value.definition).remove(value.snapshot, value.definition, runner, env);
      }
    } catch (error) {
      for (const value of processed.reverse()) {
        await registrationOf(value.definition).restore(value.snapshot, value.definition, runner, env).catch(() => {});
      }
      throw error;
    }
    // 根目录是所有宿主共用的：只卸一部分时，没被选中的宿主还指着它就留着
    const selected = new Set(definitions.map((definition) => definition.id));
    const others = await Promise.all(registry.list()
      .filter((definition) => !selected.has(definition.id))
      .map((definition) => inspectHarness(definition, runner, env)));
    const stillUsed = others.some((snapshot) => (
      snapshot.marketplaceSource != null && resolve(snapshot.marketplaceSource) === root
    ));
    return {
      action: "remove",
      root,
      removed: stillUsed ? false : await removeMarketplaceRoot(root, sourceRoot, registry),
      harnesses: await inspectSelected(),
    };
  }

  const transaction = await swapMarketplace(root, { sourceRoot, registry });
  if (options.prepareOnly) {
    await transaction.commit();
    return {
      action: options.action,
      root,
      preparedOnly: true,
      harnesses: await inspectSelected(),
    };
  }
  const processed = [];
  try {
    for (const value of snapshots.filter((item) => item.snapshot.available)) {
      processed.push(value);
      await registrationOf(value.definition).install(
        value.snapshot,
        value.definition,
        root,
        runner,
        { refresh: options.action === "update", env },
      );
    }
    await transaction.commit();
  } catch (error) {
    await transaction.rollback();
    for (const value of processed.reverse()) {
      await registrationOf(value.definition).restore(value.snapshot, value.definition, runner, env).catch(() => {});
    }
    throw error;
  }
  return {
    action: options.action,
    root,
    preparedOnly: false,
    harnesses: await inspectSelected(),
  };
}

export function parseArgs(argv, env = process.env) {
  const values = [...argv];
  const action = ACTIONS.has(values[0]) ? values.shift() : "install";
  const options = {
    action,
    harnesses: [],
    root: defaultMarketplaceRoot(env),
    prepareOnly: false,
    json: false,
  };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === "--host" || value === "--harness") options.harnesses.push(values[++index]);
    else if (value === "--root") {
      options.root = values[++index];
      if (options.root == null || options.root.startsWith("--")) throw new Error("--root requires a directory");
    }
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
