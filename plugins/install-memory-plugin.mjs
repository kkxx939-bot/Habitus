import { createHash, randomUUID } from "node:crypto";
import { cp, mkdir, open, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import path, { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { makePrivateDirectory } from "./memory-plugin-shared/lib/private-dir.mjs";
import { HARNESS_REGISTRY, MARKETPLACE_NAME, publicHarnessDescriptor } from "./harnesses.mjs";
import { CommandRunner, harnessPresent, installBlocker, registrationOf } from "./registrations.mjs";

function inspectHarness(definition, runner, env) {
  return registrationOf(definition).inspect(definition, runner, env);
}

// 出错时记下"哪个宿主、哪一步"（W5）：只往错误对象上加 habitusStep，不改原报错文字（调用方与测试照旧按原文判），
// 命令行据此告诉用户卡在哪。嵌套时最里层先记的为准。
export async function atStep(step, work) {
  try {
    return await work();
  } catch (error) {
    if (canAnnotate(error) && !error.habitusStep) error.habitusStep = step;
    throw error;
  }
}

// 失败后逐个恢复宿主注册：恢复本身失败的不再静默吞掉，挂到原错误的 restoreFailures 上，让用户知道哪个宿主要手动查
async function restoreAll(processed, runner, env, error) {
  const failures = [];
  for (const value of processed.reverse()) {
    try {
      await registrationOf(value.definition).restore(value.snapshot, value.definition, runner, env);
    } catch (restoreError) {
      failures.push({ harness: value.definition.id, message: restoreError?.message || String(restoreError) });
    }
  }
  if (failures.length && canAnnotate(error)) error.restoreFailures = failures;
}

// 被冻结或不是对象的错误不加注记，原样抛出（不能因为加注记把原错误换成 TypeError）
function canAnnotate(error) {
  return error !== null && typeof error === "object" && Object.isExtensible(error);
}

const REPOSITORY_PLUGINS = dirname(fileURLToPath(import.meta.url));
const INSTALL_SCHEMA = 1;
const ACTIONS = new Set(["install", "status", "update", "remove", "harnesses"]);

// paths 只给测试换成 path.win32（在 POSIX 上验盘符路径）；运行时就是本平台的 node:path
export function dedicatedRoot(root, sourceRoot, paths = path) {
  const resolvedRoot = paths.resolve(root);
  const resolvedSource = paths.resolve(sourceRoot);
  if (resolvedRoot === paths.parse(resolvedRoot).root || resolvedRoot === paths.resolve(homedir())) {
    throw new Error("marketplace root must be a dedicated child directory");
  }
  // 不同盘（Windows 上 C: 与 D:）时 relative() 给的是绝对路径：那也是在源目录外面
  const relation = paths.relative(resolvedSource, resolvedRoot);
  if (relation === "" || (!relation.startsWith("..") && !paths.isAbsolute(relation))) {
    throw new Error("marketplace root must be outside the source plugin directory");
  }
  return resolvedRoot;
}

// Windows 没有目录 fsync（目录句柄 FlushFileBuffers 报 EPERM）：改名的落盘靠 NTFS 元数据日志，与 Python 耐久层 Windows 份同一口径
async function syncDirectory(path) {
  if (process.platform === "win32") return;
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
  await makePrivateDirectory(parent);
  const staging = join(parent, `.${basename(root)}.staging-${randomUUID()}`);
  await makePrivateDirectory(join(staging, "plugins"));
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

// install / update 整个换掉根目录：只换不存在的、空的、或带合法安装标记的（与 remove 只删带标记的同口径），
// 别的目录里可能是用户自己的东西
async function requireReplaceable(root, sourceRoot, registry) {
  let entries;
  try { entries = await readdir(root); }
  catch (error) {
    if (error?.code === "ENOENT") return;
    if (error?.code === "ENOTDIR") throw new Error(`marketplace root ${root} exists and is not a directory`);
    throw error;
  }
  if (entries.length === 0) return;
  if ((await marketplaceStatus(root, { sourceRoot, registry })).prepared) return;
  throw new Error(
    `marketplace root ${root} is not empty and has no Habitus install marker (.habitus-install.json);`
    + " refusing to replace it. Move it aside, then run the command again",
  );
}

async function swapMarketplace(
  rootValue,
  { sourceRoot = REPOSITORY_PLUGINS, registry = HARNESS_REGISTRY } = {},
) {
  const root = dedicatedRoot(rootValue, sourceRoot);
  await requireReplaceable(root, sourceRoot, registry);
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
    env = process.env,
    runner = new CommandRunner(env),
    sourceRoot = REPOSITORY_PLUGINS,
    registry = HARNESS_REGISTRY,
  } = {},
) {
  if (options.action === "harnesses") {
    return {
      action: "harnesses",
      harnesses: await Promise.all(registry.list().map(async (definition) => publicHarnessDescriptor(
        definition,
        { available: await harnessPresent(definition, runner, env) },
      ))),
    };
  }
  const root = dedicatedRoot(options.root, sourceRoot);
  const requestedHarnesses = options.harnesses
    ?? (options.host ? [options.host] : []);
  const definitions = selectedHarnesses(requestedHarnesses, registry);
  const allowUnavailable = requestedHarnesses.length === 0 || requestedHarnesses.includes("all");
  const registering = ["install", "update"].includes(options.action) && !options.prepareOnly;
  // update 不点名宿主：只重新注册已经注册过的（装着的，或入口还在、指向的副本没了的）；没装过的、卸掉了的不碰
  const registeredOnly = options.action === "update" && requestedHarnesses.length === 0;
  const wanted = (snapshot) => snapshot.available && (!registeredOnly || snapshot.installed || snapshot.dangling === true);
  const snapshots = [];
  // 不点名时本机有、却装不了的宿主（要跑的宿主命令不在 PATH 上）跳过并说明原因，不连累别的宿主
  const skipped = new Map();
  for (const definition of definitions) {
    const snapshot = await atStep(
      { harness: definition.id, action: "检查本机状态" },
      () => inspectHarness(definition, runner, env),
    );
    if (
      !snapshot.available
      && !allowUnavailable
      && options.action !== "status"
    ) {
      throw new Error(`${definition.id} CLI is not installed`);
    }
    const blocker = registering && wanted(snapshot) ? installBlocker(definition, runner, env) : null;
    if (blocker && !allowUnavailable) throw new Error(`${definition.id} CLI is not installed: ${blocker}`);
    if (blocker) skipped.set(definition.id, blocker);
    snapshots.push({ snapshot, definition });
  }
  // 报给人看的是动手之后的状态
  const inspectSelected = async () => (await Promise.all(definitions.map((definition) => inspectHarness(definition, runner, env))))
    .map((snapshot) => (skipped.has(snapshot.harness) ? { ...snapshot, skipped: skipped.get(snapshot.harness) } : snapshot));
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
        await atStep(
          { harness: value.definition.id, action: `卸载（${value.definition.registration}）` },
          () => registrationOf(value.definition).remove(value.snapshot, value.definition, runner, env),
        );
      }
    } catch (error) {
      await restoreAll(processed, runner, env, error);
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
      removed: stillUsed
        ? false
        : await atStep({ action: `删除插件副本 ${root}` }, () => removeMarketplaceRoot(root, sourceRoot, registry)),
      harnesses: await inspectSelected(),
    };
  }

  const transaction = await atStep(
    { action: `暂存插件副本到 ${root}` },
    () => swapMarketplace(root, { sourceRoot, registry }),
  );
  if (options.prepareOnly) {
    await atStep({ action: "提交插件副本" }, () => transaction.commit());
    return {
      action: options.action,
      root,
      preparedOnly: true,
      harnesses: await inspectSelected(),
    };
  }
  const processed = [];
  try {
    for (const value of snapshots.filter((item) => wanted(item.snapshot) && !skipped.has(item.definition.id))) {
      processed.push(value);
      await atStep(
        { harness: value.definition.id, action: `${options.action === "update" ? "重新注册" : "注册"}（${value.definition.registration}）` },
        () => registrationOf(value.definition).install(
          value.snapshot,
          value.definition,
          root,
          runner,
          { refresh: options.action === "update", env },
        ),
      );
    }
    await atStep({ action: "提交插件副本" }, () => transaction.commit());
  } catch (error) {
    // 插件副本回滚失败也不能盖掉原错误、不能跳过恢复宿主注册：记在 marketplaceRollbackFailure 上一起报
    const rollbackError = await transaction.rollback().then(() => null, (value) => value);
    await restoreAll(processed, runner, env, error);
    if (rollbackError && canAnnotate(error)) {
      error.marketplaceRollbackFailure = { root, message: rollbackError?.message || String(rollbackError) };
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
    // --root 只给测试换 marketplace 根（用户只用默认根：status / doctor / update / remove 都只认默认根），不写进 --help 与 README
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
