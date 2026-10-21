// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
import { link, open, rename, unlink } from "node:fs/promises";
import { dirname } from "node:path";

const FILE_MODE = 0o600;

async function durableTemporary(target, value) {
  const temporary = `${target}.${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`;
  const handle = await open(temporary, "wx", FILE_MODE);
  try {
    await handle.writeFile(`${JSON.stringify(value)}\n`, "utf8");
    await handle.sync();
  } finally {
    await handle.close();
  }
  return temporary;
}

async function syncDirectory(target) {
  const handle = await open(dirname(target), "r");
  try {
    await handle.sync();
  } catch (error) {
    if (!["EINVAL", "ENOTSUP", "EBADF"].includes(error?.code)) throw error;
  } finally {
    await handle.close();
  }
}

export async function replaceJsonDurably(target, value) {
  const temporary = await durableTemporary(target, value);
  try {
    await rename(temporary, target);
    await syncDirectory(target);
  } catch (error) {
    await unlink(temporary).catch(() => {});
    throw error;
  }
}

export async function createJsonDurably(target, value) {
  const temporary = await durableTemporary(target, value);
  try {
    await link(temporary, target);
    await unlink(temporary);
    await syncDirectory(target);
    return true;
  } catch (error) {
    await unlink(temporary).catch(() => {});
    if (error?.code === "EEXIST") return false;
    throw error;
  }
}

export async function unlinkDurably(target) {
  await unlink(target);
  await syncDirectory(target);
}
