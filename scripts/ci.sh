#!/usr/bin/env bash
# 本地与 CI 共用的检查链。.github/workflows/ci.yml 直接调用本脚本，
# 保证本地跑的命令和远端跑的命令是同一份；推送前先在本地执行：
#
#     bash scripts/ci.sh
#
# 支持 Python 3.12 起。依赖与工具全部锁死在 requirements.txt（由 uv pip compile 从 pyproject.toml
# 生成，--universal，以 3.12 为最低版本）。本地用 uv 建环境并安装：
#
#     uv venv --python 3.12 .venv
#     uv pip install --python .venv -r requirements.txt && uv pip install --python .venv -e . --no-deps
#
# 已在某个环境里（例如 CI）时用 pip 装同一份锁文件也可以：
#
#     python -m pip install -r requirements.txt && python -m pip install -e . --no-deps
#
# 改了 pyproject.toml 的依赖后重新生成锁文件（命令见 requirements.txt 头部）。
set -euo pipefail
cd "$(dirname "$0")/.."

# 本地没激活任何虚拟环境、而项目根目录有 .venv 时，直接用它里面的解释器与工具；
# CI 上没有 .venv，走 runner 自己装好的环境。
if [[ -z "${VIRTUAL_ENV:-}" && -x .venv/bin/python ]]; then
    PATH="$PWD/.venv/bin:$PATH"
fi

PATHS=(habitus)

step() { printf '\n== %s ==\n' "$1"; }

step "Compile"
python -m compileall -q "${PATHS[@]}"

step "Lint"
# 测试也查 lint（不查格式）：新测试文件的 import 顺序、没用到的 import 以前漏过
ruff check "${PATHS[@]}" tests

step "Format"
# 源码格式由 ruff format 守：未格式化的文件直接失败，本地跑 `ruff format habitus` 修
ruff format --check "${PATHS[@]}"

step "Type check (mypy)"
mypy "${PATHS[@]}"

step "Type check (pyright)"
pyright "${PATHS[@]}"

step "Full deterministic test suite with branch coverage"
python -m coverage run --branch -m pytest -q
python -m coverage report

# 宿主插件与 npm 命令行（plugins/）：共享库副本是否与源一致由测试守住
step "Agent plugins and npm CLI (node --test)"
(cd plugins && npm test)

printf '\nAll checks passed.\n'
