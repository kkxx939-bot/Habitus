#!/usr/bin/env bash
# 本地与 CI 共用的检查链。.github/workflows/ci.yml 直接调用本脚本，
# 保证本地跑的命令和远端跑的命令是同一份；推送前先在本地执行：
#
#     bash scripts/ci.sh
#
# 依赖与工具全部锁死在 requirements.txt（Python 3.13 下由 pip-compile 从 pyproject.toml 生成），安装：
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

PATHS=(Config Runtime ModelClient pre conversation memory behavior prediction infrastructure foundation integrations benchmark)

step() { printf '\n== %s ==\n' "$1"; }

step "Compile"
python -m compileall -q "${PATHS[@]}"

step "Lint"
ruff check "${PATHS[@]}"

step "Type check (mypy)"
mypy "${PATHS[@]}"

step "Type check (pyright)"
pyright "${PATHS[@]}"

step "Full deterministic test suite with branch coverage"
python -m coverage run --branch -m pytest -q
python -m coverage report

printf '\nAll checks passed.\n'
