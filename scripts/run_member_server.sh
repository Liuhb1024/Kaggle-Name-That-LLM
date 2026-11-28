#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src"
export KAGGLEHUB_CACHE="$PWD/artifacts/v31-asset-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
# NumPy 2.2's AVX2 argsort orders equal-sized prompt groups differently on the lab CPU.
# Keep the preregistered GroupKFold assignment in both the CPU and GPU Python workers.
if [[ "$(uname -m)" == x86_64 ]]; then
  export NPY_DISABLE_CPU_FEATURES="${NPY_DISABLE_CPU_FEATURES:+$NPY_DISABLE_CPU_FEATURES,}AVX2"
fi
gpu_python=../.venv-gpu-check/bin/python
args=("$@")
want_commit=""
dry_run=0
while (($#)); do
  case "$1" in
    --commit) want_commit=${2:?missing commit}; shift 2 ;;
    --python) gpu_python=${2:?missing Python path}; shift 2 ;;
    --run-root|--reuse-stage) shift 2 ;;
    --resume) shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$want_commit" =~ ^[0-9a-f]{40}$ ]] || { echo 'Supply --commit with a full SHA' >&2; exit 2; }
if ((dry_run)); then
  exec python3 -S -m ntl_member.plan
fi
[[ "$(git rev-parse HEAD)" == "$want_commit" ]] || { echo 'Switch to the requested SHA first' >&2; exit 1; }
# A completed report may be awaiting transfer when resuming. Only an exact generated report
# is accepted later by the exporter. Source files and the index must remain clean.
if [[ -n "$(git status --porcelain --untracked-files=all -- . ':(exclude)reports/member-q8pool-v1/*')" ]]; then
  echo 'Uncommitted project files; run from a clean checkout' >&2; exit 1
fi
[[ -x "$gpu_python" ]] || { echo 'Missing dedicated CUDA Python' >&2; exit 1; }
"$gpu_python" - <<'PY'
import sys
from importlib.metadata import version
if sys.prefix == sys.base_prefix or version("torch") != "2.12.1+cu126":
    sys.exit("Expected the existing dedicated GPU environment with torch 2.12.1+cu126")
PY
if ! "$gpu_python" - <<'PY'
import importlib.metadata as m
from pathlib import Path
for line in Path("requirements-v31-gpu.txt").read_text().splitlines():
    pkg, pin = line.split("==")
    try:
        ok = m.version(pkg) == pin
    except m.PackageNotFoundError:
        ok = False
    if not ok:
        raise SystemExit(1)
PY
then
  "$gpu_python" -m pip install --only-binary=:all: -r requirements-v31-gpu.txt
fi
"$gpu_python" -m pip check
uv_bin=$(command -v uv || true)
if [[ -z "$uv_bin" ]]; then
  tools_venv="$PWD/artifacts/member-tools"
  if [[ ! -x "$tools_venv/bin/uv" ]]; then
    "$gpu_python" -m venv "$tools_venv"
    "$tools_venv/bin/python" -m pip install --only-binary=:all: 'uv==0.11.12'
  fi
  uv_bin="$tools_venv/bin/uv"
fi
export UV_PROJECT_ENVIRONMENT="$PWD/artifacts/member-cpu-venv"
"$uv_bin" sync --locked --no-dev
exec "$uv_bin" run --offline --locked --no-dev python -m ntl_member.pipeline --python "$gpu_python" "${args[@]}"
