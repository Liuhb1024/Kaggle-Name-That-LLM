#!/usr/bin/env bash
set -euo pipefail
umask 077

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
repo=$PWD
mode=${1:-}
if [[ "$mode" == --cached-auxiliary ]]; then
  shift
fi
python=${1:-"$repo/../.venv-gpu-check/bin/python"}
if [[ $# -gt 1 || ! -x "$python" ]]; then
  printf 'Usage: bash scripts/run_v31_server_smoke.sh [--cached-auxiliary] [/path/to/CUDA PyTorch virtual environment python]\n' >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$repo/src"
export PYTHONUNBUFFERED=1
export KAGGLEHUB_CACHE="$repo/artifacts/v31-asset-cache"

if [[ "$mode" == --cached-auxiliary ]]; then
  run_dir="artifacts/v31-claude-heads-$(date -u +%Y%m%dT%H%M%S)-$$"
  printf 'Completion report: %s/completed.json\n' "$run_dir"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$python" -m ntl_repro.run --synthetic --auxiliary \
    --base-dir "$KAGGLEHUB_CACHE/models/qwen-lm/qwen-3/Transformers/8b-base/1" \
    --checkpoint-dir "$KAGGLEHUB_CACHE/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/versions/1/final" \
    --anchor-data-dir data/raw/name-that-llm --run-dir "$run_dir"
  grep -E 'anchor_relative_error|claude_gate_logprobs|centered_probabilities' "$run_dir/completed.json"
  exit
fi

"$python" - <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

if sys.prefix == sys.base_prefix:
    sys.exit("Use the dedicated CUDA PyTorch virtual environment.")
try:
    installed = version("torch")
except PackageNotFoundError:
    sys.exit("PyTorch is missing from the CUDA virtual environment.")
if installed != "2.12.1+cu126":
    sys.exit("Expected the verified server PyTorch build 2.12.1+cu126.")
import torch

if torch.cuda.device_count() != 1 or (torch.ones(1, device="cuda") + 1).item() != 2:
    sys.exit("CUDA preflight failed; expose one working NVIDIA card.")
print(f"CUDA preflight: {torch.cuda.get_device_name(0)}, PyTorch {installed}")
PY

echo 'Installing/checking GPU inference dependencies in the dedicated virtual environment...'
"$python" -m pip --isolated install --index-url https://pypi.org/simple \
  --only-binary=:all: -r requirements-v31-gpu.txt
"$python" -m pip check

"$python" - <<'PY'
import os
import sys
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from uuid import uuid4

import kagglehub

from ntl_repro.run import CHECKPOINT_FILES, ROOT

import_module("ntl_repro.gpu")

base_handle = "qwen-lm/qwen-3/Transformers/8b-base/1"
checkpoint_handle = "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/versions/1"
try:
    print("Preparing the published v1 adapter, tokenizer, and readout...")
    marker = Path(kagglehub.dataset_download(checkpoint_handle, path="final/COMPLETE.json"))
    checkpoint = marker.parent
    for name in CHECKPOINT_FILES:
        if name != "COMPLETE.json":
            downloaded = Path(kagglehub.dataset_download(checkpoint_handle, path=f"final/{name}"))
            if downloaded != checkpoint / name:
                raise ValueError("Unexpected adapter cache layout")
    print("Preparing the pinned Qwen3-8B base model (about 16.4 GB on first run)...")
    base = Path(kagglehub.model_download(base_handle))
except Exception as exc:
    raise SystemExit(
        f"Kaggle asset download failed ({type(exc).__name__}). Check server network and access "
        "to both pinned v1 Kaggle resources. Do not paste credentials into logs or chat."
    ) from None

run_name = f"v31-claude-smoke-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
run_dir = ROOT / "artifacts" / run_name
print(f"Running local inference. Completion report: {run_dir / 'completed.json'}")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
os.execv(
    sys.executable,
    [
        sys.executable, "-m", "ntl_repro.run", "--synthetic", "--base-dir", str(base),
        "--checkpoint-dir", str(checkpoint), "--run-dir", str(run_dir),
    ],
)
PY
