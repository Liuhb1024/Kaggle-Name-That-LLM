"""Build an offline Notebook for the reproduced plain LoRA member."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from ntl_baseline.constants import ContractError
from ntl_baseline.data import load_test, validate_submission

from .branch import Array, branch_outputs


def write_plain_submission(
    data_dir: Path, output: Path, score: Callable[[list[str]], Array]
) -> int:
    test = load_test(data_dir)
    _, predictions = branch_outputs(test, score(test["text"].tolist()))
    with output.open("xb") as stream:
        predictions.to_parquet(stream, index=False)
    saved = pd.read_parquet(output)
    validate_submission(saved, test)
    if not saved.equals(predictions):
        raise ContractError("submission differs after Parquet readback")
    return len(saved)


def notebook_document() -> dict[str, Any]:
    from ntl_baseline.notebook import _cell

    root = Path(__file__).resolve().parents[2]
    files = (
        "src/ntl_baseline/__init__.py",
        "src/ntl_baseline/constants.py",
        "src/ntl_baseline/data.py",
        "src/ntl_repro/__init__.py",
        "src/ntl_repro/branch.py",
        "src/ntl_repro/gpu.py",
        "src/ntl_repro/notebook.py",
        "src/ntl_repro/run.py",
        "configs/v31_single_gpu.toml",
    )
    sources = {name: (root / name).read_text(encoding="utf-8") for name in files}
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()}
    bootstrap = """import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
if 'ntl_repro' in sys.modules or 'ntl_baseline' in sys.modules:
    raise RuntimeError('Restart the kernel before running this Notebook.')
"""
    bootstrap += f"SOURCES = {sources!r}\nCHECKSUMS = {hashes!r}\n"
    bootstrap += """
BUNDLE = tempfile.TemporaryDirectory(prefix='ntl-v31-plain-')
ROOT = Path(BUNDLE.name)
for name, source in SOURCES.items():
    if hashlib.sha256(source.encode()).hexdigest() != CHECKSUMS[name]:
        raise RuntimeError(f'Source checksum mismatch: {name}')
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')
sys.path.insert(0, str(ROOT / 'src'))
"""
    execute = """from unittest.mock import patch

import torch

from ntl_repro.branch import load_readout
from ntl_repro.gpu import CudaFeatures
from ntl_repro.notebook import write_plain_submission
from ntl_repro.run import check_assets, load_config

def locate(candidates, evidence, label):
    matching = [path for path in candidates if (path / evidence).is_file()]
    if len(matching) != 1:
        raise RuntimeError(f'Attach exactly the pinned {label} input; found {len(matching)}.')
    return matching[0]

INPUT = Path('/kaggle/input')
DATA = locate([INPUT / 'competitions/name-that-llm', INPUT / 'name-that-llm'],
              'test.csv', 'Name That LLM competition')
BASE = locate([INPUT / 'models/qwen-lm/qwen-3/transformers/8b-base/1',
               INPUT / 'models/qwen-lm/qwen-3/Transformers/8b-base/1'],
              'model.safetensors.index.json', 'Qwen3-8B Base v1')
CHECKPOINT = locate([
    INPUT / 'datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/final',
    INPUT / 'datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/versions/1/final',
    INPUT / 'ntl-qwen3-8b-lora-claude-head/final',
], 'COMPLETE.json', 'LoRA adapter v1')
if not torch.cuda.is_available():
    raise RuntimeError('Select a GPU accelerator.')
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
config = load_config(ROOT / 'configs/v31_single_gpu.toml')
asset_hashes = check_assets(BASE, CHECKPOINT, config)
readout = load_readout(CHECKPOINT / 'readouts.npz')
with patch('peft.tuners.lora.torchao.is_torchao_available', return_value=False):
    extractor = CudaFeatures(BASE, CHECKPOINT)

def score(texts):
    return readout.logits(extractor(texts, config.batch_size))

rows = write_plain_submission(DATA, Path('/kaggle/working/submission.parquet'), score)
print(json.dumps({'rows': rows, 'gpu': torch.cuda.get_device_name(0),
                  'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'asset_sha256': asset_hashes}))
"""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "ntl_v31_source_sha256": hashes,
        },
        "cells": [
            _cell(
                "markdown",
                """# Name That LLM — V31 plain LoRA member

Independent port of the plain `lora24_claude` member from
[V31](https://www.kaggle.com/kernels/scriptcontent/372363760/download).
Attach the competition, Qwen3-8B Base v1 and Zbigniew Kirchner's Apache 2.0
LoRA dataset v1. Use a GPU with Internet disabled. The Notebook scores the
mounted test rows directly and does not fit on competition training data.
""",
                "overview",
            ),
            _cell("code", bootstrap, "bundle-sources"),
            _cell("code", execute, "plain-inference"),
        ],
    }
