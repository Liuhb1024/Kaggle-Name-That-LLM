"""Generate a self-contained Notebook from the same reviewed package sources."""

from pathlib import Path
from typing import Any

from .artifacts import source_hashes


def _cell(kind: str, source: str, cell_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cell_type": kind,
        "id": cell_id,
        "metadata": {},
        "source": source,
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def notebook_document() -> dict[str, Any]:
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    bootstrap = """# Reviewed local package sources, embedded for offline portability.
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

if 'ntl_baseline' in sys.modules:
    raise RuntimeError('Restart the kernel before running this self-contained Notebook.')
_bundle_sources = {}
"""
    bootstrap += "\n".join(
        f"_bundle_sources[{name!r}] = {source!r}" for name, source in sources.items()
    )
    bootstrap += f"\n_expected_source_hashes = {source_hashes()!r}\n"
    bootstrap += """
for _name, _source in _bundle_sources.items():
    if hashlib.sha256(_source.encode('utf-8')).hexdigest() != _expected_source_hashes[_name]:
        raise RuntimeError('Embedded source checksum mismatch')
_bundle_tmp = tempfile.TemporaryDirectory(prefix='ntl-baseline-')
_package_dir = Path(_bundle_tmp.name) / 'ntl_baseline'
_package_dir.mkdir()
for _name, _source in _bundle_sources.items():
    (_package_dir / _name).write_text(_source, encoding='utf-8')
sys.path.insert(0, _bundle_tmp.name)

from ntl_baseline.config import BaselineConfig
from ntl_baseline.data import load_test, validate_submission
from ntl_baseline.workflow import run_baseline
"""
    configuration = """# Only explicit, known competition mounts; no recursive first-match lookup.
if os.environ.get('NTL_DATA_DIR'):
    DATA_DIR = Path(os.environ['NTL_DATA_DIR'])
else:
    _candidates = [
        Path('/kaggle/input/competitions/name-that-llm'),
        Path('/kaggle/input/name-that-llm'),
    ]
    _existing = [p for p in _candidates if (p / 'train.csv').is_file()]
    if len(_existing) != 1:
        raise RuntimeError('Attach the competition input or explicitly set NTL_DATA_DIR.')
    DATA_DIR = _existing[0]
RUN_DIR = Path(os.environ.get('NTL_RUN_DIR', str(Path.cwd() / 'baseline-run')))
CONFIG = BaselineConfig()
"""
    execute = """# Fresh directory required. No network calls, installs or pretrained weights.
# The public sample may be absent or stale during hidden-test reruns.
# Final predictions are still checked against every ID in the actual test.csv.
result = run_baseline(DATA_DIR, RUN_DIR, CONFIG, require_sample=False)
print(json.dumps(result, indent=2))
"""
    export = """import pandas as pd

_submission_source = RUN_DIR / 'prediction' / 'submission.parquet'
SUBMISSION_PATH = Path.cwd() / 'submission.parquet'
with SUBMISSION_PATH.open('xb') as _out:
    _out.write(_submission_source.read_bytes())
_checked = validate_submission(pd.read_parquet(SUBMISSION_PATH), load_test(DATA_DIR))
print(json.dumps({'submission_file': SUBMISSION_PATH.name, 'rows': len(_checked)}))
_bundle_tmp.cleanup()
"""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "ntl_baseline_source_sha256": source_hashes(),
        },
        "cells": [
            _cell(
                "markdown",
                """# Name That LLM official baseline reproduction

Word TF-IDF (1–2, 25k) + character TF-IDF (3–5, 35k), sublinear TF,
LogisticRegression(C=2.0, max_iter=1000), 5-fold GroupKFold by prompt_id.
Sources: [official Apache 2.0 Starter](https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition).
Attribution: Martyna Plomecka and Yao Yan; see repository NOTICE.md.

Attach Name That LLM inputs, select CPU and disable Internet before Save & Run All.
Uses installed pandas/numpy/scipy/scikit-learn/pyarrow/joblib/threadpoolctl, with
versions recorded in run artifacts. The exact local environment is in uv.lock.
Kaggle image parity and hidden scoring are separate checks; no score is promised.
The small visible test set is read dynamically. All original inputs stay read-only.
The package below is embedded from source, without a remote install.
Restart the kernel and use a fresh working directory for each run.
""",
                "overview",
            ),
            _cell("code", bootstrap, "package-bootstrap"),
            _cell("code", configuration, "configuration"),
            _cell("code", execute, "train-validate-predict"),
            _cell("code", export, "export-submission"),
        ],
    }
