"""Append-only artifacts and explicit environment/source provenance."""

import importlib.metadata
import json
import os
import platform
import resource
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .data import file_sha256

RUNTIME_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "pyarrow",
    "joblib",
    "threadpoolctl",
)


def write_json_new(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        frame.to_parquet(stream, engine="pyarrow", index=False)


def runtime_info() -> dict[str, Any]:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES},
        "process_peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
        "peak_rss_scope": "process high-water mark, not an isolated stage or process tree",
    }


def source_hashes() -> dict[str, str]:
    return {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def directory_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
