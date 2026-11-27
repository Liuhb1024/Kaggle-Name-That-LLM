"""Small explicit public projections. Never copy private stage reports or logs wholesale."""

import json
import math
import re
from pathlib import Path
from typing import Any

from ntl_baseline.artifacts import write_json_new
from ntl_baseline.constants import LABELS, ContractError

from .v31_bridge import validate_manifest_entries

SHA = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
HANDLE = re.compile(r"[a-z0-9_-]+/[a-z0-9_-]+")
FILE = re.compile(r"[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*")
VERSION = re.compile(r"[0-9][0-9a-zA-Z.+_-]{0,40}")


def _keys(value: Any, keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError("unexpected public fields")


def _hash(val: Any, pattern: re.Pattern[str] = SHA) -> None:
    if not isinstance(val, str) or not pattern.fullmatch(val):
        raise ContractError("invalid public identifier")


def _number(val: Any) -> None:
    if type(val) not in (int, float) or not math.isfinite(val):
        raise ContractError("expected finite public metric")


def _interval(values: Any) -> None:
    if not isinstance(values, list) or len(values) != 2:
        raise ContractError("expected confidence interval")
    for n in values:
        _number(n)
    if values[0] > values[1]:
        raise ContractError("inverted confidence interval")


def validate_public(name: str, value: Any) -> None:
    if name == "summary.json":
        _keys(
            value,
            {
                "commit",
                "config_sha256",
                "fold_sha256",
                "metrics",
                "delta_ci95",
                "by_arm",
                "gates",
                "resources",
                "versions",
            },
        )
        _hash(value["commit"], COMMIT)
        for k in ("config_sha256", "fold_sha256"):
            _hash(value[k])
        _keys(value["metrics"], {"S1", "S2", "C0", "E0", "S2_minus_C0"})
        _keys(value["by_arm"], {"S1", "S2", "C0", "E0"})
        for arm in value["by_arm"].values():
            _keys(
                arm,
                {
                    "ci95",
                    "per_fold",
                    "per_class",
                    "family_balanced_accuracy",
                    "oracle_sibling_balanced_accuracy",
                },
            )
            _keys(arm["per_fold"], set(map(str, range(5))))
            for v in arm["per_fold"].values():
                _number(v)
            _keys(arm["per_class"], set(LABELS))
            for per_label in arm["per_class"].values():
                _keys(per_label, {"recall", "support", "ci95"})
                _number(per_label["support"])
                _number(per_label["recall"])
                _interval(per_label["ci95"])
            _interval(arm["ci95"])
            _number(arm["family_balanced_accuracy"])
            _number(arm["oracle_sibling_balanced_accuracy"])
        for v in value["metrics"].values():
            _number(v)
        _interval(value["delta_ci95"])
        _keys(value["gates"], {"data", "smoke", "s1", "s2", "parity"})
        if any(v is not True for v in value["gates"].values()):
            raise ContractError("incomplete public report")
        for k, v in value["resources"].items():
            if not re.fullmatch(r"(?:00|01|02|03|04|05|06|07|08|09)", k):
                raise ContractError("unknown resource stage")
            _keys(
                v,
                {
                    "seconds",
                    "peak_reserved_bytes",
                    "peak_allocated_bytes",
                    "peak_device_mib",
                    "tokens_per_second",
                },
            )
            for n in v.values():
                _number(n)
        for k, v in value["versions"].items():
            if k not in {
                "python",
                "torch",
                "transformers",
                "peft",
                "numpy",
                "scipy",
                "scikit-learn",
                "kagglehub",
                "accelerate",
                "pyarrow",
                "pandas",
                "safetensors",
                "tokenizers",
            }:
                raise ContractError("unknown package")
            _hash(v, VERSION)
    elif name == "audit_summary.json":
        _keys(value, {"per_class", "tiers", "source_counts", "gaps"})
        _keys(value["per_class"], set(LABELS))
        for section, counts in value.items():
            if not isinstance(counts, dict):
                raise ContractError("expected aggregate counts")
            for k, v in counts.items():
                if section == "source_counts":
                    _hash(k, HANDLE)
                elif k not in ({"A", "B", "C"} if section == "tiers" else set(LABELS)):
                    raise ContractError("unknown aggregate key")
                if type(v) is not int or v < 0:
                    raise ContractError("expected nonnegative count")
    elif name in ("source_lock.json", "package_manifest.json"):
        if name == "source_lock.json":
            _keys(value, {"files", "versions"})
            for dataset, version in value["versions"].items():
                _hash(dataset, HANDLE)
                if type(version) is not int or version < 1:
                    raise ContractError("invalid source version")
            entries = value["files"]
        else:
            try:
                entries = validate_manifest_entries(value)
            except (TypeError, ValueError) as exc:
                raise ContractError("invalid public member manifest") from exc
        if not isinstance(entries, dict) or not entries:
            raise ContractError("empty manifest")
        for k, v in entries.items():
            if not FILE.fullmatch(k) or any(p in {"..", "."} for p in k.split("/")):
                raise ContractError("invalid public filename")
            fields = {"bytes", "sha256"}
            if name == "source_lock.json" and "dataset" in v:
                fields |= {"dataset", "version"}
                _hash(v["dataset"], HANDLE)
                if type(v["version"]) is not int or v["version"] < 1:
                    raise ContractError("invalid source version")
            _keys(v, fields)
            _hash(v["sha256"])
            if type(v["bytes"]) is not int or v["bytes"] < 0:
                raise ContractError("invalid file size")
    elif name == "kaggle_dataset.json":
        _keys(value, {"handle", "version", "manifest_sha256"})
        _hash(value["handle"], HANDLE)
        _hash(value["manifest_sha256"])
        if type(value["version"]) is not int or value["version"] < 1:
            raise ContractError("invalid private asset version")
    else:
        raise ContractError("unrecognized public report")


def publish(directory: Path, documents: dict[str, Any]) -> None:
    for name, value in documents.items():
        validate_public(name, value)
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in documents.items():
        write_json_new(directory / name, value)
    (directory / "summary.md").write_text(
        summary_markdown(documents["summary.json"]), encoding="utf-8"
    )


def summary_markdown(summary: dict[str, Any]) -> str:
    validate_public("summary.json", summary)
    scores = summary["metrics"]
    lines = [
        "# q8pool v1",
        "",
        "按 prompt_id 分组的官方样本评估。公开训练池经过上游基于官方样本的筛选。",
        "",
    ]
    lines.extend(f"- {key} {score:.5f}" for key, score in scores.items())
    return "\n".join(lines) + "\n"


def check_file(path: Path) -> None:
    if path.is_symlink():
        raise ContractError("unexpected link in public reports")
    if path.name == "summary.md":
        if summary_markdown(json.loads(path.with_suffix(".json").read_text())) != path.read_text():
            raise ContractError("public markdown differs from validated metrics")
    elif path.suffix == ".json":
        validate_public(path.name, json.loads(path.read_text()))
    else:
        raise ContractError("unexpected public report file")
