"""Read the preregistered experiment configuration."""

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Any

from ntl_baseline.constants import ContractError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/member_q8pool_v1.toml"


def load(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("rb") as stream:
        cfg = tomllib.load(stream)
    for key in (
        "filters",
        "pins",
        "gates",
        "model",
        "sampling",
        "evaluation",
        "train",
        "kaggle",
        "integrity",
    ):
        if not isinstance(cfg.get(key), dict):
            raise ContractError(f"missing configuration section {key}")
    pins = cfg["pins"]
    if cfg["model"]["keep_layers"] != 24 or cfg["model"]["pool_layers"] != [
        2,
        4,
        6,
        8,
        12,
        16,
        20,
        24,
    ]:
        raise ContractError("incompatible q8v5 feature contract")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", val)
        for val in [
            pins["v31_json_sha256"],
            pins["folds_digest"],
            cfg["integrity"]["official_train"],
            *cfg["integrity"]["shards"],
        ]
    ):
        raise ContractError("invalid input digest")
    if (
        cfg["train"]["token_budgets"] != [8192, 6144, 4096]
        or cfg["gates"]["s1_stop"] > cfg["gates"]["s1_continue"]
    ):
        raise ContractError("invalid preregistered budgets or gates")
    return cfg


def fold_digest(frame: Any) -> str:
    rows = frame.sort_values("id")[["id", "fold"]]
    return hashlib.sha256(
        rows.to_csv(index=False, header=False, lineterminator="\n").encode()
    ).hexdigest()
