"""Configuration for the official baseline (not tuned)."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .constants import ContractError


@dataclass(frozen=True)
class BaselineConfig:
    word_max_features: int = 25_000
    char_max_features: int = 35_000
    c: float = 2.0
    max_iter: int = 1_000
    folds: int = 5
    threads: int = 1

    def __post_init__(self) -> None:
        for name in ("word_max_features", "char_max_features", "max_iter", "threads", "folds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractError(f"config {name} must be a positive integer")
        if self.folds < 2:
            raise ContractError("config folds must be at least 2")
        if isinstance(self.c, bool) or not isinstance(self.c, int | float):
            raise ContractError("config c must be finite and positive")
        if not math.isfinite(self.c) or self.c <= 0:
            raise ContractError("config c must be finite and positive")


def load_config(path: Path | None) -> BaselineConfig:
    if path is None:
        return BaselineConfig()
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ContractError("config must be a JSON object")
        return BaselineConfig(**values)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ContractError("invalid config keys or JSON") from exc
