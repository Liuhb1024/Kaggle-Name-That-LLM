"""Numerical and tabular contracts shared by server and notebook callers."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ntl_baseline.constants import LABELS, ContractError
from ntl_baseline.data import validate_submission, validate_test

POOL_LAYERS = (2, 4, 6, 8, 12, 16, 20, 24)
HIDDEN_SIZE = 4096
FEATURE_SIZE = HIDDEN_SIZE * len(POOL_LAYERS)
TOKEN_LIMIT = 447
CLAUDE_LABELS = tuple(label for label in LABELS if label.startswith("claude-"))
Array = npt.NDArray[np.floating[Any]]


def _require_finite(values: Array, shape: tuple[int, ...], name: str) -> None:
    if values.shape != shape or values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ContractError(f"{name} must be finite floating-point values with shape {shape}")


@dataclass(frozen=True)
class PlainReadout:
    coefficients: Array
    intercept: Array

    def logits(self, features: Array) -> Array:
        _require_finite(features, (len(features), FEATURE_SIZE), "pooled features")
        return features @ self.coefficients.T + self.intercept


def load_readout(path: Path) -> PlainReadout:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive["classes"]) != LABELS:
            raise ContractError("readout class order differs from V31")
        if tuple(archive["pool_layers"]) != POOL_LAYERS:
            raise ContractError("readout pooling layers differ from V31")
        coefficients, intercept = archive["plain_coef"], archive["plain_intercept"]
    _require_finite(coefficients, (len(LABELS), FEATURE_SIZE), "plain coefficients")
    _require_finite(intercept, (len(LABELS),), "plain intercept")
    return PlainReadout(coefficients, intercept)


@dataclass(frozen=True)
class AuxiliaryReadouts:
    center: PlainReadout
    claude: PlainReadout


def load_auxiliary_readouts(path: Path) -> AuxiliaryReadouts:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive["classes"]) != LABELS or tuple(archive["claude_classes"]) != CLAUDE_LABELS:
            raise ContractError("auxiliary readout class order differs from V31")
        if tuple(archive["pool_layers"]) != POOL_LAYERS:
            raise ContractError("auxiliary readout pooling layers differ from V31")
        center = PlainReadout(archive["center_coef"], archive["center_intercept"])
        claude = PlainReadout(archive["claude_coef"], archive["claude_intercept"])
    for name, head, count in (
        ("center", center, len(LABELS)),
        ("claude", claude, len(CLAUDE_LABELS)),
    ):
        _require_finite(head.coefficients, (count, FEATURE_SIZE), f"{name} coefficients")
        _require_finite(head.intercept, (count,), f"{name} intercept")
    return AuxiliaryReadouts(center, claude)


def auxiliary_outputs(
    ids: list[str], features: Array, readouts: AuxiliaryReadouts, groups: list[str] | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(ids) != len(features) or len(set(ids)) != len(ids):
        raise ContractError("auxiliary scores need one unique ID per feature row")
    claude = _score_frame(ids, _log_softmax(readouts.claude.logits(features)), CLAUDE_LABELS)
    if groups is None:
        return claude, _score_frame([], np.empty((0, len(LABELS))), LABELS)
    if len(groups) != len(ids):
        raise ContractError("expected one group per feature row")
    group_rows: dict[str, list[int]] = {}
    for row, group in enumerate(groups):
        group_rows.setdefault(group, []).append(row)
    eligible = [rows for rows in group_rows.values() if 2 <= len(rows) <= 18]
    centered = features.astype(np.float32, copy=True)
    for rows in eligible:
        centered[rows] -= centered[rows].mean(axis=0)
    rows = sorted(row for group in eligible for row in group)
    logits = readouts.center.logits(centered[rows])
    return claude, _score_frame([ids[row] for row in rows], np.exp(_log_softmax(logits)), LABELS)


def _score_frame(ids: list[str], values: Array, labels: tuple[str, ...]) -> pd.DataFrame:
    _require_finite(values, (len(ids), len(labels)), "auxiliary scores")
    frame = pd.DataFrame(values, columns=labels)
    frame.insert(0, "id", pd.Series(ids, dtype=object))
    return frame


def _log_softmax(logits: Array) -> Array:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def encode_passages(
    tokenizer: Callable[..., dict[str, list[list[int]]]], texts: list[str], eos_id: int
) -> list[list[int]]:
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=TOKEN_LIMIT)[
        "input_ids"
    ]
    if any(not ids for ids in encoded):
        raise ContractError("a passage produced no tokens before EOS and cannot be pooled")
    return [ids + [eos_id] for ids in encoded]


def verify_anchors(observed: Array, saved: Array) -> float:
    shape = (3, FEATURE_SIZE)
    _require_finite(observed, shape, "observed anchor features")
    _require_finite(saved, shape, "saved anchor features")
    error = float(np.abs(observed - saved).mean() / (np.abs(saved).mean() + 1e-9))
    if error >= 0.02:
        raise ContractError(f"checkpoint anchor parity failed (relative error {error:.6g})")
    return error


def branch_outputs(test: pd.DataFrame, logits: Array) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = validate_test(test)
    _require_finite(logits, (len(test), len(LABELS)), "branch logits")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    scores = pd.DataFrame(exp / exp.sum(axis=1, keepdims=True), columns=list(LABELS))
    scores.insert(0, "id", test["id"])
    predictions = pd.DataFrame({"id": test["id"], "model": np.asarray(LABELS)[logits.argmax(1)]})
    return scores, validate_submission(predictions, test)
