"""Prompt-clustered inference and paired comparisons, with a fixed class universe."""

from typing import Any

import numpy as np

from ntl_baseline.constants import LABELS, ContractError

N = len(LABELS)
FAMILY = np.array([label.split("-")[0] for label in LABELS])
SIBLINGS = np.array([np.count_nonzero(FAMILY == f) > 1 for f in FAMILY])


def _counts(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (
        not len(y)
        or len(pred) != len(y)
        or len(groups) != len(y)
        or ((y < 0) | (y >= N) | (pred < 0) | (pred >= N)).any()
    ):
        raise ContractError("invalid metric rows")
    _, ix = np.unique(groups, return_inverse=True)
    total = np.zeros((int(ix.max()) + 1, N), dtype=float)
    ok = np.zeros_like(total)
    np.add.at(total, (ix, y), 1)
    np.add.at(ok, (ix, y), y == pred)
    return total, ok


def _resample(total: np.ndarray, hits: np.ndarray, repeats: int) -> np.ndarray:
    # Multinomial multiplicities equal sampling complete prompts with replacement.
    rng = np.random.default_rng(0)
    g = len(total)
    draws = rng.multinomial(g, np.full(g, 1 / g), size=repeats)
    denom = draws @ total
    return np.divide(draws @ hits, denom, out=np.full_like(denom, np.nan), where=denom > 0)


def compare(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray, repeats: int
) -> dict[str, Any]:
    total, left = _counts(y, a, groups)
    _, right = _counts(y, b, groups)
    if np.any(total.sum(0) == 0):
        raise ContractError("paired comparison requires all labels")
    ds = _resample(total, left - right, repeats)
    valid = ds[np.isfinite(ds).all(axis=1)].mean(axis=1)
    if not len(valid):
        raise ContractError("no valid prompt bootstrap replicates")
    return {
        "delta": float(((left - right).sum(0) / total.sum(0)).mean()),
        "ci95": np.quantile(valid, [0.025, 0.975]).tolist(),
        "fraction_positive": float((valid > 0).mean()),
        "valid_replicates": len(valid),
    }


def _ba(y: np.ndarray, pred: np.ndarray) -> float:
    present = np.unique(y)
    return float(np.mean([(y[y == k] == pred[y == k]).mean() for k in present]))


def score(
    y: np.ndarray,
    logp: np.ndarray,
    groups: np.ndarray,
    fold: np.ndarray,
    words: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    if logp.shape != (len(y), N) or not np.isfinite(logp).all():
        raise ContractError("invalid probability matrix")
    pred = logp.argmax(1)
    total, hits = _counts(y, pred, groups)
    if np.any(total.sum(0) == 0):
        raise ContractError("primary score requires all labels")
    replicates = _resample(total, hits, repeats)
    intervals = np.nanquantile(replicates, [0.025, 0.975], axis=0)
    valid_ba = replicates[np.isfinite(replicates).all(axis=1)].mean(axis=1)
    if not len(valid_ba):
        raise ContractError("no valid balanced accuracy bootstrap replicates")
    mat = np.zeros((N, N), dtype=int)
    np.add.at(mat, (y, pred), 1)
    norm = mat / mat.sum(1, keepdims=True)
    family_match = FAMILY[y] == FAMILY[pred]
    oracle = np.where(FAMILY[None, :] == FAMILY[y, None], logp, -np.inf).argmax(1)
    fam = float(np.mean([family_match[y == k].mean() for k in range(N)]))
    result: dict[str, Any] = {
        "balanced_accuracy": float(np.diag(norm).mean()),
        "ci95": np.quantile(valid_ba, [0.025, 0.975]).tolist(),
        "per_fold": {str(k): _ba(y[fold == k], pred[fold == k]) for k in np.unique(fold)},
        "per_class": {
            label: {
                "support": int(total[:, k].sum()),
                "recall": norm[k, k],
                "ci95": intervals[:, k].tolist(),
            }
            for k, label in enumerate(LABELS)
        },
        "confusion_counts": mat.tolist(),
        "confusion_normalized": norm.tolist(),
        "family_balanced_accuracy": fam,
        "cross_family_loss": 1 - fam,
        "within_family_loss": fam - float(np.diag(norm).mean()),
        "oracle_sibling_balanced_accuracy": _ba(y[SIBLINGS[y]], oracle[SIBLINGS[y]]),
        "by_family": {},
        "by_length": {},
    }
    for f in np.unique(FAMILY[SIBLINGS]):
        ix = FAMILY == f
        result["by_family"][f] = {
            "oracle_sibling_balanced_accuracy": _ba(y[ix[y]], oracle[ix[y]]),
            "confusion_counts": mat[np.ix_(ix, ix)].tolist(),
        }
    for low, high in zip([36, 65, 102, 158], [65, 102, 158, 275], strict=True):
        mask = (words >= low) & (words < high)
        if mask.any():
            result["by_length"][f"{low}-{high - 1}"] = {
                "rows": int(mask.sum()),
                "balanced_accuracy": _ba(y[mask], pred[mask]),
            }
    return result


def pair_diagnostic(
    y: np.ndarray,
    plain: np.ndarray,
    centered: np.ndarray,
    bias: np.ndarray,
    groups: np.ndarray,
    repeats: int,
) -> dict[str, float]:
    rng = np.random.default_rng(0)
    by_prompt = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    samples = []
    for _ in range(repeats):
        pairs = np.array([rng.choice(ix, 2, replace=False) for ix in by_prompt if len(ix) >= 2])
        c = centered[pairs]
        scores = c - c.mean(axis=1, keepdims=True) + bias[pairs]
        ids = pairs.ravel()
        samples.append(
            [_ba(y[ids], plain[ids].argmax(1)), _ba(y[ids], scores.reshape(-1, N).argmax(1))]
        )
    a, b = np.mean(samples, axis=0)
    return {"plain_balanced_accuracy": float(a), "centered_balanced_accuracy": float(b)}
