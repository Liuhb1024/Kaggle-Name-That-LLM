"""Official word/character TF-IDF + logistic regression, fitted inside each fold."""

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

from .config import BaselineConfig
from .constants import LABELS, ContractError
from .data import validate_train


def build_pipeline(config: BaselineConfig) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                max_features=config.word_max_features,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(3, 5),
                                max_features=config.char_max_features,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(C=config.c, max_iter=config.max_iter, solver="lbfgs")),
        ]
    )


def make_folds(train: pd.DataFrame, config: BaselineConfig) -> NDArray[np.int64]:
    train = validate_train(train)
    if train["prompt_id"].nunique() < config.folds:
        raise ContractError("not enough prompt groups for requested folds")
    folds = np.full(len(train), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=config.folds, shuffle=False)
    for fold, (fitting, heldout) in enumerate(
        splitter.split(train["text"], train["model"], train["prompt_id"])
    ):
        fit_frame, held_frame = train.iloc[fitting], train.iloc[heldout]
        if not set(fit_frame["prompt_id"]).isdisjoint(held_frame["prompt_id"]):
            raise ContractError("prompt group leakage")
        if set(fit_frame["model"]) != set(LABELS) or set(held_frame["model"]) != set(LABELS):
            raise ContractError("every fitting and validation fold must cover all 18 labels")
        folds[heldout] = fold
    if (folds < 0).any():
        raise ContractError("fold coverage incomplete")
    return folds


def cross_validate(
    train: pd.DataFrame, config: BaselineConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = validate_train(train)
    folds = make_folds(train, config)
    probabilities = np.zeros((len(train), len(LABELS)), dtype=np.float64)
    predictions = np.empty(len(train), dtype=object)
    reports: list[dict[str, Any]] = []
    start = perf_counter()
    with threadpool_limits(limits=config.threads):
        for fold in range(config.folds):
            fitting = train.iloc[np.flatnonzero(folds != fold)]
            heldout_indices = np.flatnonzero(folds == fold)
            heldout = train.iloc[heldout_indices]
            model = build_pipeline(config)
            fold_start = perf_counter()
            model.fit(fitting["text"], fitting["model"])
            if tuple(model.classes_) != LABELS:
                raise ContractError("model probability label order differs from candidate order")
            predictions[heldout_indices] = model.predict(heldout["text"])
            probabilities[heldout_indices] = model.predict_proba(heldout["text"])
            reports.append(
                {
                    "fold": fold,
                    "train_rows": len(fitting),
                    "validation_rows": len(heldout),
                    "train_prompts": int(fitting["prompt_id"].nunique()),
                    "validation_prompts": int(heldout["prompt_id"].nunique()),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(heldout["model"], predictions[heldout_indices])
                    ),
                    "seconds": perf_counter() - fold_start,
                    "solver_iterations": int(model.named_steps["clf"].n_iter_.max()),
                }
            )
    scores = np.array([report["balanced_accuracy"] for report in reports], dtype=float)
    recalls = recall_score(train["model"], predictions, labels=LABELS, average=None)
    oof = train[["id", "prompt_id", "model"]].copy()
    oof["fold"] = folds
    oof["prediction"] = predictions
    for index, label in enumerate(LABELS):
        oof[f"prob__{label}"] = probabilities[:, index]
    summary = {
        "metric": "balanced_accuracy",
        "splitter": "GroupKFold(shuffle=False)",
        "fold_mean_balanced_accuracy": float(scores.mean()),
        "fold_std_balanced_accuracy": float(scores.std(ddof=0)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(train["model"], predictions)),
        "per_class": {
            label: {"recall": float(recalls[i]), "support": int((train["model"] == label).sum())}
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(train["model"], predictions, labels=LABELS).tolist(),
        "folds": reports,
        "seconds": perf_counter() - start,
    }
    return oof, summary
