"""Compare a completed run against a direct, independent official-code reference.

No project model-builder imports: preserves the official pd.read_csv defaults,
FeatureUnion and cross_val_score workflow. Serial CV and one BLAS thread only
change resource scheduling. Requires the original local competition data.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

from ntl_baseline.artifacts import runtime_info, write_json_new
from ntl_baseline.data import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    before = {name: file_sha256(args.data_dir / name) for name in ("train.csv", "test.csv")}
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    model_pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                max_features=25000,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(3, 5),
                                max_features=35000,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(C=2.0, max_iter=1000)),
        ]
    )
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        scores = cross_val_score(
            model_pipeline,
            train["text"].fillna(""),
            train["model"],
            groups=train["prompt_id"],
            cv=GroupKFold(n_splits=5),
            scoring="balanced_accuracy",
            n_jobs=1,
        )
        model_pipeline.fit(train["text"].fillna(""), train["model"])
        predictions = model_pipeline.predict(test["text"].fillna(""))
    report = json.loads((args.run_dir / "validation/metrics.json").read_text())
    if before["train.csv"] != report["train_sha256"]:
        raise RuntimeError("Reference and reproduction use different training input hashes")
    reproduction_scores = np.array([fold["balanced_accuracy"] for fold in report["folds"]])
    submission = pd.read_parquet(args.run_dir / "prediction/submission.parquet").set_index("id")
    reproduction_predictions = submission.loc[test["id"], "model"].to_numpy()
    fold_delta = float(np.max(np.abs(scores - reproduction_scores)))
    predictions_equal = bool(np.array_equal(predictions, reproduction_predictions))
    after = {name: file_sha256(args.data_dir / name) for name in before}
    outcome = {
        "source_url": "https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition",
        "comparison": "direct official-code reference in the same local runtime",
        "reference_fold_scores": scores.tolist(),
        "reference_fold_mean": float(scores.mean()),
        "reference_fold_std": float(scores.std(ddof=0)),
        "reproduction_fold_scores": reproduction_scores.tolist(),
        "max_absolute_fold_score_difference": fold_delta,
        "test_predictions_identical": predictions_equal,
        "original_inputs_unchanged": before == after,
        "input_sha256": before,
        "reference_seconds": time.perf_counter() - start,
        "runtime": runtime_info(),
        "matches_reference": fold_delta <= 1e-12 and predictions_equal and before == after,
        "published_score_note": "Not a Kaggle hidden-test rerun or proof of public score 0.322.",
    }
    write_json_new(args.output, outcome)
    print(json.dumps(outcome, indent=2))
    if not outcome["matches_reference"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
