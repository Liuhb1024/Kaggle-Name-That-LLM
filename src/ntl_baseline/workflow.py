"""Audited validation, training and prediction on immutable local input files."""

import importlib.metadata
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .artifacts import (
    directory_hashes,
    runtime_info,
    source_hashes,
    write_json_new,
    write_parquet_new,
)
from .baseline import build_pipeline, cross_validate
from .config import BaselineConfig
from .constants import DATA_FILES, LABELS, ContractError
from .data import (
    file_sha256,
    input_file,
    load_competition_data,
    load_test,
    read_csv_input,
    validate_submission,
    validate_train,
)

LOGGER = logging.getLogger(__name__)


def audit_data(data_dir: Path, *, require_sample: bool = True) -> dict[str, Any]:
    train, test = load_competition_data(data_dir, require_sample=require_sample)
    audited_files = DATA_FILES if require_sample else ("train.csv", "test.csv")
    files = {
        name: {
            "sha256": file_sha256(input_file(data_dir, name)),
            "bytes": input_file(data_dir, name).stat().st_size,
        }
        for name in audited_files
    }
    if (data_dir / "source.json").is_file():
        provenance = json.loads((data_dir / "source.json").read_text(encoding="utf-8"))
        expected_hashes = provenance.get("files", {})
        if any(expected_hashes.get(name) != files[name]["sha256"] for name in audited_files):
            raise ContractError("raw input hash differs from import manifest")
    words = train["text"].str.split().str.len()
    return {
        "files": files,
        "train_rows": len(train),
        "test_rows": len(test),
        "prompt_count": int(train["prompt_id"].nunique()),
        "candidate_count": int(train["model"].nunique()),
        "train_class_counts": {label: int((train["model"] == label).sum()) for label in LABELS},
        "prompt_size_counts": {
            str(size): int(count)
            for size, count in train.groupby("prompt_id").size().value_counts().items()
        },
        "train_empty_text_rows": int((train["text"] == "").sum()),
        "test_empty_text_rows": int((test["text"] == "").sum()),
        "train_word_count_range": [int(words.min()), int(words.max())],
        "train_duplicate_text_rows": int(train["text"].duplicated().sum()),
        "train_test_exact_text_overlap": int(test["text"].isin(set(train["text"])).sum()),
        "matches_published_training_shape": len(train) == 2699
        and train["prompt_id"].nunique() == 150,
        "text_policy": "published string schema; official missing text filled with empty string",
    }


def validate_model(data_dir: Path, output_dir: Path, config: BaselineConfig) -> dict[str, Any]:
    train = validate_train(read_csv_input(data_dir, "train.csv"))
    before = file_sha256(data_dir / "train.csv")
    output_dir.mkdir(parents=True, exist_ok=False)
    LOGGER.info("Running %d prompt-grouped validation folds on %d rows", config.folds, len(train))
    oof, summary = cross_validate(train, config)
    if file_sha256(data_dir / "train.csv") != before:
        raise ContractError("training input changed during validation")
    write_parquet_new(output_dir / "oof.parquet", oof)
    write_parquet_new(output_dir / "folds.parquet", oof[["id", "prompt_id", "fold"]])
    summary.update(
        {
            "train_sha256": before,
            "config": asdict(config),
            "runtime": runtime_info(),
            "source_sha256": source_hashes(),
        }
    )
    write_json_new(output_dir / "metrics.json", summary)
    return summary


def train_model(data_dir: Path, model_dir: Path, config: BaselineConfig) -> dict[str, Any]:
    train = validate_train(read_csv_input(data_dir, "train.csv"))
    before = file_sha256(data_dir / "train.csv")
    model_dir.mkdir(parents=True, exist_ok=False)
    model = build_pipeline(config)
    LOGGER.info("Fitting full-training model on %d rows", len(train))
    start = perf_counter()
    with threadpool_limits(limits=config.threads):
        model.fit(train["text"], train["model"])
    if tuple(model.classes_) != LABELS:
        raise ContractError("fitted model must cover all 18 labels in canonical order")
    if file_sha256(data_dir / "train.csv") != before:
        raise ContractError("training input changed during fitting")
    path = model_dir / "pipeline.joblib"
    with path.open("xb") as stream:
        joblib.dump(model, stream)
    metadata = {
        "format_version": 1,
        "labels": list(LABELS),
        "config": asdict(config),
        "train_rows": len(train),
        "train_sha256": before,
        "model_sha256": file_sha256(path),
        "seconds": perf_counter() - start,
        "solver_iterations": int(model.named_steps["clf"].n_iter_.max()),
        "runtime": runtime_info(),
        "source_sha256": source_hashes(),
        "trusted_local_artifact_only": True,
    }
    write_json_new(model_dir / "metadata.json", metadata)
    return metadata


def load_trusted_model(model_dir: Path) -> tuple[Pipeline, dict[str, Any]]:
    # joblib is pickle-based. Integrity checks do NOT make third-party pickle safe.
    # Only use artifacts trained by this project from trusted local inputs.
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("labels") != list(LABELS):
        raise ContractError("incompatible model metadata or labels")
    for name, version in metadata["runtime"]["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ContractError(f"model runtime package version mismatch: {name}")
    model_path = model_dir / "pipeline.joblib"
    if file_sha256(model_path) != metadata.get("model_sha256"):
        raise ContractError("model hash mismatch; refusing to deserialize")
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline) or tuple(model.classes_) != LABELS:
        raise ContractError("invalid fitted pipeline or probability label order")
    return model, metadata


def predict_model(data_dir: Path, model_dir: Path, output_dir: Path) -> dict[str, Any]:
    test = load_test(data_dir)
    before = file_sha256(data_dir / "test.csv")
    model, metadata = load_trusted_model(model_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    with threadpool_limits(limits=metadata["config"]["threads"]):
        predictions = model.predict(test["text"])
    submission = validate_submission(pd.DataFrame({"id": test["id"], "model": predictions}), test)
    path = output_dir / "submission.parquet"
    write_parquet_new(path, submission)
    actual = pd.read_parquet(path, engine="pyarrow")
    validate_submission(actual, test)
    pd.testing.assert_frame_equal(actual, submission)
    if file_sha256(data_dir / "test.csv") != before:
        raise ContractError("test input changed during inference")
    report = {
        "test_rows": len(test),
        "test_sha256": before,
        "submission_sha256": file_sha256(path),
        "model_sha256": metadata["model_sha256"],
        "seconds": perf_counter() - start,
        "prediction_counts": {label: int((actual["model"] == label).sum()) for label in LABELS},
        "parquet_roundtrip_validated": True,
        "runtime": runtime_info(),
    }
    write_json_new(output_dir / "report.json", report)
    return report


def run_baseline(
    data_dir: Path, run_dir: Path, config: BaselineConfig, *, require_sample: bool = True
) -> dict[str, Any]:
    initial_audit = audit_data(data_dir, require_sample=require_sample)
    run_dir.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    write_json_new(
        run_dir / "started.json",
        {
            "started_at_utc": datetime.now(UTC).isoformat(),
            "config": asdict(config),
            "source_sha256": source_hashes(),
            "runtime": runtime_info(),
        },
    )
    write_json_new(run_dir / "data_manifest.json", initial_audit)
    validation = validate_model(data_dir, run_dir / "validation", config)
    training = train_model(data_dir, run_dir / "model", config)
    prediction = predict_model(data_dir, run_dir / "model", run_dir / "prediction")
    final_audit = audit_data(data_dir, require_sample=require_sample)
    if initial_audit != final_audit:
        raise ContractError("raw inputs changed during run; no completion record written")
    result = {
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "train_rows": initial_audit["train_rows"],
        "test_rows": initial_audit["test_rows"],
        "fold_mean_balanced_accuracy": validation["fold_mean_balanced_accuracy"],
        "fold_std_balanced_accuracy": validation["fold_std_balanced_accuracy"],
        "oof_balanced_accuracy": validation["oof_balanced_accuracy"],
        "validation_seconds": validation["seconds"],
        "training_seconds": training["seconds"],
        "prediction_seconds": prediction["seconds"],
        "seconds": perf_counter() - start,
        "raw_input_hashes_unchanged": True,
        "submission": "prediction/submission.parquet",
        "runtime": runtime_info(),
        "artifact_sha256": directory_hashes(run_dir),
    }
    write_json_new(run_dir / "completed.json", result)
    return result
