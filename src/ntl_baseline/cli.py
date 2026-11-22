"""Explicit local CLI; no authentication, downloads or remote submissions."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .artifacts import write_json_new
from .config import load_config
from .constants import ContractError
from .data import import_archive, load_test, validate_submission
from .workflow import audit_data, predict_model, run_baseline, train_model, validate_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "validate", "train", "predict", "run", "check-submission"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--data-dir", type=Path, required=True)
        if command in ("validate", "train", "run"):
            subparser.add_argument("--config", type=Path)
        if command in ("validate", "train", "predict"):
            subparser.add_argument("--output-dir", type=Path, required=True)
        if command == "predict":
            subparser.add_argument(
                "--model-dir",
                type=Path,
                required=True,
                help="trusted locally trained joblib artifacts only",
            )
        if command == "run":
            subparser.add_argument("--run-dir", type=Path, required=True)
        if command == "audit":
            subparser.add_argument("--output", type=Path)
        if command == "check-submission":
            subparser.add_argument("--submission", type=Path, required=True)
    importer = subparsers.add_parser("import-data")
    importer.add_argument("--archive", type=Path, required=True)
    importer.add_argument("--data-dir", type=Path, default=Path("data/raw/name-that-llm"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    try:
        if args.command == "import-data":
            result = import_archive(args.archive, args.data_dir)
        elif args.command == "audit":
            result = audit_data(args.data_dir)
            if args.output:
                write_json_new(args.output, result)
        elif args.command == "validate":
            result = validate_model(args.data_dir, args.output_dir, load_config(args.config))
        elif args.command == "train":
            result = train_model(args.data_dir, args.output_dir, load_config(args.config))
        elif args.command == "predict":
            result = predict_model(args.data_dir, args.model_dir, args.output_dir)
        elif args.command == "run":
            result = run_baseline(args.data_dir, args.run_dir, load_config(args.config))
        else:
            test = load_test(args.data_dir)
            submission = validate_submission(pd.read_parquet(args.submission), test)
            result = {"status": "valid", "rows": len(submission)}
    except (ContractError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0
