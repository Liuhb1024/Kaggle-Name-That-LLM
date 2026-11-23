"""Offline server entry point. Every attempted inference writes to a new directory."""

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ntl_baseline.constants import ContractError
from ntl_baseline.data import file_sha256, load_test, read_csv_input, validate_train

from .branch import (
    auxiliary_outputs,
    branch_outputs,
    load_auxiliary_readouts,
    load_readout,
    verify_anchors,
)

ROOT = Path(__file__).resolve().parents[2]
# Pinnable small files from Kaggle and named weight files from the same release.
CHECKPOINT_FILES = (
    "COMPLETE.json",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "heads.safetensors",
    "readouts.npz",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "anchor_features.npz",
)


@dataclass(frozen=True)
class Config:
    batch_size: int
    checkpoint_hashes: dict[str, str]
    base_hashes: dict[str, str]
    base_source: str


def load_config(path: Path) -> Config:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if (
        data.get("script_version") != 372363760
        or data.get("branch") != "lora24_claude"
        or data.get("checkpoint") != "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/1"
        or data.get("recommended_base") != "qwen-lm/qwen-3/Transformers/8b-base/1"
    ):
        raise ContractError("expected the pinned V31 lora24_claude configuration")
    batch_size = data.get("batch_size")
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ContractError("batch_size must be an integer from 1 to 8")
    checkpoint_hashes = _parse_hashes(
        data.get("sha256"), set(CHECKPOINT_FILES) - {"adapter/adapter_model.safetensors"}
    )
    base_hashes = _parse_hashes(
        data.get("base_sha256"), {"config.json", "model.safetensors.index.json"}
    )
    return Config(batch_size, checkpoint_hashes, base_hashes, data["recommended_base"])


def _parse_hashes(value: object, names: set[str]) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != names:
        raise ContractError("configuration must pin all required control file hashes")
    if not all(
        isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in value.values()
    ):
        raise ContractError("configuration contains an invalid SHA-256")
    return value


def _require_files(directory: Path, names: tuple[str, ...]) -> None:
    for name in names:
        if not (directory / name).is_file():
            raise ContractError(f"required asset is missing: {name}")


def check_assets(base: Path, checkpoint: Path, config: Config) -> dict[str, dict[str, str]]:
    _require_files(checkpoint, CHECKPOINT_FILES)
    _require_files(base, tuple(config.base_hashes))
    for directory, expected in (
        (base, config.base_hashes),
        (checkpoint, config.checkpoint_hashes),
    ):
        for name, digest in expected.items():
            if file_sha256(directory / name) != digest:
                raise ContractError(f"pinned SHA-256 mismatch: {name}")
    if (base / "model.safetensors").exists():
        raise ContractError("base directory must use the pinned sharded safetensors layout")
    index = json.loads((base / "model.safetensors.index.json").read_text())
    shards = tuple(sorted(set(index["weight_map"].values())))
    if not shards or any(
        not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name) for name in shards
    ):
        raise ContractError("unexpected base checkpoint shard names")
    _require_files(base, shards)
    adapter = checkpoint / "adapter" / "adapter_model.safetensors"
    if adapter.stat().st_size != 116434760:
        raise ContractError("adapter weight size differs from Kaggle v1 metadata")
    if (checkpoint / "adapter" / "adapter_model.bin").exists():
        raise ContractError("remove the unrecognized adapter .bin file from the asset directory")
    return {
        "base": {name: file_sha256(base / name) for name in (*config.base_hashes, *shards)},
        "checkpoint": {name: file_sha256(checkpoint / name) for name in CHECKPOINT_FILES},
    }


def _run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    test = (
        pd.DataFrame(
            {
                "id": ["synthetic-002", "synthetic-001", "synthetic-003"],
                "text": [
                    "A synthetic sentence for the first CUDA check.",
                    "Second sample.",
                    "Synthetic prose. " * 500,
                ],
            }
        )
        if args.synthetic
        else load_test(args.data_dir)
    )
    if args.limit is not None:
        test = test.head(args.limit)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT / "artifacts") or run_dir == ROOT / "artifacts":
        raise ContractError("run-dir must be a new directory under this checkout's artifacts/")
    if run_dir.exists():
        raise ContractError("run-dir already exists")
    hashes = check_assets(args.base_dir, args.checkpoint_dir, config)
    readout_path = args.checkpoint_dir / "readouts.npz"
    readout = load_readout(readout_path)
    auxiliary = load_auxiliary_readouts(readout_path) if args.auxiliary else None

    import torch

    from .gpu import CudaFeatures

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ContractError("expose exactly one CUDA GPU for inference")

    anchor_texts = None
    if args.anchor_data_dir is not None:
        train = validate_train(read_csv_input(args.anchor_data_dir, "train.csv"))
        anchor_texts = train["text"].head(3).tolist()

    run_dir.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    extractor = CudaFeatures(args.base_dir, args.checkpoint_dir)
    torch.cuda.synchronize()
    load_seconds = perf_counter() - started

    anchor = None
    if anchor_texts is not None:
        with np.load(args.checkpoint_dir / "anchor_features.npz", allow_pickle=False) as saved:
            anchor = verify_anchors(extractor(anchor_texts, 3), saved["features"])

    torch.cuda.synchronize()
    prediction_started = perf_counter()
    features = extractor(test["text"].tolist(), config.batch_size)
    scores, predictions = branch_outputs(test, readout.logits(features))
    outputs = {"probabilities.parquet": scores, "predictions.parquet": predictions}
    if auxiliary is not None:
        groups = ["pair", "pair", "single"][: len(test)] if args.synthetic else None
        gate, twin = auxiliary_outputs(test["id"].tolist(), features, auxiliary, groups)
        outputs["claude_gate_logprobs.parquet"] = gate
        if groups is not None:
            outputs["centered_probabilities.parquet"] = twin
    torch.cuda.synchronize()
    finished_prediction = perf_counter()
    prediction_seconds = finished_prediction - prediction_started
    for name, frame in outputs.items():
        frame.to_parquet(run_dir / name, index=False)
        if not pd.read_parquet(run_dir / name).equals(frame):
            raise ContractError(f"Parquet round trip differs: {name}")

    report = {
        "script_version": 372363760,
        "branch": "lora24_claude",
        "member": "plain",
        "base_source": config.base_source,
        "checkpoint_source": "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/1",
        "asset_sha256": hashes,
        "config_sha256": file_sha256(args.config),
        "outputs_sha256": {name: file_sha256(run_dir / name) for name in outputs},
        "rows": len(test),
        "synthetic": args.synthetic,
        "classes": list(scores.columns[1:]),
        "batch_size": config.batch_size,
        "anchor_relative_error": anchor,
        "auxiliary_rows": {
            name: len(frame)
            for name, frame in outputs.items()
            if name not in ("probabilities.parquet", "predictions.parquet")
        },
        "centering_groups": "synthetic_fixture"
        if auxiliary is not None and args.synthetic
        else None,
        "gpu": torch.cuda.get_device_name(),
        "cuda_build": torch.version.cuda,
        "python": sys.version.split()[0],
        "packages": {
            package: version(package)
            for package in (
                "torch",
                "transformers",
                "accelerate",
                "peft",
                "safetensors",
                "tokenizers",
                "numpy",
                "pandas",
                "pyarrow",
            )
        },
        "model_load_seconds": load_seconds,
        "prediction_seconds": prediction_seconds,
        "load_through_prediction_seconds": finished_prediction - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    marker = run_dir / ".completed.tmp"
    marker.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    marker.rename(run_dir / "completed.json")
    print(f"Saved {len(test)} rows of branch probabilities and predictions.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--synthetic", action="store_true")
    data.add_argument("--data-dir", type=Path)
    parser.add_argument("--anchor-data-dir", type=Path)
    parser.add_argument("--auxiliary", action="store_true", help="emit the published extra heads")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v31_single_gpu.toml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        _run(args)
    except ContractError as exc:
        parser.exit(1, f"V31 inference failed: {exc}\n")
    except ImportError as exc:
        parser.exit(1, f"V31 GPU environment import failed ({type(exc).__name__}).\n")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(
            1, f"V31 inference failed ({type(exc).__name__}). No completion marker written.\n"
        )


if __name__ == "__main__":
    main()
