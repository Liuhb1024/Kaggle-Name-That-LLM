"""Export the six trained members and compare the pinned V31 runner to local evaluation."""

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from ntl_baseline.artifacts import directory_hashes, write_json_new
from ntl_baseline.constants import LABELS
from ntl_baseline.data import file_sha256
from ntl_member.v31_bridge import (
    LAYERS,
    MEMBERS,
    check_members,
    extract_scripts,
    reference_command,
    verify_manifest,
)

PROJECTIONS = {
    "self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}
ANCHORS = (
    "The harbor bell rang only when the fog was dense enough to hide the last buoy.",
    "For three mornings the baker weighed the flour twice and wrote the difference in chalk.",
    "A blue notebook lay open beside the lamp, its pages moving slightly in the draft.",
)


def check_adapter_keys(keys: Iterable[str], modules: set[str]) -> None:
    found: set[tuple[str, str]] = set()
    for key in keys:
        match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", key)
        if not match or match[1] not in modules:
            raise ValueError("Invalid LoRA key or unknown backbone module")
        found.add((match[1], match[2]))
    if found != {(name, factor) for name in modules for factor in "AB"}:
        raise ValueError("Missing LoRA target or factor")


def _backbone_modules(base: Path) -> dict[str, tuple[int, int]]:
    import torch
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    cfg.num_hidden_layers = 24
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = cfg.layer_types[:24]
    with torch.device("meta"):
        model = AutoModel.from_config(cfg)
    modules = dict(model.named_modules())
    target_shapes = {}
    for layer in range(24):
        for block, projections in PROJECTIONS.items():
            for projection in projections:
                name = f"layers.{layer}.{block}.{projection}"
                output_width, input_width = modules[name].weight.shape
                target_shapes[name] = (int(output_width), int(input_width))
    return target_shapes


def _validate_weights(member: Path, modules: dict[str, tuple[int, int]]) -> None:
    from safetensors import safe_open

    config = json.loads((member / "adapter/adapter_config.json").read_text())
    if (
        config["r"] != 16
        or config["lora_alpha"] != 32
        or config["bias"] != "none"
        or config.get("use_dora")
        or config.get("use_rslora")
        or config.get("rank_pattern")
        or config.get("alpha_pattern")
        or set(config["target_modules"]) != {p for targets in PROJECTIONS.values() for p in targets}
    ):
        raise ValueError("Unsupported adapter configuration")
    with safe_open(str(member / "adapter/adapter_model.safetensors"), framework="pt") as f:
        check_adapter_keys(f.keys(), set(modules))
        for key in f.keys():
            name, factor = key.removeprefix("base_model.model.").rsplit(".lora_", 1)
            out_features, in_features = modules[name]
            expected = (16, in_features) if factor == "A.weight" else (out_features, 16)
            if tuple(f.get_slice(key).get_shape()) != expected:
                raise ValueError(f"Invalid LoRA shape for {name}")
    heads = {
        f"norms.{index}.{part}": [4096]
        for index in range(len(LAYERS))
        for part in ("weight", "bias")
    }
    for key in ("plain", "centered"):
        heads[f"{key}.weight"] = [len(LABELS), len(LAYERS) * 4096]
        heads[f"{key}.bias"] = [len(LABELS)]
    with safe_open(str(member / "heads.safetensors"), framework="pt") as f:
        if set(f.keys()) != heads.keys():
            raise ValueError("Unexpected head keys")
        for key, shape in heads.items():
            tensor = f.get_slice(key)
            if tensor.get_shape() != shape or tensor.get_dtype() != "F32":
                raise ValueError(f"Wrong shape or dtype for {key}")


def _validate_matrix(data: np.ndarray, rows: int) -> None:
    if data.shape != (rows, len(LABELS)) or not np.isfinite(data).all():
        raise ValueError("Nonfinite or misaligned log-probabilities")
    if not np.allclose(np.logaddexp.reduce(data.astype(np.float64), axis=1), 0.0, atol=1e-4):
        raise ValueError("Expected normalized log-probabilities")


def check_parity(
    reference: np.ndarray,
    fp16: np.ndarray,
    bf16: np.ndarray,
    targets: Sequence[str],
) -> dict[str, float]:
    labels = np.asarray(targets)
    if set(labels) != set(LABELS):
        raise ValueError("Fold parity requires all 18 labels")
    predictions = []
    for matrix in (reference, fp16, bf16):
        _validate_matrix(matrix, len(labels))
        predictions.append(np.asarray(LABELS)[matrix.argmax(1)])
    ref, fp, bf = predictions
    scores = [float(balanced_accuracy_score(labels, p)) for p in predictions]
    result = {
        "parity_argmax_agreement": float(np.mean(ref == fp)),
        "parity_ba_diff": abs(scores[0] - scores[1]),
        "parity_mean_abs_logp": float(np.abs(reference - fp16).mean()),
        "precision_argmax_agreement": float(np.mean(bf == fp)),
        "precision_ba_diff": abs(scores[1] - scores[2]),
    }
    if (
        result["parity_argmax_agreement"] < 0.995
        or result["parity_ba_diff"] > 0.003
        or result["parity_mean_abs_logp"] > 0.02
    ):
        raise ValueError(f"V31 parity failed: {result}")
    if result["precision_argmax_agreement"] < 0.99 or result["precision_ba_diff"] > 0.005:
        raise ValueError(f"V31 precision failed: {result}")
    return result


def reference_predict(
    scripts: Path,
    base: Path,
    adapters: list[Path],
    official_csv: Path,
    frame: pd.DataFrame,
    work: Path,
) -> np.ndarray:
    if not frame["id"].is_unique or frame.empty or frame[["id", "text"]].isna().any().any():
        raise ValueError("Invalid reference rows")
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    (work / "train.csv").symlink_to(official_csv.resolve(strict=True))
    frame[["id", "text"]].to_csv(work / "test.csv", index=False)
    cmd = reference_command(
        scripts.resolve() / "infer.py",
        base.resolve(),
        [p.resolve() for p in adapters],
        work.resolve(),
        work.resolve(),
        sys.executable,
    )
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NTL_PIPE_WORKERS="1")
    with (work / "reference.log").open("x", encoding="utf-8") as stream:
        subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    check_members((work / "reference.log").read_text().splitlines(), adapters)
    with np.load(work / "logits.npz", allow_pickle=False) as result:
        if result["ids"].tolist() != frame.id.astype(str).tolist() or result[
            "labels"
        ].tolist() != list(LABELS):
            raise ValueError("Reference ids or label order mismatch")
        values: np.ndarray = result["logits"]
    _validate_matrix(values, len(frame))
    return values


def package_member(
    output: Path,
    source: Path,
    base: Path,
    members: list[Path],
    official_csv: Path,
    validation: pd.DataFrame,
    fp16_logp: np.ndarray,
    bf16_logp: np.ndarray,
    provenance: dict[str, str],
    notice: str,
) -> dict[str, float]:
    if len(members) != 6 or len({p.resolve() for p in members}) != 6 or not notice.strip():
        raise ValueError("Expected five folds, one full member, and attribution")
    if set(provenance) != {"git_commit", "config_sha256", "data_lock_sha256"}:
        raise ValueError("Unexpected provenance fields")
    for key, width in (("git_commit", 40), ("config_sha256", 64), ("data_lock_sha256", 64)):
        if not re.fullmatch(rf"[a-f0-9]{{{width}}}", provenance[key]):
            raise ValueError(f"Invalid provenance field {key}")
    export = output / "kaggle_dataset"
    scripts = output / "v31_source"
    extract_scripts(source, scripts)
    export.mkdir(mode=0o700, parents=True, exist_ok=False)
    cfg = dict(keep=24, layers=LAYERS, max_len=384, eos_id=None, no_final_norm=False)
    modules = _backbone_modules(base)
    packed = []
    for name, member in zip(MEMBERS, members, strict=True):
        _validate_weights(member, modules)
        path = export / name
        packed.append(path)
        (path / "adapter").mkdir(mode=0o700, parents=True)
        for filename in ("adapter/adapter_model.safetensors", "heads.safetensors"):
            shutil.copyfile(member / filename, path / filename)
        adapter_config = json.loads((member / "adapter/adapter_config.json").read_text())
        adapter_config["base_model_name_or_path"] = "Qwen/Qwen3-8B-Base"
        write_json_new(path / "adapter/adapter_config.json", adapter_config)
        write_json_new(path / "meta.json", {"cfg": cfg, "labels": list(LABELS), **provenance})
    reference = reference_predict(
        scripts, base, packed[:1], official_csv, validation, output / "fold0"
    )
    metrics = check_parity(reference, fp16_logp, bf16_logp, validation["model"].tolist())
    anchors = pd.DataFrame({"id": [f"q8anchor{i}" for i in range(3)], "text": ANCHORS})
    prediction = reference_predict(scripts, base, packed, official_csv, anchors, output / "anchor")
    (export / "anchors").mkdir()
    anchors.to_csv(export / "anchors/test.csv", index=False)
    write_json_new(
        export / "anchors/expected_logits.json",
        {
            "ids": anchors.id.tolist(),
            "labels": list(LABELS),
            "logits": prediction.tolist(),
        },
    )
    (export / "NOTICE.md").write_text(
        "# q8pool training inputs\n\n"
        "Qwen3-8B-Base, Apache License 2.0. "
        "[Model card and license](https://huggingface.co/Qwen/Qwen3-8B-Base).\n\n"
        "Name That LLM official training data, CC BY-SA 4.0. "
        "[Competition data](https://www.kaggle.com/competitions/name-that-llm/data).\n\n" + notice,
        encoding="utf-8",
    )
    entries = {
        name: {"bytes": (export / name).stat().st_size, "sha256": digest}
        for name, digest in directory_hashes(export).items()
    }
    write_json_new(export / "MANIFEST.json", entries)
    verify_manifest(export, file_sha256(export / "MANIFEST.json"))
    shutil.copyfile(export / "MANIFEST.json", output / "package_manifest.json")
    return metrics
