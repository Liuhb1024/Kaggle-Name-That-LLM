"""Stage workers. All sample-level material stays in the private run directory."""

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ntl_baseline.artifacts import directory_hashes, write_json_new, write_parquet_new
from ntl_baseline.baseline import cross_validate, make_folds
from ntl_baseline.config import BaselineConfig
from ntl_baseline.constants import LABELS, ContractError
from ntl_baseline.data import file_sha256, read_csv_input, validate_train

from .config import DEFAULT_CONFIG, ROOT, fold_digest, load
from .metrics import compare, pair_diagnostic, score
from .report import publish
from .v31_bridge import extract_scripts, obtain_source


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def official(path: Path) -> pd.DataFrame:
    train = validate_train(read_csv_input(path.parent, path.name))
    folds = make_folds(train, BaselineConfig())
    return train.assign(fold=folds, label=train.model, prompt_group=train.prompt_id)


def env_versions() -> dict[str, str]:
    values = {"python": platform.python_version()}
    for pkg in (
        "torch",
        "transformers",
        "peft",
        "numpy",
        "scipy",
        "scikit-learn",
        "kagglehub",
        "accelerate",
        "pyarrow",
        "pandas",
        "safetensors",
        "tokenizers",
    ):
        try:
            values[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return values


def preflight(cfg: dict[str, Any], data: Path, base: Path, out: Path) -> None:
    import torch

    if (
        torch.cuda.device_count() != 1
        or "3090 Ti" not in torch.cuda.get_device_name()
        or not torch.cuda.is_bf16_supported()
        or not torch.ones(1, device="cuda").item()
    ):
        raise ContractError("expected a bf16-capable RTX 3090 Ti")
    if shutil.disk_usage(out).free < 60 * 1024**3:
        raise ContractError("preflight requires 60 GiB free")
    for line in (ROOT / "requirements-v31-gpu.txt").read_text().splitlines():
        package, version = line.split("==")
        if importlib.metadata.version(package) != version:
            raise ContractError(f"GPU environment version mismatch for {package}")
    if file_sha256(data) != cfg["integrity"]["official_train"]:
        raise ContractError("official train SHA-256 mismatch")
    prior = tomllib.loads((ROOT / "configs/v31_single_gpu.toml").read_text())
    expected = dict(prior["base_sha256"])
    expected.update(
        {
            f"model-{n:05d}-of-00005.safetensors": h
            for n, h in enumerate(cfg["integrity"]["shards"], 1)
        }
    )
    for name, digest in expected.items():
        if file_sha256(base / name) != digest:
            raise ContractError(f"base model digest mismatch for {name}")
    for name in ("tokenizer_config.json", "tokenizer.json"):
        expected[name] = file_sha256(base / name)
    frame = official(data)
    if fold_digest(frame) != cfg["pins"]["folds_digest"]:
        raise ContractError("official prompt folds mismatch")
    write_parquet_new(out / "folds.parquet", frame[["id", "fold"]])
    src = obtain_source(ROOT / "artifacts/v31-asset-cache/v31_372363760.json")
    shutil.copyfile(src, out / "source.json")
    write_json_new(
        out / "inputs.json",
        {
            "base": expected,
            "official_train": file_sha256(data),
            "folds": fold_digest(frame),
            "gpu_versions": env_versions(),
            "v31": file_sha256(src),
        },
    )


def prediction_file(
    base: Path, member: Path, frame: pd.DataFrame, cfg: dict[str, Any], path: Path
) -> None:
    from safetensors.numpy import load_file

    from .train import predict

    lp, cp = predict(base, member, frame, cfg)
    bias = load_file(str(member / "heads.safetensors"))["centered.bias"]
    np.savez(path, ids=frame.id.to_numpy(dtype=str), plain=lp, centered=cp, bias=bias)


def load_predictions(
    paths: list[Path], frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = []
    columns: dict[str, list[np.ndarray]] = {k: [] for k in ("plain", "centered", "bias")}
    for fold_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as a:
            if len(paths) > 1 and set(a["ids"].tolist()) != set(
                frame.loc[frame.fold == fold_index, "id"]
            ):
                raise ContractError("fold predictions must match their held-out rows")
            ids.extend(a["ids"].tolist())
            for k, v in columns.items():
                x = a[k]
                if k == "bias":
                    x = np.broadcast_to(x, (len(a["ids"]), len(LABELS)))
                if x.shape != (len(a["ids"]), len(LABELS)) or not np.isfinite(x).all():
                    raise ContractError("invalid fold predictions")
                v.append(x)
    index = pd.Index(ids)
    if not index.is_unique or set(ids) != set(frame.id):
        raise ContractError("incomplete out-of-fold row alignment")
    order = index.get_indexer(frame.id)
    return tuple(np.concatenate(columns[k])[order] for k in columns)


def evaluate_s2(
    frame: pd.DataFrame, paths: dict[str, Path], cfg: dict[str, Any], out: Path
) -> None:
    n = cfg["evaluation"]["bootstrap_resamples"]
    y = np.array([LABELS.index(x) for x in frame.label])
    groups = frame.prompt_group.to_numpy()
    folds = frame.fold.to_numpy()
    words = frame.text.str.split().str.len().to_numpy()
    scores = {}
    preds = {}
    for arm, files in {
        "S1": [paths["05"] / "pred.npz"],
        "S2": [paths["06"] / f"f{i}" / "pred.npz" for i in range(5)],
        "C0": [paths["07"] / f"f{i}" / "pred.npz" for i in range(5)],
    }.items():
        lp, cp, bias = load_predictions(files, frame)
        preds[arm] = lp.argmax(1)
        scores[arm] = score(y, lp, groups, folds, words, n)
        scores[arm]["centered_true_prompt"] = score(y, cp, groups, folds, words, n)[
            "balanced_accuracy"
        ]
        scores[arm]["pair_diagnostic"] = pair_diagnostic(
            y, lp, cp, bias, groups, cfg["evaluation"]["pair_replicates"]
        )
    e0, summary = cross_validate(frame[["id", "model", "text", "prompt_id"]], BaselineConfig())
    write_parquet_new(out / "e0.parquet", e0)
    if not np.array_equal(e0.fold, folds):
        raise ContractError("E0 fold mismatch")
    e0_lp = np.log(np.maximum(e0[[f"prob__{label}" for label in LABELS]].to_numpy(), 1e-300))
    scores["E0"] = score(y, e0_lp, groups, folds, words, n)
    scores["E0"]["seconds"] = summary["seconds"]
    comparisons = {
        f"{a}_minus_{b}": compare(y, preds[a], preds[b], groups, n)
        for a, b in (("S2", "C0"), ("S2", "S1"), ("S1", "C0"))
    }
    write_json_new(out / "evaluation.json", {"arms": scores, "comparisons": comparisons})
    gates = cfg["gates"]
    d = comparisons["S2_minus_C0"]
    if (
        scores["S2"]["balanced_accuracy"] < gates["s2_min"]
        or d["delta"] < gates["s2_minus_control_min"]
        or d["ci95"][0] <= gates["s2_minus_control_ci_low"]
    ):
        raise ContractError("S2 did not pass the preregistered packaging gate")


def fold_fit(
    base: Path,
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    out: Path,
    previous: list[Path],
    mode: str,
    budget: int,
    initial: Path | None,
) -> None:
    from .train import fit

    for fold in range(6 if mode == "stage2" else 5):
        name = f"f{fold}" if fold < 5 else "full"
        dest = out / name
        for past in previous:
            source = past / name
            if (source / "fold_done.json").is_file():
                expected = read(source / "fold_done.json")
                actual = directory_hashes(source)
                actual.pop("fold_done.json")
                if actual != expected:
                    raise ContractError("completed fold integrity failure")
                shutil.copytree(source, dest)
                break
        else:
            dest.mkdir()
            fit_rows = frame.loc[frame.fold != fold] if fold < 5 else frame
            timing = fit(
                base,
                fit_rows.reset_index(drop=True),
                cfg,
                mode,
                dest / "member",
                budget,
                init=initial,
            )
            write_json_new(dest / "timing.json", timing)
            if fold < 5:
                prediction_file(
                    base,
                    dest / "member",
                    frame.loc[frame.fold == fold].reset_index(drop=True),
                    cfg,
                    dest / "pred.npz",
                )
            write_json_new(dest / "fold_done.json", directory_hashes(dest))


def reusable_data_training(
    stage: str,
    paths: dict[str, Path],
    cfg: dict[str, Any],
    base: Path,
    data: Path,
    out: Path,
    previous: list[Path],
) -> None:
    if stage == "02":
        from .sources import download

        download(out, cfg)
    elif stage == "03":
        from .pool import prepare

        cfg["sampling"]["per_class_cap"] = read(paths["01"] / "smoke.json")["per_class_cap"]
        prepare(paths["02"], data, out, cfg)
    else:
        from .train import fit, latest_checkpoint

        frame = pd.read_parquet(paths["03"] / "stage1_sample.parquet")
        prior = None
        for p in previous:
            try:
                prior = latest_checkpoint(p / "member")
                break
            except ValueError:
                continue
        result = fit(
            base,
            frame,
            cfg,
            "stage1",
            out / "member",
            read(paths["01"] / "smoke.json")["token_budget"],
            resume=prior,
        )
        write_json_new(out / "timing.json", result)


def run(job: dict[str, Any]) -> None:
    stage = job["stage"]
    paths = {k: Path(v) for k, v in job["inputs"].items()}
    out = Path(job["output"])
    cfg = load()
    base = Path(job["base"])
    data = Path(job["official_csv"])
    previous = [Path(p) for p in job["previous"]]
    if stage == "00":
        preflight(cfg, data, base, out)
    elif stage == "01":
        from .package import ANCHORS, reference_predict
        from .train import predict, smoke

        start = time.monotonic()
        subprocess.run([sys.executable, "-m", "ntl_member.selftest"], check=True)
        result = smoke(base, cfg, out)
        anchor = pd.DataFrame(
            {
                "id": [f"a{i}" for i in range(len(ANCHORS))],
                "text": ANCHORS,
                "prompt_group": "smoke",
                "label": LABELS[0],
            }
        )
        scripts = out / "reference"
        extract_scripts(paths["00"] / "source.json", scripts)
        ref = reference_predict(scripts, base, [out / "member"], data, anchor, out / "anchor")
        own, _ = predict(base, out / "member", anchor, cfg, dtype="fp16")
        err = float(np.linalg.norm(own - ref) / max(float(np.linalg.norm(ref)), 1e-12))
        result["reference_error"] = err
        if not np.isfinite(err) or err > 0.02 or time.monotonic() - start > 2700:
            raise ContractError("synthetic smoke parity or time gate failed")
        write_json_new(out / "smoke.json", result)
    elif stage in ("02", "03", "04"):
        reusable_data_training(stage, paths, cfg, base, data, out, previous)
    elif stage == "05":
        frame = official(data)
        prediction_file(base, paths["04"] / "member", frame, cfg, out / "pred.npz")
        lp, _, _ = load_predictions([out / "pred.npz"], frame)
        y = np.array([LABELS.index(x) for x in frame.label])
        ba = float(np.mean([(lp.argmax(1)[y == k] == k).mean() for k in range(len(LABELS))]))
        write_json_new(
            out / "s1.json", {"balanced_accuracy": ba, "warning": ba < cfg["gates"]["s1_continue"]}
        )
        if ba < cfg["gates"]["s1_stop"]:
            raise ContractError("S1 did not pass the preregistered gate")
    elif stage in ("06", "07"):
        fold_fit(
            base,
            official(data),
            cfg,
            out,
            previous,
            "stage2" if stage == "06" else "control",
            read(paths["01"] / "smoke.json")["token_budget"],
            paths["04"] / "member" if stage == "06" else None,
        )
    elif stage == "08":
        evaluate_s2(official(data), paths, cfg, out)
    elif stage == "09":
        from .package import package_member
        from .train import predict

        members = [paths["06"] / name / "member" for name in ("f0", "f1", "f2", "f3", "f4", "full")]
        frame = official(data)
        valid = frame.loc[frame.fold == 0].reset_index(drop=True)
        fp, _ = predict(base, members[0], valid, cfg, "fp16")
        bf, _ = predict(base, members[0], valid, cfg, "bf16")
        provenance = {
            "git_commit": job["commit"],
            "config_sha256": file_sha256(DEFAULT_CONFIG),
            "data_lock_sha256": file_sha256(paths["02"] / "source_lock.json"),
        }
        notice = (paths["03"] / "audit/NOTICE.md").read_text()
        result = package_member(
            out, paths["00"] / "source.json", base, members, data, valid, fp, bf, provenance, notice
        )
        write_json_new(out / "parity.json", result)
    elif stage == "10":
        evaluation = read(paths["08"] / "evaluation.json")
        d = evaluation["comparisons"]["S2_minus_C0"]
        scores = {k: v["balanced_accuracy"] for k, v in evaluation["arms"].items()}
        scores["S2_minus_C0"] = d["delta"]
        summary = {
            "commit": job["commit"],
            "config_sha256": file_sha256(DEFAULT_CONFIG),
            "fold_sha256": cfg["pins"]["folds_digest"],
            "metrics": scores,
            "by_arm": {
                k: {
                    field: arm[field]
                    for field in (
                        "ci95",
                        "per_fold",
                        "per_class",
                        "family_balanced_accuracy",
                        "oracle_sibling_balanced_accuracy",
                    )
                }
                for k, arm in evaluation["arms"].items()
            },
            "delta_ci95": d["ci95"],
            "gates": dict.fromkeys(("data", "smoke", "s1", "s2", "parity"), True),
            "resources": job["resources"],
            "versions": read(paths["00"] / "inputs.json")["gpu_versions"],
        }
        publish(
            out / "public",
            {
                "summary.json": summary,
                "audit_summary.json": read(paths["03"] / "audit_summary.json"),
                "package_manifest.json": read(paths["09"] / "package_manifest.json"),
                "source_lock.json": read(paths["02"] / "source_lock.json"),
            },
        )
        (out / "REPORT.md").write_text(
            (out / "public/summary.md").read_text()
            + "\n详见阶段 08 的 evaluation.json 和阶段 09 的 parity.json。\n"
        )
    else:
        raise ContractError("unknown stage")


def main() -> None:
    job = read(Path(sys.argv[1]))
    gpu = job["stage"] in {"00", "01", "04", "05", "06", "07", "09"}
    start = time.monotonic()
    if gpu:
        import torch

        torch.cuda.reset_peak_memory_stats()
    try:
        run(job)
    finally:
        out = Path(job["output"])
        timings = [read(p) for p in out.glob("*/timing.json")]
        if (out / "timing.json").is_file():
            timings.append(read(out / "timing.json"))
        write_json_new(
            out.parent / "resources.json",
            {
                "seconds": time.monotonic() - start,
                "peak_allocated_bytes": max(
                    [torch.cuda.max_memory_allocated() if gpu else 0]
                    + [t.get("peak_allocated_bytes", 0) for t in timings]
                ),
                "peak_reserved_bytes": max(
                    [torch.cuda.max_memory_reserved() if gpu else 0]
                    + [t.get("peak_reserved_bytes", 0) for t in timings]
                ),
                "tokens_per_second": sum(t.get("tokens_per_second", 0.0) for t in timings)
                / max(1, len(timings)),
            },
        )


if __name__ == "__main__":
    main()
