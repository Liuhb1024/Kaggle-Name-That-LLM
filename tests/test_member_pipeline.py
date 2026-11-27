from pathlib import Path

import numpy as np
import pytest

from ntl_baseline.constants import LABELS, ContractError
from ntl_member.config import DEFAULT_CONFIG, load
from ntl_member.metrics import compare, score
from ntl_member.pipeline import verified_stage
from ntl_member.report import validate_public


def test_config_and_metrics():
    config = load(DEFAULT_CONFIG)
    assert config["train"]["token_budgets"] == [8192, 6144, 4096]
    y = np.tile(np.arange(len(LABELS)), 5)
    groups = np.repeat(np.arange(5), len(LABELS))
    logp = np.log(np.eye(len(LABELS))[y] * 0.982 + 0.001)
    result = score(y, logp, groups, groups, np.full(len(y), 66), 30)
    assert result["balanced_accuracy"] == 1.0
    assert compare(y, y, y, groups, 100)["ci95"] == [0.0, 0.0]


def test_untrusted_public_output():
    for value in (
        {"text": "secret"},
        {"metrics": {"gpu": "/home/someuser"}},
        {"unexpected": "anything"},
        {"metrics": {"notes": "secret"}},
    ):
        with pytest.raises(ContractError):
            validate_public("summary.json", value)


def test_incomplete_stage_is_not_resumable(tmp_path: Path):
    with pytest.raises(ContractError):
        verified_stage(tmp_path, "missing-signature")


def test_fold_files_cannot_swap_heldout_ids(tmp_path):
    import pandas as pd

    from ntl_member.stages import load_predictions

    frame = pd.DataFrame({"id": ["row0", "row1"], "fold": [0, 1]})
    paths = []
    for fold in range(2):
        p = tmp_path / f"f{fold}.npz"
        lp = np.full((1, len(LABELS)), -np.log(len(LABELS)))
        np.savez(p, ids=[f"row{1 - fold}"], plain=lp, centered=lp, bias=np.zeros(len(LABELS)))
        paths.append(p)
    with pytest.raises(ContractError, match="held-out"):
        load_predictions(paths, frame)


def test_public_guard_can_detect_forbidden_field_in_manifest():
    value = {"text": {"bytes": 1, "sha256": "0" * 64}}
    with pytest.raises(ContractError):
        validate_public("package_manifest.json", value)


def test_controller_rejects_corrupt_attempt_and_changed_signature(tmp_path, monkeypatch):
    from ntl_baseline.artifacts import directory_hashes, write_json_new
    from ntl_member import pipeline

    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    attempt = tmp_path / "artifacts/run/02-download/attempt-1"
    output = attempt / "output"
    output.mkdir(parents=True)
    (output / "file").write_text("known content")
    write_json_new(
        attempt / "completed.json", {"signature": "one", "outputs": directory_hashes(output)}
    )
    assert verified_stage(attempt, "one") == output
    with pytest.raises(ContractError, match="signature"):
        verified_stage(attempt, "two")
    (output / "file").write_text("partial transfer")
    with pytest.raises(ContractError, match="digest"):
        verified_stage(attempt, "one")


def test_synthetic_evaluation_to_public_report(tmp_path):
    import pandas as pd

    from ntl_baseline.artifacts import write_json_new
    from ntl_member import stages
    from ntl_member.v31_bridge import MEMBERS

    paths = {f"{i:02d}": tmp_path / f"{i:02d}" for i in range(11)}
    for directory in paths.values():
        directory.mkdir()
    train = tmp_path / "train.csv"
    examples = [
        {
            "id": f"synthetic-{g}-{i}",
            "prompt_id": f"p{g}",
            "model": label,
            "text": ("engineered fixture sentence about pencils " + str(i) + " ") * 7,
        }
        for g in range(10)
        for i, label in enumerate(LABELS)
    ]
    pd.DataFrame(examples).to_csv(train, index=False)
    frame = stages.official(train)
    y = np.array([LABELS.index(label) for label in frame.label])

    def fixture_prediction(out, keep, pred):
        lp = np.log(np.eye(len(LABELS))[pred[keep]] * 0.982 + 0.001)
        out.parent.mkdir(exist_ok=True)
        np.savez(
            out,
            ids=frame.id[keep].to_numpy(dtype=str),
            plain=lp,
            centered=lp,
            bias=np.zeros(len(LABELS)),
        )

    fixture_prediction(paths["05"] / "pred.npz", np.ones(len(y), dtype=bool), y)
    for fold in range(5):
        keep = frame.fold == fold
        for key, pred in [("06", y), ("07", (y + 1) % len(LABELS))]:
            fixture_prediction(paths[key] / f"f{fold}/pred.npz", keep, pred)
    cfg = load()
    cfg["evaluation"]["bootstrap_resamples"] = 100
    stages.evaluate_s2(frame, paths, cfg, paths["08"])
    result = stages.read(paths["08"] / "evaluation.json")
    assert result["comparisons"]["S2_minus_C0"]["delta"] == 1.0
    write_json_new(paths["00"] / "inputs.json", {"gpu_versions": {"python": "3.12.3"}})
    write_json_new(
        paths["02"] / "source_lock.json",
        {
            "files": {"fixture.csv": {"bytes": 1, "sha256": "a" * 64}},
            "versions": {"example/fixture": 1},
        },
    )
    write_json_new(
        paths["03"] / "audit_summary.json",
        {
            "per_class": dict.fromkeys(LABELS, 10),
            "tiers": {"A": 180},
            "source_counts": {"example/fixture": 180},
            "gaps": {},
        },
    )
    filenames = ["NOTICE.md", "anchors/test.csv", "anchors/expected_logits.json"] + [
        f"{m}/{p}"
        for m in MEMBERS
        for p in [
            "meta.json",
            "heads.safetensors",
            "adapter/adapter_model.safetensors",
            "adapter/adapter_config.json",
        ]
    ]
    write_json_new(
        paths["09"] / "package_manifest.json",
        {k: {"bytes": 1, "sha256": "b" * 64} for k in filenames},
    )
    resources = dict.fromkeys(
        [
            "seconds",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "peak_device_mib",
            "tokens_per_second",
        ],
        0,
    )
    stages.run(
        {
            "stage": "10",
            "inputs": {k: str(v) for k, v in paths.items()},
            "base": "unused",
            "official_csv": str(train),
            "previous": [],
            "output": str(paths["10"]),
            "commit": "a" * 40,
            "resources": {k: resources for k in paths if k != "10"},
        }
    )
    report = paths["10"] / "public"
    assert (report / "package_manifest.json").read_bytes() == (
        paths["09"] / "package_manifest.json"
    ).read_bytes()
    for path in report.glob("*.json"):
        validate_public(path.name, stages.read(path))
        assert "synthetic-" not in path.read_text()


def test_orchestrator_failure_then_resume_validates_earlier_work(tmp_path, monkeypatch):
    import argparse
    import subprocess
    from contextlib import nullcontext

    from ntl_baseline.artifacts import write_json_new
    from ntl_baseline.data import file_sha256
    from ntl_member import pipeline

    data = tmp_path / "data/raw/name-that-llm/train.csv"
    data.parent.mkdir(parents=True)
    data.write_text("fixture")
    root = tmp_path / "artifacts/trial"
    root.mkdir(parents=True)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "NAMES", ("preflight", "smoke"))
    monkeypatch.setattr(pipeline, "sampling", lambda *args: nullcontext())
    visited = []

    def run_worker(cmd, **kwargs):
        job = pipeline.read(Path(cmd[-1]))
        out = Path(job["output"])
        visited.append(job["stage"])
        if job["stage"] == "01":
            raise subprocess.CalledProcessError(1, cmd)
        write_json_new(out / "inputs.json", {"official_train": file_sha256(data), "base": {}})
        write_json_new(out.parent / "resources.json", {"seconds": 1})

    monkeypatch.setattr(subprocess, "run", run_worker)
    args = argparse.Namespace(commit="a" * 40, resume=False, python="unused", reuse_stage=None)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.execute(args, root)
    assert (root / "01-smoke/attempt-1/failed.json").is_file()
    assert not (root / "completed.json").exists()
    args.resume = True
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.execute(args, root)
    assert visited == ["00", "01", "01"]
    assert (root / "01-smoke/attempt-2/failed.json").is_file()
    data.write_text("changed fixture")
    with pytest.raises(ContractError, match="inputs changed"):
        pipeline.execute(args, root)
    assert visited == ["00", "01", "01"]


def test_reuse_signature_depends_on_selection(tmp_path):
    from ntl_baseline.artifacts import write_json_new
    from ntl_member.pipeline import reuse_signature

    paths = {k: tmp_path / k for k in ("00", "01", "02", "03")}
    for p in paths.values():
        p.mkdir()
    write_json_new(
        paths["00"] / "inputs.json",
        {
            "gpu_versions": {"python": "3.12.3"},
            "official_train": "a" * 64,
            "base": {"config.json": "b" * 64},
        },
    )
    write_json_new(paths["01"] / "smoke.json", {"per_class_cap": 6000, "token_budget": 8192})
    write_json_new(paths["02"] / "source_lock.json", {"example": "fixture"})
    sample = paths["03"] / "stage1_sample.parquet"
    sample.write_bytes(b"first")
    cfg = load()
    before = reuse_signature("04", cfg, paths)
    sample.write_bytes(b"changed")
    assert reuse_signature("04", cfg, paths) != before


def test_server_plan_requires_only_python_standard_library(tmp_path):
    import os
    import subprocess
    import sys

    root = DEFAULT_CONFIG.parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [sys.executable, "-S", "-m", "ntl_member.plan"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0 and result.stdout.splitlines()[-1] == "10 report"
    assert not list(tmp_path.iterdir())
