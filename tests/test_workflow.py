import hashlib
import json
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

from ntl_baseline.config import BaselineConfig
from ntl_baseline.constants import ContractError
from ntl_baseline.data import import_archive
from ntl_baseline.workflow import audit_data, predict_model, run_baseline, train_model


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def test_audit_reports_integrity_without_passage_content(data_dir):
    before = hashes(data_dir)
    audit = audit_data(data_dir)
    assert audit["train_rows"] == 90
    assert audit["prompt_count"] == 5
    assert audit["test_rows"] == 3
    assert len(audit["files"]) == 3
    assert "holdoutmarker" not in json.dumps(audit)
    assert hashes(data_dir) == before


def test_complete_run_roundtrips_model_oof_and_parquet(data_dir, tmp_path):
    before = hashes(data_dir)
    directory = tmp_path / "run"
    result = run_baseline(data_dir, directory, BaselineConfig())
    assert result["status"] == "complete"
    submission_file = directory / "prediction/submission.parquet"
    assert submission_file.read_bytes()[:4] == b"PAR1"
    expected = pd.read_csv(data_dir / "test.csv").id.tolist()
    submission = pd.read_parquet(submission_file)
    assert submission.columns.tolist() == ["id", "model"]
    assert submission.id.tolist() == expected
    assert len(pd.read_parquet(directory / "validation/oof.parquet")) == 90
    assert (directory / "model/pipeline.joblib").is_file()
    assert (directory / "completed.json").is_file()
    predict_model(data_dir, directory / "model", tmp_path / "standalone")
    pd.testing.assert_frame_equal(
        submission, pd.read_parquet(tmp_path / "standalone/submission.parquet")
    )
    assert hashes(data_dir) == before
    with pytest.raises(FileExistsError):
        run_baseline(data_dir, directory, BaselineConfig())
    assert hashes(data_dir) == before


def test_tampered_model_is_rejected_before_unpickling(data_dir, tmp_path):
    model_dir = tmp_path / "model"
    train_model(data_dir, model_dir, BaselineConfig())
    with (model_dir / "pipeline.joblib").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ContractError, match="hash"):
        predict_model(data_dir, model_dir, tmp_path / "prediction")


def test_predict_handles_arbitrary_test_row_count(data_dir, test_frame, tmp_path):
    model_dir = tmp_path / "model"
    train_model(data_dir, model_dir, BaselineConfig())
    larger = pd.concat([test_frame.assign(id=test_frame.id + f"-{i}") for i in range(7)])
    larger.to_csv(data_dir / "test.csv", index=False)
    # Inference only needs test.csv, never stale sample_submission row counts or train.csv.
    (data_dir / "train.csv").unlink()
    predict_model(data_dir, model_dir, tmp_path / "larger")
    assert len(pd.read_parquet(tmp_path / "larger/submission.parquet")) == 21


def test_cli_run_and_failure_exit_contract(data_dir, tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ntl_baseline",
            "run",
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(tmp_path / "cli"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "complete"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ntl_baseline",
            "run",
            "--data-dir",
            str(tmp_path / "missing"),
            "--run-dir",
            str(tmp_path / "bad"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "error" in proc.stderr.lower()
    assert not (tmp_path / "bad/completed.json").exists()


def test_safe_archive_import_is_non_destructive(data_dir, tmp_path):
    archive = tmp_path / "official-download.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file in data_dir.iterdir():
            zf.write(file, file.name)
    destination = tmp_path / "imported"
    manifest = import_archive(archive, destination)
    assert manifest["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    for file in data_dir.iterdir():
        assert (destination / file.name).read_bytes() == file.read_bytes()
    assert (destination / "source.json").is_file()
    with pytest.raises(FileExistsError):
        import_archive(archive, destination)


@pytest.mark.parametrize("unsafe", ["../train.csv", "/train.csv", "dir/../../train.csv"])
def test_archive_path_traversal_is_rejected_before_writes(tmp_path, unsafe):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(unsafe, b"unsafe")
    with pytest.raises(ContractError, match="archive"):
        import_archive(archive, tmp_path / "imported")
    assert not (tmp_path / "imported").exists()
