import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def guard(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/check_repository.py"
    spec = importlib.util.spec_from_file_location("repository_guard_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    snapshot = tmp_path / "docs/research/2026-11-15/snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "fixture.md").write_text("Synthetic historical fixture.\n")
    digest = hashlib.sha256((snapshot / "fixture.md").read_bytes()).hexdigest()
    (snapshot / "research-manifest-2026-11-15-v2.sha256").write_text(f"{digest}  fixture.md\n")
    return module


@pytest.mark.parametrize("relative", ["data/raw/train.csv", "artifacts/run/metrics.json"])
def test_guard_rejects_git_visible_private_inputs_or_artifacts(guard, relative):
    path = guard.ROOT / relative
    path.parent.mkdir(parents=True)
    path.write_text("synthetic only")
    with pytest.raises(SystemExit, match="Private"):
        guard.main()


def test_guard_detects_secret_without_echoing_it(guard):
    synthetic_credential = "sk-" + "A" * 30
    (guard.ROOT / "accident.py").write_text(repr(synthetic_credential))
    with pytest.raises(SystemExit, match="pattern") as error:
        guard.main()
    assert synthetic_credential not in str(error.value)


@pytest.mark.parametrize(
    "relative",
    ["backup/train.csv", "checkpoint.safetensors", "runs/submission.parquet", "secrets/.env.local"],
)
def test_guard_rejects_assets_outside_ignored_directories(guard, relative):
    path = guard.ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic only")
    with pytest.raises(SystemExit, match="Git-visible"):
        guard.main()


def test_guard_patterns_do_not_match_their_own_definitions(guard):
    script = Path(__file__).resolve().parents[1] / "scripts/check_repository.py"
    assert guard.SENSITIVE_PATTERNS.search(script.read_text()) is None
