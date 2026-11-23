import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_server_bootstrap_stops_before_install_when_python_is_missing(tmp_path):
    result = subprocess.run(
        ["bash", "scripts/run_v31_server_smoke.sh", str(tmp_path / "missing-python")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "CUDA PyTorch virtual environment" in result.stderr


def test_cached_auxiliary_run_skips_install_and_download(tmp_path):
    python = tmp_path / "fake-python"
    python.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$HF_HUB_OFFLINE" "$@"\ncase "$1" in -m) exit 7;; esac\n'
    )
    python.chmod(0o700)
    result = subprocess.run(
        ["bash", "scripts/run_v31_server_smoke.sh", "--cached-auxiliary", str(python)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 7, result.stderr
    assert "1\n-m\nntl_repro.run\n--synthetic\n--auxiliary" in result.stdout
    assert "--anchor-data-dir\n" in result.stdout
    assert "pip" not in result.stdout
