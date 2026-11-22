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
