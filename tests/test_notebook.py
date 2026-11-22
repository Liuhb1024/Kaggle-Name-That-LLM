import json
import sys

import nbformat
import pandas as pd
import pytest
from nbclient import NotebookClient

from ntl_baseline.artifacts import source_hashes
from ntl_baseline.notebook import notebook_document


def test_generated_notebook_contains_no_outputs_and_tracks_source():
    document = nbformat.from_dict(notebook_document())
    nbformat.validate(document)
    assert document.metadata["ntl_baseline_source_sha256"] == source_hashes()
    code_cells = [cell for cell in document.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(not cell.outputs and cell.execution_count is None for cell in code_cells)
    assert "pip install" not in "\n".join(cell.source for cell in code_cells)


@pytest.mark.parametrize("sample_state", ["matching", "stale", "absent"])
def test_portable_notebook_runs_with_network_blocked(data_dir, tmp_path, monkeypatch, sample_state):
    if sample_state != "matching":
        hidden = pd.read_csv(data_dir / "test.csv")
        hidden = pd.concat([hidden.assign(id=hidden.id + f"-{i}") for i in range(7)])
        hidden.to_csv(data_dir / "test.csv", index=False)
        if sample_state == "absent":
            (data_dir / "sample_submission.parquet").unlink()
    workdir = tmp_path / "isolated-notebook"
    workdir.mkdir()
    kernels = tmp_path / "kernels"
    spec = kernels / "ntl-offline"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Temporary offline test kernel",
                "language": "python",
            }
        )
    )
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path))
    document = nbformat.from_dict(notebook_document())
    guard = f"""import os, socket
os.environ['NTL_DATA_DIR'] = {str(data_dir)!r}
os.environ['NTL_RUN_DIR'] = {str(workdir / "baseline-run")!r}
def deny_network(*args, **kwargs):
    raise RuntimeError('network access blocked in offline test')
socket.socket.connect = deny_network
socket.create_connection = deny_network
"""
    document.cells.insert(0, nbformat.v4.new_code_cell(guard))
    client = NotebookClient(
        document,
        timeout=180,
        kernel_name="ntl-offline",
        resources={"metadata": {"path": str(workdir)}},
    )
    executed = client.execute()
    assert not any(
        output.output_type == "error"
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
    submission = pd.read_parquet(workdir / "submission.parquet")
    assert submission.id.tolist() == pd.read_csv(data_dir / "test.csv").id.tolist()
    report = json.loads((workdir / "baseline-run/completed.json").read_text())
    assert report["status"] == "complete"
    assert set(report["artifact_sha256"])
