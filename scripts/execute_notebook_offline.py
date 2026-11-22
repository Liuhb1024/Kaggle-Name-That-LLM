"""Execute the generated Notebook in a fresh kernel with Python networking blocked.

Executed outputs remain under ignored artifacts, never in the shared Notebook.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, default=Path("notebooks/official_baseline.ipynb"))
    args = parser.parse_args()
    workdir = args.work_dir.resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    document = nbformat.read(args.notebook, as_version=4)
    guard = f"""import os, socket
os.environ['NTL_DATA_DIR'] = {str(args.data_dir.resolve())!r}
os.environ['NTL_RUN_DIR'] = {str(workdir / "baseline-run")!r}
def deny_network(*args, **kwargs):
    raise RuntimeError('network access blocked in offline Notebook execution')
socket.socket.connect = deny_network
socket.create_connection = deny_network
socket.getaddrinfo = deny_network
"""
    document.cells.insert(0, nbformat.v4.new_code_cell(guard))
    with tempfile.TemporaryDirectory(prefix="ntl-kernel-") as temporary:
        spec = Path(temporary) / "kernels" / "ntl-offline"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Temporary ntl-baseline kernel",
                    "language": "python",
                }
            )
        )
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = temporary
        try:
            executed = NotebookClient(
                document,
                timeout=3600,
                kernel_name="ntl-offline",
                resources={"metadata": {"path": str(workdir)}},
            ).execute()
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous
    nbformat.write(executed, workdir / "executed.ipynb")
    print((workdir / "baseline-run/completed.json").read_text())


if __name__ == "__main__":
    main()
