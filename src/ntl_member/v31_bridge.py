"""Build a private V31 candidate from a digest-pinned upstream notebook."""

import copy
import hashlib
import inspect
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from ntl_baseline.artifacts import write_json_new
from ntl_baseline.data import file_sha256

SOURCE_URL = "https://www.kaggle.com/kernels/scriptcontent/372363760/download"
SOURCE_SHA256 = "1556c5110b52c917144cd961610f2fc6fd803d4c3982b4e6d2e977273eb9038d"
MEMBERS = tuple([f"q8pool_f{i}" for i in range(5)] + ["q8pool"])
LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]


def obtain_source(dest: Path) -> Path:
    """Cache the exact public release. A prior bad download is never trusted."""
    if not dest.exists():
        dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent) as tmp:
            with urlopen(SOURCE_URL, timeout=90) as stream:
                if not stream.geturl().startswith("https://"):
                    raise ValueError("Insecure source redirect")
                payload = stream.read(16 * 1024**2 + 1)
            if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
                raise ValueError("V31 source SHA-256 mismatch")
            tmp.write(payload)
            tmp.flush()
            try:
                os.link(tmp.name, dest)
            except FileExistsError:
                pass
    if file_sha256(dest) != SOURCE_SHA256:
        raise ValueError("V31 source SHA-256 mismatch")
    return dest


def _document(source: Path) -> dict[str, Any]:
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("V31 source SHA-256 mismatch")
    return json.loads(data)


def _source(cell: dict[str, Any]) -> str:
    return "".join(cell["source"])


def _index(sources: list[str], marker: str, *, starts: bool = False) -> int:
    indices = [
        i
        for i, source in enumerate(sources)
        if (source.startswith(marker) if starts else marker in source)
    ]
    if len(indices) != 1 or sources[indices[0]].count(marker) != 1:
        raise ValueError(f"Expected exactly one source marker {marker!r}")
    return indices[0]


def extract_scripts(source: Path, destination: Path) -> None:
    cells = [_source(c) for c in _document(source)["cells"]]
    scripts = {}
    for script in ("pipe.py", "gauss.py", "infer.py"):
        marker = f"%%writefile {script}\n"
        scripts[script] = cells[_index(cells, marker, starts=True)].removeprefix(marker)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, content in scripts.items():
        (destination / name).write_text(content, encoding="utf-8")


# These self-contained functions are also embedded into Kaggle's offline runtime.
def validate_manifest_entries(entries: dict[str, dict]) -> dict[str, dict]:
    """Check the public summary file before a Notebook can pin its digest."""
    required = {"NOTICE.md", "anchors/test.csv", "anchors/expected_logits.json"}
    for member in MEMBERS:
        for filename in (
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "heads.safetensors",
            "meta.json",
        ):
            required.add(f"{member}/{filename}")
    if not isinstance(entries, dict) or set(entries) != required:
        raise ValueError("Unexpected private member file set")
    for record in entries.values():
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or type(record["bytes"]) is not int
            or record["bytes"] <= 0
            or not isinstance(record["sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"])
        ):
            raise ValueError("Invalid member file manifest record")
    return entries


def read_manifest(manifest: Path) -> dict[str, dict]:
    return validate_manifest_entries(json.loads(manifest.read_text(encoding="utf-8")))


def verify_manifest(root: Path, digest: str) -> None:
    manifest = root / "MANIFEST.json"
    if not manifest.is_file() or file_sha256(manifest) != digest:
        raise ValueError("Private member MANIFEST.json SHA-256 mismatch")
    entries = read_manifest(manifest)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(entries) | {"MANIFEST.json"}:
        raise ValueError("Unexpected private member file set")
    for name, info in entries.items():
        path = root / name
        if path.stat().st_size != info["bytes"] or file_sha256(path) != info["sha256"]:
            raise ValueError(f"Private member payload mismatch: {name}")


def reference_command(
    script: Path,
    base: Path,
    adapters: list[Path],
    data_dir: Path,
    work: Path,
    python: str,
) -> list[str]:
    member = {
        "name": "q8pool",
        "type": "l24",
        "path": str(base),
        "adapters": list(map(str, adapters)),
        "keep": 24,
        "layers": LAYERS,
        "w": 1.0,
        "bs": 16,
        "max_len": 384,
        "twin": 0,
        "gauss": 0,
        "dp": False,
    }
    return [
        python,
        str(script),
        "--models",
        json.dumps([member]),
        "--data",
        str(data_dir),
        "--tfidf_w",
        "0",
        "--distort_w",
        "0",
        "--svc_w",
        "0",
        "--adapt_w",
        "0",
        "--out",
        str(work / "submission.parquet"),
        "--save_final_logits",
        str(work / "logits.npz"),
    ]


def check_members(log_lines: list[str], adapters: list[Path]) -> None:
    names = [
        match[1]
        for line in log_lines
        if (match := re.search(r"\bl24 q8pool: adapter (\S+) done\b", line))
    ]
    if names != [p.name for p in adapters] or any(
        "MEMBER FAILED q8pool" in line for line in log_lines
    ):
        raise ValueError("V31 failed to apply all expected q8pool adapters")


def _cell(source: str, name: str) -> dict[str, Any]:
    return dict(
        cell_type="code",
        id=name,
        metadata={},
        execution_count=None,
        outputs=[],
        source=source.splitlines(keepends=True),
    )


def _asset_cell(handle: str, version: int, digest: str) -> str:
    owner, slug = handle.split("/")
    return f'''from pathlib import Path
import hashlib
import json
import re
MEMBERS = {MEMBERS!r}
LAYERS = {LAYERS!r}
{inspect.getsource(file_sha256)}
{inspect.getsource(validate_manifest_entries)}
{inspect.getsource(read_manifest)}
{inspect.getsource(verify_manifest)}
# Fixed Kaggle attachment: {handle}/versions/{version}
roots = [Path("/kaggle/input/{slug}"), Path("/kaggle/input/datasets/{owner}/{slug}")]
roots = {{p.resolve() for p in roots if (p / "MANIFEST.json").is_file()}}
if len(roots) != 1:
    raise ValueError("Expected one fixed q8pool package mount")
Q8POOL_ROOT = roots.pop()
verify_manifest(Q8POOL_ROOT, "{digest}")
AQ8POOL = [str(Q8POOL_ROOT / name) for name in MEMBERS]
Q8POOL_CFG = json.loads((Q8POOL_ROOT / MEMBERS[-1] / "meta.json").read_text())["cfg"]
if Q8POOL_CFG != dict(keep=24, layers=LAYERS, max_len=384, eos_id=None, no_final_norm=False):
    raise ValueError("Unexpected q8pool feature configuration")
'''


def _anchor_cell() -> str:
    return f"""from pathlib import Path
import json
import os
import re
import subprocess
import sys
import tempfile
import numpy as np

{inspect.getsource(reference_command)}
{inspect.getsource(check_members)}
RUN_LOG = []
def run_logged(cmd, env=None):
    RUN_LOG.clear()
    with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            RUN_LOG.append(line)
            if not re.search(r"Warning|warn", line):
                print(line, end="")
        return process.wait()

with tempfile.TemporaryDirectory(dir="/kaggle/temp") as temp:
    work = Path(temp)
    (work / "train.csv").symlink_to(Path(DATA) / "train.csv")
    (work / "test.csv").symlink_to(Q8POOL_ROOT / "anchors/test.csv")
    paths = list(map(Path, AQ8POOL))
    cmd = reference_command(Path("infer.py"), Path(B8), paths, work, work, sys.executable)
    if run_logged(cmd, env=dict(os.environ, NTL_PIPE_WORKERS="1", HF_HUB_OFFLINE="1")):
        raise RuntimeError("V31 q8pool anchor inference failed")
    check_members(RUN_LOG, paths)
    expected = json.loads((Q8POOL_ROOT / "anchors/expected_logits.json").read_text())
    with np.load(work / "logits.npz", allow_pickle=False) as result:
        values = result["logits"]
        reference = np.array(expected["logits"])
        if (result["labels"].tolist() != expected["labels"]
                or result["ids"].tolist() != expected["ids"]
                or values.shape != reference.shape or not np.isfinite(values).all()
                or not np.isfinite(reference).all()
                or np.abs(values - reference).max() > 0.05
                or not np.array_equal(values.argmax(1), reference.argmax(1))):
            raise ValueError("V31 q8pool anchor differs from locked server result")
    print("V31 q8pool anchor passed")
"""


def patch(
    original: dict[str, Any],
    candidate: str,
    handle: str,
    version: int,
    manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    if candidate not in {"C1", "C2"} or not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", handle):
        raise ValueError("Invalid candidate or Kaggle dataset handle")
    if (
        type(version) is not int
        or version < 1
        or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)
    ):
        raise ValueError("Invalid package revision or digest")
    document = copy.deepcopy(original)
    cells = document["cells"]
    sources = [_source(cell) for cell in cells]
    asset = _index(sources, "AQ8V5 = [adapter_dir(")
    comp = _index(sources, "%%writefile comp_stack.py\n", starts=True)
    run = _index(sources, "def run(cmd, env=None):", starts=True)
    member = _index(sources, '"name": "q8v5"')
    if not asset < comp < run < member or "MODELS = [" not in sources[member]:
        raise ValueError("Unexpected order of V31 marker cells")
    original_member = sources[member]
    lines = original_member.splitlines(keepends=True)
    row = next(line for line in lines if '"name": "q8v5"' in line)
    if row.count("AQ8V5") != 1 or original_member.count("code = run(cmd") != 1:
        raise ValueError("Unexpected q8v5 adapter or inference call marker")
    replacement = row.replace('"q8v5"', '"q8pool"').replace("AQ8V5", "AQ8POOL")
    if candidate == "C2":
        replacement = row + replacement
    rewritten = original_member.replace(row, replacement).replace(
        "code = run(cmd", "code = run_logged(cmd"
    )
    rewritten += "\nif not FAST_COMMIT:\n    check_members(RUN_LOG, list(map(Path, AQ8POOL)))\n"
    cells[member]["source"] = rewritten.splitlines(keepends=True)
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    cells.insert(comp + 1, _cell(_anchor_cell(), "q8pool-anchor"))
    cells.insert(asset + 1, _cell(_asset_cell(handle, version, manifest_sha256), "q8pool-asset"))
    evidence = [
        f"# V31 {candidate} cell audit",
        "",
        f"Source version 372363760; member attachment {handle}/versions/{version}.",
        f"Manifest SHA-256: {manifest_sha256}",
        "",
        "| Original cell | Output cell | Change | Original SHA-256 | Output SHA-256 |",
        "|---:|---:|---|---|---|",
    ]
    for i, old in enumerate(sources):
        j = i + (i > asset) + (i > comp)
        new = _source(cells[j])
        if old != new and i != member:
            raise ValueError("Unexpected V31 source change")
        status = "member" if i == member else "same"
        evidence.append(f"| {i} | {j} | {status} | {_digest(old)} | {_digest(new)} |")
    for j in (asset + 1, comp + 2):
        evidence.append(f"| – | {j} | inserted | – | {_digest(_source(cells[j]))} |")
    return document, "\n".join(evidence) + "\n"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def build(
    source: Path,
    manifest: Path,
    candidate: str,
    handle: str,
    version: int,
    output: Path,
) -> None:
    read_manifest(manifest)
    document, evidence = patch(_document(source), candidate, handle, version, file_sha256(manifest))
    audit_path = output.with_suffix(".cells.md")
    if output.exists() or audit_path.exists():
        raise FileExistsError("Use fresh output paths for notebook and audit")
    write_json_new(output, document)
    with audit_path.open("x", encoding="utf-8") as stream:
        stream.write(evidence)
