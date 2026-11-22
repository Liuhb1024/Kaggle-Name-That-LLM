"""Local Markdown links, selected secret patterns and frozen research integrity.

Only examines Git-visible paths. No network, raw input or model deserialization.
This is a targeted check, not a general vulnerability or full secret audit.
"""

import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SENSITIVE_PATTERNS = re.compile(
    r"sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|KGAT_[A-Za-z0-9_-]{20,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"kaggleusercontent\.com/kf/|sharingControls"
    r"=true"
)
ASSET_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".parquet",
    ".arrow",
    ".jsonl",
    ".zip",
    ".gz",
    ".tar",
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".onnx",
    ".gguf",
    ".npy",
    ".npz",
}


def main() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    files = {Path(part.decode()) for part in result.stdout.split(b"\0") if part}
    failures: list[str] = []
    markdown_count = 0
    for relative in sorted(files):
        path = ROOT / relative
        if not path.is_file():
            continue
        private_path = relative.parts[:2] == ("data", "raw") or relative.parts[:1] == ("artifacts",)
        if private_path and path.name != ".gitkeep":
            failures.append(f"Private data/artifact is Git-visible: {relative}")
        if path.suffix.lower() in ASSET_EXTENSIONS:
            failures.append(f"Dataset, archive or model artifact is Git-visible: {relative}")
        env_file = path.name == ".env" or path.name.startswith(".env.")
        if path.name in {"kaggle.json", "access_token", ".netrc"} or (
            env_file and path.name != ".env.example"
        ):
            failures.append(f"Credential filename is Git-visible: {relative}")
        if path.suffix in {".py", ".md", ".json", ".toml", ".ipynb"}:
            text = path.read_text(encoding="utf-8")
            if SENSITIVE_PATTERNS.search(text):
                failures.append(f"Selected secret/temporary-URL pattern found in {relative}")
            if path.suffix == ".md":
                markdown_count += 1
                for target in re.findall(r"\]\(([^)]+)\)", text):
                    if target.startswith(("https://", "http://", "#")):
                        continue
                    local = target.split("#", 1)[0]
                    if not (path.parent / local).exists():
                        failures.append(f"Missing local link in {relative}: {target}")
    snapshot = ROOT / "docs/research/2026-11-15/snapshot"
    manifest_lines = (snapshot / "research-manifest-2026-11-15-v2.sha256").read_text().splitlines()
    for line in manifest_lines:
        expected, filename = line.split(maxsplit=1)
        actual = hashlib.sha256((snapshot / filename).read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"Frozen research snapshot hash mismatch: {filename}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"OK: {len(files)} Git-visible files, {markdown_count} Markdown files; "
        f"local links and selected secret patterns checked; {len(manifest_lines)} frozen "
        "snapshot hashes unchanged."
    )


if __name__ == "__main__":
    main()
