"""Build a locked C1 or C2 Kaggle Notebook under ignored private artifacts."""

import argparse
from pathlib import Path

from ntl_member.v31_bridge import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", choices=["C1", "C2"], default="C1")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "artifacts/notebooks"
    target = (args.output or root / f"v31-q8pool-{args.candidate.lower()}.ipynb").resolve()
    if not target.is_relative_to(root.resolve()) or target.suffix != ".ipynb":
        parser.error("Use a new .ipynb path inside artifacts/notebooks/")
    build(args.source, args.manifest, args.candidate, args.dataset, args.version, target)
    print(target)
    print(target.with_suffix(".cells.md"))


if __name__ == "__main__":
    main()
