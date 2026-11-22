"""Deterministically regenerate or check the offline Kaggle notebook."""

import argparse
import json
from pathlib import Path

from ntl_baseline.notebook import notebook_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("notebooks/official_baseline.ipynb"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = json.dumps(notebook_document(), ensure_ascii=False, indent=1) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != payload:
            raise SystemExit("Notebook differs from current sources; regenerate it.")
        print("Notebook matches current sources; outputs are empty.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Generated {args.output}")


if __name__ == "__main__":
    main()
