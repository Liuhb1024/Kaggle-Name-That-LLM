"""Deterministically regenerate or check the offline Kaggle notebook."""

import argparse
import json
from pathlib import Path

from ntl_baseline.notebook import notebook_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--v31-plain", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.v31_plain:
        from ntl_repro.notebook import notebook_document as v31_document

        document = v31_document()
        output = args.output or Path("notebooks/v31_plain.ipynb")
    else:
        document = notebook_document()
        output = args.output or Path("notebooks/official_baseline.ipynb")
    payload = json.dumps(document, ensure_ascii=False, indent=1) + "\n"
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != payload:
            raise SystemExit("Notebook differs from current sources; regenerate it.")
        print("Notebook matches current sources; outputs are empty.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"Generated {output}")


if __name__ == "__main__":
    main()
