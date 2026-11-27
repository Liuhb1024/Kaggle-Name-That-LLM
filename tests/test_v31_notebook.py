from pathlib import Path

import numpy as np
import pandas as pd

from ntl_baseline.constants import LABELS


def test_v31_notebook_exports_dynamic_ids_without_training_or_sample(tmp_path: Path) -> None:
    from ntl_repro.notebook import notebook_document, write_plain_submission

    data = tmp_path / "data"
    data.mkdir()
    (data / "test.csv").write_text("id,text\n002,one\n01,two\n009,three\n", encoding="utf-8")

    def logits(texts: list[str]) -> np.ndarray:
        assert texts == ["one", "two", "three"]
        return np.eye(len(LABELS), dtype=np.float32)[: len(texts)]

    output = tmp_path / "submission.parquet"
    assert write_plain_submission(data, output, logits) == 3
    frame = pd.read_parquet(output)
    assert frame.to_dict("list") == {"id": ["002", "01", "009"], "model": list(LABELS[:3])}
    cells = [cell for cell in notebook_document()["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells)
    for cell in cells:
        compile(cell["source"], "<v31-notebook>", "exec")
