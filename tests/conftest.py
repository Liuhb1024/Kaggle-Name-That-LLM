"""Only synthetic text fixtures; no competition passages in tests."""

from pathlib import Path

import pandas as pd
import pytest

from ntl_baseline.constants import LABELS


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": f"synthetic-{group:02d}-{i:02d}",
                "prompt_id": f"prompt-{group:02d}",
                "text": f"holdoutmarker{group} synthetic prose. voice{i} rhythm{i} choice{i}!",
                "model": label,
            }
            for group in range(5)
            for i, label in enumerate(LABELS)
        ]
    )


@pytest.fixture
def test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["synthetic-test-2", "synthetic-test-1", "synthetic-test-3"],
            "text": ["An unseen paragraph voice3!", "Another voice1 paragraph.", ""],
        }
    )


@pytest.fixture
def data_dir(tmp_path: Path, train: pd.DataFrame, test_frame: pd.DataFrame) -> Path:
    directory = tmp_path / "synthetic-input"
    directory.mkdir()
    train.to_csv(directory / "train.csv", index=False)
    test_frame.to_csv(directory / "test.csv", index=False)
    pd.DataFrame({"id": test_frame["id"], "model": LABELS[0]}).to_parquet(
        directory / "sample_submission.parquet", index=False
    )
    return directory
