from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ntl_baseline.constants import LABELS, ContractError
from ntl_repro.branch import (
    FEATURE_SIZE,
    POOL_LAYERS,
    PlainReadout,
    branch_outputs,
    encode_passages,
    load_readout,
    verify_anchors,
)
from ntl_repro.run import check_assets, load_config


@pytest.mark.parametrize("rows", [1, 3, 9])
def test_dynamic_scores_keep_input_order(rows: int) -> None:
    test = pd.DataFrame({"id": [f"{i:03}" for i in reversed(range(rows))], "text": ["x"] * rows})
    logits = np.full((rows, 18), -1500.0)
    winners = np.arange(rows) % 18
    logits[np.arange(rows), winners] = 1500
    scores, predictions = branch_outputs(test, logits)
    assert tuple(scores.columns) == ("id", *LABELS)
    assert list(scores.id) == list(predictions.id) == list(test.id)
    assert list(predictions.model) == [LABELS[i] for i in winners]
    assert np.isfinite(scores.iloc[:, 1:]).all().all()
    np.testing.assert_allclose(scores.iloc[:, 1:].sum(axis=1), 1.0)


@pytest.mark.parametrize("shape, value", [((1, 18), 0), ((2, 17), 0), ((2, 18), np.nan)])
def test_invalid_scores_fail(shape: tuple[int, int], value: float) -> None:
    test = pd.DataFrame({"id": ["a", "b"], "text": ["x", "y"]})
    with pytest.raises(ContractError, match="logits"):
        branch_outputs(test, np.full(shape, value))


def test_duplicate_input_ids_fail() -> None:
    test = pd.DataFrame({"id": ["a", "a"], "text": ["x", "y"]})
    with pytest.raises(ContractError, match="unique"):
        branch_outputs(test, np.zeros((2, 18)))


def test_pinned_readout_order_and_shape(tmp_path: Path) -> None:
    path = tmp_path / "readouts.npz"
    coef = np.zeros((18, FEATURE_SIZE), dtype=np.float32)
    coef[5, 1] = 2
    bias = np.arange(18, dtype=np.float32)
    for classes in (LABELS, LABELS[::-1]):
        np.savez(
            path, classes=classes, pool_layers=POOL_LAYERS, plain_coef=coef, plain_intercept=bias
        )
        if classes == LABELS:
            readout = load_readout(path)
            features = np.zeros((3, FEATURE_SIZE), dtype=np.float32)
            features[:, 1] = [1, 2, 3]
            np.testing.assert_array_equal(readout.logits(features)[:, 5], [7, 9, 11])
        else:
            with pytest.raises(ContractError, match="class order"):
                load_readout(path)
    bad = PlainReadout(coef, bias)
    with pytest.raises(ContractError, match="features"):
        bad.logits(np.zeros((1, 4)))


def test_tokenization_keeps_raw_text_and_appends_eos() -> None:
    def tokenizer(texts: list[str], **options: object) -> dict[str, list[list[int]]]:
        assert texts[0] == "  # a\n"
        assert options == {"add_special_tokens": False, "truncation": True, "max_length": 447}
        return {"input_ids": [[3, 4] if text else [] for text in texts]}

    assert encode_passages(tokenizer, ["  # a\n", "b"], 9) == [[3, 4, 9], [3, 4, 9]]
    with pytest.raises(ContractError, match="no tokens"):
        encode_passages(tokenizer, ["  # a\n", ""], 9)


def test_anchor_threshold_and_shape() -> None:
    saved = np.ones((3, FEATURE_SIZE), dtype=np.float32)
    assert verify_anchors(saved, saved) == 0.0
    with pytest.raises(ContractError, match="parity"):
        verify_anchors(saved * 1.03, saved)
    with pytest.raises(ContractError, match="anchor features"):
        verify_anchors(saved[:1], saved)


def test_missing_assets_fail_before_gpu_import(tmp_path: Path) -> None:
    config = load_config(Path("configs/v31_single_gpu.toml"))
    with pytest.raises(ContractError, match="missing|directory"):
        check_assets(tmp_path, tmp_path, config)


def test_cli_failure_preserves_existing_run(tmp_path: Path, monkeypatch, capsys) -> None:
    from ntl_repro import run

    monkeypatch.setattr(run, "ROOT", tmp_path)
    destination = tmp_path / "artifacts" / "previous"
    destination.mkdir(parents=True)
    saved = destination / "completed.json"
    saved.write_text("historical", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "ntl_repro.run",
            "--synthetic",
            "--base-dir",
            str(tmp_path),
            "--checkpoint-dir",
            str(tmp_path),
            "--config",
            "configs/v31_single_gpu.toml",
            "--run-dir",
            str(destination),
        ],
    )
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 1
    assert "already exists" in capsys.readouterr().err
    assert saved.read_text(encoding="utf-8") == "historical"
