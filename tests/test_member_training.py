from pathlib import Path

import pytest

from ntl_member.batching import padded_batches, plan_steps
from ntl_member.train import latest_checkpoint


def test_padded_batches_account_for_padding_and_preserve_row_order() -> None:
    lengths = [2, 4, 10, 3]
    assert padded_batches(range(4), lengths, 10) == [[0, 1], [2], [3]]
    with pytest.raises(ValueError, match="budget"):
        padded_batches(range(4), lengths, 9)


def test_official_groups_kept_whole_even_when_they_need_several_forwards() -> None:
    groups = [0, 0, 0, 1, 1, 2]
    steps = plan_steps([4] * 6, groups, 8, seed=5, pool=False, target=2)
    micros = [m for s in steps for m in s]
    assert sorted(sorted(m) for m in micros) == [[0, 1, 2], [3, 4], [5]]
    assert [len(s) for s in steps] == [2, 1]


def test_pool_epoch_is_reproducible_and_covers_each_row_within_budget() -> None:
    lengths = [2, 3, 7, 2, 2, 4, 5, 3]
    groups = [1, 1, 1, 3, 5, 5, 6, 7]
    kwargs = dict(seed=31, pool=True, target=4)
    steps = plan_steps(lengths, groups, 9, **kwargs)
    assert steps == plan_steps(lengths, groups, 9, **kwargs)
    assert sorted(i for s in steps for m in s for i in m) == list(range(8))
    assert all(len(m) * max(lengths[i] for i in m) <= 9 for s in steps for m in s)
    with pytest.raises(ValueError):
        plan_steps(lengths[:-1], groups, 9, **kwargs)


def test_partial_or_tampered_checkpoint_never_restored(tmp_path: Path) -> None:
    step = tmp_path / "checkpoints" / "step-000005"
    step.mkdir(parents=True)
    (step / "checkpoint.json").write_text('{"global_step":5,"files":{}}')
    with pytest.raises(ValueError, match="valid checkpoint"):
        latest_checkpoint(tmp_path)


def test_restart_uses_newest_intact_checkpoint_and_never_accepts_unchecked_paths(
    tmp_path: Path,
) -> None:
    import hashlib
    import json

    files = [
        "optimizer.pt",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "heads.safetensors",
    ]
    for step_num in (10, 20):
        step = tmp_path / "checkpoints" / f"step-{step_num:06d}"
        (step / "adapter").mkdir(parents=True)
        for name in files:
            (step / name).write_bytes(b"state")
        (step / "checkpoint.json").write_text(
            json.dumps(
                {
                    "global_step": step_num,
                    "data_cursor": step_num,
                    "signature": "fixture",
                    "files": dict.fromkeys(files, hashlib.sha256(b"state").hexdigest()),
                }
            )
        )
    checkpoint = latest_checkpoint(tmp_path)
    assert checkpoint.name == "step-000020"
    (checkpoint / "optimizer.pt").write_bytes(b"truncated")
    assert latest_checkpoint(tmp_path).name == "step-000010"
    with pytest.raises(ValueError):
        latest_checkpoint(checkpoint)


def test_missing_pool_prompt_is_always_singleton():
    import pandas as pd

    from ntl_member.train import _groups

    groups = _groups(pd.DataFrame({"prompt_group": ["", None, " ", "p", "p", ""]}))
    assert len(set(groups[:3] + groups[5:])) == 4
    assert groups[3] == groups[4] and all(groups[i] < 0 for i in (0, 1, 2, 5))
