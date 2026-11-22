import pandas as pd
import pytest

from ntl_baseline.constants import ContractError
from ntl_baseline.data import (
    load_competition_data,
    validate_submission,
    validate_test,
    validate_train,
)


@pytest.mark.parametrize("column", ["id", "prompt_id", "text", "model"])
def test_train_requires_each_column(train, column):
    with pytest.raises(ContractError, match="column"):
        validate_train(train.drop(columns=column))


@pytest.mark.parametrize("kind", ["duplicate", "blank", "null"])
def test_invalid_ids_are_rejected(train, kind):
    if kind == "duplicate":
        train.loc[1, "id"] = train.loc[0, "id"]
    elif kind == "blank":
        train.loc[1, "id"] = " "
    else:
        train.loc[1, "id"] = None
    with pytest.raises(ContractError, match="id"):
        validate_train(train)


def test_unknown_label_and_incomplete_candidate_set_rejected(train):
    invalid = train.copy()
    invalid.loc[0, "model"] = "invented-model"
    with pytest.raises(ContractError, match="label"):
        validate_train(invalid)
    with pytest.raises(ContractError, match="18"):
        validate_train(train.iloc[:3])


def test_null_text_matches_official_fillna_policy_without_mutating_input(train, test_frame):
    train.loc[0, "text"] = None
    test_frame.loc[0, "text"] = None
    cleaned = validate_train(train)
    assert cleaned.loc[0, "text"] == ""
    assert validate_test(test_frame).loc[0, "text"] == ""
    assert train.loc[0, "text"] is None


def test_non_string_text_is_rejected(test_frame):
    test_frame.at[0, "text"] = {"untrusted": "object"}
    with pytest.raises(ContractError, match="text"):
        validate_test(test_frame)


def test_load_real_schema_preserves_id_order_and_blank_text(data_dir, test_frame):
    train, test = load_competition_data(data_dir)
    assert len(train) == 90
    assert test["id"].tolist() == test_frame["id"].tolist()
    assert test["text"].iloc[-1] == ""


@pytest.mark.parametrize("kind", ["missing", "duplicate", "extra", "invalid", "null", "index"])
def test_rejects_malformed_submission(test_frame, kind):
    from ntl_baseline.constants import LABELS

    frame = pd.DataFrame({"id": test_frame["id"], "model": LABELS[0]})
    if kind == "missing":
        frame = frame.iloc[:-1]
    elif kind == "duplicate":
        frame.loc[1, "id"] = frame.loc[0, "id"]
    elif kind == "extra":
        frame.loc[len(frame)] = ["extra-id", LABELS[0]]
    elif kind == "invalid":
        frame.loc[0, "model"] = "not-a-label"
    elif kind == "null":
        frame.loc[0, "model"] = None
    elif kind == "index":
        frame["index"] = frame.index
    with pytest.raises(ContractError):
        validate_submission(frame, test_frame)


def test_submission_is_aligned_by_id_not_position(test_frame):
    from ntl_baseline.constants import LABELS

    frame = pd.DataFrame({"id": test_frame["id"], "model": LABELS[:3]}).iloc[::-1]
    actual = validate_submission(frame, test_frame)
    assert actual["id"].tolist() == test_frame["id"].tolist()
    assert actual["model"].tolist() == list(LABELS[:3])
