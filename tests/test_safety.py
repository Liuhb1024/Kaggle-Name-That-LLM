import json
import stat
import zipfile

import pandas as pd
import pytest

from ntl_baseline.config import BaselineConfig, load_config
from ntl_baseline.constants import ContractError
from ntl_baseline.data import import_archive, load_test, read_csv_input


def test_csv_id_strings_keep_leading_zeros(tmp_path):
    (tmp_path / "test.csv").write_text("id,text\n0001,A paragraph.\n0002,Another paragraph.\n")
    assert load_test(tmp_path).id.tolist() == ["0001", "0002"]


def test_duplicate_csv_headers_are_rejected(tmp_path):
    (tmp_path / "test.csv").write_text("id,text,text\nx,a,b\n")
    with pytest.raises(ContractError, match="column"):
        read_csv_input(tmp_path, "test.csv")


@pytest.mark.parametrize("kwargs", [{"folds": 1}, {"c": float("nan")}, {"threads": 0}])
def test_invalid_config_fails_before_computation(kwargs):
    with pytest.raises(ContractError):
        BaselineConfig(**kwargs)


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"unreviewed_tuning_option": True}))
    with pytest.raises(ContractError, match="config"):
        load_config(path)


def test_archive_symlink_rejected(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        info = zipfile.ZipInfo("train.csv")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(info, "../../secret")
    with pytest.raises(ContractError, match="archive"):
        import_archive(archive, tmp_path / "imported")
    assert not (tmp_path / "imported").exists()


def test_inference_empty_text_and_numeric_csv_text_are_strings(tmp_path):
    pd.DataFrame({"id": ["0001", "0002"], "text": [123, None]}).to_csv(
        tmp_path / "test.csv", index=False
    )
    assert load_test(tmp_path).text.tolist() == ["123.0", ""]
