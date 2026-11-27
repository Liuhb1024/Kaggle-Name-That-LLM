from hashlib import sha256
from pathlib import Path

import pytest

from ntl_baseline.constants import ContractError
from ntl_member.sources import check_lock, normalize_license, public_metadata, safe_filename


def test_public_metadata_uses_anonymous_fixed_endpoint():
    requested = []

    def request(url):
        requested.append(url)
        return {
            "ref": "owner/data",
            "isPrivate": False,
            "currentVersionNumber": 3,
            "licenseName": "Attribution 4.0 International (CC BY 4.0)",
            "lastUpdated": "2026-10-21T00:00:00Z",
        }

    info = public_metadata("owner/data", request)
    assert requested == ["https://www.kaggle.com/api/v1/datasets/view/owner/data"]
    assert info["public"] and info["license"] == "CC BY 4.0" and info["version"] == 3
    with pytest.raises(ContractError):
        public_metadata("../../secrets", request)
    with pytest.raises(ContractError):
        public_metadata("owner/data", lambda _: {"ref": "owner/data", "licenseName": "CC0 1.0"})


@pytest.mark.parametrize("name", ["../other", "/etc/passwd", "dir/../../private", "x\\foo"])
def test_source_filename_confined(name):
    with pytest.raises(ContractError):
        safe_filename(name)


def test_lock_changed_hash_rejected(tmp_path: Path):
    (tmp_path / "input.csv").write_bytes(b"abc")

    lock = {"files": {"input.csv": {"sha256": sha256(b"abc").hexdigest(), "bytes": 3}}}
    check_lock(tmp_path, lock)
    (tmp_path / "input.csv").write_bytes(b"abd")
    with pytest.raises(ContractError):
        check_lock(tmp_path, lock)
    assert normalize_license("Other (specified in description)") == ""
    assert normalize_license("CC BY-NC 4.0") == ""
