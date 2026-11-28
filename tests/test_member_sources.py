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


def test_transient_404_retries_before_marking_a_source_private(monkeypatch):
    import io
    from urllib.error import HTTPError

    from ntl_member import sources

    url = "https://www.kaggle.com/api/v1/datasets/view/owner/data"
    calls = []

    def open_response(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(url, 404, "missing", {}, None)
        stream = io.BytesIO(b'{"ref":"owner/data","isPrivate":false}')
        stream.geturl = lambda: url
        return stream

    monkeypatch.setattr(sources, "urlopen", open_response)
    monkeypatch.setattr(sources.time, "sleep", lambda _: None)
    assert sources._request(url)["isPrivate"] is False
    assert calls == [url, url]


@pytest.mark.parametrize("invalid", [{"public": False}, {"license": ""}])
def test_contributing_source_losing_access_or_license_fails_by_name(tmp_path, monkeypatch, invalid):
    from ntl_member import sources
    from ntl_member.config import load

    cfg = load()
    ref = "tadeuhayashida/ntl-gen003-claude-panel"
    good = {"public": True, "license": "CC BY 4.0", "version": 99, "last_updated": "2030-01-01"}
    monkeypatch.setattr(
        sources, "public_metadata", lambda name: good | (invalid if name == ref else {})
    )

    def unexpected_download(*args):
        pytest.fail("file transfer before compliance verdict")

    monkeypatch.setattr(sources, "download_file", unexpected_download)
    with pytest.raises(ContractError, match=ref):
        sources.download(tmp_path, cfg)


def test_pinned_snapshot_hash_checked_before_online_requests(tmp_path, monkeypatch):
    import shutil

    from ntl_member import sources
    from ntl_member.config import load, source_lock_path

    cfg = load()
    approved = source_lock_path(cfg)
    for original in (
        approved,
        approved.with_name("source_metadata.json"),
        approved.with_name("spot_plan.json"),
    ):
        shutil.copyfile(original, tmp_path / original.name)
    with (tmp_path / "spot_plan.json").open("ab") as stream:
        stream.write(b" ")
    cfg["pins"]["source_lock"] = str(tmp_path / approved.name)
    monkeypatch.setattr(
        sources, "public_metadata", lambda _: pytest.fail("unexpected network request")
    )
    with pytest.raises(ContractError, match="spot_plan.json"):
        sources.download(tmp_path / "out", cfg)
