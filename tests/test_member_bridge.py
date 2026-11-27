import ast
import hashlib
import json

import pytest

from ntl_member import v31_bridge as bridge


def fixture():
    sources = [
        "# unchanged",
        "AQ8V5 = [adapter_dir('x')]",
        "%%writefile comp_stack.py\n...",
        "def run(cmd, env=None):\n    pass",
        'MODELS = [\n    {"name": "q8v5", "adapters": AQ8V5},\n]\n'
        "if FAST_COMMIT:\n    pass\nelse:\n    code = run(cmd, env=None)",
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            dict(cell_type="code", source=[s], metadata={}, outputs=["x"], execution_count=1)
            for s in sources
        ],
    }


@pytest.mark.parametrize("candidate", ["C1", "C2"])
def test_patch_locates_source_and_keeps_original_cells(candidate):
    original = fixture()
    snapshot = json.dumps(original)
    result, evidence = bridge.patch(original, candidate, "owner/member", 4, "c" * 64)
    assert json.dumps(original) == snapshot
    sources = ["".join(c["source"]) for c in result["cells"]]
    assert [sources[i] for i in (0, 1, 3, 5)] == [
        "".join(original["cells"][i]["source"]) for i in (0, 1, 2, 3)
    ]
    assert "MANIFEST.json" in sources[2] and "c" * 64 in sources[2]
    assert "expected_logits.json" in sources[4] and "def run_logged(" in sources[4]
    assert ('"name": "q8v5"' in sources[6]) == (candidate == "C2")
    assert '"name": "q8pool"' in sources[6] and "run_logged(cmd" in sources[6]
    for i in (2, 4, 6):
        ast.parse(sources[i])
    assert all(not c["outputs"] and c["execution_count"] is None for c in result["cells"])
    assert hashlib.sha256(sources[0].encode()).hexdigest() in evidence


@pytest.mark.parametrize("duplicate", [True, False])
def test_rejects_ambiguous_or_absent_markers(duplicate):
    payload = fixture()
    if duplicate:
        payload["cells"].append(payload["cells"][1])
    else:
        del payload["cells"][1]
    with pytest.raises(ValueError, match="marker"):
        bridge.patch(payload, "C1", "owner/member", 1, "a" * 64)


def test_rejects_unpinned_source_before_extraction(tmp_path):
    source = tmp_path / "source"
    source.write_text(json.dumps(fixture()))
    with pytest.raises(ValueError, match="SHA-256"):
        bridge.extract_scripts(source, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_logged_runtime_rejects_missing_repeated_reordered_and_failed_members(tmp_path):
    adapters = [tmp_path / name for name in bridge.MEMBERS]
    log = [f"[0s] l24 q8pool: adapter {p.name} done (bf16 redo batches 0)" for p in adapters]
    bridge.check_members(log, adapters)
    for malformed in [log[:-1], log + log[:1], log[::-1], log + ["MEMBER FAILED q8pool: OOM"]]:
        with pytest.raises(ValueError, match="adapters"):
            bridge.check_members(malformed, adapters)


def test_asset_cell_executes_hash_guard_with_a_mounted_fixture(tmp_path):
    from ntl_baseline.artifacts import directory_hashes, write_json_new
    from ntl_baseline.data import file_sha256

    dataset = tmp_path / "member"
    dataset.mkdir()
    cfg = dict(keep=24, layers=bridge.LAYERS, max_len=384, eos_id=None, no_final_norm=False)
    for member in bridge.MEMBERS:
        root = dataset / member
        (root / "adapter").mkdir(parents=True)
        for name in (
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "heads.safetensors",
        ):
            (root / name).write_bytes(b"fixture")
        write_json_new(root / "meta.json", {"cfg": cfg})
    (dataset / "anchors").mkdir()
    for path in ("NOTICE.md", "anchors/test.csv", "anchors/expected_logits.json"):
        (dataset / path).write_text("fixture")
    manifest = dataset / "MANIFEST.json"
    write_json_new(
        manifest,
        {
            name: {"bytes": (dataset / name).stat().st_size, "sha256": digest}
            for name, digest in directory_hashes(dataset).items()
        },
    )
    digest = file_sha256(manifest)
    payload, _ = bridge.patch(fixture(), "C1", "owner/member", 1, digest)
    source = "".join(payload["cells"][2]["source"]).replace("/kaggle/input", str(tmp_path))
    env = {}
    exec(source, env)
    assert env["Q8POOL_CFG"] == cfg
    assert len(env["AQ8POOL"]) == 6
    (dataset / "q8pool" / "heads.pt").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="file set"):
        exec(source, {})
    (dataset / "q8pool" / "heads.pt").unlink()
    (dataset / "q8pool" / "heads.safetensors").write_bytes(b"fixturF")
    with pytest.raises(ValueError, match="payload"):
        exec(source, {})


def test_rejects_invalid_server_manifest_before_deployment(tmp_path):
    bad = tmp_path / "package_manifest.json"
    bad.write_text('{"../outside":{"bytes":3,"sha256":"deadbeef"}}')
    with pytest.raises(ValueError, match="file set"):
        bridge.read_manifest(bad)
