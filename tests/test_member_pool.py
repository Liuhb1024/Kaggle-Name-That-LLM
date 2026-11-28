import pandas as pd
import pytest

from ntl_member.pool import classify_row, data_gate, deduplicate, overlap_audit, sample_pool

LABEL = "qwen3.8-max"
CONFIG = {
    "filters": {
        "min_words": 2,
        "max_words": 20,
        "gemini_drift_cutoff": "2026-11-01",
        "kimi_reasoning": ["high"],
        "allowed_licenses": ["CC BY 4.0"],
    }
}
INFO = {"public": True, "license": "CC BY 4.0", "version": 1, "last_updated": "2026-10-01"}
ROW = dict(
    text="Fresh sample words",
    model=LABEL,
    source_dataset="owner/data",
    source_file="input.csv",
    gen_route="api",
    gen_served_model=LABEL,
    prompt_canon="p123",
    gen_date="2026-10-12",
)
PROTOCOL = {"allowed": True, "verified": False, "first_party": False}


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"model": "unknown"}, "F1"),
        ({"text": "one"}, "F2"),
        ({"text": "first paragraph\n\nsecond paragraph"}, "F3"),
        ({"text": "go https://example.com"}, "F3"),
        ({"gen_served_model": "gpt-5.5"}, "F4"),
        ({"model": "gemini-3.7-flash", "gen_served_model": "", "gen_date": "2026-11-08"}, "F5"),
        ({"model": "kimi-k3", "gen_served_model": "", "gen_reasoning": "off"}, "F6"),
    ],
)
def test_row_exclusions(changes, reason):
    assert classify_row({**ROW, **changes}, INFO, PROTOCOL, CONFIG)[1] == reason


def test_source_evidence_and_snapshot_tiers():
    row, reason = classify_row(ROW, INFO, PROTOCOL, CONFIG)
    assert not reason and row["tier"] == "A"
    snap = {**ROW, "gen_served_model": LABEL + "-0902"}
    assert classify_row(snap, INFO, PROTOCOL, CONFIG)[0]["tier"] == "B"
    assert classify_row(ROW, {**INFO, "public": False}, PROTOCOL, CONFIG)[1] == "F7"
    assert classify_row(ROW, INFO, {**PROTOCOL, "allowed": False}, CONFIG)[1] == "F7"


def test_dedup_and_sampling_are_permutation_invariant():
    base, _ = classify_row(ROW, INFO, PROTOCOL, CONFIG)
    other = {**base, "tier": "B", "source_dataset": "other/pub"}
    bad = {**base, "label": "glm-5.3"}
    empty, _, _ = deduplicate(pd.DataFrame([base, bad]))
    assert empty.empty
    clean, conflicts, dupes = deduplicate(pd.DataFrame([other, base]))
    assert conflicts == 0 and dupes == 1 and clean.iloc[0]["tier"] == "A"
    variants = pd.DataFrame(
        [
            {**base, "text_sha256": str(i), "tier": t}
            for i, t in enumerate(["A", "B", "C", "A", "B"])
        ]
    )
    sampled = sample_pool(variants, 3, 0)
    assert sorted(sampled.tier) == ["A", "A", "B"]
    assert list(sampled.text_sha256) == list(sample_pool(variants.iloc[::-1], 3, 0).text_sha256)


def test_overlap_audit_independent_of_official_labels():
    official = pd.DataFrame(
        {
            "text": [
                "airplane flying above ocean",
                "little desert plant",
                "theater stage lights",
                "actors perform tonight",
            ],
            "prompt_id": [0, 0, 1, 1],
            "model": ["x", "y", "x", "y"],
        }
    )
    pool = pd.DataFrame({"text": ["airplane flying above ocean", "separate vocabulary"]})
    hits, counts = overlap_audit(pool, official, 0.8, 0.50)
    assert hits.tolist() == [True, False] and counts["near_text"] == 1
    assert (
        overlap_audit(pool, official.drop(columns="model"), 0.8, 0.5)[0].tolist() == hits.tolist()
    )


def test_gate_counts_all_eighteen_and_warns_for_one_small_class():
    from ntl_baseline.constants import LABELS

    gates = {
        "data_min_rows_per_class": 10,
        "data_stop_rows_per_class": 5,
        "data_stop_class_count": 3,
    }
    counts = dict.fromkeys(LABELS, 10)
    counts[LABELS[0]] = 4
    assert data_gate(counts, gates)["status"] == "warning"
    assert data_gate({}, gates)["status"] == "stop"


def test_protocol_rewrite_and_provider_names_reject_unlabelled_imitation():
    from ntl_member.pool import protocol_evidence

    baseline = dict(
        source_dataset="owner/data",
        served_model_check="unknown",
        route="router",
        post_processing="none",
        notes="",
        instruction="write a paragraph",
    )
    for edit in ["rewritten by another model", "style transfer", "polished outputs"]:
        assert not protocol_evidence({**baseline, "post_processing": edit})["allowed"]
    spoof = {**ROW, "gen_served_model": "anthropic/qwen3.8-max"}
    assert classify_row(spoof, INFO, PROTOCOL, CONFIG)[1] == "F4"


def test_spot_check_missing_files_and_mismatch_count(tmp_path):
    from ntl_member.pool import spot_checks

    picked = pd.DataFrame(
        {
            "source_file": ["rows.csv"] * 3,
            "text": ["one  text", "other text", "missing"],
            "label": [LABEL] * 3,
        }
    )
    orig = tmp_path / "spot" / "owner" / "data" / "rows.csv"
    orig.parent.mkdir(parents=True)
    pd.DataFrame({"text": ["one text", "other text"], "label": [LABEL, "glm-5.3"]}).to_csv(
        orig, index=False
    )
    metadata = {"owner/data": INFO}
    result = spot_checks({"owner/data": picked}, tmp_path, metadata)
    assert result["owner/data"]["sampled"] == 3 and result["owner/data"]["matched"] == 1


def test_extra_and_pool_end_to_end_emit_only_aggregate_audit(tmp_path, monkeypatch):
    import json
    import os
    import shutil
    from hashlib import sha256
    from pathlib import Path

    from ntl_member import pool, sources
    from ntl_member.config import ROOT
    from ntl_member.report import validate_public

    dataset = tmp_path / "fixtures"
    dataset.mkdir()
    source = "owner/data"
    cfg = {
        "seed": 0,
        "filters": {
            **CONFIG["filters"],
            "near_dup_char_cosine": 0.80,
            "prompt_centroid_cosine": 0.50,
            "spot_check_rows": 200,
            "spot_check_max_mismatch": 0.02,
        },
        "pins": {
            "open_pool": sources.OPEN_POOL + "/versions/19",
            "extra_api": sources.EXTRA + "/versions/2",
        },
        "sampling": {"per_class_cap": 4},
        "gates": {
            "data_min_rows_per_class": 0,
            "data_stop_rows_per_class": 0,
            "data_stop_class_count": 3,
        },
    }
    df = pd.DataFrame(
        [
            {
                **dict.fromkeys(pool.POOL_COLUMNS, ""),
                **ROW,
                "source_dataset": source,
                "flag_kb_proxy": False,
                "flag_kb_proxy_after_ruling": False,
            }
        ]
    )
    excluded = "owner/unlicensed"
    df["model"] = "gemini-3.7-flash"
    df["gen_served_model"] = "gemini-3.7-flash"
    df["gen_date"] = ""
    df = pd.concat([df, df.assign(source_dataset=excluded, text="Another simple passage")])
    df.to_parquet(dataset / "ntl_open_pool.parquet", index=False)
    pd.DataFrame(
        [{"text": "Different little words", "model": "claude-fable-5", "prompt_id": "zero"}]
    ).to_csv(dataset / "ntl_extra_passages_claude.csv", index=False)
    proto = {
        "post_processing": "none",
        "instruction": "write",
        "notes": "",
        "route": "first_party_api",
        "served_model_check": "yes",
    }
    pd.DataFrame(
        [{**proto, "source_dataset": ref} for ref in [source, sources.EXTRA, excluded]]
    ).to_csv(dataset / "source_protocol.csv", index=False)
    df.to_csv(dataset / "input.csv", index=False)

    calls = []

    def download_file(ref: str, version: int, filename: str, output: Path):
        calls.append((ref, version, filename))
        output.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(dataset / filename, output)

    def info(ref):
        return {
            **INFO,
            "version": 2 if ref == sources.EXTRA else 1,
            "raw_license": "" if ref == excluded else INFO["license"],
            "license": "" if ref == excluded else INFO["license"],
        }

    monkeypatch.setattr(sources, "download_file", download_file)
    monkeypatch.setattr(sources, "public_metadata", info)
    monkeypatch.setattr(
        sources,
        "PINNED",
        {
            name: (
                ref,
                ver,
                (dataset / name).stat().st_size,
                sha256((dataset / name).read_bytes()).hexdigest(),
            )
            for name, (ref, ver, _, _) in sources.PINNED.items()
        },
    )
    inputs = tmp_path / "download"
    output = tmp_path / "prepare"
    sources.download(inputs, cfg)
    # The weight package references the exact bytes of the approved source lock.
    approved = tmp_path / "approved.json"
    approved.write_text(
        json.dumps(json.loads((inputs / "source_lock.json").read_text()), indent=3, sort_keys=True)
    )
    for name in ("source_metadata.json", "spot_plan.json"):
        shutil.copyfile(inputs / name, tmp_path / name)
    (tmp_path / "retained_sources.json").write_text(json.dumps([source, sources.EXTRA]))
    cfg["pins"]["source_lock"] = os.path.relpath(approved, ROOT)
    monkeypatch.chdir(dataset)
    monkeypatch.setattr(
        sources,
        "public_metadata",
        lambda ref: {
            **info(ref),
            "version": 99,
            "last_updated": "2030-01-01",
            "public": ref != excluded,
        },
    )
    calls.clear()
    replay = tmp_path / "replay"
    sources.download(replay, cfg)
    assert (replay / "source_lock.json").read_bytes() == approved.read_bytes()
    assert (source, 1, "input.csv") in calls and all(v != 99 for _, v, _ in calls)
    live = json.loads((replay / "source_status_live.json").read_bytes())
    assert live[source]["version"] == 99 and live[source]["last_updated"] == "2030-01-01"
    official = tmp_path / "official.csv"
    pd.DataFrame(
        {
            "text": ["alpha beta gamma", "delta epsilon zeta"],
            "prompt_id": [1, 2],
            "model": ["claude-fable-5", "gemini-3.7-flash"],
        }
    ).to_csv(official, index=False)
    before = pool.prepare(inputs, official, tmp_path / "before", {**cfg, "pins": {}})
    summary = pool.prepare(replay, official, output, cfg)
    assert (output / "source_lock.json").read_bytes() == approved.read_bytes()
    assert before["source_counts"] == summary["source_counts"]
    detail = json.loads((output / "audit/per_source.json").read_text())[excluded]
    assert summary["excluded_sources"][excluded] == {
        "raw_license": "",
        "f7_rows": detail["discarded"]["F7"],
        "protocol_reason": "none",
    }
    assert detail["discarded"]["F7"] == 1
    validate_public("audit_summary.json", summary)
    assert summary["source_counts"] == {source: 1, sources.EXTRA: 1}
    assert pd.read_parquet(output / "stage1_sample.parquet").shape[0] == 2
    assert {source, sources.EXTRA} <= set(
        json.loads((output / "audit/per_source.json").read_text())
    )
    assert "Different little words" not in json.dumps(summary)
    assert "CC BY 4.0" in (output / "audit/NOTICE.md").read_text()


def test_reworded_format_instruction_is_distinct_from_rewritten_passage():
    from ntl_member.pool import protocol_evidence

    fields = dict(
        source_dataset="owner/data",
        route="router",
        served_model_check="unknown",
        post_processing="none",
        instruction='paraphrase ("Write in about N words")',
        notes="",
    )
    assert protocol_evidence(fields)["allowed"]
    assert not protocol_evidence({**fields, "instruction": "Rewrite that passage more concisely"})[
        "allowed"
    ]


def test_zero_eligible_sources_still_yield_auditable_empty_frame(tmp_path):
    from ntl_member.pool import POOL_COLUMNS, classify_inputs, spot_plan
    from ntl_member.sources import EXTRA

    pd.DataFrame([{**dict.fromkeys(POOL_COLUMNS, ""), **ROW}]).to_parquet(
        tmp_path / "ntl_open_pool.parquet", index=False
    )
    pd.DataFrame([{"text": "extra unlicensed words", "model": LABEL, "prompt_id": "1"}]).to_csv(
        tmp_path / "ntl_extra_passages_claude.csv", index=False
    )
    pd.DataFrame(
        [
            dict(
                source_dataset=r,
                post_processing="none",
                notes="",
                route="router",
                served_model_check="unknown",
                instruction="write",
            )
            for r in [ROW["source_dataset"], EXTRA]
        ]
    ).to_csv(tmp_path / "source_protocol.csv", index=False)
    frame, counts, total = classify_inputs(tmp_path, {}, CONFIG)
    assert total == 2 and frame.empty and sum(x["F7"] for x in counts.values()) == 2
    assert deduplicate(frame)[0].empty and not spot_plan(frame, 200, 0)


def test_prompt_centroid_removal_and_per_source_audit_without_label_use():
    from collections import Counter, defaultdict

    official = pd.DataFrame(
        {
            "prompt_id": [0, 0, 1, 1],
            "text": ["flute piano trumpet", "flute oboe lyre", "hiking river", "hiking mountain"],
        }
    )
    samples = pd.DataFrame(
        {
            "text": ["flute piano lyre oboe trumpet", "extraneous symbols"],
            "source_dataset": ["test/source"] * 2,
        }
    )
    counts = defaultdict(Counter)
    hit, details = overlap_audit(samples, official, 0.99, 0.6, per_source=counts)
    assert list(hit) == [True, False]
    assert details == {"near_text": 0, "near_prompt": 1}
    assert counts["test/source"]["F10"] == 1


def test_two_percent_source_mismatch_is_the_accepted_boundary(tmp_path):
    from ntl_member.pool import spot_checks

    sample = pd.DataFrame(
        {"text": [f"row {i}" for i in range(200)], "label": LABEL, "source_file": "examples.csv"}
    )
    source = tmp_path / "spot" / "owner" / "data"
    source.mkdir(parents=True)
    sample.iloc[:196].to_csv(source / "examples.csv", index=False)
    result = spot_checks({"owner/data": sample}, tmp_path, {"owner/data": INFO})
    assert result["owner/data"]["mismatch_rate"] == 0.02
