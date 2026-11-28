"""Normalize external observations and audit source fidelity before any model training."""

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ntl_baseline.artifacts import write_json_new, write_parquet_new
from ntl_baseline.constants import LABELS, ContractError

from .config import source_lock_path
from .sources import EXTRA, OPEN_POOL, check_lock, retained_sources

FORMAT = re.compile(
    r"(?:\n\s*\n)|(?:^|\n)\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)|```|https?://|"
    r'(?:^\s*[\[{]|"[^"\n]+"\s*:)',
    re.I,
)
EDITS = re.compile(
    r"\brewrit(?:e|es|ing|ten)\b|\brewrot[e]\b|\bpolish(?:ed|ing)?\b|"
    r"\bparaphras\w*|style[- ]transfer|imitation|non[- ]target model|"
    r"(?:dash|punctuation) replaced by",
    re.I,
)
SNAPSHOT = re.compile(r"-(?:\d{4}(?:-\d{2}-\d{2})?|\d{8})$")
PROVIDERS = {
    "claude": "anthropic",
    "gpt": "openai",
    "gemini": "google",
    "kimi": "moonshotai",
    "glm": "z-ai",
    "qwen": "qwen",
}
GROUP_UNKNOWN = {"unknown", "none", "nan", "null", ""}
POOL_COLUMNS = [
    "text",
    "model",
    "source_dataset",
    "source_file",
    "gen_route",
    "gen_served_model",
    "gen_date",
    "gen_reasoning",
    "prompt_canon",
    "prompt_group",
    "flag_kb_proxy",
    "flag_kb_proxy_after_ruling",
]
MARKERS = {"em_dash": "—", "spaced_em_dash": " — ", "curly_apostrophe": "’", "curly_double": "[“”]"}


def family(label: str) -> str:
    return next((name for name in PROVIDERS if label.startswith(name)), "")


def protocol_evidence(row: dict[str, Any]) -> dict[str, bool]:
    check = str(row["served_model_check"]).lower()
    # In v19, "paraphrase (...)" in this column refers to the author's prompt
    # template. A quoted instruction to rewrite an answer still counts as an edit.
    instruction = str(row["instruction"]).removeprefix("paraphrase (")
    evidence = " ".join([str(row["post_processing"]), str(row["notes"]), instruction])
    first_party = str(row["route"]).lower().startswith("first_party_api")
    verified = "all match" in check or "every row" in check or "0 mismatch" in check
    if row["source_dataset"] == EXTRA and check == "yes" and first_party:
        # Pinned v2 publisher documentation states 0 mismatches for all responses.
        verified = True
    return {
        "allowed": not bool(EDITS.search(evidence)),
        "first_party": first_party,
        "verified": verified,
        "checked": verified or check == "yes" or check.startswith("row ("),
    }


def classify_row(
    row: dict[str, Any], info: dict[str, Any], protocol: dict[str, bool], config: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    filters = config["filters"]
    text, label = row["text"], row["model"]
    if label not in LABELS:
        return {}, "F1"
    count = len(text.split()) if isinstance(text, str) else 0
    if not filters["min_words"] <= count <= filters["max_words"]:
        return {}, "F2"
    if FORMAT.search(text):
        return {}, "F3"
    served = str(row.get("gen_served_model", "")).strip()
    route = str(row.get("gen_route", "")).strip()
    route = "" if route.lower() in GROUP_UNKNOWN else route
    served = "" if served.lower() in GROUP_UNKNOWN else served
    lab_family = family(label)
    provider = PROVIDERS[lab_family]
    # A provider namespace and a generation snapshot are recorded as variants.
    namespace_free = served.removeprefix(provider + "/").removesuffix("@default")
    variant = served != label and (
        namespace_free == label
        or (
            namespace_free.startswith(label + "-")
            and SNAPSHOT.fullmatch(namespace_free[len(label) :])
        )
    )
    if served and served != label and not variant:
        return {}, "F4"
    raw_date = str(row.get("gen_date", "")).strip()
    row_date = raw_date[:10] if re.fullmatch(r"\d{4}-\d\d-\d\d(?:[T ].*)?", raw_date) else ""
    date = row_date or str(info.get("last_updated", ""))[:10]
    if lab_family == "gemini" and (not date or date >= filters["gemini_drift_cutoff"]):
        return {}, "F5"
    reasoning = str(row.get("gen_reasoning", "")).strip().lower()
    if lab_family == "kimi" and reasoning not in filters["kimi_reasoning"]:
        return {}, "F6"
    if (
        not info.get("public")
        or info.get("license") not in filters["allowed_licenses"]
        or not protocol.get("allowed")
    ):
        return {}, "F7"
    tier = "C"
    if route or protocol.get("checked") or protocol.get("verified") or variant:
        tier = "B"
    if not variant and (
        (served == label and route) or (protocol["first_party"] and protocol["verified"])
    ):
        tier = "A"
    digest = hashlib.sha256(
        " ".join(unicodedata.normalize("NFKC", text).lower().split()).encode("utf-8")
    ).hexdigest()
    source = row["source_dataset"]
    canonical = str(row.get("prompt_canon", "")).strip()
    local_group = str(row.get("prompt_group", "")).strip()
    group = (
        canonical
        if canonical.lower() not in GROUP_UNKNOWN
        else (source + "::" + local_group if local_group.lower() not in GROUP_UNKNOWN else "")
    )
    return {
        "row_key": digest,
        "text_sha256": digest,
        "text": text,
        "label": label,
        "family": lab_family,
        "word_count": count,
        "source_dataset": source,
        "source_file": row["source_file"],
        "source_version": info["version"],
        "source_license": info["license"],
        "license_basis": "public_metadata",
        "prompt_group": group,
        "tier": tier,
        "gen_route": route,
        "gen_served_model": served,
        "served_model_variant": bool(variant),
        "gen_reasoning": reasoning,
        "gen_date": date,
        "date_basis": "row" if row_date else "upper_bound",
        "flag_kb_proxy": bool(row.get("flag_kb_proxy", False)),
        "flag_kb_proxy_after_ruling": bool(row.get("flag_kb_proxy_after_ruling", False)),
    }, ""


def classify_inputs(
    root: Path, metadata: dict[str, Any], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Counter[str]], int]:
    protocols = pd.read_csv(root / "source_protocol.csv", keep_default_na=False)
    if protocols.source_dataset.duplicated().any():
        raise ContractError("Ambiguous source protocol")
    audit = {
        item["source_dataset"]: protocol_evidence(item) for item in protocols.to_dict("records")
    }
    open_rows = pd.read_parquet(root / "ntl_open_pool.parquet", columns=POOL_COLUMNS).fillna("")
    extra = pd.read_csv(
        root / "ntl_extra_passages_claude.csv",
        usecols=["text", "model", "prompt_id"],
        keep_default_na=False,
        dtype=str,
    )
    extra = extra.rename(columns={"prompt_id": "prompt_group"})
    extra["source_dataset"] = EXTRA
    extra["source_file"] = "ntl_extra_passages_claude.csv"
    extra["gen_route"] = "first_party_api"
    total = len(open_rows) + len(extra)
    retained = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for frame in (open_rows, extra):
        for r in frame.itertuples(index=False):
            fields = r._asdict()
            ref = fields["source_dataset"]
            result, reason = classify_row(fields, metadata.get(ref, {}), audit.get(ref, {}), config)
            if reason:
                counts[ref][reason] += 1
            else:
                retained.append(result)
    frame = pd.DataFrame(retained)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "label",
                "text",
                "text_sha256",
                "tier",
                "source_dataset",
                "source_file",
                "prompt_group",
            ]
        )
    return frame, counts, total


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    conflicting = frame.groupby("text_sha256").label.transform("nunique").gt(1)
    valid = frame.loc[~conflicting].sort_values(
        ["text_sha256", "tier", "source_dataset", "source_file", "text", "prompt_group"]
    )
    unique = valid.drop_duplicates("text_sha256").reset_index(drop=True)
    return unique, int(conflicting.sum()), len(valid) - len(unique)


def sample_pool(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    ordered = frame.sort_values(["label", "text_sha256"]).sample(frac=1, random_state=seed)
    return (
        ordered.sort_values(["label", "tier"], kind="stable")
        .groupby("label")
        .head(cap)
        .sort_values("text_sha256")
        .reset_index(drop=True)
    )


def spot_plan(frame: pd.DataFrame, n: int, seed: int) -> dict[str, pd.DataFrame]:
    counts = frame.groupby("source_dataset").size().sort_values(ascending=False, kind="stable")
    tier_a = set(frame.loc[frame.tier == "A", "source_dataset"])
    sources = list(
        dict.fromkeys([*counts.index[:5], *[ref for ref in counts.index if ref in tier_a]])
    )
    result = {}
    for ref in sources[:10]:
        group = frame.loc[frame.source_dataset == ref].sort_values(["text_sha256", "source_file"])
        result[ref] = group.sample(min(n, len(group)), random_state=seed)
    return result


def _raw_rows(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def spot_checks(
    plan: dict[str, pd.DataFrame], root: Path, metadata: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    out = {}
    for ref, sample in plan.items():
        hits = 0
        for filename, rows in sample.groupby("source_file"):
            path = root / "spot" / ref / filename
            if (
                ref == EXTRA
                and filename == "ntl_extra_passages_claude.csv"
                and metadata[ref]["version"] == 2
            ):
                path = root / filename
            if not path.is_relative_to(root) or not path.resolve().is_relative_to(root.resolve()):
                raise ContractError("Unconfined spot-check path")
            if not path.is_file():
                continue
            raw = _raw_rows(path)
            txt = next(
                (
                    c
                    for c in (
                        "text",
                        "passage",
                        "response_text",
                        "output_text",
                        "response",
                        "completion",
                    )
                    if c in raw
                ),
                None,
            )
            label = next(
                (
                    c
                    for c in (
                        "model",
                        "label",
                        "model_label",
                        "target_model",
                        "model_name",
                        "model_id",
                    )
                    if c in raw
                ),
                None,
            )
            if not txt or not label:
                continue
            lookup = {
                (" ".join(supplied_text.split()), supplied_label)
                for supplied_text, supplied_label in zip(raw[txt], raw[label], strict=True)
                if isinstance(supplied_text, str) and isinstance(supplied_label, str)
            }
            hits += sum((" ".join(r.text.split()), r.label) in lookup for r in rows.itertuples())
        out[ref] = {
            "sampled": len(sample),
            "matched": hits,
            "mismatch_rate": (len(sample) - hits) / len(sample),
        }
    return out


def overlap_audit(
    pool: pd.DataFrame,
    official: pd.DataFrame,
    near_cut: float,
    prompt_cut: float,
    *,
    per_source: dict[str, Counter[str]] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), dtype=np.float32)
    word = TfidfVectorizer(stop_words="english", dtype=np.float32)
    train_char = char.fit_transform(official.text).T.tocsr()
    train_word = word.fit_transform(official.text)
    groups, indices = np.unique(official.prompt_id, return_inverse=True)
    # L2 normalization of the mean is identical to L2 normalization of its sum.
    membership = csr_matrix(
        (np.ones(len(indices), np.float32), (indices, np.arange(len(indices)))),
        shape=(len(groups), len(indices)),
    )
    centroids = normalize(membership @ train_word).T.tocsr()
    hits = np.zeros(len(pool), dtype=bool)
    counts = {"near_text": 0, "near_prompt": 0}
    for i in range(0, len(pool), 256):
        docs = pool.text.iloc[i : i + 256]
        near = (char.transform(docs) @ train_char).max(axis=1).toarray().ravel() >= near_cut
        prompt = (word.transform(docs) @ centroids).max(axis=1).toarray().ravel() >= prompt_cut
        if per_source is not None:
            providers = pool.source_dataset.iloc[i : i + len(docs)]
            for rule, flagged in (("F9", near), ("F10", prompt & ~near)):
                for name, size in providers.loc[flagged].value_counts().items():
                    per_source[name][rule] += int(size)
        counts["near_text"] += int(near.sum())
        counts["near_prompt"] += int((prompt & ~near).sum())
        hits[i : i + len(docs)] = near | prompt
        done = i + len(docs)
        if done % 8192 == 0 or done == len(pool):
            print(f"External overlap audit: {done}/{len(pool)} passages", flush=True)
    return hits, counts


def data_gate(counts: dict[str, int], gates: dict[str, Any]) -> dict[str, Any]:
    gaps = {
        label: int(counts.get(label, 0))
        for label in LABELS
        if counts.get(label, 0) < gates["data_min_rows_per_class"]
    }
    critical = sum(x < gates["data_stop_rows_per_class"] for x in gaps.values())
    status = "stop" if critical >= gates["data_stop_class_count"] else "warning" if gaps else "pass"
    return {"status": status, "gaps": gaps, "critical_classes": critical}


def _stats(group: pd.DataFrame) -> dict[str, Any]:
    dates = group.gen_date.map(
        lambda d: "unknown"
        if not d
        else "October_or_earlier"
        if d < "2026-11-01"
        else "November_or_later"
    )
    return {
        "rows": len(group),
        "tier": group.tier.value_counts().to_dict(),
        "route": group.gen_route.value_counts().to_dict(),
        "date": dates.value_counts().to_dict(),
        "reasoning": group.gen_reasoning.value_counts().to_dict(),
        "kb_proxy": int(group.flag_kb_proxy.sum()),
        "kb_proxy_after_ruling": int(group.flag_kb_proxy_after_ruling.sum()),
        "words_quartiles": group.word_count.quantile([0, 0.25, 0.5, 0.75, 1]).tolist(),
        "typography_rate": {
            k: float(group.text.str.contains(v).mean()) for k, v in MARKERS.items()
        },
    }


def prepare(
    download_dir: Path, official_train: Path, output_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    lock_bytes = (download_dir / "source_lock.json").read_bytes()
    pinned = config["pins"].get("source_lock")
    if pinned and lock_bytes != source_lock_path(config).read_bytes():
        raise ContractError("Downloaded source lock differs from approved bytes")
    lock = json.loads(lock_bytes)
    check_lock(download_dir, lock)
    metadata = json.loads((download_dir / "source_metadata.json").read_text())
    frame, counts, total = classify_inputs(download_dir, metadata, config)
    filters = config["filters"]
    plan = spot_plan(frame, filters["spot_check_rows"], config["seed"])
    if list(plan) != json.loads((download_dir / "spot_plan.json").read_text())["sources"]:
        raise ContractError("Spot audit plan differs from acquisition")
    checks = spot_checks(plan, download_dir, metadata)
    clean, conflicts, duplicates = deduplicate(frame)
    per_ref = frame.source_dataset.value_counts().subtract(
        clean.source_dataset.value_counts(), fill_value=0
    )
    conflicting = frame.groupby("text_sha256").label.transform("nunique").gt(1)
    conflict_counts = frame.loc[conflicting, "source_dataset"].value_counts()
    for ref, number in per_ref.items():
        counts[ref]["F8"] = int(conflict_counts.get(ref, 0))
        counts[ref]["duplicates"] = int(number) - counts[ref]["F8"]
    official = pd.read_csv(
        official_train, keep_default_na=False, usecols=["text", "prompt_id", "model"]
    )
    hits, overlap = overlap_audit(
        clean,
        official,
        filters["near_dup_char_cosine"],
        filters["prompt_centroid_cosine"],
        per_source=counts,
    )
    clean = clean.loc[~hits]
    bad = {
        ref for ref, a in checks.items() if a["mismatch_rate"] > filters["spot_check_max_mismatch"]
    }
    bad_rows = clean.source_dataset.isin(bad)
    for ref, count in clean.loc[bad_rows, "source_dataset"].value_counts().items():
        counts[ref]["F12"] = int(count)
    clean = clean.loc[~bad_rows].reset_index(drop=True)
    per_class = clean.label.value_counts().to_dict()
    gate = data_gate(per_class, config["gates"])
    audit = output_dir / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    write_json_new(
        audit / "per_class.json",
        {
            label: {
                **_stats(rows),
                "official_typography_rate": {
                    k: float(official.loc[official.model == label, "text"].str.contains(v).mean())
                    for k, v in MARKERS.items()
                },
            }
            for label, rows in clean.groupby("label")
        },
    )
    protocols = pd.read_csv(download_dir / "source_protocol.csv", keep_default_na=False)
    details = {
        rec["source_dataset"]: {
            **metadata[rec["source_dataset"]],
            "discarded": dict(counts[rec["source_dataset"]]),
            "spot_check": checks.get(rec["source_dataset"]),
            "protocol": protocol_evidence(rec),
        }
        for rec in protocols.to_dict("records")
    }
    write_json_new(audit / "per_source.json", details)
    removed = sum(sum(n.values()) for n in counts.values())
    filtered = removed - conflicts - duplicates - int(hits.sum())
    if len(clean) + removed != total:
        raise ContractError("Pool row-level audit counts do not reconcile")
    write_json_new(
        audit / "filters.json",
        {
            "input": total,
            "row_filters": filtered,
            "F8_conflicting_rows": conflicts,
            "duplicates": duplicates,
            "retained": len(clean),
            "F9_F10": int(hits.sum()),
            "F11": "audit_only",
        },
    )
    write_json_new(
        audit / "leakage.json",
        {
            **overlap,
            "retained_overlap": 0,
            "reference": "all official texts and prompt groups, labels excluded",
            "upstream_caveat": "Public pool already filtered using official data",
        },
    )
    write_json_new(audit / "gaps.json", gate)
    source_counts = clean.source_dataset.value_counts().to_dict()
    if pinned and set(source_counts) != retained_sources(source_lock_path(config)):
        raise ContractError("Actual contributing sources differ from reviewed compliance coverage")
    summary = {
        "per_class": {k: int(per_class.get(k, 0)) for k in LABELS},
        "tiers": clean.tier.value_counts().to_dict(),
        "source_counts": source_counts,
        "excluded_sources": {
            ref: {
                "raw_license": detail.get("raw_license", ""),
                "f7_rows": detail["discarded"].get("F7", 0),
                "protocol_reason": "none"
                if detail["protocol"]["allowed"]
                else "rewrite_or_style_transfer",
            }
            for ref, detail in details.items()
            if ref not in source_counts
        },
        "gaps": gate["gaps"],
    }
    write_json_new(output_dir / "audit_summary.json", summary)
    with (output_dir / "source_lock.json").open("xb") as stream:
        stream.write(lock_bytes)
    with (audit / "NOTICE.md").open("x") as notice:
        notice.write("# External data attribution\n\n")
        for ref in sorted(set(clean.source_dataset) | {OPEN_POOL, EXTRA}):
            version = lock["versions"].get(ref, metadata[ref]["version"])
            notice.write(
                f"- {ref} ({metadata[ref]['license']}). "
                f"[Dataset version {version}](https://www.kaggle.com/datasets/"
                f"{ref}/versions/{version})\n"
            )
        notice.write(
            "\nFor pooled rows the upstream version identifies the original source "
            "at audit time. Open Pool v19 pins the redistributed contents.\n"
        )
    if gate["status"] == "stop":
        raise ContractError("Public training pool failed per-class coverage; see audit/gaps.json")
    sample = sample_pool(clean, config["sampling"]["per_class_cap"], config["seed"])
    write_parquet_new(output_dir / "pool_rows.parquet", clean)
    write_parquet_new(output_dir / "stage1_sample.parquet", sample)
    return summary
