"""Fetch immutable public training inputs and keep a reproducible source audit."""

import json
import re
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from ntl_baseline.artifacts import write_json_new
from ntl_baseline.constants import ContractError
from ntl_baseline.data import file_sha256

OPEN_POOL = "crtproduction/name-that-llm-open-pool"
EXTRA = "seyitkaangunes/ntl-extra-api-passages"
PINNED = {
    "ntl_open_pool.parquet": (
        OPEN_POOL,
        19,
        119227255,
        "fe6ece20e6f9f2fe5fb3c47f2aaa9749394c1056bc688a8eab865e0dc2bb56df",
    ),
    "source_protocol.csv": (
        OPEN_POOL,
        19,
        26447,
        "aad5be493cad260eb365d685c817dac961feb321257dd1e25afe1e6988ecc95e",
    ),
    "ntl_extra_passages_claude.csv": (
        EXTRA,
        2,
        4387162,
        "710dbe484fda23f0dee9b67c20e5bef9376bf754af05584ad142bc17845cd4b9",
    ),
}
HANDLE = re.compile(r"[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*")
ALLOWED_LICENSE_NAMES = {
    "CC BY 4.0": {"CC BY 4.0", "Attribution 4.0 International (CC BY 4.0)"},
    "CC BY-SA 4.0": {"CC BY-SA 4.0", "Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)"},
    "CC0 1.0": {
        "CC0 1.0",
        "CC0: Public Domain",
        "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    },
}


def normalize_license(name: str) -> str:
    return next((k for k, names in ALLOWED_LICENSE_NAMES.items() if name.strip() in names), "")


def safe_filename(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or name != str(path)
        or path.is_absolute()
        or name == "."
        or ".." in path.parts
        or not re.fullmatch(r"[a-zA-Z0-9_./() -]+", name)
    ):
        raise ContractError("Unsafe upstream filename")
    return name


def _request(url: str) -> dict[str, Any]:
    # No Kaggle tokens, cookies, netrc, or Authorization headers are used by urllib.
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (NTL public-source audit)",
            "Accept": "application/json",
        },
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                if response.geturl().split("?")[0] != url:
                    raise ContractError("Unexpected metadata redirect")
                return json.load(response)
        except HTTPError as exc:
            if exc.code in (401, 403, 404):
                return {"ref": url.split("/view/")[-1], "isPrivate": True}
            if exc.code != 429 and exc.code < 500:
                raise
        except (TimeoutError, URLError):
            pass
        if attempt < 2:
            time.sleep(2**attempt)
    raise ContractError("Kaggle public metadata unavailable after retries")


def public_metadata(
    ref: str, request: Callable[[str], dict[str, Any]] = _request
) -> dict[str, Any]:
    if not HANDLE.fullmatch(ref):
        raise ContractError("Invalid Kaggle dataset handle")
    result = request("https://www.kaggle.com/api/v1/datasets/view/" + ref)
    if result.get("ref") != ref or not isinstance(result.get("isPrivate"), bool):
        raise ContractError(f"Cannot verify source identity or visibility for {ref}")
    if result["isPrivate"]:
        return {"public": False, "license": "", "version": None, "last_updated": ""}
    version = result.get("currentVersionNumber")
    if type(version) is not int or version < 1:
        raise ContractError(f"Cannot verify current source version for {ref}")
    return {
        "public": True,
        "version": version,
        "license": normalize_license(result.get("licenseName") or ""),
        "raw_license": result.get("licenseName") or "",
        "last_updated": str(result.get("lastUpdated") or "")[:10],
    }


def check_lock(root: Path, lock: dict[str, Any]) -> None:
    for name, record in lock["files"].items():
        path = root / safe_filename(name)
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ContractError("Source lock contains an absent or escaped file")
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ContractError(f"Source file disagrees with lock: {name}")


def download_file(ref: str, version: int, filename: str, output: Path) -> None:
    import kagglehub

    safe_filename(filename)
    if not HANDLE.fullmatch(ref) or version <= 0:
        raise ContractError("Invalid versioned dataset handle")
    src = Path(kagglehub.dataset_download(f"{ref}/versions/{version}", path=filename))
    if not src.is_file() or src.name != Path(filename).name:
        raise ContractError("Kaggle returned an unexpected single-file download")
    output.parent.mkdir(exist_ok=True, parents=True)
    with src.open("rb") as stream, output.open("xb") as destination:
        shutil.copyfileobj(stream, destination)


def download(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    from .pool import classify_inputs, spot_plan

    versions = {OPEN_POOL: 19, EXTRA: 2}
    if config["pins"]["open_pool"] != OPEN_POOL + "/versions/19" or (
        config["pins"]["extra_api"] != EXTRA + "/versions/2"
    ):
        raise ContractError("Data handles differ from the reviewed versions")
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    metadata = {ref: public_metadata(ref) for ref in versions}
    for ref, info in metadata.items():
        if not info["public"] or info["license"] not in config["filters"]["allowed_licenses"]:
            raise ContractError(f"Main public dataset unavailable or disallowed: {ref}")
    for filename, (ref, version, size, digest) in PINNED.items():
        download_file(ref, version, filename, output_dir / filename)
        files[filename] = {"dataset": ref, "version": version, "bytes": size, "sha256": digest}
    check_lock(output_dir, {"files": files})
    protocol = pd.read_csv(output_dir / "source_protocol.csv", keep_default_na=False)
    sources = sorted(set(protocol.source_dataset) - metadata.keys())
    with ThreadPoolExecutor(max_workers=4) as executor:
        metadata.update(zip(sources, executor.map(public_metadata, sources), strict=True))
    write_json_new(output_dir / "source_metadata.json", dict(sorted(metadata.items())))
    eligible, _, _ = classify_inputs(output_dir, metadata, config)
    plan = spot_plan(eligible, config["filters"]["spot_check_rows"], config["seed"])
    missing: list[dict[str, str]] = []
    for ref, sample in plan.items():
        version = metadata[ref]["version"]
        for filename in sorted(set(sample.source_file)):
            # Some upstream identifiers describe joins, not recoverable raw files.
            if Path(filename).suffix.lower() not in {".csv", ".parquet", ".jsonl"}:
                missing.append({"dataset": ref, "file": filename})
                continue
            name = f"spot/{ref}/{filename}"
            path = output_dir / safe_filename(name)
            if ref == EXTRA and filename == "ntl_extra_passages_claude.csv" and version == 2:
                continue
            try:
                download_file(ref, version, filename, path)
            except Exception as exc:
                # A verified API 404 can be audited as absent; network errors must be retried.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status != 404:
                    raise
                missing.append({"dataset": ref, "file": filename})
                continue
            files[name] = {
                "dataset": ref,
                "version": version,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    write_json_new(output_dir / "spot_plan.json", {"sources": list(plan), "missing": missing})
    for name in ("source_metadata.json", "spot_plan.json"):
        path = output_dir / name
        files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    lock = {"files": dict(sorted(files.items())), "versions": versions}
    previous = config["pins"].get("source_lock", "")
    if previous:
        pinned_bytes = Path(previous).read_bytes()
        if json.loads(pinned_bytes) != lock:
            raise ContractError("Pinned source lock differs from current acquisition")
        # All downstream provenance uses the exact bytes of the approved manifest.
        with (output_dir / "source_lock.json").open("xb") as destination:
            destination.write(pinned_bytes)
    else:
        write_json_new(output_dir / "source_lock.json", lock)
    return {"datasets": len(metadata), "spot_sources": len(plan), "files": len(files)}
