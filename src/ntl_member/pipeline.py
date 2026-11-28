"""Run the preregistered CPU / GPU stages with verified, append-only attempts."""

import argparse
import fcntl
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ntl_baseline.artifacts import directory_hashes, write_json_new
from ntl_baseline.constants import ContractError
from ntl_baseline.data import file_sha256

from .config import DEFAULT_CONFIG, ROOT, load, source_lock_path
from .plan import NAMES, print_plan
from .stages import env_versions, read, reusable_data_training

GPU = {"00", "01", "02", "04", "05", "06", "07", "09"}
REUSE = {
    "02": ("sources", "pool"),
    "03": ("sources", "pool"),
    "04": ("train", "features", "batching"),
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_marker(path: Path, value: dict[str, Any]) -> None:
    """A killed controller leaves a fresh attempt or a complete marker, never half a marker."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, path)


def verified_stage(attempt: Path, signature: str) -> Path:
    marker = attempt / "completed.json"
    if not marker.is_file() or (attempt / "failed.json").exists():
        raise ContractError("stage is incomplete")
    record = read(marker)
    if record.get("signature") != signature:
        raise ContractError("stage signature mismatch")
    payload = Path(record.get("reference", str(attempt / "output")))
    if not payload.is_dir() or not payload.resolve().is_relative_to((ROOT / "artifacts").resolve()):
        raise ContractError("invalid stage payload location")
    if directory_hashes(payload) != record["outputs"]:
        raise ContractError("completed stage output digest mismatch")
    return payload


def attempts(root: Path, stage: str) -> list[Path]:
    parent = root / f"{stage}-{NAMES[int(stage)]}"
    return sorted(
        (p for p in parent.glob("attempt-*") if re.fullmatch(r"attempt-[1-9][0-9]*", p.name)),
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,
    )


def resource_record(attempt: Path) -> dict[str, Any]:
    record = read(attempt / "resources.json")
    record.setdefault("peak_device_mib", 0.0)
    gpu = attempt / "gpu_samples.csv"
    if gpu.exists():
        for line in gpu.read_text().splitlines():
            cols = line.split(",")
            if len(cols) >= 2 and cols[1].strip().isdigit():
                record["peak_device_mib"] = max(record["peak_device_mib"], int(cols[1]))
    return record


@contextmanager
def sampling(enabled: bool, path: Path) -> Iterator[None]:
    if not enabled:
        yield
        return
    with path.open("x") as stream:
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
                "-lms",
                "1000",
            ],
            stdout=stream,
            stderr=subprocess.DEVNULL,
        )
        try:
            yield
        finally:
            proc.terminate()
            proc.wait()


def reuse_signature(stage: str, cfg: dict[str, Any], paths: dict[str, Path]) -> str:
    # Only inputs capable of changing these three stage results participate.
    code = {name: file_sha256(ROOT / "src/ntl_member" / f"{name}.py") for name in REUSE[stage]}
    code["stage_driver"] = digest(inspect.getsource(reusable_data_training))
    code["config"] = file_sha256(ROOT / "src/ntl_member/config.py")
    for shared in ("constants", "artifacts", "data"):
        code["baseline_" + shared] = file_sha256(ROOT / "src/ntl_baseline" / f"{shared}.py")
    code["cpu_lock"] = file_sha256(ROOT / "uv.lock")
    inputs = read(paths["00"] / "inputs.json")
    locked = cfg["pins"].get("source_lock")
    values: dict[str, Any] = {
        "implementation": code,
        "file_pins": cfg["pins"],
        "filters": cfg["filters"],
        "source_lock_sha256": file_sha256(source_lock_path(cfg)) if locked else None,
        "retained_sources_sha256": (
            file_sha256(source_lock_path(cfg).with_name("retained_sources.json"))
            if locked
            else None
        ),
        "environment": inputs["gpu_versions"],
        "seed": cfg["seed"],
    }
    if stage in ("03", "04"):
        values.update(
            official=inputs["official_train"],
            pool=file_sha256(paths["02"] / "source_lock.json"),
            sampling=cfg["sampling"],
            data_gates={k: v for k, v in cfg["gates"].items() if k.startswith("data_")},
            cap=read(paths["01"] / "smoke.json")["per_class_cap"],
        )
    if stage == "04":
        values.update(
            sample=file_sha256(paths["03"] / "stage1_sample.parquet"),
            base=inputs["base"],
            token_budget=read(paths["01"] / "smoke.json")["token_budget"],
            model=cfg["model"],
            train=cfg["train"],
        )
    return digest(values)


def log(root: Path, message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with (root / "pipeline.log").open("a") as stream:
        print(f"{stamp} {message}", file=stream)
    print(message, flush=True)


def install_reports(source: Path, target: Path) -> None:
    if target.exists():
        if directory_hashes(source) != directory_hashes(target):
            raise ContractError("a different report already exists; retain the historical report")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        staging = Path(tmp) / "public"
        shutil.copytree(source, staging)
        staging.rename(target)


def recheck_sources(cfg: dict[str, Any], root: Path) -> None:
    from .sources import check_live

    check_dir = Path(tempfile.mkdtemp(prefix="compliance-", dir=root))
    report = check_dir / "source_status_live.json"
    log(root, f"Rechecking contributing sources; report {report}")
    check_live(cfg, report)


def execute(args: argparse.Namespace, root: Path, lock_fd: int | None = None) -> None:
    cfg = load()
    base = ROOT / "artifacts/v31-asset-cache/models/qwen-lm/qwen-3/Transformers/8b-base/1"
    data = ROOT / "data/raw/name-that-llm/train.csv"
    cfg_hash = file_sha256(DEFAULT_CONFIG)
    metadata = {"commit": args.commit, "config_sha256": cfg_hash}
    if args.resume:
        if read(root / "run.json") != metadata:
            raise ContractError("resume needs identical commit and configuration")
    else:
        write_marker(root / "run.json", metadata)
    paths: dict[str, Path] = {}
    records: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    reuse_root = None
    last_reuse = -1
    if args.reuse_stage:
        old, _, number = args.reuse_stage.rpartition(":")
        reuse_root = Path(old).resolve()
        if (
            number not in REUSE
            or not old
            or not reuse_root.is_relative_to((ROOT / "artifacts").resolve())
            or reuse_root == root
        ):
            raise ContractError("reuse expects a different artifacts run and stage 02, 03 or 04")
        last_reuse = int(number)
    for i, name in enumerate(NAMES):
        stage = f"{i:02d}"
        existing = attempts(root, stage)
        signature = (
            reuse_signature(stage, cfg, paths)
            if stage in REUSE
            else digest({"run": metadata, "upstream": hashes, "stage": stage})
        )
        completed = next((p for p in existing if (p / "completed.json").exists()), None)
        if completed:
            attempt = completed
            output = verified_stage(attempt, signature)
            if stage == "02":
                recheck_sources(cfg, root)
            log(root, f"{stage} {name} verified; using completed stage")
        else:
            attempt = (
                root
                / f"{stage}-{name}"
                / f"attempt-{int(existing[0].name.split('-')[1]) + 1 if existing else 1}"
            )
            attempt.mkdir(parents=True)
            out = attempt / "output"
            write_marker(
                attempt / "started.json",
                {
                    **metadata,
                    "signature": signature,
                    "stage": name,
                    "versions": env_versions(),
                    "resumed_from": [x.name for x in existing],
                },
            )
            try:
                if reuse_root and stage in REUSE and i <= last_reuse:
                    source = next(
                        (a for a in attempts(reuse_root, stage) if (a / "completed.json").exists()),
                        None,
                    )
                    if source is None:
                        raise ContractError("requested source stage was never completed")
                    output = verified_stage(source, signature)
                    if stage == "02":
                        recheck_sources(cfg, root)
                    write_json_new(attempt / "resources.json", resource_record(source))
                    extra = {"reference": str(output)}
                else:
                    out.mkdir()
                    job = {
                        "commit": args.commit,
                        "stage": stage,
                        "inputs": {k: str(v) for k, v in paths.items()},
                        "output": str(out),
                        "base": str(base),
                        "official_csv": str(data),
                        "resources": records,
                        "previous": [
                            str(p / "output")
                            for p in existing
                            if (p / "started.json").is_file()
                            and read(p / "started.json")["signature"] == signature
                        ],
                    }
                    write_json_new(attempt / "job.json", job)
                    python = args.python if stage in GPU else sys.executable
                    log(root, f"{stage} {name} running; log {attempt / 'stage.log'}")
                    with (
                        (attempt / "stage.log").open("x") as stream,
                        sampling(stage in GPU and stage != "02", attempt / "gpu_samples.csv"),
                    ):
                        subprocess.run(
                            [python, "-m", "ntl_member.stages", str(attempt / "job.json")],
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=True,
                            pass_fds=(() if lock_fd is None else (lock_fd,)),
                        )
                    output = out
                    extra = {}
                write_marker(
                    attempt / "completed.json",
                    {"signature": signature, "outputs": directory_hashes(output), **extra},
                )
            except BaseException as exc:
                write_json_new(
                    attempt / "failed.json", {"stage": stage, "error_type": type(exc).__name__}
                )
                log(root, f"{stage} {name} stopped; see {attempt / 'stage.log'}")
                raise
        paths[stage] = output
        hashes[stage] = digest(read(attempt / "completed.json")["outputs"])
        records[stage] = resource_record(attempt)
        if stage == "00" and completed:
            # An old report cannot attest to external input caches today.
            inputs = read(output / "inputs.json")
            if file_sha256(data) != inputs["official_train"] or any(
                file_sha256(base / k) != v for k, v in inputs["base"].items()
            ):
                raise ContractError("server inputs changed since preflight")
    install_reports(paths["10"] / "public", ROOT / "reports/member-q8pool-v1")
    if not (root / "completed.json").exists():
        write_marker(
            root / "completed.json",
            {
                **metadata,
                "stages": hashes,
                "package": str(paths["09"] / "kaggle_dataset"),
                "public_report": "reports/member-q8pool-v1",
            },
        )
    log(root, f"Completed. Report {root / 'completed.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-stage")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("expected a full commit SHA")
    if args.dry_run:
        print_plan()
        return
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != args.commit:
        parser.error("checked-out commit differs from requested SHA")
    root = args.run_root.resolve()
    private = (ROOT / "artifacts").resolve()
    if not root.is_relative_to(private) or root == private:
        parser.error("run-root must be inside artifacts")
    os.umask(0o077)
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, root, lock.fileno())


if __name__ == "__main__":
    main()
