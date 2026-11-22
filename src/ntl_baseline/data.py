"""Local-only input, archive and submission contracts. Never log passage content."""

import csv
import hashlib
import json
import stat
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pandas as pd

from .constants import DATA_FILES, LABELS, ContractError

MAX_INPUT_BYTES = 64 * 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_columns(frame: pd.DataFrame, expected: tuple[str, ...]) -> None:
    if not frame.columns.is_unique or set(frame.columns) != set(expected):
        raise ContractError(f"columns must be exactly {', '.join(expected)}")
    if frame.empty:
        raise ContractError("input must contain at least one row")


def _check_strings(series: pd.Series, column: str) -> None:
    if (
        series.isna().any()
        or not series.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    ):
        raise ContractError(f"{column} must contain non-empty strings")


def _check_ids(frame: pd.DataFrame) -> None:
    _check_strings(frame["id"], "id")
    if frame["id"].duplicated().any():
        raise ContractError("id must be unique")


def _normalize_text(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True).reset_index(drop=True)
    result["text"] = result["text"].fillna("")
    if not result["text"].map(lambda value: isinstance(value, str)).all():
        raise ContractError("text must contain strings or missing values")
    return result


def validate_train(frame: pd.DataFrame) -> pd.DataFrame:
    _check_columns(frame, ("id", "prompt_id", "text", "model"))
    _check_ids(frame)
    _check_strings(frame["prompt_id"], "prompt_id")
    _check_strings(frame["model"], "model label")
    if not frame["model"].isin(LABELS).all():
        raise ContractError("unknown model label")
    if set(frame["model"]) != set(LABELS):
        raise ContractError("training data must cover all 18 labels")
    return _normalize_text(frame)


def validate_test(frame: pd.DataFrame) -> pd.DataFrame:
    _check_columns(frame, ("id", "text"))
    _check_ids(frame)
    return _normalize_text(frame)


def validate_submission(frame: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    test = validate_test(test)
    _check_columns(frame, ("id", "model"))
    _check_ids(frame)
    _check_strings(frame["model"], "model label")
    if not frame["model"].isin(LABELS).all():
        raise ContractError("submission contains an invalid model label")
    if len(frame) != len(test) or set(frame["id"]) != set(test["id"]):
        raise ContractError("submission id set must equal test id set")
    return frame.set_index("id").loc[test["id"], ["model"]].reset_index()


def input_file(data_dir: Path, name: str) -> Path:
    path = data_dir / name
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"missing regular input file: {name}")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ContractError(f"input exceeds local size budget: {name}")
    return path


def read_csv_input(data_dir: Path, name: str) -> pd.DataFrame:
    path = input_file(data_dir, name)
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), [])
        if not header or len(set(header)) != len(header):
            raise ContractError(f"invalid or duplicate columns in {name}")
        # Enforce the published string schema (including leading-zero IDs).
        # Missing text is still normalized with the official fillna('') policy.
        return pd.read_csv(path, dtype={name: str for name in ("id", "prompt_id", "text", "model")})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ContractError(f"invalid CSV input: {name}") from exc


def load_test(data_dir: Path) -> pd.DataFrame:
    return validate_test(read_csv_input(data_dir, "test.csv"))


def load_competition_data(
    data_dir: Path, *, require_sample: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = validate_train(read_csv_input(data_dir, "train.csv"))
    test = load_test(data_dir)
    if not require_sample:
        # Hidden reruns replace test.csv; the public sample is not its ID authority.
        return train, test
    sample_path = input_file(data_dir, "sample_submission.parquet")
    try:
        sample = pd.read_parquet(sample_path, engine="pyarrow")
    except (ValueError, OSError) as exc:
        raise ContractError("invalid sample_submission.parquet") from exc
    validate_submission(sample, test)
    return train, test


def import_archive(archive: Path, destination: Path) -> dict[str, object]:
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"refusing to replace existing data: {destination}")
    if destination.is_symlink() or not archive.is_file():
        raise ContractError("archive and destination must be local regular paths")
    if archive.stat().st_size > MAX_INPUT_BYTES:
        raise ContractError("archive exceeds local size budget")
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            names = [member.filename for member in members]
            for member in members:
                path = PurePosixPath(member.filename)
                is_link = stat.S_ISLNK(member.external_attr >> 16)
                if path.is_absolute() or ".." in path.parts or "\\" in member.filename or is_link:
                    raise ContractError("unsafe archive member path or symlink")
            if len(members) != 3 or set(names) != set(DATA_FILES):
                raise ContractError("archive must contain exactly the three official files at root")
            if sum(member.file_size for member in members) > MAX_INPUT_BYTES:
                raise ContractError("archive expansion exceeds local size budget")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".ntl-import-", dir=destination.parent) as temp:
                staging = Path(temp) / "data"
                staging.mkdir()
                for member in members:
                    payload = bundle.read(member)
                    if len(payload) != member.file_size:
                        raise ContractError("archive member size mismatch")
                    (staging / member.filename).write_bytes(payload)
                load_competition_data(staging)
                manifest: dict[str, object] = {
                    "source_url": "https://www.kaggle.com/competitions/name-that-llm/data",
                    "imported_at_utc": datetime.now(UTC).isoformat(),
                    "archive_name": archive.name,
                    "archive_sha256": file_sha256(archive),
                    "files": {name: file_sha256(staging / name) for name in DATA_FILES},
                }
                (staging / "source.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
                staging.rename(destination)
                return manifest
    except zipfile.BadZipFile as exc:
        raise ContractError("invalid ZIP archive") from exc
