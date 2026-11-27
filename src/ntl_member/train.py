"""Fit the V31-compatible member, resume verified checkpoints and predict in row order."""

import gc
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd

from ntl_baseline.constants import LABELS
from ntl_baseline.data import file_sha256

from .batching import padded_batches, plan_steps

_STATE_FILES = {
    "optimizer.pt",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "heads.safetensors",
}


def _json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)


def _verified(path: Path) -> bool:
    try:
        manifest = json.loads((path / "checkpoint.json").read_text())
        return (
            manifest["global_step"] > 0
            and set(manifest["files"]) == _STATE_FILES
            and path.name == f"step-{manifest['global_step']:06d}"
            and all(
                not (path / name).is_symlink() and file_sha256(path / name) == digest
                for name, digest in manifest["files"].items()
            )
        )
    except (KeyError, ValueError, OSError, TypeError):
        return False


def latest_checkpoint(path: Path) -> Path:
    if path.name.startswith("step-"):
        candidates = [path]
    else:
        directory = path if path.name == "checkpoints" else path / "checkpoints"
        candidates = sorted(directory.glob("step-*"), reverse=True)
    for candidate in candidates:
        if _verified(candidate):
            return candidate
    raise ValueError("no valid checkpoint for resume")


def _groups(frame: pd.DataFrame) -> list[int]:
    values = frame["prompt_group"].replace(r"^\s*$", pd.NA, regex=True)
    groups, _ = pd.factorize(values, sort=False)
    # Unspecified groups only contribute to the plain loss.
    return [int(g) if g >= 0 else -1 - i for i, g in enumerate(groups)]


def _prepare(
    base: Path, frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[Any, list[list[int]]]:
    from .features import load_tokenizer

    tokenizer = load_tokenizer(base)
    if frame.empty or frame["text"].isna().any():
        raise ValueError("expected a nonempty frame of texts")
    texts = frame.text.tolist()
    encoded = []
    for start in range(0, len(texts), 1024):
        encoded.extend(
            tokenizer(
                texts[start : start + 1024], truncation=True, max_length=config["model"]["max_len"]
            )["input_ids"]
        )
    if any(len(e) < 2 for e in encoded):
        raise ValueError("a passage has no tokens after first-token masking")
    return tokenizer, encoded


def _fingerprint(
    base: Path,
    config: dict[str, Any],
    encoded: list[list[int]],
    labels: list[int],
    groups: list[int],
    mode: str,
    budget: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [config["model"], config["train"], config["seed"], mode, budget], sort_keys=True
        ).encode()
    )
    digest.update((base / "config.json").read_bytes())
    for tokens, label, group in zip(encoded, labels, groups, strict=True):
        digest.update(np.asarray([len(tokens), label, group, *tokens], dtype="<i8").tobytes())
    return digest.hexdigest()


def _save_checkpoint(
    directory: Path,
    step: int,
    member: Any,
    optimizer: Any,
    scheduler: Any,
    signature: str,
    device: str,
) -> Path:
    import torch

    from .features import export_weights

    path = directory / "checkpoints" / f"step-{step:06d}"
    path.mkdir(parents=True, exist_ok=False)
    export_weights(member, path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.startswith("cuda") else None,
            "python_rng": random.getstate(),
            "global_step": step,
        },
        path / "optimizer.pt",
    )
    _json(
        path / "checkpoint.json",
        {
            "global_step": step,
            "data_cursor": step,
            "signature": signature,
            "files": {name: file_sha256(path / name) for name in sorted(_STATE_FILES)},
        },
    )
    return path


def _optimizer(member: Any, recipe: dict[str, Any], decay: float) -> Any:
    import torch

    groups = []
    for module, rate in [(member.trunk, recipe["lr_lora"]), (member.heads, recipe["lr_head"])]:
        for regularize in [True, False]:
            params = [
                p
                for _, p in module.named_parameters()
                if p.requires_grad and (p.ndim > 1) == regularize
            ]
            if params:
                groups.append(
                    {"params": params, "lr": rate, "weight_decay": decay if regularize else 0.0}
                )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def _rate(step: int, steps: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


def _learn_step(
    member: Any,
    microbatches: list[list[int]],
    encoded: list[list[int]],
    lengths: list[int],
    labels: list[int],
    groups: list[int],
    budget: int,
    weights: np.ndarray,
    device: str,
) -> tuple[float, int]:
    import torch
    from torch.nn.functional import cross_entropy
    from torch.utils.checkpoint import checkpoint

    from .features import center_features, tensors

    # Match weighted CE over the accumulated optimizer step, even when microbatch sizes differ.
    plain_mass, centered_mass = 0.0, 0.0
    for rows in microbatches:
        counts = Counter(groups[i] for i in rows)
        plain_mass += sum(weights[labels[i]] for i in rows)
        centered_mass += sum(weights[labels[i]] for i in rows if counts[groups[i]] > 1)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    total = torch.zeros((), device=device)
    tokens = 0
    for rows in microbatches:
        chunks = padded_batches(rows, lengths, budget)
        features = []
        for part in chunks:
            ids, mask = tensors(encoded, part, member.pad, device)
            tokens += ids.numel()
            if len(chunks) == 1:
                features.append(member(ids, mask))
            else:
                # A long official prompt still needs one joint centering mean.
                features.append(checkpoint(member, ids, mask, use_reentrant=False))
        z = torch.cat(features)
        target = torch.tensor([labels[i] for i in rows], device=device)
        loss = cross_entropy(member.heads.plain(z), target, weight=w, reduction="sum") / plain_mass
        centered, eligible = center_features(z, [groups[i] for i in rows])
        if eligible:
            loss = (
                loss
                + cross_entropy(
                    member.heads.centered(centered[eligible]),
                    target[eligible],
                    weight=w,
                    reduction="sum",
                )
                / centered_mass
            )
        loss.backward()
        total += loss.detach()
    return float(total.item()), tokens


def fit(
    base: Path,
    frame: pd.DataFrame,
    config: dict[str, Any],
    mode: str,
    output: Path,
    budget: int,
    init: Path | None = None,
    resume: Path | None = None,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    import torch

    from .features import export_weights, make_member

    if mode not in {"stage1", "stage2", "control"}:
        raise ValueError("unknown training mode")
    if (mode == "stage2") != (init is not None) or (resume is not None and mode != "stage1"):
        raise ValueError("invalid initialization or resume mode")
    recipe = config["train"][mode]
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in ("adapter", "checkpoints", "train_curve.jsonl", "meta.json")
    ):
        raise FileExistsError("fit requires a fresh attempt directory")
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer, encoded = _prepare(base, frame, config)
    lengths = [len(x) for x in encoded]
    labels = [LABELS.index(x) for x in frame["label"]]
    counts = np.bincount(labels, minlength=len(LABELS))
    weights = 1.0 / np.maximum(counts, 1)
    groups = _groups(frame)
    target = (
        config["train"]["min_rows_per_step"] if mode == "stage1" else recipe["prompts_per_step"]
    )
    steps = [
        step
        for epoch in range(recipe["epochs"])
        for step in plan_steps(
            lengths, groups, budget, seed=seed + epoch, pool=mode == "stage1", target=target
        )
    ]
    signature = _fingerprint(base, config, encoded, labels, groups, mode, budget)
    checkpoint_dir = latest_checkpoint(resume) if resume is not None else None
    if checkpoint_dir:
        manifest = json.loads((checkpoint_dir / "checkpoint.json").read_text())
        if manifest["signature"] != signature:
            raise ValueError("checkpoint differs from configuration, inputs or batch plan")
    gpu = device.startswith("cuda")
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    member = make_member(
        base,
        config,
        tokenizer.pad_token_id,
        torch.bfloat16 if gpu else torch.float32,
        initial=checkpoint_dir or init,
        training=True,
        device=device,
    )
    optimizer = _optimizer(member, recipe, config["train"]["weight_decay"])
    warmup = math.ceil(len(steps) * config["train"]["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _rate(step, len(steps), warmup)
    )
    start = 0
    if checkpoint_dir:
        state = torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
        start = state["global_step"]
        if not 0 < start <= len(steps) or start != manifest["data_cursor"]:
            raise ValueError("invalid checkpoint cursor")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        random.setstate(state["python_rng"])
        if gpu:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        del state
    t0 = monotonic()
    total_tokens = 0
    with (output / "train_curve.jsonl").open("x", buffering=1) as log:
        for index in range(start, len(steps)):
            optimizer.zero_grad(set_to_none=True)
            lr = float(optimizer.param_groups[0]["lr"])
            loss, n_tokens = _learn_step(
                member, steps[index], encoded, lengths, labels, groups, budget, weights, device
            )
            if not math.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            torch.nn.utils.clip_grad_norm_(
                member.parameters(), config["train"]["grad_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            total_tokens += n_tokens
            cursor = index + 1
            log.write(
                json.dumps(
                    dict(
                        global_step=cursor,
                        data_cursor=cursor,
                        loss=loss,
                        lr=lr,
                        padded_tokens=n_tokens,
                    )
                )
                + "\n"
            )
            if mode == "stage1" and (
                cursor % recipe["checkpoint_every"] == 0 or cursor == len(steps)
            ):
                _save_checkpoint(output, cursor, member, optimizer, scheduler, signature, device)
    if gpu:
        torch.cuda.synchronize(device)
    seconds = monotonic() - t0
    export_weights(member, output)
    _json(
        output / "meta.json",
        {
            "labels": list(LABELS),
            "cfg": {
                "keep": config["model"]["keep_layers"],
                "layers": config["model"]["pool_layers"],
                "max_len": config["model"]["max_len"],
                "eos_id": None,
                "no_final_norm": False,
            },
            "training_signature": signature,
        },
    )
    result = {
        "global_step": len(steps),
        "resumed_step": start,
        "padded_tokens": total_tokens,
        "seconds": seconds,
        "tokens_per_second": total_tokens / max(1e-9, seconds),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if gpu else 0,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if gpu else 0,
    }
    _json(output / "fit.json", result)
    del member, optimizer, scheduler
    gc.collect()
    if gpu:
        torch.cuda.empty_cache()
    return result


@contextmanager
def _v31_matmul() -> Iterator[None]:
    import torch

    matmul = torch.backends.cuda.matmul
    previous = matmul.allow_fp16_reduced_precision_reduction
    matmul.allow_fp16_reduced_precision_reduction = False
    try:
        yield
    finally:
        matmul.allow_fp16_reduced_precision_reduction = previous


def predict(
    base: Path,
    member: Path,
    frame: pd.DataFrame,
    config: dict[str, Any],
    dtype: str = "bf16",
    *,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from .features import center_features, make_member, tensors

    precision = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]
    tokenizer, encoded = _prepare(base, frame, config)
    network = make_member(
        base,
        config,
        tokenizer.pad_token_id,
        precision,
        initial=member,
        training=False,
        device=device,
    )
    order = sorted(range(len(frame)), key=lambda i: len(encoded[i]))
    batch_size = int(config["kaggle"]["batch_size"])
    blocks = []
    with torch.inference_mode(), _v31_matmul():
        for start in range(0, len(order), batch_size):
            part = order[start : start + batch_size]
            ids, mask = tensors(encoded, part, network.pad, device)
            z = network(ids, mask)
            if not torch.isfinite(z).all():
                if precision != torch.float16:
                    raise FloatingPointError("nonfinite readout")
                network.trunk.to(torch.bfloat16)
                z = network(ids, mask)
                network.trunk.to(torch.float16)
                if not torch.isfinite(z).all():
                    raise FloatingPointError("nonfinite readout after bf16 retry")
            blocks.append(z.cpu())
        features = torch.cat(blocks)[np.argsort(order)]
        network.heads.to("cpu")
        centered, _ = center_features(features, _groups(frame))
        plain_lp = network.heads.plain(features).log_softmax(-1).numpy()
        centered_lp = network.heads.centered(centered).log_softmax(-1).numpy()
    return plain_lp, centered_lp


def smoke(base: Path, config: dict[str, Any], out: Path) -> dict[str, Any]:
    """Benchmark an actual optimizer step at maximum sequence length on the target trunk."""
    import shutil

    import torch

    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("smoke requires a bf16-capable CUDA device")
    if (out / "member").exists():
        raise FileExistsError("smoke requires a fresh attempt directory")
    start = monotonic()
    for budget in config["train"]["token_budgets"]:
        width = config["model"]["max_len"]
        # Two optimizer steps, at least one almost full padded microbatch per step.
        n = 2 * max(config["train"]["min_rows_per_step"], budget // width)
        frame = pd.DataFrame(
            {
                "text": [
                    f"Synthetic token benchmark case {i}. " + " measurement" * width
                    for i in range(n)
                ],
                "label": [LABELS[i % len(LABELS)] for i in range(n)],
                "prompt_group": [i // 2 for i in range(n)],
            }
        )
        attempt = out / f"budget-{budget}"
        try:
            stats = fit(base, frame, config, "stage1", attempt, budget)
        except torch.cuda.OutOfMemoryError:
            stats = None
        # Clear failed forward/backward graphs after leaving the exception scope.
        gc.collect()
        torch.cuda.empty_cache()
        if stats is None:
            _json(attempt / "budget_result.json", {"budget": budget, "outcome": "oom"})
            continue
        peak = stats["peak_reserved_bytes"] / 2**30
        if peak > config["gates"]["smoke_peak_reserved_gib"]:
            _json(
                attempt / "budget_result.json",
                {"budget": budget, "outcome": "over_budget", "peak_reserved_gib": peak},
            )
            continue
        cap = config["sampling"]["per_class_cap"]

        # Project the padded corpus size from the preregistered 140 * 1.3 prior.
        hours_per_cap_row = len(LABELS) * 140 * 1.3 / stats["tokens_per_second"] / 3600

        if (hours_per_cap_row * cap) > config["gates"]["smoke_max_projected_stage1_hours"]:
            cap = config["sampling"]["fallback_per_class_cap"]
        if (hours_per_cap_row * cap) > config["gates"]["smoke_max_projected_stage1_hours"]:
            raise RuntimeError("even the fallback class cap exceeds the stage1 time budget")
        if monotonic() - start > 45 * 60:
            raise RuntimeError("training smoke exceeded 45 minutes")
        target = out / "member"
        target.mkdir()
        shutil.copytree(attempt / "adapter", target / "adapter")
        for name in ["heads.safetensors", "meta.json"]:
            shutil.copy2(attempt / name, target / name)
        return {
            **stats,
            "token_budget": budget,
            "per_class_cap": cap,
            "peak_reserved_gib": peak,
            "projected_stage1_hours": (hours_per_cap_row * cap),
            "smoke_seconds": monotonic() - start,
        }
    raise RuntimeError("no preregistered token budget passed the GPU smoke")
