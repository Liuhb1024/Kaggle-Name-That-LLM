"""Exercise checkpoint restart and V31 member layout with a local, random two-layer Qwen."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ntl_baseline.constants import LABELS

from .train import fit, predict


def run(directory: Path, device: str = "cuda:0") -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(71)
    base = directory / "base"
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "sample": 3, "text": 4}
    vocab.update({f"s{i}": 5 + i for i in range(40)})
    model = Qwen3Model(
        Qwen3Config(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            attention_dropout=0.0,
            use_cache=False,
        )
    )
    model.save_pretrained(base, safe_serialization=True)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]"
    ).save_pretrained(base)
    config: dict[str, Any] = {
        "seed": 0,
        "model": {
            "keep_layers": 2,
            "pool_layers": [1, 2],
            "max_len": 16,
            "lora_r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "train": {
            "min_rows_per_step": 2,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "grad_clip": 1.0,
            "stage1": {"epochs": 1, "lr_lora": 1e-4, "lr_head": 1e-3, "checkpoint_every": 10},
        },
        "kaggle": {"batch_size": 2},
    }
    frame = pd.DataFrame(
        {
            "text": [f"sample text s{i}" for i in range(40)],
            "label": [LABELS[i % len(LABELS)] for i in range(40)],
            "prompt_group": [i // 2 for i in range(40)],
        }
    )
    direct, resumed = directory / "direct", directory / "resumed"
    # Two three-token rows exhaust each (six-token) budget. Each step uses one pair.
    stats = fit(base, frame, config, "stage1", direct, 6, device=device)
    checkpoint = direct / "checkpoints" / "step-000010"
    restarted = fit(base, frame, config, "stage1", resumed, 6, resume=checkpoint, device=device)
    if stats["global_step"] != 20 or restarted["resumed_step"] != 10:
        raise ValueError("toy selftest did not exercise the required restart boundary")

    def curve(d: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (d / "train_curve.jsonl").read_text().splitlines()]

    before, after = curve(direct)[10:], curve(resumed)
    max_error = 0.0
    for ref, run_step in zip(before, after, strict=True):
        if any(ref[k] != run_step[k] for k in ("global_step", "data_cursor", "lr")):
            raise ValueError("resumed schedule or cursor differs")
        max_error = max(
            max_error, abs(ref["loss"] - run_step["loss"]) / max(1e-8, abs(ref["loss"]))
        )
    if max_error > 0.02:
        raise ValueError("resumed loss differs by more than 2%")

    with safe_open(direct / "heads.safetensors", framework="pt") as weights:
        expected = {"plain.weight", "plain.bias", "centered.weight", "centered.bias"} | {
            f"norms.{i}.{key}" for key in ("weight", "bias") for i in range(2)
        }
        if set(weights.keys()) != expected or any(
            weights.get_tensor(k).dtype != torch.float32 for k in expected
        ):
            raise ValueError("unexpected V31 head state")
    modules = {
        n for n, _ in model.named_modules() if n.split(".")[-1] in config["model"]["target_modules"]
    }
    expected_lora = {
        f"base_model.model.{n}.lora_{side}.weight" for n in modules for side in ["A", "B"]
    }
    with safe_open(direct / "adapter" / "adapter_model.safetensors", framework="pt") as weights:
        if set(weights.keys()) != expected_lora:
            raise ValueError("exported LoRA keys cannot be mapped by V31")
        if not any(
            torch.count_nonzero(weights.get_tensor(k)).item()
            for k in expected_lora
            if ".lora_B." in k
        ):
            raise ValueError("training did not update the zero-initialized LoRA B matrices")
    # Exercise the official-data arm, including prompts larger than a physical microbatch.
    grouped = frame.iloc[:12].copy()
    grouped["prompt_group"] = [i // 3 for i in range(len(grouped))]
    for mode, initial in [("control", None), ("stage2", direct)]:
        config["train"][mode] = {
            "epochs": 1,
            "lr_lora": 5e-5,
            "lr_head": 5e-4,
            "prompts_per_step": 2,
        }
        arm = fit(base, grouped, config, mode, directory / mode, 6, init=initial, device=device)
        if arm["global_step"] != 2:
            raise ValueError("official prompt accumulation must update once for every two prompts")
    outputs = predict(
        base,
        direct,
        frame.iloc[:4],
        config,
        "fp16" if device.startswith("cuda") else "fp32",
        device=device,
    )
    if any(a.shape != (4, len(LABELS)) or not np.isfinite(a).all() for a in outputs):
        raise ValueError("toy member produced invalid log probabilities")
    return {
        "steps": 20,
        "resumed_step": 10,
        "max_relative_loss_error": max_error,
        "mapped_lora_keys": len(expected_lora),
        "stage2_steps": 2,
        "control_steps": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="q8pool-selftest-") as tmp:
        print(json.dumps(run(Path(tmp), device=args.device), sort_keys=True))


if __name__ == "__main__":
    main()
