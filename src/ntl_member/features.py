"""LoRA Qwen3 trunk and readout compatible with V31 l24_predict."""

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ntl_baseline.constants import LABELS


def load_tokenizer(base: Path) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("base tokenizer requires an EOS or padding token")
    return tokenizer


class Readouts(nn.Module):
    def __init__(self, hidden: int, layers: list[int]) -> None:
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in layers)
        self.plain = nn.Linear(hidden * len(layers), len(LABELS))
        self.centered = nn.Linear(hidden * len(layers), len(LABELS))


class Member(nn.Module):
    def __init__(self, trunk: Any, heads: Readouts, layers: list[int], pad: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.heads = heads
        self.layers = layers
        self.pad = pad

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        dtype = next(self.trunk.parameters()).dtype
        with torch.autocast("cuda", dtype=dtype, enabled=ids.is_cuda and self.training):
            states = self.trunk(
                input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False
            ).hidden_states
        # Only the trunk is autocast; the pooling, normalization and heads are FP32 in V31.
        active = mask.unsqueeze(-1).float()
        active[:, 0] = 0.0
        denom = active.sum(dim=1)
        return torch.cat(
            [
                norm((states[index].float() * active).sum(dim=1) / denom)
                for index, norm in zip(self.layers, self.heads.norms, strict=True)
            ],
            dim=-1,
        )


def make_member(
    base: Path,
    config: dict[str, Any],
    pad: int,
    dtype: torch.dtype,
    *,
    initial: Path | None,
    training: bool,
    device: str = "cuda:0",
) -> Member:
    model = config["model"]
    cfg = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    keep = model["keep_layers"]
    layers = model["pool_layers"]
    if cfg.model_type != "qwen3" or cfg.num_hidden_layers < keep or max(layers) > keep:
        raise ValueError("base is incompatible with the configured Qwen3 readout")
    cfg.num_hidden_layers = keep
    cfg.use_cache = False
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = cfg.layer_types[:keep]
    trunk, info = AutoModel.from_pretrained(
        base,
        config=cfg,
        dtype=dtype,
        device_map={"": device},
        use_safetensors=True,
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("incomplete base checkpoint")
    if initial is None and training:
        adapter = LoraConfig(
            r=model["lora_r"],
            lora_alpha=model["lora_alpha"],
            lora_dropout=model["lora_dropout"],
            target_modules=model["target_modules"],
            bias="none",
            use_dora=False,
            use_rslora=False,
        )
        trunk = get_peft_model(trunk, adapter)
    elif training:
        assert initial is not None
        trunk = PeftModel.from_pretrained(
            trunk,
            initial / "adapter",
            local_files_only=True,
            is_trainable=training,
        )
    if training:
        trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        # V31 multiplies B@A in fp32, casts the delta to base dtype, and adds in place.
        # PEFT merge behavior varies by dtype, so merge the same delta explicitly.
        if initial is None:
            raise ValueError("inference requires a trained member")
        mods = dict(trunk.named_modules())
        weights = load_file(initial / "adapter" / "adapter_model.safetensors")
        scale = model["lora_alpha"] / model["lora_r"]
        with torch.no_grad():
            for key, val in weights.items():
                if ".lora_A." not in key:
                    continue
                name = key.removeprefix("base_model.model.").split(".lora_A.")[0]
                weight = mods[name].weight
                b = weights[key.replace(".lora_A.", ".lora_B.")]
                delta = (b.to(device, torch.float32) @ val.to(device, torch.float32)) * scale
                weight.add_(delta.to(weight.dtype))
    heads = Readouts(cfg.hidden_size, layers).to(device=device, dtype=torch.float32)
    if initial is not None:
        heads.load_state_dict(load_file(initial / "heads.safetensors"), strict=True)
    member = Member(trunk, heads, layers, pad)
    member.train(training)
    return member


def export_weights(member: Member, output: Path) -> None:
    member.trunk.save_pretrained(
        output / "adapter", safe_serialization=True, save_embedding_layers=False
    )
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in member.heads.state_dict().items()},
        output / "heads.safetensors",
    )


def tensors(
    encoded: list[list[int]], rows: list[int], pad: int, device: str
) -> tuple[Tensor, Tensor]:
    width = max(len(encoded[r]) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for position, row in enumerate(rows):
        tokens = encoded[row]
        ids[position, : len(tokens)] = torch.tensor(tokens)
        mask[position, : len(tokens)] = 1
    return ids.to(device), mask.to(device)


def center_features(z: Tensor, groups: list[int]) -> tuple[Tensor, list[int]]:
    positions: dict[int, list[int]] = {}
    for i, group in enumerate(groups):
        positions.setdefault(group, []).append(i)
    centered = torch.zeros_like(z)
    eligible = []
    for indices in positions.values():
        if len(indices) >= 2:
            block = z[indices]
            centered[indices] = block - block.mean(0)
            eligible.extend(indices)
    return centered, sorted(eligible)
