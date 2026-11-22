"""Load the author's local checkpoint and reproduce the plain classifier readout."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ntl_baseline.constants import ContractError

from .branch import FEATURE_SIZE, HIDDEN_SIZE, POOL_LAYERS, Array, encode_passages


class CudaFeatures:
    def __init__(self, base: Path, checkpoint: Path) -> None:
        self.device = torch.device("cuda:0")
        config = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
        keep = POOL_LAYERS[-1]
        if (
            config.model_type != "qwen3"
            or config.hidden_size != HIDDEN_SIZE
            or config.num_hidden_layers < keep
            or getattr(config, "quantization_config", None) is not None
        ):
            raise ContractError("expected an unquantized Qwen3 trunk with 4096 hidden units")
        config.num_hidden_layers = keep
        config.use_cache = False
        if getattr(config, "layer_types", None) is not None:
            config.layer_types = config.layer_types[:keep]
        trunk, info = AutoModel.from_pretrained(
            base,
            config=config,
            dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="sdpa",
            use_safetensors=True,
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise ContractError("base checkpoint did not load all truncated trunk weights")
        self.model = PeftModel.from_pretrained(
            trunk, checkpoint / "adapter", local_files_only=True
        ).eval()
        with safe_open(
            checkpoint / "adapter" / "adapter_model.safetensors", framework="pt"
        ) as weights:
            loaded_keys = get_peft_model_state_dict(self.model, save_embedding_layers=False)
            if set(weights.keys()) != set(loaded_keys):
                raise ContractError("adapter tensor keys differ from the loaded LoRA modules")

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.eos_token_id is None:
            raise ContractError("checkpoint tokenizer is missing EOS")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        state = load_file(checkpoint / "heads.safetensors")
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(HIDDEN_SIZE) for _ in POOL_LAYERS)
        self.norms.load_state_dict(
            {
                key.removeprefix("norms."): value
                for key, value in state.items()
                if key.startswith("norms.")
            }
        )
        self.norms.to(device=self.device, dtype=torch.float32).eval()

    @torch.no_grad()
    def __call__(self, texts: list[str], batch_size: int) -> Array:
        ids = encode_passages(self.tokenizer, texts, self.tokenizer.eos_token_id)
        order = np.argsort([len(tokens) for tokens in ids], kind="stable")
        features = np.empty((len(texts), FEATURE_SIZE), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            batch = self.tokenizer.pad(
                {"input_ids": [ids[row] for row in rows]}, return_tensors="pt"
            ).to(self.device)
            features[rows] = self._pool(batch)
        return features

    def _pool(self, batch: Any) -> Array:
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = self.model(**batch, output_hidden_states=True, use_cache=False).hidden_states
        mask = batch["attention_mask"].unsqueeze(-1).float()
        mask[:, 0] = 0
        counts = mask.sum(dim=1)
        pooled = [
            norm((hidden[layer].float() * mask).sum(dim=1) / counts)
            for layer, norm in zip(POOL_LAYERS, self.norms, strict=True)
        ]
        return torch.cat(pooled, dim=1).cpu().numpy()
