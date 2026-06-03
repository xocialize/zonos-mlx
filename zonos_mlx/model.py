"""Port of upstream ``zonos/model.py`` — the top-level ``Zonos`` module.

Wires PrefixConditioner → backbone (AR transformer, KV-cached) → per-codebook heads
(with delay pattern) → sampling → DAC decode. CUDA-graph/torch.compile paths are
dropped (we always take the eager path). Delayed-codes bookkeeping is done on the
host (numpy ints) since it's sequential and integer-only; heavy compute stays in MLX.
"""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .autoencoder import DAC
from .backbone.transformer import KVCache, TorchZonosBackbone
from .codebook_pattern import apply_delay_pattern, revert_delay_pattern
from .conditioning import PrefixConditioner
from .config import ZonosConfig
from .sampling import sample_from_logits

DEFAULT_SAMPLING = dict(min_p=0.1)


class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        self.autoencoder = DAC()
        self.backbone = TorchZonosBackbone(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        n_cb = self.autoencoder.num_codebooks
        self.embeddings = [nn.Embedding(1026, dim) for _ in range(n_cb)]
        self.heads = [nn.Linear(dim, 1025, bias=False) for _ in range(n_cb)]

    # ------------------------------------------------------------------ core
    def embed_codes(self, codes: mx.array) -> mx.array:
        return sum(emb(codes[:, i]) for i, emb in enumerate(self.embeddings))

    def apply_heads(self, hidden_states: mx.array) -> mx.array:
        return mx.stack([head(hidden_states) for head in self.heads], axis=1)

    def _compute_logits(self, hidden_states, cache, cfg_scale: float) -> mx.array:
        last = self.backbone(hidden_states, cache)[:, -1:, :]      # (B, 1, d)
        logits = self.apply_heads(last).squeeze(2).astype(mx.float32)  # (B, n_cb, V)
        if cfg_scale != 1.0:
            cond, uncond = logits[: logits.shape[0] // 2], logits[logits.shape[0] // 2 :]
            logits = uncond + (cond - uncond) * cfg_scale
        return logits

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None) -> mx.array:
        if uncond_dict is None:
            uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}
        return mx.concatenate([self.prefix_conditioner(cond_dict), self.prefix_conditioner(uncond_dict)], axis=0)

    # ------------------------------------------------------------------ generate
    def generate(
        self,
        prefix_conditioning: mx.array,   # (2, L, d) cond+uncond
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        sampling_params: dict = DEFAULT_SAMPLING,
        progress: bool = False,
    ) -> mx.array:
        assert cfg_scale != 1.0, "cfg_scale==1 not supported"
        n_cb = self.autoencoder.num_codebooks
        unknown = -1
        audio_seq_len = max_new_tokens
        cache = [KVCache() for _ in self.backbone.layers]

        codes = np.full((1, n_cb, audio_seq_len), unknown, dtype=np.int64)
        delayed = np.array(apply_delay_pattern(mx.array(codes), self.masked_token_id))  # (1,n_cb,audio+n_cb)

        # prefill on prefix + first (delayed) audio frame.
        first = mx.array(delayed[..., :1])
        hidden = mx.concatenate([prefix_conditioning, self.embed_codes(mx.broadcast_to(first, (2, n_cb, 1)))], axis=1)
        logits = self._compute_logits(hidden, cache, cfg_scale)
        next_token = np.array(sample_from_logits(logits, **sampling_params))  # (1,n_cb,1)

        offset = 1
        frame = delayed[..., offset : offset + 1]
        frame[frame == unknown] = next_token[frame == unknown]

        # codebook 1..n-1 may never emit EOS.
        eos = self.eos_token_id
        max_steps = delayed.shape[2] - offset
        remaining = np.full((1,), max_steps)
        stopping = np.zeros((1,), dtype=bool)

        step = 0
        while remaining.max() > 0 and offset < delayed.shape[2] - 1:
            offset += 1
            input_ids = mx.array(delayed[..., offset - 1 : offset])
            hidden = self.embed_codes(mx.broadcast_to(input_ids, (2, n_cb, 1)))
            logits = self._compute_logits(hidden, cache, cfg_scale)
            # bias: only codebook 0 may predict EOS
            bias = np.zeros((1, n_cb, logits.shape[-1]), np.float32)
            bias[:, 1:, eos] = -np.inf
            logits = logits + mx.array(bias)

            gen = mx.array(delayed[..., :offset])
            next_token = np.array(sample_from_logits(logits, generated_tokens=gen, **sampling_params))

            eos_in_cb0 = next_token[:, 0, 0] == eos
            remaining[eos_in_cb0] = np.minimum(remaining[eos_in_cb0], 9)
            stopping |= eos_in_cb0
            eos_cb_idx = np.clip(9 - remaining, a_min=None, a_max=n_cb - 1)
            for i in range(next_token.shape[0]):
                if stopping[i]:
                    idx = int(eos_cb_idx[i])
                    next_token[i, :idx, 0] = self.masked_token_id
                    next_token[i, idx, 0] = eos

            frame = delayed[..., offset : offset + 1]
            frame[frame == unknown] = next_token[frame == unknown]
            remaining -= 1
            step += 1
            if progress and step % 50 == 0:
                print(f"  step {step}/{max_steps}")

        out = revert_delay_pattern(mx.array(delayed))
        out = mx.where(out >= 1024, 0, out)
        out = out[..., : max(0, offset - n_cb)]
        return out  # (1, n_cb, T)

    # ------------------------------------------------------------------ I/O
    # DAC is a separate checkpoint (upstream loads descript/dac_44khz independently).
    DAC_REPO = "descript/dac_44khz"

    @staticmethod
    def sanitize(zonos_weights: dict, dac_weights: dict) -> dict:
        """Combined weight dict matching the (decode-only) model tree.

        Zonos body keys (backbone./prefix_conditioner./embeddings./heads.) already match
        param names 1:1. DAC body comes from the separate descript/dac_44khz checkpoint,
        remapped to autoencoder.* with conv transposes via DAC.sanitize (encoder excluded
        until encode() lands — end deliverable).
        """
        out = {k: v for k, v in zonos_weights.items() if not k.startswith("autoencoder.")}
        for k, v in DAC.sanitize(dac_weights).items():  # decoder.* / quantizer.*
            out["autoencoder." + k] = v
        return out

    @classmethod
    def from_pretrained(cls, repo_id: str, dtype=mx.float32):
        from huggingface_hub import hf_hub_download

        config = ZonosConfig.from_dict(json.load(open(hf_hub_download(repo_id, "config.json"))))
        zonos_w = mx.load(hf_hub_download(repo_id, "model.safetensors"))
        dac_w = mx.load(hf_hub_download(cls.DAC_REPO, "model.safetensors"))
        model = cls(config)
        w = {k: v.astype(dtype) for k, v in cls.sanitize(zonos_w, dac_w).items()}
        model.load_weights(list(w.items()), strict=True)
        model.eval()
        return model

    @classmethod
    def from_mlx(cls, path: str, dtype=mx.bfloat16):
        """Load a pre-converted MLX repo (model.safetensors already in our param layout)."""
        import os

        def get(fn):
            if os.path.isdir(path):
                return os.path.join(path, fn)
            from huggingface_hub import hf_hub_download

            return hf_hub_download(path, fn)

        config = ZonosConfig.from_dict(json.load(open(get("config.json"))))
        w = {k: v.astype(dtype) for k, v in mx.load(get("model.safetensors")).items()}
        model = cls(config)
        model.load_weights(list(w.items()), strict=True)
        model.eval()
        return model
