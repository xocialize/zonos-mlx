"""Port of upstream ``zonos/backbone/_torch.py`` — ``TorchZonosBackbone``.

1:1 with upstream (skill isomorphic-structure rule). PyTorch ops → MLX:
  - F.scaled_dot_product_attention(enable_gqa=True) → mx.fast.scaled_dot_product_attention
    (native GQA: 16 q-heads share 4 kv-heads).
  - gpt-fast interleaved rotary (apply_rotary_emb pairs adjacent dims) → mx.fast.rope(
    traditional=True, base=10000) — verified against the golden per-layer states.
  - nn.LayerNorm(eps) identical. Linear weights are (out,in) on both sides → no transpose.

Config (Zonos-v0.1-transformer): d_model=2048, n_layer=26, num_heads=16, num_heads_kv=4,
head_dim=128 (=rotary_emb_dim), attn_mlp_d_intermediate=8192, LayerNorm eps=1e-5, no biases.
"""

import mlx.core as mx
import mlx.nn as nn

from ..config import BackboneConfig

# gpt-fast precompute_freqs_cis default base (upstream _torch.precompute_freqs_cis).
ROPE_BASE = 10000.0


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = ROPE_BASE, offset: int = 0) -> mx.array:
    """Mirror of upstream gpt-fast precompute_freqs_cis → (seq_len, n_elem//2, 2) [cos, sin]."""
    freqs = 1.0 / (base ** (mx.arange(0, n_elem, 2)[: n_elem // 2].astype(mx.float32) / n_elem))
    t = mx.arange(offset, offset + seq_len).astype(mx.float32)
    freqs = mx.outer(t, freqs)
    return mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Interleaved (gpt-fast) rotary. x: (B, H, S, Dh); freqs_cis: (S, Dh//2, 2).

    Matches upstream exactly: pairs adjacent dims (x0,x1) and rotates
    (x0*cos - x1*sin, x1*cos + x0*sin). mx.fast.rope(traditional=True) was ~6e-2 off.
    """
    b, h, s, dh = x.shape
    xshaped = x.astype(mx.float32).reshape(b, h, s, dh // 2, 2)
    cos = freqs_cis[..., 0].reshape(1, 1, s, dh // 2)
    sin = freqs_cis[..., 1].reshape(1, 1, s, dh // 2)
    x0 = xshaped[..., 0]
    x1 = xshaped[..., 1]
    out = mx.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], axis=-1)
    return out.reshape(b, h, s, dh).astype(x.dtype)


class KVCache:
    """Minimal incremental K/V cache for the AR decode loop. k/v: (B, n_kv, S, Dh)."""

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, k: mx.array, v: mx.array):
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        self.offset = self.keys.shape[2]
        return self.keys, self.values


def causal_mask(seq_len: int, offset: int = 0, dtype=mx.float32) -> mx.array:
    """Additive causal mask, query positions [offset, offset+seq_len)."""
    q_idx = mx.arange(offset, offset + seq_len).reshape(-1, 1)
    k_idx = mx.arange(offset + seq_len).reshape(1, -1)
    return mx.where(k_idx <= q_idx, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))


class Attention(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.attn_cfg["num_heads"]
        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // self.num_heads
        self.layer_idx = layer_idx
        self.rotary_emb_dim = config.attn_cfg["rotary_emb_dim"]
        self.scale = self.head_dim**-0.5

        total_head_dim = (self.num_heads + 2 * self.num_heads_kv) * self.head_dim
        self.in_proj = nn.Linear(config.d_model, total_head_dim, bias=config.attn_cfg.get("qkv_proj_bias", False))
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=config.attn_cfg.get("out_proj_bias", False))

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        batch_size, seqlen, _ = x.shape

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_heads_kv * self.head_dim
        qkv = self.in_proj(x)
        q = qkv[..., :q_size]
        k = qkv[..., q_size : q_size + kv_size]
        v = qkv[..., q_size + kv_size :]

        # (B, S, H, Dh) -> (B, H, S, Dh)
        q = q.reshape(batch_size, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seqlen, self.num_heads_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seqlen, self.num_heads_kv, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        freqs_cis = precompute_freqs_cis(seqlen, self.rotary_emb_dim, ROPE_BASE, offset)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seqlen, q_size)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 2 * config.attn_mlp_d_intermediate, bias=False)
        self.fc2 = nn.Linear(config.attn_mlp_d_intermediate, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        y, gate = mx.split(self.fc1(x), 2, axis=-1)
        return self.fc2(y * nn.silu(gate))


class TransformerBlock(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mixer = Attention(config, layer_idx)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mlp = FeedForward(config)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        x = x + self.mixer(self.norm(x), mask=mask, cache=cache)
        x = x + self.mlp(self.norm2(x))
        return x


class TorchZonosBackbone(nn.Module):
    supported_architectures = ["transformer"]

    def __init__(self, config: BackboneConfig):
        super().__init__()
        assert not config.ssm_cfg, "This backbone implementation only supports the Transformer model."
        self.config = config
        self.layers = [TransformerBlock(config, i) for i in range(config.n_layer)]
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def __call__(self, hidden_states: mx.array, cache=None) -> mx.array:
        seqlen = hidden_states.shape[1]
        if cache is None:
            cache = [None] * len(self.layers)
        offset = cache[0].offset if cache[0] is not None else 0
        mask = causal_mask(seqlen, offset, hidden_states.dtype) if seqlen > 1 else None
        for layer, c in zip(self.layers, cache):
            hidden_states = layer(hidden_states, mask=mask, cache=c)
        return self.norm_f(hidden_states)

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Strip the top-level ``backbone.`` prefix; Linear/LayerNorm need no transpose."""
        out = {}
        for k, v in weights.items():
            if k.startswith("backbone."):
                out[k[len("backbone.") :]] = v
        return out
