"""Smoke test: ZonosConfig.from_dict round-trips the real checkpoint config and the
verified architecture invariants hold. Runs without weights, torch, or mlx-the-model
(only imports the pure-dataclass config module).
"""

from zonos_mlx.config import ZonosConfig

# Subset of Zyphra/Zonos-v0.1-transformer/config.json verified 2026-06-03.
TRANSFORMER_CONFIG = {
    "backbone": {
        "d_model": 2048,
        "d_intermediate": 0,
        "attn_mlp_d_intermediate": 8192,
        "n_layer": 26,
        "ssm_cfg": {},
        "attn_layer_idx": list(range(26)),
        "attn_cfg": {
            "causal": True,
            "num_heads": 16,
            "num_heads_kv": 4,
            "rotary_emb_dim": 128,
            "rotary_emb_interleaved": True,
            "qkv_proj_bias": False,
            "out_proj_bias": False,
        },
        "rms_norm": False,
        "residual_in_fp32": False,
        "norm_epsilon": 1e-5,
    },
    "prefix_conditioner": {
        "conditioners": [
            {"type": "EspeakPhonemeConditioner", "name": "espeak"},
            {"type": "PassthroughConditioner", "name": "speaker", "cond_dim": 128,
             "uncond_type": "learned", "projection": "linear"},
            {"type": "FourierConditioner", "name": "emotion", "input_dim": 8,
             "uncond_type": "learned"},
            {"type": "FourierConditioner", "name": "fmax", "min_val": 0, "max_val": 24000,
             "uncond_type": "learned"},
            {"type": "FourierConditioner", "name": "pitch_std", "min_val": 0, "max_val": 400,
             "uncond_type": "learned"},
            {"type": "FourierConditioner", "name": "speaking_rate", "min_val": 0, "max_val": 40,
             "uncond_type": "learned"},
            {"type": "IntegerConditioner", "name": "language_id", "min_val": -1, "max_val": 126,
             "uncond_type": "learned"},
        ],
        "projection": "linear",
    },
    "eos_token_id": 1024,
    "masked_token_id": 1025,
}


def test_config_from_dict():
    cfg = ZonosConfig.from_dict(TRANSFORMER_CONFIG)

    # Backbone invariants (the transformer checkpoint is pure-attention).
    assert cfg.backbone.n_layer == 26
    assert cfg.backbone.attn_layer_idx == list(range(26)), "all layers must be attention"
    assert cfg.backbone.d_model == 2048
    assert cfg.backbone.rms_norm is False, "uses LayerNorm, not RMSNorm"

    attn = cfg.backbone.attn_cfg
    assert attn["num_heads"] == 16 and attn["num_heads_kv"] == 4, "GQA 16/4"
    assert cfg.backbone.d_model // attn["num_heads"] == attn["rotary_emb_dim"], "head_dim==128"
    assert attn["rotary_emb_interleaved"] is True, "interleaved-RoPE trap"

    # Conditioner inventory + the headline 8-D emotion vector.
    names = [c["name"] for c in cfg.prefix_conditioner.conditioners]
    assert names == ["espeak", "speaker", "emotion", "fmax", "pitch_std",
                     "speaking_rate", "language_id"]
    emotion = next(c for c in cfg.prefix_conditioner.conditioners if c["name"] == "emotion")
    assert emotion["input_dim"] == 8

    # Resolved-config default not present in config.json.
    assert cfg.pad_vocab_to_multiple_of == 8
    assert cfg.eos_token_id == 1024 and cfg.masked_token_id == 1025
