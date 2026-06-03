"""Port of upstream ``zonos/config.py`` (Zyphra/Zonos).

Faithful 1:1 mirror of the upstream dataclasses so a reader can diff this against
the reference. The only deviation: the upstream ``InferenceParams`` dataclass is
Mamba-generation machinery typed against ``torch.Tensor`` and is not used on the
``torch`` (pure-transformer) backbone path we are porting; KV-cache state is held
MLX-side instead. It is intentionally omitted here.

Verified against Zyphra/Zonos-v0.1-transformer/config.json on 2026-06-03
(see ../_research/PREFLIGHT.md).
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BackboneConfig:
    d_model: int = 1024
    d_intermediate: int = 0
    attn_mlp_d_intermediate: int = 0
    n_layer: int = 16
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = False
    residual_in_fp32: bool = False
    norm_epsilon: float = 1e-5


@dataclass
class PrefixConditionerConfig:
    conditioners: list[dict]
    projection: Literal["none", "linear", "mlp"]


@dataclass
class ZonosConfig:
    backbone: BackboneConfig
    prefix_conditioner: PrefixConditionerConfig
    eos_token_id: int = 1024
    masked_token_id: int = 1025
    # NOTE (resolved-config trap): not present in config.json; the upstream class
    # injects this default. The vocab is padded to a multiple of 8.
    pad_vocab_to_multiple_of: int = 8

    @classmethod
    def from_dict(cls, d: dict) -> "ZonosConfig":
        d = d.copy()
        backbone_config = BackboneConfig(**d.pop("backbone"))
        prefix_conditioner_config = PrefixConditionerConfig(**d.pop("prefix_conditioner"))
        config = cls(backbone_config, prefix_conditioner_config, **d)
        return config
