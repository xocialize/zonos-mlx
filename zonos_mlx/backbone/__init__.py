"""Backbone registry — mirrors upstream ``zonos/backbone/__init__.py``.

Upstream registers two backbones: ``mamba_ssm`` (hybrid checkpoint) and ``torch``
(pure-transformer checkpoint). This port targets the ``torch`` backbone only:
Zonos-v0.1-transformer has attn_layer_idx = [0..25] (all 26 layers attention), so
no Mamba2/SSM is needed. The hybrid backbone is deferred (see PREFLIGHT.md).
"""

BACKBONES = {}

from .transformer import TorchZonosBackbone  # noqa: E402

BACKBONES["torch"] = TorchZonosBackbone
