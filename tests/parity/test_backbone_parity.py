"""Gate A — backbone parity vs the PyTorch golden run.

Feeds the oracle's golden ``backbone_input`` through the MLX TorchZonosBackbone and
asserts per-layer hidden states, final norm_f, and pre-sampling logits match the
PyTorch reference (goldens/golden_transformer.npz). Both sides fp32, so thresholds
are tight. Needs only mlx + numpy + huggingface_hub (NO torch) + the cached weights.

Run:  pytest tests/parity/test_backbone_parity.py -s
"""

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

# Run on CPU: MLX's Apple-GPU fp32 matmul uses tf32-like reduced precision
# (~3.8e-3 per matmul, ~6e-2 accumulated over 26 layers) — a platform numerics
# trait, NOT a port bug (verified: MLX-CPU matmul matches the fp32 oracle bitwise).
# CPU gives true fp32 so this gate isolates "is the math correct". GPU/bf16
# numerics are validated separately at the bf16 e2e stage.
mx.set_default_device(mx.cpu)

from zonos_mlx.backbone.transformer import TorchZonosBackbone, causal_mask
from zonos_mlx.config import ZonosConfig

GOLDEN = Path(__file__).resolve().parents[2] / "goldens" / "golden_transformer.npz"
REPO_ID = "Zyphra/Zonos-v0.1-transformer"

# RELATIVE thresholds: this model has massive activations (a channel hits ~16k at
# layer 5+), so raw max_abs is misleading — true-fp32 rel error is ~7e-7 everywhere.
# Gate on max_abs / golden_max_abs instead (skill lesson: rel error for massive-act models).
TOL_LAYER = 1e-5
TOL_NORMF = 1e-5
TOL_LOGITS = 1e-5


def _rel_max(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9))


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"missing {GOLDEN} — run scripts/capture_golden.py")
    return np.load(GOLDEN)


@pytest.fixture(scope="module")
def backbone_and_weights():
    from huggingface_hub import hf_hub_download

    config = ZonosConfig.from_dict(json.load(open(hf_hub_download(REPO_ID, "config.json"))))
    weights = mx.load(hf_hub_download(REPO_ID, "model.safetensors"))

    backbone = TorchZonosBackbone(config.backbone)
    bb = {k: v.astype(mx.float32) for k, v in TorchZonosBackbone.sanitize(weights).items()}
    backbone.load_weights(list(bb.items()), strict=True)
    backbone.eval()

    heads = [weights[f"heads.{i}.weight"].astype(mx.float32) for i in range(9)]
    return backbone, heads


def test_backbone_parity(golden, backbone_and_weights):
    backbone, heads = backbone_and_weights

    h = mx.array(golden["backbone_input"].astype(np.float32))
    seqlen = h.shape[1]
    mask = causal_mask(seqlen, 0, h.dtype)

    # Per-layer hidden states.
    worst_layer = 0.0
    for i, layer in enumerate(backbone.layers):
        h = layer(h, mask=mask)
        mx.eval(h)
        d = _rel_max(np.array(h), golden[f"backbone_layer_{i:02d}"])
        worst_layer = max(worst_layer, d)
        print(f"layer {i:02d}: rel={d:.3e}")
        assert d < TOL_LAYER, f"layer {i} diverges: rel={d:.3e}"

    # Final norm.
    norm_f = backbone.norm_f(h)
    mx.eval(norm_f)
    d_norm = _rel_max(np.array(norm_f), golden["backbone_norm_f"])
    print(f"norm_f: rel={d_norm:.3e}")
    assert d_norm < TOL_NORMF

    # Pre-sampling logits (last position), compare real vocab [:1025].
    last = norm_f[:, -1, :]                                  # (1, d)
    logits = mx.stack([last @ w.T for w in heads], axis=1)   # (1, 9, 1025)
    mx.eval(logits)
    gold_logits = golden["logits_presample"][..., :1025]
    d_logits = _rel_max(np.array(logits), gold_logits)
    print(f"logits[:1025]: rel={d_logits:.3e}")
    assert d_logits < TOL_LOGITS

    print(f"\nGate A PASS — worst layer rel {worst_layer:.3e}, norm_f {d_norm:.3e}, logits {d_logits:.3e}")
