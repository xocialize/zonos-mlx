"""Conditioning-stack parity vs the PyTorch golden run.

Builds the cond_dict with the golden's fixed speaker embedding + make_cond_dict
defaults, runs the MLX PrefixConditioner, and diffs prefix_cond / prefix_uncond
against the oracle. Matching prefix_cond also confirms espeak G2P + tokenization
parity (wrong phoneme IDs would change the sequence). CPU for true fp32.

Run:  pytest tests/parity/test_conditioning_parity.py -s
"""

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from zonos_mlx.conditioning import PrefixConditioner, make_cond_dict
from zonos_mlx.config import ZonosConfig

GOLDEN = Path(__file__).resolve().parents[2] / "goldens" / "golden_transformer.npz"
REPO_ID = "Zyphra/Zonos-v0.1-transformer"
TOL = 1e-5  # relative


def _rel_max(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9))


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("run scripts/capture_golden.py")
    return np.load(GOLDEN)


@pytest.fixture(scope="module")
def prefix_conditioner():
    from huggingface_hub import hf_hub_download

    config = ZonosConfig.from_dict(json.load(open(hf_hub_download(REPO_ID, "config.json"))))
    weights = mx.load(hf_hub_download(REPO_ID, "model.safetensors"))
    pc = PrefixConditioner(config.prefix_conditioner, config.backbone.d_model)
    pref = "prefix_conditioner."
    w = {k[len(pref):]: v.astype(mx.float32) for k, v in weights.items() if k.startswith(pref)}
    pc.load_weights(list(w.items()), strict=True)
    pc.eval()
    return pc


def test_conditioning_parity(golden, prefix_conditioner):
    speaker = mx.array(golden["speaker_input"].astype(np.float32))
    cond_dict = make_cond_dict(speaker=speaker)  # text/lang/emotion/prosody defaults match the golden

    prefix = prefix_conditioner(cond_dict)
    mx.eval(prefix)
    d_cond = _rel_max(np.array(prefix), golden["prefix_cond"])
    print(f"prefix_cond:   rel={d_cond:.3e}  shape={prefix.shape}")
    assert prefix.shape == golden["prefix_cond"].shape
    assert d_cond < TOL

    uncond_dict = {k: cond_dict[k] for k in prefix_conditioner.required_keys}
    prefix_u = prefix_conditioner(uncond_dict)
    mx.eval(prefix_u)
    d_uncond = _rel_max(np.array(prefix_u), golden["prefix_uncond"])
    print(f"prefix_uncond: rel={d_uncond:.3e}  shape={prefix_u.shape}")
    assert d_uncond < TOL

    print(f"\nConditioning PASS — cond {d_cond:.3e}, uncond {d_uncond:.3e}")
