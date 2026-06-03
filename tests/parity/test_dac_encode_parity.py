"""DAC encode parity — encode the golden waveform and require exact integer code match.

The oracle's golden dac_codes were produced by encoding dac_wav_input, so a correct
MLX encoder must reproduce them exactly (integer codes). CPU true-fp32.
"""

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from zonos_mlx.autoencoder import DAC

GOLDEN = Path(__file__).resolve().parents[2] / "goldens" / "golden_transformer.npz"


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("run scripts/capture_golden.py")
    return np.load(GOLDEN)


@pytest.fixture(scope="module")
def dac():
    from huggingface_hub import hf_hub_download

    w = mx.load(hf_hub_download("descript/dac_44khz", "model.safetensors"))
    model = DAC()
    model.load_weights(list(DAC.sanitize(w).items()), strict=True)
    model.eval()
    return model


def test_dac_encode_parity(golden, dac):
    wav = mx.array(golden["dac_wav_input"].astype(np.float32))  # (1, 1, N) already hop-aligned
    codes = np.array(dac.encode(wav))
    gold = golden["dac_codes"].astype(np.int64)

    print(f"codes mlx={codes.shape} golden={gold.shape}")
    assert codes.shape == gold.shape
    match = float((codes == gold).mean())
    print(f"DAC encode: exact code match = {match:.4%}")
    assert match == 1.0, f"codes differ ({match:.2%} match)"
    print("Gate (encode) PASS — exact integer code match")
