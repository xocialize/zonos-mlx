"""Gate C — DAC decode parity vs the PyTorch golden run.

Decodes the oracle's golden DAC codes through the MLX DAC and compares the waveform
(max-abs + SI-SDR) against the golden decoded waveform. CPU for true fp32.

Weights are sourced from the transformers DacModel state_dict (canonical keys),
so torch is needed here (dev-only [parity] extra).

Run:  pytest tests/parity/test_dac_decode_parity.py -s
"""

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from zonos_mlx.autoencoder import DAC

GOLDEN = Path(__file__).resolve().parents[2] / "goldens" / "golden_transformer.npz"


def _si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est, ref = est.flatten().astype(np.float64), ref.flatten().astype(np.float64)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    proj = alpha * ref
    noise = est - proj
    return float(10 * np.log10((np.dot(proj, proj) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("run scripts/capture_golden.py")
    return np.load(GOLDEN)


@pytest.fixture(scope="module")
def dac():
    torch = pytest.importorskip("torch")
    from transformers import DacModel

    sd = DacModel.from_pretrained("descript/dac_44khz").state_dict()
    weights = {k: mx.array(v.float().numpy()) for k, v in sd.items()}
    model = DAC()
    model.load_weights(list(DAC.sanitize(weights).items()), strict=True)
    model.eval()
    return model


def test_dac_decode_parity(golden, dac):
    codes = mx.array(golden["dac_codes"].astype(np.int32))
    wav = np.array(dac.decode(codes))
    gold = golden["dac_wav_decoded"]

    print(f"shape mlx={wav.shape} golden={gold.shape}")
    assert wav.shape == gold.shape, "decoded length mismatch (check ConvTranspose padding)"

    max_abs = float(np.max(np.abs(wav.astype(np.float64) - gold.astype(np.float64))))
    si_sdr = _si_sdr(wav, gold)
    print(f"DAC decode: max_abs={max_abs:.3e}  SI-SDR={si_sdr:.1f} dB")
    assert max_abs < 1e-3, f"waveform diverges: max_abs={max_abs:.3e}"
    assert si_sdr > 40.0, f"SI-SDR too low: {si_sdr:.1f} dB"
    print(f"\nGate C PASS — max_abs {max_abs:.3e}, SI-SDR {si_sdr:.1f} dB")
