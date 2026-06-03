"""Speaker-embedding (ResNet293 + LDA) parity vs the PyTorch golden. CPU true-fp32.

Needs weights/zonos_speaker/*.safetensors + goldens/golden_speaker.npz
(scripts/capture_speaker_golden.py). No torch.
"""

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from zonos_mlx.speaker_cloning import SpeakerEmbeddingLDA

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "goldens" / "golden_speaker.npz"
WDIR = ROOT / "weights" / "zonos_speaker"
TOL = 1e-4  # relative; deep BN/conv stack accumulates a bit more than a single layer


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9))


@pytest.fixture(scope="module")
def spk():
    if not GOLDEN.exists() or not (WDIR / "resnet293.safetensors").exists():
        pytest.skip("run scripts/capture_speaker_golden.py")
    return SpeakerEmbeddingLDA.from_local(
        str(WDIR / "resnet293.safetensors"), str(WDIR / "lda.safetensors"), str(WDIR / "featcal.safetensors")
    )


def test_speaker_parity(spk):
    g = np.load(GOLDEN)
    emb, lda = spk(mx.array(g["wav_input"].astype(np.float32)))
    mx.eval(emb, lda)

    d_emb = _rel(np.array(emb), g["emb_256"])
    d_lda = _rel(np.array(lda), g["lda_128"])
    print(f"emb_256 rel={d_emb:.3e}  lda_128 rel={d_lda:.3e}")
    assert d_emb < TOL, f"speaker emb diverges: {d_emb:.3e}"
    assert d_lda < TOL, f"LDA diverges: {d_lda:.3e}"
    print(f"\nSpeaker PASS — emb {d_emb:.3e}, lda {d_lda:.3e}")
