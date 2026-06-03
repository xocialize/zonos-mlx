"""End-to-end smoke test: text → audio runs and produces a sane waveform.

Functional check only (not a quality/parity gate — generation RNG differs from torch).
Asserts the pipeline runs, output is finite, non-silent, and ~the right length.
Needs the cached weights; skipped if unavailable.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


@pytest.fixture(scope="module")
def pipeline():
    pytest.importorskip("huggingface_hub")
    from zonos_mlx.pipeline_mlx import ZonosPipeline

    try:
        return ZonosPipeline.from_pretrained()
    except Exception as e:  # offline / weights missing
        pytest.skip(f"weights unavailable: {e}")


def test_e2e_generates_audio(pipeline):
    mx.random.seed(0)
    speaker = mx.random.normal((1, 1, 128))
    max_new = 80
    wav = pipeline.generate("Hello world, this is a test.", speaker=speaker, max_new_tokens=max_new)
    mx.eval(wav)
    w = np.array(wav)

    assert w.shape[0] == 1 and w.ndim == 2
    assert np.isfinite(w).all(), "non-finite samples"
    # length ≤ requested frames * hop; > 0
    assert 0 < w.shape[1] <= max_new * pipeline.model.autoencoder.hop_length
    rms = float(np.sqrt((w**2).mean()))
    assert rms > 1e-3, f"output is silent (rms={rms:.2e})"
    assert np.abs(w).max() <= 1.0 + 1e-4, "waveform exceeds tanh range"
    print(f"e2e PASS — wav {w.shape}, rms={rms:.4f}, peak={np.abs(w).max():.4f}")
