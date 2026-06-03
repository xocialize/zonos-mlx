"""Codebook delay-pattern parity: MLX vs upstream torch on a fixed input."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from zonos_mlx.codebook_pattern import apply_delay_pattern, revert_delay_pattern

MASK = 1025


def test_delay_pattern_matches_torch():
    torch = pytest.importorskip("torch")
    from zonos.codebook_pattern import apply_delay_pattern as t_apply, revert_delay_pattern as t_revert

    rng = np.random.default_rng(0)
    codes = rng.integers(0, 1024, size=(2, 9, 20)).astype(np.int64)

    mx_delayed = np.array(apply_delay_pattern(mx.array(codes), MASK))
    t_delayed = t_apply(torch.tensor(codes), MASK).numpy()
    assert mx_delayed.shape == t_delayed.shape
    assert np.array_equal(mx_delayed, t_delayed), "apply_delay_pattern mismatch"

    mx_rev = np.array(revert_delay_pattern(mx.array(t_delayed)))
    t_rev = t_revert(torch.tensor(t_delayed)).numpy()
    assert np.array_equal(mx_rev, t_rev), "revert_delay_pattern mismatch"

    # round-trip recovers the original (within the valid window).
    assert np.array_equal(mx_rev, codes[..., : mx_rev.shape[-1]])
    print(f"delay pattern PASS — delayed {mx_delayed.shape}, reverted {mx_rev.shape}")
