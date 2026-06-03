"""Port of upstream ``zonos/codebook_pattern.py`` — DAC multi-codebook delay pattern.

1:1 with upstream: torch.roll → mx.roll, F.pad → mx.pad. Staggers the n_q DAC
codebooks so the AR transformer predicts each with the correct per-codebook offset.
"""

import mlx.core as mx


def apply_delay_pattern(codes: mx.array, mask_token: int) -> mx.array:
    n_q = codes.shape[1]
    codes = mx.pad(codes, [(0, 0), (0, 0), (0, n_q)], constant_values=mask_token)
    return mx.stack([mx.roll(codes[:, k], k + 1, axis=-1) for k in range(n_q)], axis=1)


def revert_delay_pattern(codes: mx.array) -> mx.array:
    _, n_q, seq_len = codes.shape
    return mx.stack([codes[:, k, k + 1 : seq_len - n_q + k + 1] for k in range(n_q)], axis=1)
