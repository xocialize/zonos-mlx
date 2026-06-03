"""Port of upstream ``zonos/sampling.py`` — AR token sampling.

RNG note: MLX mx.random is not seed-compatible with torch, so generated token
sequences will differ from the oracle even at matching settings (validated instead
via pre-sampling logits, Gate A). The filtering math is a faithful port.
"""

import mlx.core as mx


def apply_top_k(probs: mx.array, k: int) -> mx.array:
    k = min(k, probs.shape[-1])
    pivot = mx.sort(probs, axis=-1)[..., -k][..., None]
    probs = mx.where(probs < pivot, 0.0, probs)
    return probs / probs.sum(axis=-1, keepdims=True)


def apply_top_p(probs: mx.array, p: float) -> mx.array:
    idx = mx.argsort(probs, axis=-1)[..., ::-1]          # descending order indices
    probs_sort = mx.take_along_axis(probs, idx, axis=-1)
    probs_sum = mx.cumsum(probs_sort, axis=-1)
    mask = (probs_sum - probs_sort) > p
    probs_sort = probs_sort * (1.0 - mask.astype(probs.dtype))
    inv = mx.argsort(idx, axis=-1)                        # unsort
    probs = mx.take_along_axis(probs_sort, inv, axis=-1)
    return probs / probs.sum(axis=-1, keepdims=True)


def apply_min_p(probs: mx.array, min_p: float) -> mx.array:
    top = probs.max(axis=-1, keepdims=True)
    probs = mx.where(probs < (min_p * top), 0.0, probs)
    return probs / probs.sum(axis=-1, keepdims=True)


def modify_logit_for_repetition_penalty(logits, generated_tokens, penalty, window):
    """logits: (B, n_cb, V); generated_tokens: (B, n_cb, T). factors = penalty**count."""
    V = logits.shape[-1]
    gen = mx.clip(generated_tokens[..., -window:], 0, V - 1).astype(mx.int32)
    # count occurrences of each vocab id within the window (one-hot sum).
    onehot = (gen[..., None] == mx.arange(V).reshape(1, 1, 1, V)).astype(logits.dtype)
    counts = onehot.sum(axis=2)                           # (B, n_cb, V)
    factors = penalty**counts
    return mx.where(logits <= 0, logits * factors, logits / factors)


def _multinomial1(probs: mx.array) -> mx.array:
    """num_samples=1 via Gumbel-style argmax(probs / Exp(1)) — matches upstream form."""
    q = -mx.log(mx.random.uniform(shape=probs.shape) + 1e-20)  # Exponential(1)
    return mx.argmax(probs / q, axis=-1, keepdims=True).astype(mx.int32)


def sample_from_logits(
    logits: mx.array,
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    generated_tokens: mx.array | None = None,
    repetition_penalty: float = 3.0,
    repetition_penalty_window: int = 2,
) -> mx.array:
    if repetition_penalty != 1.0 and generated_tokens is not None:
        logits = modify_logit_for_repetition_penalty(
            logits, generated_tokens, repetition_penalty, repetition_penalty_window
        )
    if temperature > 0:
        probs = mx.softmax(logits / temperature, axis=-1)
        if top_p > 0:
            probs = apply_top_p(probs, top_p)
        if top_k > 0:
            probs = apply_top_k(probs, top_k)
        if min_p > 0:
            probs = apply_min_p(probs, min_p)
        return _multinomial1(probs)
    return mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)
