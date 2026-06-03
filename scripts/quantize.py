"""Phase 5 — quantize the AR-transformer Linears to 4-bit and validate.

Per the skill: quantize transformer Linears only; keep DAC, conditioner, embeddings,
heads (and speaker model) at higher precision. Validate by pre-sampling-logit cosine
vs the fp32 golden (int4 target ≈ 0.99+) and a listening sample.

Run:  python scripts/quantize.py
"""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf

from zonos_mlx.backbone.transformer import causal_mask
from zonos_mlx.conditioning import make_cond_dict
from zonos_mlx.model import Zonos

GROUP_SIZE, BITS = 64, 4


def logits_on_golden(model, golden) -> np.ndarray:
    h = mx.array(golden["backbone_input"].astype(np.float32))
    mask = causal_mask(h.shape[1], 0, h.dtype)
    for layer in model.backbone.layers:
        h = layer(h, mask=mask)
    norm_f = model.backbone.norm_f(h)[:, -1, :]
    logits = mx.stack([head(norm_f) for head in model.heads], axis=1)
    mx.eval(logits)
    return np.array(logits)[..., :1025].reshape(-1)


def main():
    golden = np.load("goldens/golden_transformer.npz")
    gold_logits = golden["logits_presample"][..., :1025].reshape(-1)

    from mlx.utils import tree_flatten

    def nbytes(m):
        return sum(v.nbytes for _, v in tree_flatten(m.parameters()))

    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", dtype=mx.float32)
    base = logits_on_golden(model, golden)
    cos_fp = float(np.dot(base, gold_logits) / (np.linalg.norm(base) * np.linalg.norm(gold_logits)))
    print(f"fp32 backbone logits cosine vs golden = {cos_fp:.6f}")
    bb_before = nbytes(model.backbone)

    # Quantize ONLY the backbone (its layers contain only Linears + LayerNorms).
    nn.quantize(model.backbone, group_size=GROUP_SIZE, bits=BITS)
    bb_after = nbytes(model.backbone)
    print(f"backbone size: {bb_before/1e6:.1f} MB (fp32) -> {bb_after/1e6:.1f} MB (int{BITS}), {bb_before/bb_after:.1f}x")
    q = logits_on_golden(model, golden)
    cos_q = float(np.dot(q, gold_logits) / (np.linalg.norm(q) * np.linalg.norm(gold_logits)))
    print(f"int{BITS} backbone logits cosine vs golden = {cos_q:.6f}")

    # Listening sample with the quantized model.
    mx.random.seed(0)
    cond = make_cond_dict(text="The quick brown fox jumps over the lazy dog.", speaker=mx.random.normal((1, 1, 128)))
    prefix = model.prepare_conditioning(cond)
    codes = model.generate(prefix, max_new_tokens=256, cfg_scale=2.0)
    wav = np.array(model.autoencoder.decode(codes))[0, 0]
    sf.write("outputs/sample_int4.wav", wav, model.autoencoder.sampling_rate)
    print(f"int4 sample: {len(wav)/44100:.2f}s rms={np.sqrt((wav**2).mean()):.4f} -> outputs/sample_int4.wav")
    print(f"\nQuantize {'PASS' if cos_q > 0.99 else 'CHECK'} — int{BITS} logit cosine {cos_q:.4f} (target >0.99)")


if __name__ == "__main__":
    main()
