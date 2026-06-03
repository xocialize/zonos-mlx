"""Capture a deterministic PyTorch golden run for MLX parity (CONFIRM-5 oracle).

Loads upstream Zonos (Zyphra/Zonos-v0.1-transformer) in fp32 on CPU and saves the
intermediate tensors the MLX port will diff against:

  - prefix_conditioning  (conditioning stack output, cond + uncond)
  - backbone per-layer hidden states (26)  + final norm_f  -> Gate A
  - pre-sampling logits  (apply_heads)                     -> Gate A
  - DAC encode->decode round-trip on a fixed waveform      -> Gate C (autoencoder)
  - full state_dict key inventory                          -> sanitize() target

Everything is seeded and uses a FIXED speaker embedding (random speaker model
deferred to its own gate), so the run is reproducible. Outputs -> goldens/.

Run:  PYTHONPATH=refs/Zonos python scripts/capture_golden.py
(espeak-ng must be installed; we point phonemizer at the brew dylib below.)
"""

import json
import os
from pathlib import Path

# Point phonemizer at the Homebrew espeak-ng before importing zonos.conditioning.
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")

import numpy as np
import torch

REPO_ID = "Zyphra/Zonos-v0.1-transformer"
SEED = 1234
TEXT = "It would be nice to have time for testing, indeed."
LANGUAGE = "en-us"
OUT = Path(__file__).resolve().parent.parent / "goldens"
OUT.mkdir(exist_ok=True)


def stats(t: torch.Tensor) -> dict:
    f = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "max_abs": float(f.abs().max()),
        "mean": float(f.mean()),
        "std": float(f.std()),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict

    print(f"Loading {REPO_ID} on CPU ...")
    model = Zonos.from_pretrained(REPO_ID, device="cpu")
    model = model.float().eval()  # fp32 oracle (loaded as bf16 by from_local)

    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, object] = {
        "repo_id": REPO_ID, "seed": SEED, "text": TEXT, "language": LANGUAGE,
        "dtype": "float32", "device": "cpu", "tensors": {},
    }

    def save(name: str, t: torch.Tensor):
        a = t.detach().float().cpu().numpy()
        arrays[name] = a
        manifest["tensors"][name] = stats(t)
        print(f"  saved {name:32s} {list(t.shape)} {t.dtype}")

    # ---- 1. Key inventory (sanitize() target) -------------------------------
    keys = sorted(model.state_dict().keys())
    (OUT / "key_inventory.txt").write_text("\n".join(keys) + "\n")
    print(f"key inventory: {len(keys)} keys -> goldens/key_inventory.txt")

    # ---- 2. Conditioning ----------------------------------------------------
    # Fixed speaker embedding (cond_dim=128) instead of the random speaker model.
    speaker = torch.randn(1, 1, 128, generator=torch.Generator().manual_seed(SEED))
    cond_dict = make_cond_dict(text=TEXT, language=LANGUAGE, speaker=speaker, device="cpu")
    manifest["emotion"] = cond_dict["emotion"].flatten().tolist()

    with torch.inference_mode():
        prefix_cond = model.prefix_conditioner(cond_dict)            # [1, L, d]
        uncond_dict = {k: cond_dict[k] for k in model.prefix_conditioner.required_keys}
        prefix_uncond = model.prefix_conditioner(uncond_dict)        # [1, L, d]
    save("prefix_cond", prefix_cond)
    save("prefix_uncond", prefix_uncond)
    save("speaker_input", speaker)

    # ---- 3. Backbone (Gate A): per-layer hidden states + logits -------------
    # Build a fixed audio-code prefix and run the full backbone, batch=1.
    n_cb = model.autoencoder.num_codebooks
    T_audio = 16
    g = torch.Generator().manual_seed(SEED + 1)
    codes = torch.randint(0, 1024, (1, n_cb, T_audio), generator=g)
    save("audio_codes_input", codes)

    captured: dict[str, torch.Tensor] = {}
    hooks = []
    for i, layer in enumerate(model.backbone.layers):
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out, i=i: captured.__setitem__(f"layer_{i:02d}", out)))
    hooks.append(model.backbone.norm_f.register_forward_hook(
        lambda m, inp, out: captured.__setitem__("norm_f", out)))
    # Layer-0 sub-step internals (for localizing parity divergence).
    l0 = model.backbone.layers[0]
    hooks.append(l0.norm.register_forward_hook(lambda m, i, o: captured.__setitem__("l0_norm", o)))
    hooks.append(l0.mixer.register_forward_hook(lambda m, i, o: captured.__setitem__("l0_mixer", o)))
    hooks.append(l0.norm2.register_forward_hook(lambda m, i, o: captured.__setitem__("l0_norm2", o)))
    hooks.append(l0.mlp.register_forward_hook(lambda m, i, o: captured.__setitem__("l0_mlp", o)))
    hooks.append(l0.mixer.in_proj.register_forward_hook(lambda m, i, o: captured.__setitem__("l0_inproj", o)))
    hooks.append(l0.mixer.out_proj.register_forward_pre_hook(lambda m, i: captured.__setitem__("l0_attn_y", i[0])))

    with torch.inference_mode():
        audio_emb = model.embed_codes(codes)                          # [1, T_audio, d]
        hidden_states = torch.cat([prefix_cond, audio_emb], dim=1)    # [1, L+T, d]
        inference_params = model.setup_cache(
            batch_size=1, max_seqlen=hidden_states.shape[1] + 16, dtype=torch.float32)
        backbone_out = model.backbone(hidden_states, inference_params)
        logits = model.apply_heads(backbone_out[:, -1:, :]).squeeze(2).float()  # [1, n_cb, 1025+pad]
    for h in hooks:
        h.remove()

    save("backbone_input", hidden_states)
    for name in sorted(captured):
        save(f"backbone_{name}", captured[name])
    save("logits_presample", logits)

    # ---- 4. DAC round-trip (Gate C) -----------------------------------------
    gw = torch.Generator().manual_seed(SEED + 2)
    wav = 0.1 * torch.randn(1, 1, 44_100, generator=gw)               # 1s @ 44.1kHz
    with torch.inference_mode():
        wav_pp = model.autoencoder.preprocess(wav, 44_100)
        dac_codes = model.autoencoder.encode(wav_pp)
        dac_wav = model.autoencoder.decode(dac_codes)
    save("dac_wav_input", wav_pp)
    save("dac_codes", dac_codes.float())
    save("dac_wav_decoded", dac_wav)

    # ---- write ---------------------------------------------------------------
    np.savez(OUT / "golden_transformer.npz", **arrays)
    (OUT / "golden_transformer.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(arrays)} tensors -> goldens/golden_transformer.npz")
    print("Manifest -> goldens/golden_transformer.manifest.json")


if __name__ == "__main__":
    main()
