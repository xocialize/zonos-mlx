"""Capture a deterministic golden for the speaker-embedding model (ResNet293 + LDA)
and convert its .pt weights to safetensors (so the MLX port loads without torch).

Outputs:
  goldens/golden_speaker.npz         — input wav, mel features, 256-d emb, 128-d LDA
  goldens/speaker_key_inventory.txt  — ResNet + LDA key inventory (sanitize target)
  weights/zonos_speaker/{resnet293.safetensors, lda.safetensors}

Run:  PYTHONPATH=refs/Zonos python scripts/capture_speaker_golden.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")

import numpy as np
import torch
from safetensors.torch import save_file

SEED = 1234
OUT = Path(__file__).resolve().parent.parent / "goldens"
WDIR = Path(__file__).resolve().parent.parent / "weights" / "zonos_speaker"
OUT.mkdir(exist_ok=True)
WDIR.mkdir(parents=True, exist_ok=True)


def main():
    torch.manual_seed(SEED)
    from zonos.speaker_cloning import SpeakerEmbeddingLDA

    spk = SpeakerEmbeddingLDA(device="cpu")
    spk.eval()

    # Fixed 2s @ 16kHz input (resampler is identity at 16k).
    g = torch.Generator().manual_seed(SEED)
    wav = 0.1 * torch.randn(1, 32000, generator=g)

    captured = {}
    mm = spk.model.model
    hooks = [
        mm.featCal.register_forward_hook(lambda m, i, o: captured.__setitem__("mel", o)),
        mm.front.register_forward_hook(lambda m, i, o: captured.__setitem__("front", o)),
        mm.pooling.register_forward_hook(lambda m, i, o: captured.__setitem__("pooling", o)),
        mm.front.layer1[0].register_forward_hook(lambda m, i, o: captured.__setitem__("l1b0", o)),
        mm.front.conv1.register_forward_hook(lambda m, i, o: captured.__setitem__("conv1", o)),
    ]
    with torch.inference_mode():
        emb, lda = spk(wav, 16000)
    for h in hooks:
        h.remove()

    arrays = {
        "wav_input": wav.numpy(),
        "mel": captured["mel"].float().numpy(),
        "conv1": captured["conv1"].float().numpy(),
        "l1b0": captured["l1b0"].float().numpy(),
        "front": captured["front"].float().numpy(),
        "pooling": captured["pooling"].float().numpy(),
        "emb_256": emb.float().numpy(),
        "lda_128": lda.float().numpy(),
    }
    np.savez(OUT / "golden_speaker.npz", **arrays)
    for k, v in arrays.items():
        print(f"  {k:12s} {v.shape}")

    # Convert weights to safetensors.
    resnet_sd = {k: v.contiguous() for k, v in spk.model.model.state_dict().items() if not k.startswith("featCal")}
    lda_sd = {k: v.contiguous() for k, v in spk.lda.state_dict().items()}
    save_file(resnet_sd, str(WDIR / "resnet293.safetensors"))
    save_file(lda_sd, str(WDIR / "lda.safetensors"))

    # Bake the fixed mel frontend constants (torchaudio MelSpectrogram) so the MLX
    # port reproduces them exactly without a torchaudio dependency.
    fc = spk.model.model.featCal.fbankCal
    save_file(
        {"mel_fb": fc.mel_scale.fb.contiguous().float(), "window": fc.spectrogram.window.contiguous().float()},
        str(WDIR / "featcal.safetensors"),
    )
    print(f"featcal: mel_fb {tuple(fc.mel_scale.fb.shape)} window {tuple(fc.spectrogram.window.shape)}")
    (OUT / "speaker_key_inventory.txt").write_text(
        "\n".join(sorted(resnet_sd)) + "\n\n[LDA]\n" + "\n".join(sorted(lda_sd)) + "\n"
    )
    print(f"resnet keys: {len(resnet_sd)}, lda keys: {len(lda_sd)} -> weights/zonos_speaker/")
    print(f"emb_256 stats: mean={arrays['emb_256'].mean():.4f} std={arrays['emb_256'].std():.4f}")


if __name__ == "__main__":
    main()
