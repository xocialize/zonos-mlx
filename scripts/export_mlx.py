"""Phase 6 — export a self-contained MLX weight repo (bf16) for mlx-community.

Produces dist/<name>/ with everything needed to run with NO torch and NO multi-repo
fetch: pre-sanitized Zonos body + DAC (our layout) + the converted speaker model +
config + model card. Loadable via ZonosPipeline.from_pretrained_mlx(dir).

Run:  python scripts/export_mlx.py
"""

import json
import shutil
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download

from zonos_mlx.autoencoder import DAC
from zonos_mlx.model import Zonos

NAME = "Zonos-v0.1-transformer-bf16"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / NAME
SPK = ROOT / "weights" / "zonos_speaker"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dtype = mx.bfloat16

    # Zonos body (backbone/conditioner/embeddings/heads) + DAC, in our param layout.
    zonos_w = mx.load(hf_hub_download("Zyphra/Zonos-v0.1-transformer", "model.safetensors"))
    dac_w = mx.load(hf_hub_download("descript/dac_44khz", "model.safetensors"))
    combined = {k: v.astype(dtype) for k, v in Zonos.sanitize(zonos_w, dac_w).items()}
    mx.save_safetensors(str(OUT / "model.safetensors"), combined)
    print(f"model.safetensors: {len(combined)} tensors (bf16)")

    # Speaker model (already converted .pt → safetensors); ship bf16 body, fp32 featcal buffers.
    for fn in ("resnet293.safetensors", "lda.safetensors", "featcal.safetensors"):
        w = {k: (v if "featcal" in fn else v.astype(dtype)) for k, v in mx.load(str(SPK / fn)).items()}
        mx.save_safetensors(str(OUT / f"speaker_{fn}"), w)
    print("speaker_*.safetensors written")

    cfg_path = hf_hub_download("Zyphra/Zonos-v0.1-transformer", "config.json")
    shutil.copy(cfg_path, OUT / "config.json")

    card = (ROOT / "scripts" / "model_card.md").read_text() if (ROOT / "scripts" / "model_card.md").exists() else ""
    (OUT / "README.md").write_text(card)
    print(f"exported -> {OUT}")
    print("files:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
