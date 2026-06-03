---
license: apache-2.0
language:
- en
- ja
- zh
- fr
- de
library_name: mlx
pipeline_tag: text-to-speech
tags:
- mlx
- text-to-speech
- tts
- voice-cloning
- zonos
base_model: Zyphra/Zonos-v0.1-transformer
---

# Zonos-v0.1-transformer (MLX, bf16)

Apple-Silicon **MLX** port of [Zyphra/Zonos-v0.1-transformer](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) —
an autoregressive transformer TTS over [DAC](https://huggingface.co/descript/dac_44khz)
codec tokens at 44.1 kHz, with zero-shot **voice cloning**, an **8-D emotion vector**,
and prosody controls (pitch, fmax, speaking-rate). Self-contained: no PyTorch, no
multi-repo fetch.

Code: https://github.com/xocialize/zonos-mlx

## Verification

Every component is numerically parity-tested against the PyTorch reference (CPU fp32):

| Component | Metric |
|---|---|
| AR backbone (GQA transformer) | rel err ~7e-7 |
| Conditioning + espeak G2P | rel err ~2e-7 |
| DAC decode | SI-SDR 108 dB |
| DAC encode | exact integer code match |
| Speaker encoder (ResNet293) | rel err ~1.2e-5 |
| Duration (speaking_rate) | monotonic |
| Emotion (8-D vector) | content + energy change, speaker stable |

## Usage

```python
import mlx.core as mx, soundfile as sf
from zonos_mlx.pipeline_mlx import ZonosPipeline

tts = ZonosPipeline.from_pretrained_mlx("mlx-community/Zonos-v0.1-transformer-bf16")

# voice cloning from a 16 kHz mono reference clip:
ref = mx.array(...)                       # (N,) float32 @ 16 kHz
spk = tts.make_speaker_embedding(ref)
wav = tts.generate("Hello from MLX.", speaker=spk, speaking_rate=15.0,
                   emotion=[0.0,0,0,0,0,0,0.1,0.9])  # happy..neutral 8-D
sf.write("out.wav", mx.array(wav)[0], tts.sampling_rate)
```

## License & attribution

Apache-2.0, inherited from the upstream Zonos model and code (Zyphra). Bundled
third-party components, all redistributable:
- **DAC** (Descript Audio Codec) — `descript/dac_44khz`, via HF transformers `DacModel`.
- **Speaker embedding** — ResNet293 (SimAM/ASP) + LDA from `Zyphra/Zonos-v0.1-speaker-embedding`.
- **espeak-ng** phonemizer frontend (GPL-3.0 tool, invoked at runtime — not redistributed here).

Please cite Zyphra's Zonos for the model weights and architecture.
