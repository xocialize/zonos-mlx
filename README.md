# zonos-mlx

Apple MLX port of [Zyphra/Zonos-v0.1](https://github.com/Zyphra/Zonos) — an
autoregressive transformer TTS that generates [DAC](https://github.com/descriptinc/descript-audio-codec)
codec tokens at 44.1 kHz, with **voice cloning**, an **8-D emotion vector**, and
prosody controls (pitch, fmax, speaking-rate) — all Apache-2.0.

> **Status:** Pre-alpha. Scaffold + pre-flight complete; module translation in progress.
> Target checkpoint: **`Zyphra/Zonos-v0.1-transformer`** (pure-attention, 26 layers —
> no Mamba needed). The Mamba `-hybrid` variant is deferred.

## Why this port

Picked as TTS port #1 after the originally-planned IndexTTS-2 turned out to be
non-Apache (bilibili license). Zonos is Apache-2.0, has the same 8-D emotion-vector
paradigm, native Japanese, and had no working MLX port. See
[`_research/PREFLIGHT.md`](_research/PREFLIGHT.md) for the full pre-flight (gates,
architecture teardown, traps) and [`_research/pr2_zonos.diff`](_research/pr2_zonos.diff)
for the abandoned upstream attempt (mlx-audio PR #2) we mine from.

## Architecture (at a glance)

```
text ──espeak G2P──┐
speaker clip ──────┤
8-D emotion ───────┼─► PrefixConditioner ─► 26-layer AR transformer ─► DAC codes ─► DAC decode ─► 44.1 kHz wav
pitch/fmax/rate ───┤                          (GQA 16/4, interleaved RoPE,
language_id ───────┘                           LayerNorm, no biases)
```

## Layout

```
zonos_mlx/            # module names mirror upstream zonos/ 1:1
  config.py           # ✅ ported (dataclasses)
  backbone/transformer.py   # AR transformer (upstream _torch.py)
  conditioning.py     # espeak / speaker / emotion(8-D) / prosody / language conditioners
  speaker_cloning.py  # 128-d speaker embedding model
  autoencoder.py      # DAC codec wrapper
  codebook_pattern.py # multi-codebook delay pattern
  sampling.py model.py utils.py pipeline_mlx.py
tests/parity/         # PT (oracle) ↔ MLX numerical parity (torch = dev-only extra)
tests/smoke/          # config / shape / e2e-noise smoke
```

## Install (dev)

```bash
uv venv && uv pip install -e ".[dev]"   # includes torch + DAC for the parity oracle
pytest tests/smoke                       # runs without weights
```

## Parity gates (see PREFLIGHT.md §gates)

A — pre-sampling logits vs PyTorch oracle · B — DAC codes exact · C — waveform SI-SDR/PESQ ·
D — duration (speaking-rate) accuracy · E — emotion-vector sweep monotonic, speaker stable.

License: Apache-2.0 (matches upstream Zonos).
