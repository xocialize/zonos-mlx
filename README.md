# zonos-mlx

Apple MLX port of [Zyphra/Zonos-v0.1](https://github.com/Zyphra/Zonos) — an
autoregressive transformer TTS that generates [DAC](https://github.com/descriptinc/descript-audio-codec)
codec tokens at 44.1 kHz, with **voice cloning**, an **8-D emotion vector**, and
prosody controls (pitch, fmax, speaking-rate) — all Apache-2.0.

> **Status:** Feature-complete and published. All components parity-verified vs the
> PyTorch oracle; e2e voice cloning, encode + decode, int4 quant, Gates A–E.
> Checkpoint: **`Zyphra/Zonos-v0.1-transformer`** (pure-attention, 26 layers — no Mamba),
> exported to **[`mlx-community/Zonos-v0.1-transformer-bf16`](https://huggingface.co/mlx-community/Zonos-v0.1-transformer-bf16)**.
> The Mamba `-hybrid` variant is deferred.

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
8-D emotion ───────┼─► PrefixConditioner ─► 26-layer AR transformer ─► 9-codebook DAC ─► DAC decode ─► 44.1 kHz wav
pitch/fmax/rate ───┤                          (GQA 16/4, interleaved RoPE,    (delay pattern)
language_id ───────┘                           SwiGLU, LayerNorm, no biases)
```

DAC is a separate checkpoint (`descript/dac_44khz`), not part of the Zonos weights.

## Layout

```
zonos_mlx/            # module names mirror upstream zonos/ 1:1
  config.py           # dataclasses
  backbone/transformer.py   # AR transformer (upstream _torch.py)
  conditioning.py     # espeak / speaker / emotion(8-D) / prosody / language conditioners
  speaker_cloning.py  # 128-d speaker embedding model (ResNet293 + LDA)
  autoencoder.py      # DAC codec wrapper (encode + decode)
  codebook_pattern.py # multi-codebook delay pattern
  sampling.py model.py utils.py pipeline_mlx.py
tests/parity/         # PT (oracle) ↔ MLX numerical parity (torch = dev-only extra)
tests/smoke/          # config / shape / e2e smoke (e2e skips if weights unavailable)
scripts/              # capture_golden, capture_speaker_golden, export_mlx, export_swift,
                      #   quantize, gates_de, model_card.md
```

## Install (dev)

```bash
uv venv && uv pip install -e ".[dev]"   # includes torch + DAC for the parity oracle
pytest tests/smoke                       # config smoke runs without weights
```

Runtime deps are MLX-only (`mlx`, `mlx-lm`, `safetensors`, `huggingface_hub`, `numpy`,
`soundfile`, `phonemizer`). PyTorch / `transformers` (the DAC + Zonos oracle) live in the
`[parity]` extra and are **dev-only** — end users on MLX never install them.

There is no console-script entry point; drive the pipeline via the library
(`zonos_mlx.pipeline_mlx`) or the helper scripts under `scripts/`.

## Scripts

- `scripts/export_mlx.py` — export the self-contained bf16 repo (`Zonos-v0.1-transformer-bf16`) for mlx-community.
- `scripts/export_swift.py` — dump CPU goldens/weights for the `zonos-mlx-swift` sibling.
- `scripts/quantize.py` — int4/int8 quantization.
- `scripts/capture_golden.py`, `scripts/capture_speaker_golden.py` — regenerate oracle goldens.
- `scripts/gates_de.py` — duration / emotion gates (D, E).

## Parity gates (see PREFLIGHT.md §gates)

A — pre-sampling logits vs PyTorch oracle · B — DAC codes exact · C — waveform SI-SDR/PESQ ·
D — duration (speaking-rate) accuracy · E — emotion-vector sweep monotonic, speaker stable.

Parity tests run on the CPU stream (MLX GPU fp32 matmul is tf32-like and drifts over depth).

License: Apache-2.0 (matches upstream Zonos).
