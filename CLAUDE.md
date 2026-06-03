# CLAUDE.md — zonos-mlx

MLX port of Zyphra/Zonos-v0.1 (transformer variant). Driven by the `mlx-porting` skill.

## Ground rules
- **Port = transpose, not redesign.** Upstream `zonos/` is the oracle. Preserve module
  names, file paths, class/method names, call order 1:1 (skill's isomorphic-structure rule).
  No "cleanups" until fp16 parity is locked.
- **Never skip parity (Step 5).** Gate each stage on the *oracle's* intermediate tensors,
  not on your own prior stage's output.
- PyTorch + DAC are **dev-only** (`.[parity]`); end users on MLX never install them.

## Where things are
- Pre-flight (gates, arch, traps): `_research/PREFLIGHT.md`
- Abandoned upstream MLX attempt to mine: `_research/pr2_zonos.diff` (mlx-audio PR #2)
- Upstream source to translate against: clone Zyphra/Zonos into `refs/` (or read raw on GH)
- Weights: `Zyphra/Zonos-v0.1-transformer` (+ `-speaker-embedding`); local override via
  `ZONOS_MLX_WEIGHTS_DIR`.

## Verified config (Zonos-v0.1-transformer, 2026-06-03)
d_model=2048, n_layer=26 (ALL attention), GQA 16/4, head_dim=128, **interleaved RoPE**,
LayerNorm (rms_norm=False), no qkv/out biases. Conditioners: espeak, speaker(128),
emotion(8-D Fourier), fmax, pitch_std, speaking_rate, language_id. eos=1024, mask=1025.

## Build / test
`uv venv && uv pip install -e ".[dev]"` · `pytest tests/smoke` (no weights) ·
`pytest tests/parity` (needs `.[parity]` + a PyTorch golden run).

## Current status
Scaffold done; `config.py` ported. Next: PyTorch oracle + golden run, then translate
backbone → Gate A. Module files are honest skeletons (raise/TODO) — fill line-by-line.
