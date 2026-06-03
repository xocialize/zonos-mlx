# CLAUDE.md — zonos-mlx

MLX port of Zyphra/Zonos-v0.1 (transformer variant) — autoregressive TTS over DAC codec
with voice cloning, an 8-D emotion vector, and prosody controls. Driven by the
`mlx-porting` skill. **Status: feature-complete + published** to
[`mlx-community/Zonos-v0.1-transformer-bf16`](https://huggingface.co/mlx-community/Zonos-v0.1-transformer-bf16).

## Ground rules
- **Port = transpose, not redesign.** Upstream `zonos/` (cloned into `refs/`) is the oracle.
  Preserve module/file/class/method names + call order 1:1. No "cleanups" until parity is locked.
- **Never skip parity.** Gate each stage on the *oracle's* intermediate tensors, not your
  own prior stage's output. Run correctness parity on **CPU** (`mx.set_default_device(mx.cpu)`)
  — MLX GPU fp32 matmul is tf32-like and accumulates over layers (looked like a bug; isn't).
- **Gate on relative error** for this model — it grows a ~16k massive-activation channel at
  layer 5; absolute `max_abs` misfires.
- PyTorch + DAC are **dev-only** (`.[parity]`); end users on MLX never install them.

## Architecture (verified)
26-layer AR transformer (d_model 2048, GQA 16/4, head_dim 128, **interleaved RoPE** →
hand-rolled `apply_rotary_emb`, NOT mx.fast.rope; SwiGLU FFN; LayerNorm, no biases) →
9-codebook DAC delay pattern → DAC decode @44.1kHz. Conditioners: espeak G2P, speaker(128,
ResNet293+LDA), emotion(8-D Fourier), fmax, pitch_std, speaking_rate, language_id.
**DAC is a separate checkpoint** (`descript/dac_44khz`), not in the Zonos weights.

## Where things are
- Full story + parity gates + numerics lessons: `_research/PREFLIGHT.md`
- Mined upstream MLX attempt: `_research/pr2_zonos.diff` (abandoned mlx-audio PR #2)
- Golden oracle tensors: `goldens/` (gitignored); regenerate via `scripts/capture_golden.py`
- Speaker weights `.pt`→safetensors: `scripts/capture_speaker_golden.py` → `weights/zonos_speaker/`
- Quantize: `scripts/quantize.py` · Publish export: `scripts/export_mlx.py`

## Build / test
`uv venv && uv pip install -e ".[dev]"` (incl. torch + upstream `refs/Zonos`) ·
`pytest tests/` (8 tests: config smoke, backbone Gate A, conditioning, codebook, DAC
decode+encode, speaker, e2e). Parity tests run on CPU stream.

## Validation
Circular check via the sibling `tts-validation` project (MLX Whisper STT): synthesized
output transcribes back at WER 0% (cloning / int4 / emotion all preserve content).

## Status / open items
Feature-complete: all components parity-verified, e2e voice cloning, Gates A–E, encode +
decode, int4 quant, bf16 published. Open: minor end-of-clip pop (post-EOS tail trim),
torch-free resampler in pipeline, emotion2vec-based Gate E. Swift mirror → deferred to the
post-WWDC mlx-engine work (see `whisper-mlx-swift` for the STT half).
