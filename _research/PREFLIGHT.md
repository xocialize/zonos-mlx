# Zonos → MLX port — Pre-flight findings

**Date:** 2026-06-03 · **Backlog:** #1 (pivoted from IndexTTS-2 — bilibili license, not Apache) · **License:** Apache-2.0 ✅

## Gate results

- **CONFIRM-1 (prior art):** mlx-audio Zonos **PR #2** (author: maintainer Blaizzy) — **CLOSED, never merged**, 2026-02-21. Only Zonos MLX attempt; Zonos is NOT on mlx-audio main (38 other TTS models are). Preliminary/abandoned, **transformer-backbone only** (no Mamba). Cross-refs (#137 Spark, #694 Voxtral) unrelated. → **Net-new. Proceed with Python port; mine PR #2 as a starting diff** (`_research/pr2_zonos.diff`). Head SHA on Blaizzy fork.
- **CONFIRM-2 (checkpoint/backbone):** Target **`Zyphra/Zonos-v0.1-transformer`** (Apache-2.0, 8.4k dl, single `model.safetensors` + `config.json`). Its 26 layers are **all attention** (`attn_layer_idx: 0–25`) → **no Mamba2/SSM needed in MLX**. The `-hybrid` (Mamba) variant is deferred. ✅
- **CONFIRM-3 (license):** Zyphra/Zonos code + all weights (`-transformer`, `-hybrid`, `-speaker-embedding`) Apache-2.0. ✅
- **CONFIRM-4 (config truth):** captured below from real `config.json`.
- **CONFIRM-5 (oracle):** ✅ DONE. PyTorch Zonos installed (`.venv`, upstream editable in `refs/Zonos`; espeak-ng via brew). Golden run captured by `scripts/capture_golden.py` → `goldens/golden_transformer.npz` (36 tensors, fp32, seed 1234): prefix_cond/uncond, fixed speaker emb, audio codes input, backbone input + all 26 layer hidden states + norm_f, pre-sampling logits, DAC encode/decode round-trip. Key inventory (246 keys, sanitize target) → `goldens/key_inventory.txt`. NOTE: saved logits width = 1026 (vocab pad quirk; mask ≥1025 for parity).
  - **Key structure:** `backbone.layers.{0-25}.{norm,norm2}.{weight,bias}` (LayerNorm w/ bias), `.mixer.{in_proj,out_proj}.weight` (no bias), `.mlp.{fc1,fc2}.weight` (no bias); `backbone.norm_f.{weight,bias}`; `embeddings.{0-8}.weight`, `heads.{0-8}.weight`; `prefix_conditioner.conditioners.{0 phoneme_embedder / 1 speaker project+uncond / 2-5 Fourier weight+uncond / 6 int_embedder+uncond}`, `prefix_conditioner.{norm,project}.{weight,bias}`.
  - **Backbone confirmed (`_torch.py`):** fused QKV `in_proj` split `[q,k,v]` (clean, NOT interleaved); GQA via SDPA `enable_gqa=True`; **interleaved RoPE** (`apply_rotary_emb` pairs adjacent dims, gpt-fast `precompute_freqs_cis` base 10000) → MLX `mx.fast.rope(traditional=True)`; FFN = SwiGLU `fc2(y * silu(gate))` w/ `fc1`→2×8192; pre-norm blocks, LayerNorm everywhere.

## Architecture (AR transformer over DAC codec, NOT flow-matching)

text → **espeak phonemes** → prefix conditioning ─┐
                                                  ├─→ **26-layer AR transformer** → DAC codes (delay pattern) → **DAC decode @44.1kHz** → wav
speaker emb + emotion + prosody scalars ──────────┘

**Backbone (`backbone._torch` → port to `backbone/transformer.py`):**
- d_model 2048, n_layer 26, attn MLP d_intermediate 8192, `d_intermediate: 0` (no mamba MLP)
- **GQA: num_heads 16, num_heads_kv 4**; head_dim = 2048/16 = 128 = `rotary_emb_dim`
- **`rotary_emb_interleaved: true`** ⚠️ interleaved-RoPE trap — replicate exactly
- `rms_norm: false` → **LayerNorm** (eps 1e-5), `residual_in_fp32: false`
- `qkv_proj_bias: false`, `out_proj_bias: false`; `causal: true`
- `eos_token_id: 1024`, `masked_token_id: 1025`

**Prefix conditioners (`conditioning.py`), in order:**
1. `espeak` — **EspeakPhonemeConditioner** (espeak-ng G2P; how JP/ZH/multi-lang work) ⚠️ Python-only — the hard Swift lift
2. `speaker` — PassthroughConditioner, cond_dim 128 (from `speaker_cloning.py`)
3. `emotion` — **FourierConditioner, input_dim 8** = the 8-D vector [happiness, sadness, disgust, fear, surprise, anger, other, neutral]
4. `fmax` — FourierConditioner (0–24000)
5. `pitch_std` — FourierConditioner (0–400)
6. `speaking_rate` — FourierConditioner (0–40) = duration/AV-sync lever
7. `language_id` — IntegerConditioner (-1–126)

**Codec:** `autoencoder.py` wraps **DAC (Descript Audio Codec)** @44.1kHz. `codebook_pattern.py` = multi-codebook delay pattern. Keep DAC at bf16/fp16 (don't quantize).

## Upstream → port file map (preserve 1:1 structure)
`zonos/model.py`→model, `backbone/_torch.py`→transformer, `conditioning.py`, `sampling.py`, `speaker_cloning.py`, `autoencoder.py`, `codebook_pattern.py`, `config.py`, `utils.py`.

## Traps flagged (skill Step 1)
1. Interleaved RoPE (`rotary_emb_interleaved: true`) — use matching interleaved variant, fp32 then cast.
2. GQA 16/4 — `mlx_lm` SDPA handles natively.
3. LayerNorm not RMSNorm (rms_norm false).
4. No biases on qkv/out proj.
5. DAC codec layout (Conv/ConvTranspose transpose rules) + delay-pattern codebook offsets.
6. espeak G2P reproducibility — fine in Python; the main Swift-mirror risk.
7. RNG: MLX vs torch not seed-compatible — compare pre-sampling logits, not sampled codes.

## Progress
1. ✅ Scaffold `-mlx` layout + `parity_helpers.py`.
2. ✅ PyTorch oracle + golden run (CONFIRM-5).
3. ✅ **Backbone translated → Gate A PASS** (`zonos_mlx/backbone/transformer.py`, `tests/parity/test_backbone_parity.py`): all 26 layers + norm_f + pre-sampling logits at ~7e-7 relative error (true fp32). RoPE hand-rolled interleaved (mx.fast.rope traditional also matches); GQA via mx.fast SDPA; SwiGLU FFN; LayerNorm.
4. ✅ Conditioning stack → parity 2e-7 (incl. espeak G2P/phonemizer parity confirmed). ✅ Codebook delay pattern → exact match to torch.
5. ▶ DAC autoencoder — **DECIDED: port HF transformers `DacModel` from scratch** (golden used `DacModel("descript/dac_44khz")`; 1:1 keys, self-contained). **Decode first** (Gate C, SI-SDR/PESQ), **encode is a REQUIRED end-of-project deliverable** (audio-prefix/voice-continuation — do NOT leave for a future project).
6. ✅ DAC decode → Gate C PASS (max_abs 3e-6, SI-SDR 107.8 dB; ConvTranspose length matched).
7. ✅ **Functional end-to-end TTS**: `sampling.py` + `model.py` (embed_codes, CFG `_compute_logits`, AR `generate` loop w/ KVCache + delay pattern + EOS, `from_pretrained`, `sanitize`) + `pipeline_mlx.py`. DAC loads from separate `descript/dac_44khz` safetensors (no torch at runtime). e2e smoke green: text→audio, finite/non-silent/right length. Sample: `outputs/sample_random_speaker.wav` (~3s in 6s). **6 tests pass.**
8. ✅ **speaker_cloning.py** (ResNet293 SimAM + ASP + LDA + mel frontend) → parity emb 1.2e-5, LDA 9.6e-6. Mel frontend 1.7e-6 (baked torchaudio filterbank+window buffers; manual reflect-pad since mx.pad lacks reflect). Weights converted .pt→safetensors in `weights/zonos_speaker/`. **7 tests pass.** Bug found+fixed: ASP attention order is conv→ReLU→BN→conv (not conv→BN→ReLU).
9. ✅ Speaker wired into pipeline + real cloned-voice demo (`outputs/sample_cloned_voice.wav`).
10. ✅ **Gate D + Gate E PASS** (`scripts/gates_de.py`): D — speaking_rate→duration strictly monotonic (10→5.8s, 15→3.8s, 20→2.9s, 30→2.1s). E — 8-D emotion vector alters content (1.4–1.5× rel waveform diff vs neutral) + energy (~18% RMS spread), speaker fixed. Emotion samples in `outputs/sample_emotion_*.wav`.
11. ✅ **DAC encode** (end deliverable) — exact 100% integer code match vs oracle (`autoencoder.py` encoder + quantizer.encode/decode_latents; `tests/parity/test_dac_encode_parity.py`). 8 tests pass.
12. ✅ **Quantize (Phase 5)** — int4 backbone Linears (group_size 64), logit cosine 0.9993 vs golden, backbone 6.4× smaller (6326→989 MB). `scripts/quantize.py`, `outputs/sample_int4.wav`.
13. ✅ **Publish artifacts ready (Phase 6)** — `scripts/export_mlx.py` → `dist/Zonos-v0.1-transformer-bf16/` (3.4 GB self-contained: model.safetensors bf16 our-layout + speaker_* + config + card). `Zonos.from_mlx` / `ZonosPipeline.from_pretrained_mlx` load it with NO torch; e2e voice-clone verified (`outputs/sample_bf16_published.wav`). HF auth = `xocialize` (member of `mlx-community`). ▶ AWAITING user confirm to push to `mlx-community/Zonos-v0.1-transformer-bf16` (outward-facing). Then optional int4 variant.
14. DEFERRED: Swift mirror → mlx-engine work (pending WWDC ~2026-06-10). Minor: end-clip pop, torch-free resampler, emotion2vec Gate E.

## ⚠️ Numerics lessons (fold into mlx-porting skill)
- **MLX Apple-GPU fp32 matmul is tf32-like** (~3.8e-3 abs err/matmul; not bf16, not true fp32). Accumulated ~6e-2 over 26 layers and looked like a port bug. **MLX-CPU matmul is bitwise-exact fp32.** → Run correctness parity on `mx.set_default_device(mx.cpu)`; treat GPU/bf16 as a separate (looser) e2e concern.
- **Gate on RELATIVE error for massive-activation models.** Zonos grows a massive-activation channel (~16,104 at layer 5+; cf. ~333 at layer 4). Raw `max_abs` of 1.2e-2 was just 7e-7 relative. Absolute thresholds misfire; use `max_abs / golden_max_abs`.
