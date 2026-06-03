"""Gate D (speaking-rate → duration) + Gate E (emotion sweep) behavioral validation.

Generation is stochastic (RNG differs from torch), so these are behavioral gates:
  D — duration decreases monotonically as speaking_rate increases (fixed text/speaker/seed).
  E — distinct emotion vectors produce measurably different audio, all valid speech,
      with the speaker embedding held fixed (identity stable by construction).
Saves samples to outputs/ for listening.

Run:  python scripts/gates_de.py
"""

import numpy as np
import mlx.core as mx
import soundfile as sf

from zonos_mlx.pipeline_mlx import ZonosPipeline

TEXT = "The quick brown fox jumps over the lazy dog and runs away."
EMOTIONS = {
    # happy, sad, disgust, fear, surprise, anger, other, neutral
    "neutral": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
    "happy":   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad":     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "angry":   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
}


def main():
    p = ZonosPipeline.from_pretrained()
    speaker = mx.random.normal((1, 1, 128))

    # ---- Gate D ----
    print("Gate D — speaking_rate → duration (mean over 3 seeds):")
    rates, durs = [10.0, 15.0, 20.0, 30.0], []
    for rate in rates:
        ds = []
        for seed in (1, 2, 3):
            mx.random.seed(seed)
            wav = p.generate(TEXT, speaker=speaker, speaking_rate=rate, max_new_tokens=86 * 10)
            mx.eval(wav)
            ds.append(wav.shape[1] / p.sampling_rate)
        durs.append(float(np.mean(ds)))
        print(f"  rate={rate:5.1f}  duration={durs[-1]:.2f}s")
    monotonic = all(durs[i] > durs[i + 1] for i in range(len(durs) - 1))
    print(f"  Gate D {'PASS' if monotonic else 'FAIL'} — strictly decreasing: {monotonic}\n")

    # ---- Gate E ----
    print("Gate E — emotion sweep (fixed text/speaker/seed):")
    wavs = {}
    for name, vec in EMOTIONS.items():
        mx.random.seed(0)
        wav = p.generate(TEXT, speaker=speaker, emotion=vec, max_new_tokens=86 * 5)
        mx.eval(wav)
        w = np.array(wav)[0]
        wavs[name] = w
        sf.write(f"outputs/sample_emotion_{name}.wav", w, p.sampling_rate)
        print(f"  {name:8s} dur={len(w)/p.sampling_rate:.2f}s rms={np.sqrt((w**2).mean()):.4f} -> outputs/sample_emotion_{name}.wav")

    # Emotion should (a) change content (waveform differs from neutral) and (b) change
    # energy/arousal (RMS spread), with speaker held fixed. Same seed → identical only
    # if the knob is inert.
    base = wavs["neutral"]
    rms = {k: float(np.sqrt((w**2).mean())) for k, w in wavs.items()}
    print("  content difference vs neutral + energy:")
    content_changed = True
    for name, w in wavs.items():
        if name == "neutral":
            continue
        n = min(len(w), len(base))
        rel = float(np.sqrt(((w[:n] - base[:n]) ** 2).mean()) / (np.sqrt((base[:n] ** 2).mean()) + 1e-9))
        print(f"    {name:8s} rel_waveform_diff={rel:.2f}  rms={rms[name]:.4f}")
        content_changed &= rel > 0.1
    rms_spread = (max(rms.values()) - min(rms.values())) / (np.mean(list(rms.values())) + 1e-9)
    print(f"  rms spread across emotions = {rms_spread:.2%}  (arousal: " +
          ", ".join(f"{k}={v:.3f}" for k, v in sorted(rms.items(), key=lambda x: -x[1])) + ")")
    ok = content_changed and rms_spread > 0.1
    print(f"  Gate E {'PASS' if ok else 'WEAK'} — emotion alters content & energy: {ok}")


if __name__ == "__main__":
    main()
