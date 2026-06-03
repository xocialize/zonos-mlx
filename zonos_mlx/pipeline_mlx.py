"""High-level entrypoint for the MLX Zonos port (mirrors upstream sample.py usage)."""

import mlx.core as mx

from .conditioning import make_cond_dict
from .model import Zonos
from .speaker_cloning import SpeakerEmbeddingLDA


class ZonosPipeline:
    def __init__(self, model: Zonos, speaker_model: SpeakerEmbeddingLDA | None = None):
        self.model = model
        self.speaker_model = speaker_model
        self.sampling_rate = model.autoencoder.sampling_rate

    @classmethod
    def from_pretrained(cls, repo_id: str = "Zyphra/Zonos-v0.1-transformer", dtype=mx.float32, speaker_dir: str | None = None):
        model = Zonos.from_pretrained(repo_id, dtype=dtype)
        spk = None
        if speaker_dir is not None:
            import os
            spk = SpeakerEmbeddingLDA.from_local(
                os.path.join(speaker_dir, "resnet293.safetensors"),
                os.path.join(speaker_dir, "lda.safetensors"),
                os.path.join(speaker_dir, "featcal.safetensors"),
            )
        return cls(model, spk)

    @classmethod
    def from_pretrained_mlx(cls, path: str, dtype=mx.bfloat16):
        """Load a self-contained MLX repo (local dir or HF id) exported by scripts/export_mlx.py."""
        import os

        model = Zonos.from_mlx(path, dtype=dtype)

        def get(fn):
            if os.path.isdir(path):
                return os.path.join(path, fn)
            from huggingface_hub import hf_hub_download

            return hf_hub_download(path, fn)

        spk = SpeakerEmbeddingLDA.from_local(
            get("speaker_resnet293.safetensors"), get("speaker_lda.safetensors"), get("speaker_featcal.safetensors")
        )
        return cls(model, spk)

    def make_speaker_embedding(self, wav_16k_mono: mx.array) -> mx.array:
        """Reference clip (mono, 16 kHz) → (1, 1, 128) speaker embedding. Resample upstream."""
        assert self.speaker_model is not None, "load with speaker_dir=... to enable voice cloning"
        if wav_16k_mono.ndim == 1:
            wav_16k_mono = wav_16k_mono[None, :]
        _, lda = self.speaker_model(wav_16k_mono)
        return lda.reshape(1, 1, -1)

    def generate(
        self,
        text: str,
        speaker: mx.array,                       # (1, 1, 128) speaker embedding
        language: str = "en-us",
        emotion=None,                            # 8-D [happy, sad, disgust, fear, surprise, anger, other, neutral]
        fmax: float = 22050.0,
        pitch_std: float = 20.0,
        speaking_rate: float = 15.0,
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        sampling_params: dict | None = None,
    ) -> mx.array:
        kw = dict(text=text, language=language, speaker=speaker, fmax=fmax, pitch_std=pitch_std, speaking_rate=speaking_rate)
        if emotion is not None:
            kw["emotion"] = emotion
        cond = make_cond_dict(**kw)
        prefix = self.model.prepare_conditioning(cond)
        codes = self.model.generate(
            prefix, max_new_tokens=max_new_tokens, cfg_scale=cfg_scale,
            sampling_params=sampling_params or dict(min_p=0.1),
        )
        wav = self.model.autoencoder.decode(codes)   # (1, 1, T)
        return wav[0].astype(mx.float32)              # (1, T) — fp32 for audio I/O
