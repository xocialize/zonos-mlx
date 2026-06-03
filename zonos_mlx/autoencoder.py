"""Port of the HF transformers ``DacModel`` (descript/dac_44khz) — the codec Zonos
wraps in ``zonos/autoencoder.py``. Module/param names mirror transformers 1:1 so
weights load directly. MLX is channels-last (B, T, C); torch DAC is channels-first
(B, C, T) — we stay channels-last throughout and transpose conv weights in sanitize.

DECODE path (codes → 44.1 kHz waveform) is implemented + Gate-C tested. ENCODE is a
required end-of-project deliverable (audio-prefix / voice-continuation) — see encode().

Conv weight layouts: Conv1d torch (O,I,K) → MLX (O,K,I); ConvTranspose1d torch (I,O,K)
→ MLX (O,K,I). Snake1d.alpha stays (1,C,1) (reshaped to channels-last at use).
"""

import math

import mlx.core as mx
import mlx.nn as nn

UPSAMPLING_RATIOS = [8, 8, 4, 2]
DOWNSAMPLING_RATIOS = [2, 4, 8, 8]
HIDDEN_SIZE = 1024
ENCODER_HIDDEN_SIZE = 64
DECODER_HIDDEN_SIZE = 1536
N_CODEBOOKS = 9
CODEBOOK_SIZE = 1024
CODEBOOK_DIM = 8
SAMPLING_RATE = 44100
HOP_LENGTH = 512


class Snake1d(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.alpha = mx.ones((1, hidden_dim, 1))  # matches checkpoint key shape

    def __call__(self, x: mx.array) -> mx.array:  # x: (B, T, C)
        alpha = self.alpha.reshape(1, 1, -1)
        return x + (1.0 / (alpha + 1e-9)) * mx.square(mx.sin(alpha * x))


class DacResidualUnit(nn.Module):
    def __init__(self, dimension: int, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad)
        self.snake2 = Snake1d(dimension)
        self.conv2 = nn.Conv1d(dimension, dimension, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv1(self.snake1(x))
        y = self.conv2(self.snake2(y))
        pad = (x.shape[1] - y.shape[1]) // 2  # (B, T, C): time is axis 1
        if pad > 0:
            x = x[:, pad:-pad, :]
        return x + y


class DacDecoderBlock(nn.Module):
    def __init__(self, stride: int, stride_index: int):
        super().__init__()
        input_dim = DECODER_HIDDEN_SIZE // 2**stride_index
        output_dim = DECODER_HIDDEN_SIZE // 2 ** (stride_index + 1)
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
        )
        self.res_unit1 = DacResidualUnit(output_dim, dilation=1)
        self.res_unit2 = DacResidualUnit(output_dim, dilation=3)
        self.res_unit3 = DacResidualUnit(output_dim, dilation=9)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_t1(self.snake1(x))
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        return x


class DacEncoderBlock(nn.Module):
    def __init__(self, stride: int, stride_index: int):
        super().__init__()
        dim = ENCODER_HIDDEN_SIZE * 2**stride_index
        self.res_unit1 = DacResidualUnit(dim // 2, dilation=1)
        self.res_unit2 = DacResidualUnit(dim // 2, dilation=3)
        self.res_unit3 = DacResidualUnit(dim // 2, dilation=9)
        self.snake1 = Snake1d(dim // 2)
        self.conv1 = nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))

    def __call__(self, x: mx.array) -> mx.array:
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.snake1(self.res_unit3(x))
        return self.conv1(x)


class DacEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, ENCODER_HIDDEN_SIZE, kernel_size=7, padding=3)
        self.block = [DacEncoderBlock(s, i + 1) for i, s in enumerate(DOWNSAMPLING_RATIOS)]
        d_model = ENCODER_HIDDEN_SIZE * 2 ** len(DOWNSAMPLING_RATIOS)
        self.snake1 = Snake1d(d_model)
        self.conv2 = nn.Conv1d(d_model, HIDDEN_SIZE, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:  # x: (B, N, 1) -> (B, T, HIDDEN_SIZE)
        x = self.conv1(x)
        for layer in self.block:
            x = layer(x)
        return self.conv2(self.snake1(x))


class DacDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(HIDDEN_SIZE, DECODER_HIDDEN_SIZE, kernel_size=7, padding=3)
        self.block = [DacDecoderBlock(s, i) for i, s in enumerate(UPSAMPLING_RATIOS)]
        out_dim = DECODER_HIDDEN_SIZE // 2 ** len(UPSAMPLING_RATIOS)
        self.snake1 = Snake1d(out_dim)
        self.conv2 = nn.Conv1d(out_dim, 1, kernel_size=7, padding=3)

    def __call__(self, x: mx.array) -> mx.array:  # x: (B, T, HIDDEN_SIZE)
        x = self.conv1(x)
        for layer in self.block:
            x = layer(x)
        x = self.conv2(self.snake1(x))
        return mx.tanh(x)


class DacVectorQuantize(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Conv1d(HIDDEN_SIZE, CODEBOOK_DIM, kernel_size=1)
        self.out_proj = nn.Conv1d(CODEBOOK_DIM, HIDDEN_SIZE, kernel_size=1)
        self.codebook = nn.Embedding(CODEBOOK_SIZE, CODEBOOK_DIM)

    def decode_latents(self, projected: mx.array):
        """projected: (B, T, CODEBOOK_DIM) → (quantized (B,T,Dc), indices (B,T))."""
        b, t, d = projected.shape
        enc = projected.reshape(b * t, d)
        enc = enc / (mx.linalg.norm(enc, axis=1, keepdims=True) + 1e-12)   # L2 normalize (ViT-VQGAN)
        cb = self.codebook.weight
        cbn = cb / (mx.linalg.norm(cb, axis=1, keepdims=True) + 1e-12)
        l2 = (enc**2).sum(axis=1, keepdims=True)
        dist = -(l2 - 2 * (enc @ cbn.T)) + (cbn**2).sum(axis=1, keepdims=True).T
        idx = mx.argmax(dist, axis=1).reshape(b, t)
        return self.codebook(idx), idx

    def encode(self, hidden: mx.array):
        """hidden (B,T,HIDDEN_SIZE) → (quantized_full (B,T,HIDDEN_SIZE), indices (B,T))."""
        projected = self.in_proj(hidden)
        quantized, idx = self.decode_latents(projected)
        return self.out_proj(quantized), idx


class DacResidualVectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizers = [DacVectorQuantize() for _ in range(N_CODEBOOKS)]

    def from_codes(self, audio_codes: mx.array) -> mx.array:
        """audio_codes: (B, n_q, T) int → quantized_representation (B, T, HIDDEN_SIZE)."""
        quantized = 0.0
        for i in range(audio_codes.shape[1]):
            latents_i = self.quantizers[i].codebook(audio_codes[:, i, :])  # (B, T, CODEBOOK_DIM)
            quantized = quantized + self.quantizers[i].out_proj(latents_i)
        return quantized

    def encode(self, hidden: mx.array) -> mx.array:
        """hidden (B,T,HIDDEN_SIZE) → audio_codes (B, n_q, T) via residual quantization."""
        residual = hidden
        codes = []
        for q in self.quantizers:
            quantized_i, idx = q.encode(residual)
            residual = residual - quantized_i
            codes.append(idx)
        return mx.stack(codes, axis=1)


class DAC(nn.Module):
    """transformers DacModel (decode path). Channels-last MLX port."""

    def __init__(self):
        super().__init__()
        self.encoder = DacEncoder()
        self.quantizer = DacResidualVectorQuantizer()
        self.decoder = DacDecoder()
        self.sampling_rate = SAMPLING_RATE
        self.hop_length = HOP_LENGTH
        self.num_codebooks = N_CODEBOOKS

    def decode(self, audio_codes: mx.array) -> mx.array:
        """(B, n_q, T) codes → (B, 1, T*hop) waveform (channels-first, like the oracle)."""
        quantized = self.quantizer.from_codes(audio_codes)   # (B, T, HIDDEN_SIZE)
        audio = self.decoder(quantized)                      # (B, T_audio, 1)
        return audio.transpose(0, 2, 1)                      # (B, 1, T_audio)

    def preprocess(self, wav: mx.array) -> mx.array:
        """Right-pad (B, N) to a multiple of hop_length (assumes already 44.1 kHz mono)."""
        n = wav.shape[-1]
        pad = (math.ceil(n / self.hop_length) * self.hop_length) - n
        return mx.pad(wav, [(0, 0), (0, pad)]) if pad else wav

    def encode(self, wav: mx.array) -> mx.array:
        """(B, 1, N) or (B, N) waveform @44.1kHz → audio_codes (B, n_q, T)."""
        if wav.ndim == 3:
            wav = wav[:, 0, :]
        x = self.preprocess(wav)[..., None]                  # (B, N', 1) channels-last
        hidden = self.encoder(x)                             # (B, T, HIDDEN_SIZE)
        return self.quantizer.encode(hidden)                 # (B, n_q, T)

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Transpose conv weights to MLX layout. Keys come prefixed with `autoencoder.dac.`
        in the full Zonos checkpoint, or bare for a standalone DAC checkpoint."""
        out = {}
        for k, v in weights.items():
            key = k
            for pref in ("autoencoder.dac.", "dac."):
                if key.startswith(pref):
                    key = key[len(pref):]
            if not (key.startswith("decoder.") or key.startswith("quantizer.") or key.startswith("encoder.")):
                continue
            if key.endswith(".weight") and v.ndim == 3:
                if ".conv_t1." in key:           # ConvTranspose1d: torch (I,O,K) → MLX (O,K,I)
                    v = v.transpose(1, 2, 0)
                else:                            # Conv1d: torch (O,I,K) → MLX (O,K,I)
                    v = v.transpose(0, 2, 1)
            out[key] = v
        return out
