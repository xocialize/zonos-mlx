"""Port of upstream ``zonos/speaker_cloning.py`` — ResNet293 (SimAM) speaker encoder
+ attentive-statistics pooling + LDA → 128-d speaker embedding.

MLX is channels-last; torch is channels-first. Data flows as (B, H, W, C) here
(torch (B, C, H, W)). Conv weight transposes + Sequential-index key remaps are in
sanitize(). The mel frontend reproduces torchaudio MelSpectrogram via baked
filterbank+window buffers (verified to ~1.7e-6 vs the oracle).

Weights: Zyphra/Zonos-v0.1-speaker-embedding (.pt → safetensors via
scripts/capture_speaker_golden.py): resnet293.safetensors, lda.safetensors,
featcal.safetensors.
"""

import mlx.core as mx
import mlx.nn as nn

N_FFT, HOP, WIN = 512, 160, 400


# ------------------------------------------------------------------ mel frontend
def _reflect_pad(x: mx.array, p: int) -> mx.array:
    return mx.concatenate([x[:, 1 : p + 1][:, ::-1], x, x[:, -p - 1 : -1][:, ::-1]], axis=1)


class LogFbankCal(nn.Module):
    """torchaudio MelSpectrogram(n_fft=512, win=400, hop=160, n_mels=80) + log + mean-sub."""

    def __init__(self):
        super().__init__()
        self.mel_fb = mx.zeros((N_FFT // 2 + 1, 80))   # (257, 80) loaded buffer
        self.window = mx.zeros((WIN,))                  # (400,) loaded buffer

    def __call__(self, wav: mx.array) -> mx.array:      # wav: (B, N) -> (B, 80, T)
        x = _reflect_pad(wav, N_FFT // 2)
        win = mx.pad(self.window, [((N_FFT - WIN) // 2, (N_FFT - WIN) // 2)])
        n = (x.shape[1] - N_FFT) // HOP + 1
        frames = mx.stack([x[:, t * HOP : t * HOP + N_FFT] * win for t in range(n)], axis=1)
        power = mx.abs(mx.fft.rfft(frames, n=N_FFT, axis=-1)) ** 2     # (B, n, 257)
        mel = mx.transpose(power @ self.mel_fb, (0, 2, 1))            # (B, 80, n)
        mel = mx.log(mel + 1e-6)
        return mel - mel.mean(axis=2, keepdims=True)


# ------------------------------------------------------------------ ResNet293 (SimAM)
class SimAMBasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.downsample = []
        if stride != 1 or in_planes != planes:
            self.downsample = [nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False), nn.BatchNorm(planes)]

    def _simam(self, x: mx.array, lam: float = 1e-4) -> mx.array:  # x: (B,H,W,C)
        n = x.shape[1] * x.shape[2] - 1
        d = mx.square(x - x.mean(axis=(1, 2), keepdims=True))
        v = d.sum(axis=(1, 2), keepdims=True) / n
        return x * mx.sigmoid(d / (4 * (v + lam)) + 0.5)

    def __call__(self, x: mx.array) -> mx.array:
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self._simam(out)
        identity = x
        for layer in self.downsample:
            identity = layer(identity)
        return nn.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, in_planes: int, num_blocks: list[int]):
        super().__init__()
        self.conv1 = nn.Conv2d(1, in_planes, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(in_planes)
        self.in_planes = in_planes
        self.layer1 = self._make_layer(in_planes, num_blocks[0], 1)
        self.layer2 = self._make_layer(in_planes * 2, num_blocks[1], 2)
        self.layer3 = self._make_layer(in_planes * 4, num_blocks[2], 2)
        self.layer4 = self._make_layer(in_planes * 8, num_blocks[3], 2)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> list:
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(SimAMBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return layers

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.bn1(self.conv1(x)))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x = block(x)
        return x


class ASP(nn.Module):
    """Attentive statistics pooling. Input torch (B,C,H,W) → here (B,H,W,C)."""

    def __init__(self, in_planes: int, acoustic_dim: int):
        super().__init__()
        outmap = acoustic_dim // 8
        feat = in_planes * 8 * outmap
        self.att_conv1 = nn.Conv1d(feat, 128, 1)
        self.att_bn = nn.BatchNorm(128)
        self.att_conv2 = nn.Conv1d(128, feat, 1)

    def __call__(self, x: mx.array) -> mx.array:                 # x: (B, H, W, C)
        b, h, w, c = x.shape
        x = mx.transpose(x, (0, 2, 3, 1)).reshape(b, w, c * h)   # (B, T, C*H) idx=c*H+h
        a = self.att_conv2(self.att_bn(nn.relu(self.att_conv1(x))))  # conv→ReLU→BN→conv (upstream order)
        wgt = mx.softmax(a, axis=1)                               # over time
        mu = (x * wgt).sum(axis=1)
        sg = mx.sqrt(mx.maximum((x * x * wgt).sum(axis=1) - mu * mu, 1e-5))
        return mx.concatenate([mu, sg], axis=1)                   # (B, 2*C*H)


class ResNet293Speaker(nn.Module):
    def __init__(self, in_planes: int = 64, embd_dim: int = 256, acoustic_dim: int = 80):
        super().__init__()
        self.featCal = LogFbankCal()
        self.front = ResNet(in_planes, [10, 20, 64, 3])
        self.pooling = ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(in_planes * 8 * (acoustic_dim // 8) * 2, embd_dim)

    def __call__(self, wav: mx.array) -> mx.array:    # wav: (B, N) @16kHz
        mel = self.featCal(wav)                       # (B, 80, T)
        x = mel[..., None]                            # (B, 80, T, 1) channels-last
        x = self.front(x)
        x = self.pooling(x)
        return self.bottleneck(x)


class SpeakerEmbeddingLDA:
    """ResNet293 speaker encoder + LDA → 128-d embedding (the `speaker` conditioner input)."""

    def __init__(self, model: ResNet293Speaker, lda: nn.Linear):
        self.model = model
        self.lda = lda

    def __call__(self, wav: mx.array, sample_rate: int = 16000):
        emb = self.model(wav)
        return emb, self.lda(emb)

    @classmethod
    def from_local(cls, resnet_path: str, lda_path: str, featcal_path: str):
        model = ResNet293Speaker()
        rw = {k: v for k, v in mx.load(resnet_path).items()}
        rw.update(mx.load(featcal_path))                          # mel_fb, window → featCal.*
        model.load_weights(list(_sanitize_resnet(rw, model).items()), strict=True)

        lda_w = mx.load(lda_path)
        lda = nn.Linear(lda_w["weight"].shape[1], lda_w["weight"].shape[0])
        lda.load_weights([("weight", lda_w["weight"]), ("bias", lda_w["bias"])], strict=True)

        model.eval()
        lda.eval()
        return cls(model, lda)


def _sanitize_resnet(weights: dict, model) -> dict:
    out = {}
    for k, v in weights.items():
        if k.endswith("num_batches_tracked"):
            continue
        key = k
        if key in ("mel_fb", "window"):
            key = f"featCal.{key}"
        # Sequential-index remaps for the ASP attention (0=conv,2=bn,3=conv).
        key = key.replace("pooling.attention.0.", "pooling.att_conv1.")
        key = key.replace("pooling.attention.2.", "pooling.att_bn.")
        key = key.replace("pooling.attention.3.", "pooling.att_conv2.")
        if key.endswith(".weight") and v.ndim == 4:               # Conv2d (O,I,kH,kW)→(O,kH,kW,I)
            v = v.transpose(0, 2, 3, 1)
        elif key.endswith(".weight") and v.ndim == 3:             # Conv1d (O,I,k)→(O,k,I)
            v = v.transpose(0, 2, 1)
        out[key] = v
    return out
