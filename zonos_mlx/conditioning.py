"""Port of upstream ``zonos/conditioning.py`` — prefix conditioners.

The espeak G2P + text-normalization helpers are pure Python (phonemizer/inflect/
sudachi) and are copied verbatim from upstream (the one piece that must be
reproduced for any future Swift mirror). The conditioner Modules are MLX.

Deviation from upstream (identical output, verified by parity): upstream
``Conditioner.forward`` does ``apply_cond(*inputs)``, which unpacks a (1,1,D)
tensor to (1,D) and relies on torch ``expand(1,-1,-1)`` re-adding the dim. We
operate on the 3D array directly so every conditioner yields (B, seq, D) cleanly.

Conditioner order (config): espeak, speaker, emotion(8-D), fmax, pitch_std,
speaking_rate, language_id. Output is concatenated along seq, then norm∘project.
"""

import math
import os
import re
import sys
import unicodedata
from functools import cache
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from .config import PrefixConditionerConfig

if sys.platform == "darwin":
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")


# ------------------------------------------------------------------ Conditioner base
class Conditioner(nn.Module):
    def __init__(
        self,
        output_dim: int,
        name: str,
        cond_dim: int | None = None,
        projection: Literal["none", "linear", "mlp"] = "none",
        uncond_type: Literal["learned", "none"] = "none",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self.cond_dim = cond_dim = cond_dim or output_dim

        if projection == "linear":
            self.project = nn.Linear(cond_dim, output_dim)
        elif projection == "mlp":
            self.project = nn.Sequential(nn.Linear(cond_dim, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim))
        else:
            self.project = lambda x: x  # Identity

        self.uncond_vector = None
        if uncond_type == "learned":
            self.uncond_vector = mx.zeros((output_dim,))

    def apply_cond(self, *inputs):
        raise NotImplementedError()

    def __call__(self, inputs) -> mx.array:
        if inputs is None:
            assert self.uncond_vector is not None
            return self.uncond_vector.reshape(1, 1, -1)
        cond = self.apply_cond(*inputs) if isinstance(inputs, tuple) else self.apply_cond(inputs)
        return self.project(cond)


# ------------------------------------------------------------------ espeak G2P (verbatim)
import inflect  # noqa: E402
from kanjize import number2kanji  # noqa: E402
from sudachipy import Dictionary, SplitMode  # noqa: E402

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        return "%s %s, %s %s" % (dollars, "dollar" if dollars == 1 else "dollars", cents, "cent" if cents == 1 else "cents")
    elif dollars:
        return "%s %s" % (dollars, "dollar" if dollars == 1 else "dollars")
    elif cents:
        return "%s %s" % (cents, "cent" if cents == 1 else "cents")
    return "zero dollars"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            return "two thousand"
        elif 2000 < num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    return _inflect.number_to_words(num, andword="")


def normalize_numbers(text: str) -> str:
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKEN_IDS = [PAD_ID, UNK_ID, BOS_ID, EOS_ID]

_punctuation = ';:,.!?¡¿—…"«»“”() *~-/\\&'
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

symbols = [*_punctuation, *_letters, *_letters_ipa]
_symbol_to_id = {s: i for i, s in enumerate(symbols, start=len(SPECIAL_TOKEN_IDS))}


def _get_symbol_id(s: str) -> int:
    return _symbol_to_id.get(s, 1)


def get_symbol_ids(text: str) -> list[int]:
    return list(map(_get_symbol_id, text))


def tokenize_phonemes(phonemes: list[str]):
    phoneme_ids = [[BOS_ID, *get_symbol_ids(p), EOS_ID] for p in phonemes]
    lengths = list(map(len, phoneme_ids))
    longest = max(lengths)
    phoneme_ids = [[PAD_ID] * (longest - len(ids)) + ids for ids in phoneme_ids]
    return mx.array(phoneme_ids), lengths


def normalize_jp_text(text: str, tokenizer=None) -> str:
    if tokenizer is None:
        tokenizer = Dictionary(dict="full").create()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\d+", lambda m: number2kanji(int(m[0])), text)
    return " ".join([x.reading_form() for x in tokenizer.tokenize(text, SplitMode.A)])


def clean(texts: list[str], languages: list[str]) -> list[str]:
    out = []
    for text, language in zip(texts, languages):
        out.append(normalize_jp_text(text) if "ja" in language else normalize_numbers(text))
    return out


@cache
def get_backend(language: str):
    import logging

    from phonemizer.backend import EspeakBackend

    logger = logging.getLogger("phonemizer")
    backend = EspeakBackend(
        language, preserve_punctuation=True, with_stress=True, punctuation_marks=_punctuation, logger=logger
    )
    logger.setLevel(logging.ERROR)
    return backend


def phonemize(texts: list[str], languages: list[str]) -> list[str]:
    texts = clean(texts, languages)
    return [get_backend(lang).phonemize([t], strip=True)[0] for t, lang in zip(texts, languages)]


# ------------------------------------------------------------------ conditioners
class EspeakPhonemeConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, **kwargs)
        self.phoneme_embedder = nn.Embedding(len(SPECIAL_TOKEN_IDS) + len(symbols), output_dim)

    def apply_cond(self, texts: list[str], languages: list[str]) -> mx.array:
        phoneme_ids, _ = tokenize_phonemes(phonemize(texts, languages))
        return self.phoneme_embedder(phoneme_ids)


class FourierConditioner(Conditioner):
    def __init__(self, output_dim: int, input_dim: int = 1, std: float = 1.0, min_val: float = 0.0, max_val: float = 1.0, **kwargs):
        assert output_dim % 2 == 0
        super().__init__(output_dim, **kwargs)
        self.weight = mx.zeros((output_dim // 2, input_dim))  # loaded from checkpoint (upstream: buffer)
        self.input_dim, self.min_val, self.max_val = input_dim, min_val, max_val

    def apply_cond(self, x: mx.array) -> mx.array:
        assert x.shape[-1] == self.input_dim
        x = (x - self.min_val) / (self.max_val - self.min_val)
        f = 2 * math.pi * (x @ self.weight.T)
        return mx.concatenate([mx.cos(f), mx.sin(f)], axis=-1)


class IntegerConditioner(Conditioner):
    def __init__(self, output_dim: int, min_val: int = 0, max_val: int = 512, **kwargs):
        super().__init__(output_dim, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim)

    def apply_cond(self, x: mx.array) -> mx.array:
        assert x.shape[-1] == 1
        return self.int_embedder((x[..., 0] - self.min_val).astype(mx.int32))


class PassthroughConditioner(Conditioner):
    def apply_cond(self, x: mx.array) -> mx.array:
        assert x.shape[-1] == self.cond_dim
        return x


_cond_cls_map = {
    "PassthroughConditioner": PassthroughConditioner,
    "EspeakPhonemeConditioner": EspeakPhonemeConditioner,
    "FourierConditioner": FourierConditioner,
    "IntegerConditioner": IntegerConditioner,
}


def build_conditioners(conditioners: list[dict], output_dim: int) -> list[Conditioner]:
    return [_cond_cls_map[c["type"]](output_dim, **c) for c in conditioners]


class PrefixConditioner(Conditioner):
    def __init__(self, config: PrefixConditionerConfig, output_dim: int):
        super().__init__(output_dim, "prefix", projection=config.projection)
        self.conditioners = build_conditioners(config.conditioners, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.required_keys = {c.name for c in self.conditioners if c.uncond_vector is None}

    def __call__(self, cond_dict: dict) -> mx.array:
        if not set(cond_dict).issuperset(self.required_keys):
            raise ValueError(f"Missing required keys: {self.required_keys - set(cond_dict)}")
        conds = [c(cond_dict.get(c.name)) for c in self.conditioners]
        max_bsz = max(c.shape[0] for c in conds)
        assert all(c.shape[0] in (max_bsz, 1) for c in conds)
        conds = [mx.broadcast_to(c, (max_bsz, *c.shape[1:])) for c in conds]
        return self.norm(self.project(mx.concatenate(conds, axis=1)))


# ------------------------------------------------------------------ cond-dict helper
supported_language_codes = [
    'af', 'am', 'an', 'ar', 'as', 'az', 'ba', 'bg', 'bn', 'bpy', 'bs', 'ca', 'cmn',
    'cs', 'cy', 'da', 'de', 'el', 'en-029', 'en-gb', 'en-gb-scotland', 'en-gb-x-gbclan',
    'en-gb-x-gbcwmd', 'en-gb-x-rp', 'en-us', 'eo', 'es', 'es-419', 'et', 'eu', 'fa',
    'fa-latn', 'fi', 'fr-be', 'fr-ch', 'fr-fr', 'ga', 'gd', 'gn', 'grc', 'gu', 'hak',
    'hi', 'hr', 'ht', 'hu', 'hy', 'hyw', 'ia', 'id', 'is', 'it', 'ja', 'jbo', 'ka',
    'kk', 'kl', 'kn', 'ko', 'kok', 'ku', 'ky', 'la', 'lfn', 'lt', 'lv', 'mi', 'mk',
    'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'nci', 'ne', 'nl', 'om', 'or', 'pa', 'pap',
    'pl', 'pt', 'pt-br', 'py', 'quc', 'ro', 'ru', 'ru-lv', 'sd', 'shn', 'si', 'sk',
    'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tn', 'tr', 'tt', 'ur', 'uz', 'vi',
    'vi-vn-x-central', 'vi-vn-x-south', 'yue',
]


def make_cond_dict(
    text: str = "It would be nice to have time for testing, indeed.",
    language: str = "en-us",
    speaker=None,
    emotion=[0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],
    fmax: float = 22050.0,
    pitch_std: float = 20.0,
    speaking_rate: float = 15.0,
    language_id_unused=None,
    unconditional_keys=("vqscore_8", "dnsmos_ovrl", "ctc_loss", "speaker_noised"),
) -> dict:
    assert language.lower() in supported_language_codes, "Please pick a supported language"
    lang_id = supported_language_codes.index(language)
    cond_dict = {
        "espeak": ([text], [language]),
        "speaker": speaker,
        "emotion": emotion,
        "fmax": fmax,
        "pitch_std": pitch_std,
        "speaking_rate": speaking_rate,
        "language_id": lang_id,
    }
    for k in unconditional_keys:
        cond_dict.pop(k, None)
    for k, v in cond_dict.items():
        if k == "espeak":
            continue
        if isinstance(v, (float, int, list)):
            v = mx.array(v, dtype=mx.float32)
        if isinstance(v, mx.array):
            cond_dict[k] = v.reshape(1, 1, -1)
        if k == "emotion":
            cond_dict[k] = cond_dict[k] / cond_dict[k].sum(axis=-1)
    return cond_dict
