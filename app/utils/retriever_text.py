"""Small deterministic text frontend for the Lite retriever.

This is intentionally dependency-light. It produces a fixed hashed bag of
words plus explicit speed/energy/genre slots, keeping local inference dependency-light. It is not presented as a pretrained language
model.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

import unicodedata
from collections.abc import Iterable

def _compact(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).strip().lower()
    value = value.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", value)


# Canonical labels observed in FineDance style2 plus the user-facing aliases.
_ALIASES = {
    "hiphop": "hiphop",
    "hiphopdance": "hiphop",
    "hiphopstyle": "hiphop",
    "hiphopdancing": "hiphop",
    "rap": "hiphop",
    "jazz": "jazz",
    "popping": "popping",
    "pop": "popping",
    "breaking": "breaking",
    "breakdance": "breaking",
    "bboy": "breaking",
    "bboying": "breaking",
    "locking": "locking",
    "street": "street",
    "classic": "classic",
    "mix": "mix",
    "folk": "folk",
    "jiewu": "jiewu",
    "shenyun": "shenyun",
    "korean": "korean",
    "hantang": "hantang",
    "dai": "dai",
    "miao": "miao",
    "wei": "wei",
    "urban": "urban",
    "choreography": "choreography",
    "chinese": "chinese",
    "dunhuang": "dunhuang",
    "kun": "kun",
}


def normalize_genre(value: str) -> str:
    """Return the canonical compact genre label.

    Hip Hop, hip-hop and hiphop all become hiphop; Jazz and jazz become jazz.
    For a non-empty unknown label, the compact form is returned so manifests
    remain inspectable, but strict queries should use known_genre.
    """

    compact = _compact(value)
    if not compact:
        raise ValueError("genre is empty")
    return _ALIASES.get(compact, compact)


def known_genre(value: str, allowed: Iterable[str] | None = None) -> str:
    canonical = normalize_genre(value)
    allowed_set = set(allowed) if allowed is not None else set(_ALIASES.values())
    if canonical not in allowed_set:
        choices = ", ".join(sorted(allowed_set))
        raise ValueError(f"unknown genre {value!r}; use one of: {choices}")
    return canonical




TEXT_DIM = 256
_GENRE_OFFSET = 8
_GENRE_SLOTS = 80
_TOKEN_OFFSET = 96
_TOKEN_SLOTS = TEXT_DIM - _TOKEN_OFFSET
_TOKEN_RE = re.compile(r"[a-z0-9]+")

SPEEDS = ("slow", "medium", "fast")
ENERGIES = ("calm", "energetic")


def _slot(token: str, count: int) -> int:
    # CRC32 is deterministic across Python processes and is not used for
    # artifact identity or integrity verification.
    return zlib.crc32(token.encode("utf-8")) % count


def text_to_vector(text: str, genre: str = "", *, speed: str, energy: str) -> np.ndarray:
    """Encode text into a deterministic float32 feature vector."""

    vec = np.zeros(TEXT_DIM, dtype=np.float32)
    vec[0] = 1.0
    if speed not in SPEEDS or energy not in ENERGIES:
        raise ValueError("speed/energy must be validated API labels")
    vec[1 + SPEEDS.index(speed)] = 1.0
    vec[4 + ENERGIES.index(energy)] = 1.0

    if genre:
        canonical = normalize_genre(genre)
        vec[_GENRE_OFFSET + (_slot(canonical, _GENRE_SLOTS))] = 1.0

    tokens = _TOKEN_RE.findall(text.lower())
    if genre:
        tokens.extend(_TOKEN_RE.findall(normalize_genre(genre)))
    for token in tokens:
        index = _TOKEN_OFFSET + _slot(token, _TOKEN_SLOTS)
        vec[index] += 1.0
    if len(tokens) >= 2:
        for left, right in zip(tokens, tokens[1:]):
            index = _TOKEN_OFFSET + _slot(left + "_" + right, _TOKEN_SLOTS)
            vec[index] += 0.5

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec
