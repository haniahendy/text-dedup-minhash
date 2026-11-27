"""Tokenization and word k-shingling."""

from __future__ import annotations

import hashlib
import re
from typing import List, Set

_WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer; punctuation and digits-only noise dropped."""
    return _WORD_RE.findall(text.lower())


def shingles(tokens: List[str], k: int = 3) -> Set[tuple]:
    """Set of word k-shingles. Short docs yield a single whole-doc shingle."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(tokens) <= k:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def shingle_hashes(text: str, k: int = 3) -> Set[int]:
    """64-bit integer hashes of a document's word k-shingles."""
    out: Set[int] = set()
    for shingle in shingles(tokenize(text), k):
        payload = " ".join(shingle).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        out.add(int.from_bytes(digest, "big"))
    return out
