"""Locality-sensitive hashing over MinHash signatures.

Signatures are split into ``bands`` bands of ``rows`` rows; two documents
become candidates when they collide in any whole band. Candidate pairs are
then verified with the exact Jaccard of their shingle sets, so the final
output has no false positives above the threshold (only possible misses
from banding, which the band/row tradeoff controls).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .minhash import MinHasher, jaccard
from .shingles import shingle_hashes


class LSHIndex:
    """Index documents and retrieve verified near-duplicate pairs."""

    def __init__(
        self,
        num_perm: int = 128,
        bands: int = 32,
        k: int = 3,
        threshold: float = 0.7,
        seed: int = 0,
    ) -> None:
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.k = k
        self.threshold = threshold
        self.hasher = MinHasher(num_perm=num_perm, seed=seed)
        self._docs: Dict[str, Set[int]] = {}
        self._sigs: Dict[str, List[int]] = {}
        self._buckets: List[Dict[Tuple[int, ...], List[str]]] = [
            {} for _ in range(bands)
        ]

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._docs:
            raise ValueError(f"duplicate doc_id {doc_id!r}")
        sh = shingle_hashes(text, k=self.k)
        sig = self.hasher.signature(sh)
        self._docs[doc_id] = sh
        self._sigs[doc_id] = sig
        for band in range(self.bands):
            start = band * self.rows
            key = tuple(sig[start:start + self.rows])
            self._buckets[band].setdefault(key, []).append(doc_id)

    def candidate_pairs(self) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()
        for bucket in self._buckets:
            for ids in bucket.values():
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pairs.add((ids[i], ids[j]))
        return pairs

    def similar_pairs(self) -> List[Tuple[str, str, float]]:
        """Verified pairs with exact Jaccard >= threshold, sorted desc."""
        out = []
        for a, b in self.candidate_pairs():
            sim = jaccard(self._docs[a], self._docs[b])
            if sim >= self.threshold:
                out.append((a, b, sim))
        out.sort(key=lambda t: (-t[2], t[0], t[1]))
        return out

    def query(self, text: str, top: int = 5) -> List[Tuple[str, float]]:
        """Return the top index documents most similar to a new text."""
        sh = shingle_hashes(text, k=self.k)
        sig = self.hasher.signature(sh)
        candidates: Set[str] = set()
        for band in range(self.bands):
            start = band * self.rows
            key = tuple(sig[start:start + self.rows])
            candidates.update(self._buckets[band].get(key, []))
        scored = [(doc_id, jaccard(sh, self._docs[doc_id])) for doc_id in candidates]
        scored = [item for item in scored if item[1] >= self.threshold]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:top]
