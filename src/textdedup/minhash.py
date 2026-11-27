"""MinHash signatures over 64-bit shingle hashes.

Each permutation is simulated by a keyed blake2b hash of the shingle hash,
which is deterministic across runs (unlike Python's salted ``hash``).
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Set

MASK64 = (1 << 64) - 1


def jaccard(a: Set[int], b: Set[int]) -> float:
    """Exact Jaccard similarity of two shingle-hash sets."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class MinHasher:
    """Compute fixed-length MinHash signatures for shingle sets."""

    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        if num_perm < 1:
            raise ValueError("num_perm must be >= 1")
        self.num_perm = num_perm
        self.seed = seed

    def _permuted(self, value: int, perm: int) -> int:
        payload = value.to_bytes(8, "big")
        key = (self.seed.to_bytes(4, "big") + perm.to_bytes(4, "big"))
        digest = hashlib.blake2b(payload, digest_size=8, key=key).digest()
        return int.from_bytes(digest, "big")

    def signature(self, shingle_set: Set[int]) -> List[int]:
        """MinHash signature: the min permuted hash per permutation."""
        if not shingle_set:
            return [MASK64] * self.num_perm
        mins = [MASK64] * self.num_perm
        for value in shingle_set:
            for perm in range(self.num_perm):
                h = self._permuted(value, perm)
                if h < mins[perm]:
                    mins[perm] = h
        return mins

    @staticmethod
    def estimate(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
        """Jaccard estimate from two equal-length signatures."""
        if len(sig_a) != len(sig_b):
            raise ValueError("signatures must have equal length")
        if not sig_a:
            return 1.0
        matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
        return matches / len(sig_a)
