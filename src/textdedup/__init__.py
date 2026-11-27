"""Near-duplicate text detection with MinHash and LSH banding.

Pure-Python pipeline: word k-shingles -> MinHash signatures -> LSH
candidate pairs -> exact Jaccard verification. Suitable for de-duplicating
training corpora, dedup before indexing, and clustering near-identical
documents without a vector database.
"""

from .minhash import MinHasher, jaccard
from .lsh import LSHIndex
from .shingles import shingle_hashes, tokenize

__all__ = ["LSHIndex", "MinHasher", "jaccard", "shingle_hashes", "tokenize"]

__version__ = "0.1.0"
