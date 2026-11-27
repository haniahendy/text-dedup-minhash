# text-dedup-minhash

Near-duplicate text detection with MinHash signatures and LSH banding.
Finds pairs of documents whose word-shingle Jaccard similarity exceeds a
threshold, at corpus scale, without comparing every pair.

Pure Python 3.10+, zero dependencies.

## How it works

1. **Shingling**: each document becomes a set of word k-shingles
   (default k=3), hashed to 64-bit integers with blake2b.
2. **MinHash**: each shingle set gets a fixed-length signature
   (default 128 permutations, keyed-hash simulation, deterministic
   across runs).
3. **LSH banding**: signatures split into bands (default 32 bands of
   4 rows); documents colliding in any whole band become candidates.
4. **Exact verification**: candidate pairs are scored with the true
   Jaccard of their shingle sets, so reported pairs have no false
   positives above the threshold.

## Usage

```bash
PYTHONPATH=src python3 -m textdedup.cli documents.txt --threshold 0.5
```

```
documents: 3
near-duplicate pairs (jaccard >= 0.5): 1
  line-0 <-> line-1  jaccard=0.833
```

Input is a UTF-8 file with one document per line. `--json` emits
machine-readable output; `--num-perm`, `--bands`, and `-k` tune the
accuracy/recall tradeoff. The library API (`LSHIndex.add/query`,
`MinHasher`) supports incremental indexing and ad-hoc queries.

## Honest limits

- LSH banding can *miss* true near-duplicates (false negatives); the
  band/row split controls recall. Verify with exact Jaccard after
  raising band count if recall matters more than speed.
- Word-level shingles make it sensitive to heavy paraphrase: semantic
  duplicates with different wording score low. This catches
  copy-with-edits, not rewritten content.
- Very short documents (fewer than k words) collapse to a single
  whole-document shingle, which coarsens their similarity to 0 or 1.
- Signature comparison cost is linear in corpus size for banding but
  the exact-verification pass scales with candidate pairs; adversarial
  inputs (many documents sharing one band) degrade to pairwise work.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

12 tests: tokenizer and shingle counts, signature determinism and
estimate bounds, planted near-duplicate recovery, duplicate-ID and
band-divisibility validation, and query ranking.
