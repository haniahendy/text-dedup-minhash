import unittest

from textdedup import LSHIndex, MinHasher, jaccard, shingle_hashes, tokenize

DOC_A = "the quick brown fox jumps over the lazy dog near the river bank"
# ~80% shingle overlap with DOC_A: one phrase changed.
DOC_B = "the quick brown fox jumps over the lazy dog near the river side"
DOC_C = "quantum entanglement enables correlations that classical systems cannot"


class TestShingles(unittest.TestCase):
    def test_tokenize_lowercases_and_strips(self):
        self.assertEqual(tokenize("Hello, WORLD! 123"), ["hello", "world", "123"])

    def test_shingle_count(self):
        hashes = shingle_hashes("one two three four five", k=3)
        self.assertEqual(len(hashes), 3)

    def test_short_doc_single_shingle(self):
        self.assertEqual(len(shingle_hashes("tiny doc", k=3)), 1)


class TestMinHash(unittest.TestCase):
    def test_identical_sets_identical_signatures(self):
        h = MinHasher(num_perm=64)
        s = shingle_hashes(DOC_A)
        self.assertEqual(h.signature(s), h.signature(s))

    def test_estimate_bounds(self):
        h = MinHasher(num_perm=128)
        sa = h.signature(shingle_hashes(DOC_A))
        sb = h.signature(shingle_hashes(DOC_B))
        sc = h.signature(shingle_hashes(DOC_C))
        self.assertEqual(MinHasher.estimate(sa, sa), 1.0)
        est_ab = MinHasher.estimate(sa, sb)
        est_ac = MinHasher.estimate(sa, sc)
        true_ab = jaccard(shingle_hashes(DOC_A), shingle_hashes(DOC_B))
        self.assertGreater(est_ab, est_ac)
        # Estimate should be within a loose band of the true value.
        self.assertLess(abs(est_ab - true_ab), 0.2)

    def test_empty_set_signature(self):
        h = MinHasher(num_perm=8)
        self.assertEqual(len(h.signature(set())), 8)


class TestLSHIndex(unittest.TestCase):
    def test_finds_planted_near_duplicate(self):
        index = LSHIndex(num_perm=128, bands=32, threshold=0.5)
        index.add("a", DOC_A)
        index.add("b", DOC_B)
        index.add("c", DOC_C)
        pairs = index.similar_pairs()
        self.assertEqual(len(pairs), 1)
        a, b, sim = pairs[0]
        self.assertEqual({a, b}, {"a", "b"})
        self.assertGreater(sim, 0.5)

    def test_unrelated_docs_not_reported(self):
        index = LSHIndex(threshold=0.7)
        index.add("a", DOC_A)
        index.add("c", DOC_C)
        self.assertEqual(index.similar_pairs(), [])

    def test_exact_duplicates_score_one(self):
        index = LSHIndex(threshold=0.99)
        index.add("a", DOC_A)
        index.add("a2", DOC_A)
        pairs = index.similar_pairs()
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0][2], 1.0)

    def test_duplicate_id_rejected(self):
        index = LSHIndex()
        index.add("a", DOC_A)
        with self.assertRaises(ValueError):
            index.add("a", DOC_C)

    def test_bands_must_divide_num_perm(self):
        with self.assertRaises(ValueError):
            LSHIndex(num_perm=100, bands=32)

    def test_query_returns_match(self):
        index = LSHIndex(threshold=0.5)
        index.add("a", DOC_A)
        index.add("c", DOC_C)
        hits = index.query("the quick brown fox jumps over the lazy dog near the river bank")
        self.assertEqual(hits[0][0], "a")
        self.assertAlmostEqual(hits[0][1], 1.0)


if __name__ == "__main__":
    unittest.main()
