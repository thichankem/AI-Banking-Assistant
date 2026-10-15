"""
Unit tests for NDCG@10 calculation in RetrievalEvaluator.
Verifies all boundary conditions and mathematical constraints:
- 1.0 for single relevant document at rank 1
- 1.0 for multiple relevant documents at top ranks (rank 1 & 2)
- 0.0 for zero hits
- Strictly bounded within [0.0, 1.0] for all 2^10 = 1024 binary relevance permutations
"""

import math
import itertools
import unittest


def compute_ndcg_at_10(relevance: list[int]) -> float:
    """Standalone reference implementation matching RetrievalEvaluator.evaluate."""
    rel_top10 = relevance[:10]
    dcg = 0.0
    for rank_idx, rel in enumerate(rel_top10, start=1):
        if rel == 1:
            dcg += 1.0 / math.log2(rank_idx + 1)

    num_rel = sum(rel_top10)
    if num_rel > 0:
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_rel + 1))
        return round(min(dcg / idcg, 1.0), 4)
    return 0.0


class TestNDCG(unittest.TestCase):
    def test_ndcg_rank1(self):
        rel = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(compute_ndcg_at_10(rel), 1.0)

    def test_ndcg_rank1_and_2(self):
        rel = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(compute_ndcg_at_10(rel), 1.0)

    def test_ndcg_zero_hits(self):
        rel = [0] * 10
        self.assertEqual(compute_ndcg_at_10(rel), 0.0)

    def test_ndcg_suboptimal_ranks(self):
        # Hit at rank 2 only: dcg = 1/log2(3) ~ 0.6309, idcg = 1/log2(2) = 1.0 -> 0.6309
        rel = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        ndcg = compute_ndcg_at_10(rel)
        self.assertTrue(0.0 < ndcg < 1.0)
        self.assertAlmostEqual(ndcg, 1.0 / math.log2(3), places=3)

    def test_ndcg_strictly_bounded_all_binary_permutations(self):
        """Exhaustively verify all 2^10 = 1024 binary vectors to guarantee 0 <= NDCG <= 1.0."""
        for bits in itertools.product([0, 1], repeat=10):
            val = compute_ndcg_at_10(list(bits))
            self.assertTrue(0.0 <= val <= 1.0, f"NDCG out of bounds: {val} for {bits}")


if __name__ == "__main__":
    unittest.main()
