import unittest, numpy as np
from sems.rsi.feedback import outcome_reward


class Emb:
    def encode(self, s):
        return {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}[s]


class TestFeedback(unittest.TestCase):
    def test_wrong_product_rating_never_used(self):
        d = {"purchase": True, "product_id": "b", "rating_probabilities": [0, 0, 0, 0, 1]}
        t = {"asin": "a", "rating": 5}
        r = outcome_reward(d, t, {"a": "A", "b": "B"}, Emb(), 1)
        self.assertIsNone(r["matched_rating_quality"])
        self.assertAlmostEqual(r["reward"], -0.001)

    def test_match_and_cost(self):
        d = {"purchase": True, "product_id": "a", "rating_probabilities": [0, 0, 0, 0, 1]}
        r = outcome_reward(d, {"asin": "a", "rating": 5}, {"a": "A"}, Emb(), 2)
        self.assertAlmostEqual(r["reward"], 1 + 0.2 + 0.1 - 0.002)
