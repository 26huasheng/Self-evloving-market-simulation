import unittest
from sems.rsi.reward import compute_reward, RewardConfig, centered_advantages


def step(reviews, cost=1):
    return {
        "tool": "get_reviews",
        "asin": "a",
        "result_status": "ok",
        "cost": cost,
        "result": {"data": {"reviews": [{"id": x, "text": "review"} for x in reviews]}},
    }


class RewardTests(unittest.TestCase):
    def test_repeat_and_valid_pagination(self):
        a = compute_reward(
            [step(["1", "2"]), step(["1", "2"]), step(["3", "4"]), {"tool": "STOP"}],
            {"purchase": True, "product_id": "b"},
            {"asin": "a"},
        )
        self.assertEqual([q["new_information_units"] for q in a["queries"]], [2, 0, 2])
        self.assertAlmostEqual(a["reward"], 0.04 - 0.015)

    def test_cap_and_match(self):
        a = compute_reward(
            [step(["1", "2", "3", "4"]), step(["5", "6", "7", "8"])],
            {"purchase": True, "product_id": "a"},
            {"asin": "a"},
            RewardConfig(max_information=4),
        )
        self.assertEqual(a["information_units"], 4)
        self.assertEqual(a["raw_information_units"], 6)
        self.assertAlmostEqual(a["reward"], 1.03)

    def test_centered_advantage(self):
        baseline, adv = centered_advantages([0, 1, 2])
        self.assertEqual(baseline, 1)
        self.assertEqual(adv, [-1, 0, 1])
        with self.assertRaises(ValueError):
            centered_advantages([0.2, 0.2])

    def test_nonreview_repeat(self):
        s = {
            "tool": "get_rating_info",
            "asin": "a",
            "result_status": "ok",
            "cost": 1,
            "result": {"data": {"average_rating": {"state": "known", "value": 4}}},
        }
        a = compute_reward([s, s], {"purchase": True, "product_id": "b"}, {"asin": "a"})
        self.assertEqual([q["new_information_units"] for q in a["queries"]], [1, 0])

    def test_information_independent_of_target(self):
        tr = [step(["1", "2"])]
        decision = {"purchase": True, "product_id": "c"}
        a = compute_reward(tr, decision, {"asin": "a"})
        b = compute_reward(tr, decision, {"asin": "b"})
        self.assertEqual(a["information_units"], b["information_units"])
        self.assertEqual(a["reward"], b["reward"])

    def test_empty_sort_has_no_information_reward(self):
        s = {
            "tool": "sort_shelf",
            "asin": None,
            "result_status": "ok",
            "cost": 1,
            "result": {"data": {"products": [{"asin": "a"}]}},
        }
        a = compute_reward([s], {"purchase": True, "product_id": "b"}, {"asin": "a"})
        self.assertEqual(a["information_units"], 0)
        self.assertEqual(a["reward"], -0.005)

    def test_invalid_and_unavailable(self):
        with self.assertRaises(ValueError):
            compute_reward([], {"purchase": False}, {"asin": "a"})
        a = compute_reward(
            [
                {
                    "tool": "get_rating_info",
                    "asin": "a",
                    "result_status": "invalid_request",
                    "cost": 0,
                    "result": {"data": {"average_rating": {"state": "known", "value": 5}}},
                }
            ],
            {"purchase": True, "product_id": "a"},
            {"asin": "a"},
        )
        self.assertEqual(a["information_units"], 0)


if __name__ == "__main__":
    unittest.main()


class AblationRewardTests(unittest.TestCase):
    def test_scheme_c_requires_product_match(self):
        tr = [step(["x"])]
        d = {"purchase": True, "product_id": "wrong", "rating_probabilities": [0, 0, 0, 0, 1]}
        target = {"asin": "true", "rating": 5}
        r = compute_reward(tr, d, target, RewardConfig(scheme="C"))
        self.assertIsNone(r["rating_quality"])
        self.assertEqual(r["rating_quality_term"], 0)
        d["product_id"] = "true"
        r = compute_reward(tr, d, target, RewardConfig(scheme="C"))
        self.assertEqual(r["rating_quality"], 1)
        self.assertAlmostEqual(r["reward"], 1.045)

    def test_a_has_no_information_bonus(self):
        r = compute_reward(
            [step(["x"])],
            {"purchase": True, "product_id": "b"},
            {"asin": "a"},
            RewardConfig(scheme="A"),
        )
        self.assertEqual(r["information_term"], 0)
        self.assertEqual(r["reward"], -0.005)
