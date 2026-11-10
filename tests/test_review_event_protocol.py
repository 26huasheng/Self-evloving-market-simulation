import unittest
from sems.protocol import validate_review_event
from sems.evaluation.review_events import evaluate_review_events, ndcg5_ties


class ProtocolTests(unittest.TestCase):
    def test_rating_validation(self):
        x = {
            "selected_asin": "a",
            "rating_probabilities": [0, 0, 0, 0.5, 0.5],
            "rating": 5,
            "review_text": "fine",
        }
        d, actions = validate_review_event(x, {"a"})
        self.assertEqual(len(actions), 3)
        self.assertEqual(d["rating_probabilities"], x["rating_probabilities"])
        for bad in ([0, 0, 0, 0, 0], [0, 0, 0, 0, float("nan")], [1, 0], [-1, 1, 1, 0, 0]):
            with self.assertRaises(ValueError):
                validate_review_event({**x, "rating_probabilities": bad}, {"a"})

    def test_formula_event_pair_even_wrong_product(self):
        r = {
            "session_id": "s",
            "user_id": "u",
            "scheduled_at_ms": 1,
            "status": "ok",
            "decision": {
                "product_id": "b",
                "rating_probabilities": [0, 0, 0, 0.5, 0.5],
                "rating": 5,
                "review_text": "ok",
            },
        }
        truth = {"s": {"user_id": "u", "timestamp_ms": 1, "asin": "a", "rating": 4}}
        m = evaluate_review_events({"days": [{"sessions": [r]}]}, truth, ["a", "b"])["metrics"]
        self.assertAlmostEqual(m["JSD_full"]["value"], 1)
        self.assertEqual(m["TVD"]["value"], 1)
        self.assertEqual(m["selection_rate_error"]["value"], 0)
        self.assertAlmostEqual(m["MAE"]["value"], 0.5)
        self.assertAlmostEqual(m["W1"]["value"], 0.5)
        self.assertEqual(m["sentiment_consistency"]["status"], "not_evaluated")

    def test_optional_judge_exact_match_normalizes(self):
        r = {
            "session_id": "s",
            "user_id": "u",
            "scheduled_at_ms": 1,
            "status": "ok",
            "decision": {
                "product_id": "a",
                "rating_probabilities": [0, 0, 0, 0, 1],
                "rating": 5,
                "review_text": "good",
            },
        }
        t = {"s": {"user_id": "u", "timestamp_ms": 1, "asin": "a", "rating": 5, "text": "great"}}

        class Judge:
            def score(self, payload):
                return {
                    "sentiment_consistency": 4,
                    "product_factual_consistency": 2,
                    "content_alignment": 0,
                }

        m = evaluate_review_events(
            {"days": [{"sessions": [r]}]},
            t,
            ["a"],
            judge=Judge(),
            t0_facts={"a": {"title": "x"}},
            judge_max=1,
        )["metrics"]
        self.assertEqual(m["sentiment_consistency"]["value"], 1)
        self.assertEqual(m["product_factual_consistency"]["value"], 0.5)
        self.assertEqual(m["content_alignment"]["value"], 0)
        t["s"]["asin"] = "b"
        m = evaluate_review_events(
            {"days": [{"sessions": [r]}]}, t, ["a", "b"], judge=Judge(), judge_max=1
        )["metrics"]
        self.assertEqual(m["sentiment_consistency"]["status"], "insufficient_data")

    def test_failed_event_not_counted_as_none(self):
        good = {
            "session_id": "s",
            "user_id": "u",
            "scheduled_at_ms": 1,
            "status": "ok",
            "decision": {
                "product_id": "a",
                "rating_probabilities": [0, 0, 0, 0, 1],
                "rating": 5,
                "review_text": "ok",
            },
        }
        failed = {
            "session_id": "t",
            "user_id": "v",
            "scheduled_at_ms": 2,
            "status": "decision_failed",
        }
        truth = {
            "s": {"user_id": "u", "timestamp_ms": 1, "asin": "a", "rating": 5},
            "t": {"user_id": "v", "timestamp_ms": 2, "asin": "a", "rating": 1},
        }
        e = evaluate_review_events({"days": [{"sessions": [good, failed]}]}, truth, ["a"])
        self.assertIsNone(e["metrics"]["JSD_full"]["value"])
        self.assertEqual(e["metrics"]["JSD_full"]["status"], "incomplete_cohort")
        self.assertEqual(e["coverage"]["valid_users"], 1)

    def test_wasserstein_distribution(self):
        r = {
            "session_id": "s",
            "user_id": "u",
            "scheduled_at_ms": 1,
            "status": "ok",
            "decision": {
                "product_id": "a",
                "rating_probabilities": [0.5, 0, 0, 0, 0.5],
                "rating": 5,
                "review_text": "ok",
            },
        }
        truth = {"s": {"user_id": "u", "timestamp_ms": 1, "asin": "a", "rating": 1}}
        m = evaluate_review_events({"days": [{"sessions": [r]}]}, truth, ["a"])["metrics"]
        self.assertEqual(m["W1"]["value"], 2)
        self.assertEqual(m["MAE"]["value"], 2)

    def test_ndcg_ties_and_duplicate_users(self):
        import math

        self.assertAlmostEqual(
            ndcg5_ties({"a": 1}, {"a": 1, "b": 1}, ["a", "b"]), (1 + 1 / math.log2(3)) / 2
        )
        r = {"session_id": "s", "user_id": "u"}
        with self.assertRaises(ValueError):
            evaluate_review_events(
                {"days": [{"sessions": [r, {**r, "session_id": "x"}]}]}, {}, ["a"]
            )


if __name__ == "__main__":
    unittest.main()
