import unittest
from sems.decision import DecisionModel, REVIEW_EVENT_SYSTEM


class TestDecisionInput(unittest.TestCase):
    def test_all_observed_data_preserved_and_mapped(self):
        class Client:
            mode = "mock"

            def complete_json(self, system, payload):
                self.system = system
                self.payload = payload
                return {
                    "selected_asin": "a",
                    "rating_probabilities": [0, 0, 0, 0, 1],
                    "rating": 5,
                    "review_text": "test",
                }

        c = Client()
        profile = {"user_id": "u", "narrative": "x" * 100000}
        observation = {
            "initial_shelf": {"products": [{"asin": "a"}]},
            "shelf": [{"product_id": "a"}],
            "tool_results": [
                {
                    "action": {"asin": "a", "tool_name": "get_product_details"},
                    "result": {"data": {"description": {"value": "long detail", "state": "known"}}},
                }
            ],
            "read_reviews": {},
            "remaining_budget": 3,
        }
        DecisionModel(c).decide_review_event("u", profile, observation)
        self.assertEqual(c.payload["profile"], profile)
        self.assertEqual(c.payload["observation"]["tool_results"], observation["tool_results"])
        self.assertEqual(c.payload["observation"]["shelf"], observation["shelf"])
        self.assertIn("ASIN", c.system)
        self.assertIn("unknown", c.system)
        self.assertNotIn("truth", c.payload)

    def test_only_exact_duplicate_profile_and_reviews_are_removed(self):
        class Client:
            mode = "mock"

            def complete_json(self, system, payload):
                self.payload = payload
                return {}

        review = {"id": "r1", "text": "visible review"}
        extra = {"id": "r2", "text": "extra review"}
        profile = {"user_id": "u", "narrative": "full"}
        obs = {
            "profile": profile,
            "tool_results": [
                {
                    "action": {"tool_name": "get_reviews", "asin": "a"},
                    "result": {"data": {"reviews": [review]}},
                }
            ],
            "read_reviews": {"a": [review, extra]},
        }
        client = Client()
        DecisionModel(client).decide_review_event("u", profile, obs)
        sent = client.payload["observation"]
        self.assertNotIn("profile", sent)
        self.assertEqual(sent["read_reviews"], {"a": [extra]})
        self.assertEqual(obs["read_reviews"], {"a": [review, extra]})


if __name__ == "__main__":
    unittest.main()
