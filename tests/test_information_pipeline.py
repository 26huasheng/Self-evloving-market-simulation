import tempfile
import unittest
import numpy as np
from sems_attention.observation import Observation
from sems_attention.actions import ToolAction
from sems.simulation_runner import SimulationRunner
from sems_attention.profiles.cache import CachedProfiles as Profiles


def observation():
    return Observation(
        {"products": [{"asin": "a", "name": "A", "new_price": {"value": 10, "state": "known"}}]},
        3,
        3,
        "2020-01-01",
        {"user_id": "u"},
    )


class InformationPipelineTests(unittest.TestCase):
    def test_feature_reads_do_not_fabricate_review_coverage(self):
        obs = observation()
        before = obs.shelf_stats().copy()
        obs.product_features("a")
        obs.product_texts("a")
        self.assertEqual(dict(obs.reviews), {})
        np.testing.assert_array_equal(before, obs.shelf_stats())
        self.assertEqual(float(obs.shelf_stats()[6]), 0)

    def test_empty_review_page_is_not_observed_review_content(self):
        obs = observation()
        obs.update(
            ToolAction("get_reviews", "a", {}),
            {"status": "ok", "remaining": 2, "data": {"reviews": []}},
        )
        self.assertEqual(float(obs.shelf_stats()[6]), 0)
        self.assertEqual(float(obs.query_state()[12]), 0)

    def test_price_query_updates_attention_card(self):
        obs = observation()
        new = {"value": 20, "state": "known", "source": "observed"}
        obs.update(
            ToolAction("get_price_info", "a", {"view": "offers"}),
            {"status": "ok", "remaining": 2, "data": {"offers": {"new": new}}},
        )
        self.assertEqual(obs.cards["a"]["new_price"], new)
        self.assertEqual(obs.initial_shelf["products"][0]["new_price"]["value"], 10)

    def test_formal_profile_missing_is_not_silently_replaced(self):
        history = [{"user_id": "u", "review_id": "r", "timestamp_ms": 1, "text": "evidence"}]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                Profiles(directory).load("u", history)

    def test_invalid_budgets_and_modes_rejected(self):
        for kwargs in (
            {"budget": -1},
            {"budget": True},
            {"budget": 1.5},
            {"mode": "bogus"},
            {"budget": 1, "min_read_queries": 2},
        ):
            with self.assertRaises(ValueError):
                SimulationRunner("unused", None, None, None, **kwargs)

    def test_resume_rejects_changed_configuration(self):
        from sems.offline import FixtureEmbeddings
        from sems.offline import OfflineProfiles, OfflineDecision

        case = __import__("pathlib").Path(__import__("os").environ["SEMS_CASE_PATH"])
        result = SimulationRunner(
            case, FixtureEmbeddings(), OfflineDecision(), OfflineProfiles(), budget=2
        ).run(max_sessions=1, max_days=1)
        with self.assertRaisesRegex(ValueError, "resume configuration"):
            SimulationRunner(
                case,
                FixtureEmbeddings(),
                OfflineDecision(),
                OfflineProfiles(),
                budget=3,
                resume_days=result["days"],
            ).run(max_sessions=1, max_days=2)

    def test_ordinary_invalid_decision_is_repaired(self):
        from sems.offline import FixtureEmbeddings
        from sems.offline import OfflineProfiles, OfflineDecision

        case = __import__("pathlib").Path(__import__("os").environ["SEMS_CASE_PATH"])

        class Decision(OfflineDecision):
            def decide_simulation(self, agent_id, profile, observation, correction=None):
                if not correction:
                    return {"invalid": "first response"}
                return super().decide_simulation(agent_id, profile, observation)

        result = SimulationRunner(
            case, FixtureEmbeddings(), Decision(), OfflineProfiles(), budget=2
        ).run(max_sessions=1)
        row = next(s for d in result["days"] for s in d["sessions"])
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["decision_format_repairs"], 1)


if __name__ == "__main__":
    unittest.main()
