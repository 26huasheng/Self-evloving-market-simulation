"""Public-channel and day-commit integration for the versioned trend policy."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from sems.offline import FixtureEmbeddings
from sems.offline import OfflineDecision, OfflineProfiles
from sems.simulation_runner import SimulationRunner
from sems_tools import MarketEnvironment
from sems_attention import NeuralAttentionPolicy, TrendAttentionPolicy, Observation

CASE = Path(__import__("os").environ["SEMS_CASE_PATH"])
DAY = 86400000


class TrendSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def runner(self, **kwargs):
        return SimulationRunner(
            CASE,
            FixtureEmbeddings(),
            OfflineDecision(),
            OfflineProfiles(),
            seed=7,
            attention_version="trend_v3",
            **kwargs,
        )

    def test_public_channel_is_explicit_and_independent_of_private_queries(self):
        market = MarketEnvironment(CASE, {"simulation_mode": True})
        try:
            self.assertNotIn(
                "public_market_context", market.open_session("v2").initial_observation()
            )
            market.enable_public_trends()
            first = market.open_session("first")
            second = market.open_session("second")
            before = copy.deepcopy(first.initial_observation()["public_market_context"])
            asin = market.data.asins[0]
            for tool, args in [
                ("get_product_details", {"section": "overview"}),
                ("get_reviews", {"sort": "recent", "rating_filter": "all"}),
            ]:
                self.assertEqual(first.call(tool, asin=asin, **args)["status"], "ok")
            self.assertNotEqual(first.remaining, second.remaining)
            self.assertEqual(before, second.initial_observation()["public_market_context"])
            self.assertEqual(before, market.public_market_context())
            obs = Observation(
                second.initial_observation(), 6, 6, market.current_time.isoformat(), {}
            )
            self.assertEqual(obs.decision_observation()["public_market_context"], before)
            self.assertEqual(obs.for_decision_model()["public_market_context"], before)
            self.assertIn("not sales", before["feature_definitions"]["review_activity_change"])
        finally:
            market.close()

    def test_review_trend_uses_commit_publication_not_private_or_real_future_events(self):
        market = MarketEnvironment(CASE, {"simulation_mode": True, "public_trends": True})
        try:
            for day in range(13):
                market.commit_day(market.data.t0_ms + (day + 1) * DAY)
            session = market.open_session("writer")
            asin = market.data.asins[0]
            before = market.public_market_context()
            session.call("publish_review", asin=asin, text="PRIVATE UNTIL COMMIT")
            session.call("purchase_product", asin=asin)
            self.assertEqual(before, market.public_market_context())
            self.assertFalse(before["features"]["review_activity_change"]["known"])
            market.commit_day(market.data.t0_ms + 14 * DAY)
            public = market.public_market_context()
            self.assertEqual(public["features"]["review_activity_change"]["value"], 0.5)
            market.data.reviews[asin].append(
                {
                    "timestamp_ms": market.data.t0_ms + 30 * DAY,
                    "text": "REAL FUTURE REVIEW",
                    "rating": 1,
                }
            )
            market.events.append({"kind": "purchase", "visible_at": market.current_time})
            self.assertEqual(public, market.public_market_context())
            self.assertNotIn("PRIVATE UNTIL COMMIT", json.dumps(public))
            self.assertNotIn("REAL FUTURE REVIEW", json.dumps(public))
        finally:
            market.close()

    def test_sample_resume_reconstructs_trend_history_and_rng(self):
        full = self.runner(mode="sample").run(max_sessions=3, max_days=18)
        prefix = self.runner(mode="sample").run(max_sessions=3, max_days=9)
        resumed = self.runner(
            mode="sample", resume_days=json.loads(json.dumps(prefix["days"]))
        ).run(max_sessions=3, max_days=18)
        self.assertEqual(full, resumed)
        self.assertTrue(
            full["days"][14]["public_market_context"]["features"]["review_activity_change"]["known"]
        )
        with self.assertRaisesRegex(ValueError, "resume configuration"):
            SimulationRunner(
                CASE,
                FixtureEmbeddings(),
                OfflineDecision(),
                OfflineProfiles(),
                seed=7,
                mode="sample",
                attention_version="state_v2",
                resume_days=prefix["days"],
            ).run(max_sessions=3, max_days=18)

    def test_checkpoint_inference_and_version_mismatch(self):
        market = MarketEnvironment(CASE)
        try:
            registry = market.open_session("init").registry
            with tempfile.TemporaryDirectory() as directory:
                v2 = Path(directory) / "v2.pt"
                v3 = Path(directory) / "v3.pt"
                NeuralAttentionPolicy.from_registry(registry).save(v2)
                TrendAttentionPolicy.from_registry(registry).save(v3)
                with self.assertRaisesRegex(ValueError, "representation differ"):
                    self.runner(policy_path=v2).run(max_sessions=1, max_days=1)
                result = SimulationRunner(
                    CASE, FixtureEmbeddings(), OfflineDecision(), OfflineProfiles(), policy_path=v3
                ).run(max_sessions=1, max_days=1)
                self.assertEqual(result["attention_representation"], "trend_v3")
                self.assertIn("public_market_context", result["days"][0])
        finally:
            market.close()

    def test_real_session_budget_caps_remain_authoritative(self):
        for budget in (0, 1, 6):
            result = self.runner(budget=budget, mode="sample").run(max_sessions=1)
            rows = [row for day in result["days"] for row in day["sessions"]]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["status"], "ok")
            self.assertLessEqual(row["budget_used"], budget)
            self.assertGreaterEqual(row["observation"]["remaining_budget"], 0)
            for step in row["trajectory"]:
                if "environment_delta_after_query" in step:
                    self.assertEqual(step["environment_delta_after_query"], 0)
                    self.assertEqual(step["lambda_delta_after_query"], 0)


if __name__ == "__main__":
    unittest.main()
