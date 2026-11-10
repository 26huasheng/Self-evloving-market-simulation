import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from sems_tools.public_trends import PublicTrendHistory, TREND_NAMES, TREND_SCHEMA
from sems_attention.observation import Observation
from sems_attention.actions import ActionSpace, ToolAction
from sems_attention.trend_encoder import trend_features
from sems_attention.trend_policy import TrendAttentionPolicy
from sems_attention.policy import NeuralAttentionPolicy
from sems_attention.runner import AttentionRunner

from support.trend import (
    Embeddings,
    Session,
    NOW,
    bulletin,
    observation,
    train_demo,
)


class TrendTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.policy = TrendAttentionPolicy(["STOP", "get_reviews"], seed=7)
        self.emb = Embeddings()
        self.x = np.ones(776, dtype=np.float32)

    def test_queries_change_search_state_but_not_market_vector(self):
        a = observation()
        b = observation(steps=1)
        before = self.policy.encode(self.x, a, self.emb)
        after = self.policy.encode(self.x, b, self.emb)
        for name in ("e", "lambda_matrix", "z", "public_trend_features"):
            self.assertTrue(torch.equal(before[name], after[name]), name)
        self.assertFalse(torch.equal(before["H"], after["H"]))
        self.assertFalse(torch.equal(before["query_state"], after["query_state"]))
        self.assertFalse(
            torch.equal(self.policy.routing_state(before, a), self.policy.routing_state(after, b))
        )
        space = ActionSpace(Session(False), a)
        self.assertFalse(
            torch.equal(
                self.policy.forward(self.x, a, self.emb, space)["tool_probs"],
                self.policy.forward(self.x, b, self.emb, space)["tool_probs"],
            )
        )

    def test_same_trends_for_different_profiles_and_product_content(self):
        a = observation()
        b = observation()
        b.cards["a"]["name"] = "A totally different product"
        b.cards["a"]["new_price"]["value"] = 500
        self.assertTrue(
            torch.equal(
                self.policy.encode(self.x, a, self.emb)["e"],
                self.policy.encode(self.x * 2, b, self.emb)["e"],
            )
        )

    def test_trend_changes_lambda_while_product_state_is_fixed(self):
        a = self.policy.encode(self.x, observation(False), self.emb)
        b = self.policy.encode(self.x, observation(True), self.emb)
        self.assertTrue(torch.equal(a["H"], b["H"]))
        for name in ("e", "lambda_matrix", "z"):
            self.assertFalse(torch.equal(a[name], b[name]), name)

    def test_unknown_is_distinct_from_zero_and_stale_is_rejected(self):
        a = observation()
        b = observation()
        for item in b.initial_shelf["public_market_context"]["features"].values():
            item.update(value=None, known=False)
        self.assertFalse(torch.equal(trend_features(a), trend_features(b)))
        b.initial_shelf["public_market_context"]["as_of"] = (NOW + timedelta(days=1)).isoformat()
        with self.assertRaises(ValueError):
            trend_features(b)

    def test_checkpoint_version_and_gradients(self):
        obs = observation()
        space = ActionSpace(Session(False), obs)
        loss = -self.policy.forward(self.x, obs, self.emb, space)["tool_probs"][1].log()
        loss.backward()
        for module in (
            self.policy.environment_encoder.trend,
            self.policy.lambda_layer,
            self.policy.search_fusion,
        ):
            self.assertTrue(
                any(
                    p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                    for p in module.parameters()
                )
            )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "v3.pt"
            self.policy.save(path)
            restored = NeuralAttentionPolicy.load(path)
            self.assertIsInstance(restored, TrendAttentionPolicy)
            self.assertEqual(
                restored.get_evolvable_state()["config"],
                self.policy.get_evolvable_state()["config"],
            )
            self.assertTrue(
                all(
                    torch.equal(v, restored.state_dict()[k])
                    for k, v in self.policy.state_dict().items()
                )
            )
            with self.assertRaises(ValueError):
                self.policy.load_evolvable_state(
                    NeuralAttentionPolicy(["STOP", "get_reviews"]).get_evolvable_state()
                )

    def test_public_history_warmup_and_no_future_or_purchase_leakage(self):
        start = NOW - timedelta(days=14)
        history = PublicTrendHistory(start)
        history.observe_prices(start, {"a": 10})
        early = history.bulletin(start, [])
        self.assertTrue(all(not x["known"] for x in early["features"].values()))
        for day in range(1, 15):
            history.observe_prices(start + timedelta(days=day), {"a": 10 if day < 8 else 20})
        events = [{"kind": "review", "visible_at": NOW - timedelta(days=1)}]
        a = history.bulletin(NOW, events)
        self.assertGreater(a["features"]["price_change"]["value"], 0)
        self.assertGreater(a["features"]["review_activity_change"]["value"], 0)
        future = [
            {"kind": "review", "visible_at": NOW + timedelta(days=1)},
            {"kind": "purchase", "visible_at": NOW},
        ]
        self.assertEqual(a, history.bulletin(NOW, events + future))
        history.observe_prices(NOW, {"a": 999})
        self.assertEqual(a, history.bulletin(NOW, events))

    def test_controlled_learning_can_change_reads_without_hardcoded_rule(self):
        _, report = train_demo()
        self.assertEqual([r["paid_reads"] for r in report["records"]], [3, 5])
        self.assertTrue(all(r["stop_reason"] == "policy_stop" for r in report["records"]))
        self.assertEqual(report["min_read_queries"], 0)


if __name__ == "__main__":
    unittest.main()
