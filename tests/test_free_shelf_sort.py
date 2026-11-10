import unittest
from pathlib import Path
from sems_tools import MarketEnvironment
from sems_attention.actions import ActionSpace, ToolAction
from sems_attention.observation import Observation
from sems.rsi.reward import information_units

CASE = Path(__import__("os").environ["SEMS_CASE_PATH"])


class SortTests(unittest.TestCase):
    def setUp(self):
        self.market = MarketEnvironment(CASE, {"simulation_mode": True, "query_budget": 4})
        self.addCleanup(self.market.close)
        self.s = self.market.open_session("test", "test")

    def test_sort_free_and_no_duplicate_or_hidden_rank(self):
        acts = self.s.registry.actions()
        available = [a for a in acts if a["tool"] == "sort_shelf" and a["allowed"]]
        self.assertTrue(all(a["cost"] == 0 for a in available))
        self.assertFalse(any(a["args"]["sort_by"] in ("rating", "rating_count") for a in available))
        hidden = self.s.registry.call("sort_shelf", sort_by="rating", order="desc")
        self.assertEqual(hidden["status"], "invalid_request")
        self.assertEqual(self.s.remaining, 4)
        self.assertTrue(available)
        args = available[0]["args"]
        before = self.s.remaining
        result = self.s.registry.call("sort_shelf", **args)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["cost"], 0)
        self.assertEqual(self.s.remaining, before)
        self.assertEqual(self.s.registry.call("sort_shelf", **args)["status"], "invalid_request")
        self.assertEqual(
            information_units({"tool": "sort_shelf", "result_status": "ok", "result": result}),
            set(),
        )
        asin = self.market.data.asins[0]
        read = self.s.registry.call("get_rating_info", asin=asin)
        self.assertEqual(read["cost"], 1)
        self.assertEqual(self.s.remaining, before - 1)

    def test_sort_bound_and_observation(self):
        obs = Observation(
            self.s.initial_observation(),
            self.s.remaining,
            4,
            self.market.current_time.isoformat(),
            {},
        )
        for _ in range(3):
            available = [
                x for x in self.s.registry.actions() if x["tool"] == "sort_shelf" and x["allowed"]
            ]
            if not available:
                break
            a = ToolAction("sort_shelf", None, available[0]["args"])
            r = self.s.registry.call("sort_shelf", **a.arguments)
            obs.update(a, r)
            self.assertEqual(obs.order, self.s.order)
            self.assertNotIn(a.key(), {x.key() for x in ActionSpace(self.s.registry, obs).actions})
        if len(self.s.sort_history) == 3:
            self.assertFalse(
                any(x["allowed"] for x in self.s.registry.actions() if x["tool"] == "sort_shelf")
            )

    def test_sort_mask_log_probability_and_environment(self):
        import numpy as np, torch
        from sems_attention import NeuralAttentionPolicy
        from sems.rsi.trainer import BehaviorCloningTrainer
        from test_attention_formula import Embeddings

        obs = Observation(
            self.s.initial_observation(),
            self.s.remaining,
            4,
            self.market.current_time.isoformat(),
            {},
        )
        policy = NeuralAttentionPolicy.from_registry(self.s.registry, seed=7)
        emb = Embeddings()
        x = np.ones(776, dtype=np.float32)
        space = ActionSpace(self.s.registry, obs)
        action = next(a for a in space.actions if a.tool_name == "sort_shelf")
        before = policy.encode(x, obs, emb)["e"].detach()
        lp = BehaviorCloningTrainer.action_log_prob(
            policy,
            {"profile_x": x, "observation": obs, "action_space": space, "action": action},
            emb,
        )
        out = policy.forward(x, obs, emb, space, "sort_shelf", None)
        expected = (
            out["tool_probs"][policy.tool_names.index("sort_shelf")].log()
            + out["parameter_probs"][[a.key() for a in out["actions"]].index(action.key())].log()
        )
        self.assertTrue(torch.allclose(lp, expected))
        (-lp).backward()
        self.assertTrue(torch.isfinite(policy.temperature.grad))
        result = self.s.registry.call("sort_shelf", **action.arguments)
        obs.update(action, result)
        after = policy.encode(x, obs, emb)["e"].detach()
        self.assertFalse(torch.allclose(before, after, atol=1e-8, rtol=1e-8))
        self.assertEqual(obs.remaining, 4)

    def test_free_sort_cannot_satisfy_min_read_queries(self):
        import numpy as np, torch
        from sems_attention import NeuralAttentionPolicy, AttentionRunner
        from test_attention_formula import Embeddings

        p = NeuralAttentionPolicy.from_registry(self.s.registry, seed=3)
        with torch.no_grad():
            p.tool_bias.fill_(-20)
            p.tool_bias[p.tool_names.index("sort_shelf")] = 20
            p.tool_bias[0] = 10
        result = AttentionRunner(
            p, Embeddings(), mode="greedy", min_read_queries=2, min_query_kinds=("read", "shelf")
        ).run(self.s, {}, np.ones(776, dtype=np.float32))
        steps = [s for s in result["trajectory"] if s["tool"] != "STOP"]
        reads = [s for s in steps if s["cost"] > 0]
        sorts = [s for s in steps if s["tool"] == "sort_shelf"]
        self.assertGreaterEqual(len(reads), 2)
        self.assertLessEqual(len(sorts), 3)
        self.assertTrue(all(s["cost"] == 0 for s in sorts))
        self.assertEqual(4 - self.s.remaining, len(reads))

    def test_hidden_ratings_cannot_control_sort(self):
        first = self.s.registry.call("sort_shelf", sort_by="rating", order="desc")
        self.assertEqual(first["status"], "invalid_request")
        self.assertEqual(self.s.remaining, 4)


if __name__ == "__main__":
    unittest.main()
