import copy
import tempfile
import unittest
import warnings
from pathlib import Path
import numpy as np
import torch
from sems_attention import NeuralAttentionPolicy
from sems_attention.actions import ActionSpace
from sems_attention.observation import Observation


class Embeddings:
    def encode(self, text):
        return np.full(384, sum(text.encode()) % 17 / 17, dtype=np.float32)

    def mean(self, texts):
        return (
            np.mean([self.encode(t) for t in texts], axis=0)
            if texts
            else np.zeros(384, dtype=np.float32)
        )


class Registry:
    def actions(self):
        return [
            {"tool": "STOP", "args": {}, "allowed": True},
            *[
                {"tool": "get_reviews", "args": {"asin": a, "page": p}, "allowed": True}
                for a in ("a", "b", "c")
                for p in (1, 2)
            ],
            {"tool": "sort_shelf", "args": {"sort": "price"}, "allowed": True},
            {"tool": "sort_shelf", "args": {"sort": "rating"}, "allowed": False},
        ]


def observation():
    card = lambda x: {
        "asin": x,
        "name": x,
        "new_price": {"state": "known", "source": "observed", "value": 10},
    }
    return Observation({"products": [card(x) for x in ("a", "b", "c")]}, 3, 3, "2025-01-01", {})


class FormulaTest(unittest.TestCase):
    def setUp(self):
        self.p = NeuralAttentionPolicy(["STOP", "get_reviews", "sort_shelf"], seed=4)
        self.obs = observation()
        self.emb = Embeddings()
        self.x = np.ones(776, dtype=np.float32)
        self.space = ActionSpace(Registry(), self.obs)

    def test_formula_and_environment(self):
        out = self.p.encode(self.x, self.obs, self.emb)
        self.assertEqual(out["z"].shape, (64,))
        self.assertTrue(torch.allclose(out["lambda_p"], out["lambda_matrix"] @ out["p"], atol=1e-6))
        self.assertTrue(
            torch.allclose(
                out["z"], self.p.norm(self.p.W + out["lambda_matrix"] @ out["p"]), atol=1e-6
            )
        )
        self.obs.remaining = 1
        new = self.p.encode(self.x, self.obs, self.emb)
        self.assertFalse(torch.equal(out["e"], new["e"]))
        self.assertFalse(torch.equal(out["lambda_matrix"], new["lambda_matrix"]))
        self.assertFalse(torch.equal(out["z"], new["z"]))
        self.assertFalse(any(k.startswith(("W_e.", "W_q.")) for k in self.p.state_dict()))

    def test_heads_mask_and_gradients(self):
        out = self.p.forward(self.x, self.obs, self.emb, self.space, "get_reviews", "b")
        self.assertEqual(len(out["targets"]), 3)
        self.assertEqual(len(out["actions"]), 2)
        self.assertTrue(all(self.space.contains(a) for a in out["actions"]))
        loss = (
            -out["tool_probs"][1].log()
            - out["product_probs"][1].log()
            - out["parameter_probs"][0].log()
        )
        loss.backward()
        for module in (
            self.p.tool_head,
            self.p.product_head,
            self.p.parameter_head,
            self.p.profile_encoder,
            self.p.lambda_layer,
        ):
            self.assertTrue(
                any(
                    v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
                    for v in module.parameters()
                )
            )
        with torch.no_grad():
            self.p.tool_bias[:] = torch.tensor([100.0, -100.0, -100.0])
        action, info = self.p.select_action(self.x, self.obs, self.emb, self.space)
        self.assertTrue(action.is_stop)
        self.assertNotIn("targets", info["forward"])
        with torch.no_grad():
            self.p.tool_bias[:] = torch.tensor([-100.0, -100.0, 100.0])
        action, info = self.p.select_action(self.x, self.obs, self.emb, self.space)
        self.assertIsNone(action.asin)
        self.assertEqual(action.parameters, {"sort": "price"})
        self.assertTrue(self.space.contains(action))

    def test_masked_temperature_gradient_finite(self):
        self.space.actions = [a for a in self.space.actions if not a.is_stop]
        self.space.tools = [t for t in self.space.tools if t != "STOP"]
        out = self.p.forward(self.x, self.obs, self.emb, self.space)
        (-out["tool_probs"][1].log()).backward()
        self.assertTrue(torch.isfinite(self.p.temperature.grad))

    def test_checkpoint_versions(self):
        state = self.p.get_evolvable_state()
        legacy = copy.deepcopy(state)
        legacy["config"].pop("model_version")
        legacy["state_dict"]["W_e.weight"] = torch.zeros(64, 64)
        legacy["state_dict"]["W_q.weight"] = torch.zeros(64, 16)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "old.pt"
            torch.save(legacy, path)
            with self.assertWarnsRegex(UserWarning, "W_e.weight"):
                p = NeuralAttentionPolicy.load(path)
            self.assertTrue(
                all(torch.equal(v, p.state_dict()[k]) for k, v in state["state_dict"].items())
            )
            path = Path(d) / "new.pt"
            p.save(path)
            self.assertEqual(
                NeuralAttentionPolicy.load(path).get_evolvable_state()["config"]["model_version"],
                "attention_policy_v2",
            )
            bad = copy.deepcopy(legacy)
            bad["state_dict"]["extra"] = torch.zeros(1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaises(ValueError):
                    p.load_evolvable_state(bad)
            bad = copy.deepcopy(state)
            bad["config"]["model_version"] = "unknown"
            with self.assertRaises(ValueError):
                p.load_evolvable_state(bad)


if __name__ == "__main__":
    unittest.main()
