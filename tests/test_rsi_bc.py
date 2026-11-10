import tempfile
import unittest
from pathlib import Path
import torch
from test_attention_formula import FormulaTest
from sems.rsi.history import History
from sems.rsi.trainer import BehaviorCloningTrainer, ReinforceTrainer
from sems.rsi.types import TrainingConfig
from sems.rsi.pilot_api import PilotDecisionClient


class BCTest(unittest.TestCase):
    def test_training_and_persistence(self):
        fixture = FormulaTest()
        fixture.setUp()
        policy = fixture.p
        space = fixture.space
        sample = {
            "profile_x": fixture.x,
            "observation": fixture.obs,
            "action_space": space,
            "action": next(
                a
                for a in space.actions
                if a.tool_name == "get_reviews" and a.asin == "b" and a.parameters["page"] == 1
            ),
        }
        stop = dict(sample, action=next(a for a in space.actions if a.is_stop))
        trainer = BehaviorCloningTrainer([sample, stop], fixture.emb)
        with tempfile.TemporaryDirectory() as d:
            hist = History(d)
            hist.initialize(policy)
            initial = {k: v.clone() for k, v in policy.state_dict().items()}
            candidate = trainer.train(
                policy,
                None,
                None,
                None,
                TrainingConfig(seed=1, options={"epochs": 3, "learning_rate": 0.001}),
                "test",
                hist,
            )
            self.assertTrue(Path(candidate.checkpoint).exists())
            self.assertTrue(Path(candidate.optimizer_checkpoint).exists())
            self.assertTrue(
                any(not torch.equal(v, policy.state_dict()[k]) for k, v in initial.items())
            )
            self.assertLess(trainer.log["loss_after"], trainer.log["loss_before_steps"][0])
            for prefix in (
                "tool_head",
                "product_head",
                "parameter_head",
                "profile_encoder",
                "lambda_layer",
            ):
                self.assertTrue(
                    any(
                        v > 0
                        for k, v in trainer.log["gradient_norm_max"].items()
                        if k.startswith(prefix)
                    )
                )
            self.assertTrue(
                all(
                    torch.equal(v, hist.restore("initial").state_dict()[k])
                    for k, v in initial.items()
                )
            )

    def test_reinforce_recomputes_gradient_and_requires_on_policy(self):
        fixture = FormulaTest()
        fixture.setUp()
        policy = fixture.p
        sample = {
            "profile_x": fixture.x,
            "observation": fixture.obs,
            "action_space": fixture.space,
            "action": next(
                a
                for a in fixture.space.actions
                if a.tool_name == "get_reviews" and a.asin == "b" and a.parameters["page"] == 1
            ),
        }
        with tempfile.TemporaryDirectory() as d:
            hist = History(d)
            hist.initialize(policy, "bc")
            before = {k: v.clone() for k, v in policy.state_dict().items()}
            trainer = ReinforceTrainer(
                [{"policy_version": "bc", "samples": [sample], "reward": 1.0}], fixture.emb, "bc"
            )
            candidate = trainer.train(
                policy, None, None, None, TrainingConfig(seed=1), "feedback", hist
            )
            self.assertTrue(Path(candidate.checkpoint).exists())
            self.assertTrue(
                any(not torch.equal(v, policy.state_dict()[k]) for k, v in before.items())
            )
            bad = ReinforceTrainer(
                [{"policy_version": "old", "samples": [sample], "reward": 1.0}], fixture.emb, "bc"
            )
            with self.assertRaisesRegex(ValueError, "off-policy"):
                bad.train(policy, None, None, None, TrainingConfig(), "other", hist)

    def test_hard_cap_and_disabled_roles(self):
        with self.assertRaises(ValueError):
            PilotDecisionClient(max_requests=151)
        client = object.__new__(PilotDecisionClient)
        with self.assertRaises(ValueError):
            client.chat(module="profile")


if __name__ == "__main__":
    unittest.main()
