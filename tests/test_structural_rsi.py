import tempfile, unittest
from pathlib import Path
import torch
from sems_attention import NeuralAttentionPolicy
from sems.rsi.proposer import StructuralExecutor, StructuralProposal
from test_attention_formula import FormulaTest


class StructuralTests(unittest.TestCase):
    def test_migration_gradient_and_checkpoint(self):
        f = FormulaTest()
        f.setUp()
        p = f.p
        q = StructuralExecutor().apply(p, StructuralProposal("candidate"))
        x = p.encode(f.x, f.obs, f.emb)
        y = q.encode(f.x, f.obs, f.emb)
        self.assertTrue(torch.equal(x["z"], y["z"]))
        self.assertTrue(all(torch.equal(v, q.state_dict()[k]) for k, v in p.state_dict().items()))
        result = q.forward(f.x, f.obs, f.emb, f.space, "get_reviews", "b")
        (-result["tool_probs"][1].log() - result["product_probs"][1].log()).backward()
        self.assertGreater(float(q.environment_encoder.condition_adapter.weight.grad.norm()), 0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.pt"
            q.save(path)
            loaded = NeuralAttentionPolicy.load(path)
            self.assertEqual(loaded.structure, {"query_context_v1": True})
            self.assertTrue(torch.equal(q.W, loaded.W))
        with self.assertRaises(ValueError):
            StructuralExecutor().apply(p, StructuralProposal("x", feature="bad"))


if __name__ == "__main__":
    unittest.main()


class AutoControllerTests(unittest.TestCase):
    def test_rejection_does_not_stop_rounds(self):
        from sems.rsi.controller import AutoEvolutionController
        from sems.rsi.history import History
        from sems.rsi.types import ValidationResult

        class Pipe:
            def __init__(self):
                self.rounds = 0
                self.events = []

            def train(self, p, label, batches):
                with torch.no_grad():
                    p.W.add_(0.01)
                return p, {"batches": batches, "simulation_ref": None, "evaluation_ref": None}

            def compare(self, a, b, label):
                return ValidationResult("rejected", "offline test")

            def diagnose(self, log):
                return {"unmatched_fraction": 1}

            def should_propose(self, d):
                return True

            def usage(self):
                return {"actual_requests": 0}

            def completed_rounds(self):
                return self.rounds

            def record_event(self, e):
                self.events.append(e)

            def finish_round(self, n):
                self.rounds = n

        with tempfile.TemporaryDirectory() as d:
            hist = History(d)
            hist.initialize(NeuralAttentionPolicy(["STOP", "get_reviews"], seed=1))
            pipe = Pipe()
            result = AutoEvolutionController(
                hist, pipe, rounds=2, numeric_batches=2, structural_batches=1
            ).run()
            self.assertEqual(pipe.rounds, 2)
            self.assertEqual(len(hist.data["records"]), 4)
            self.assertEqual(hist.current.version_id, "initial")
            structural = [r for r in hist.data["records"] if r["candidate"]["kind"] == "structure"]
            self.assertEqual(len(structural), 2)
            self.assertTrue(
                all(
                    NeuralAttentionPolicy.load(r["candidate"]["checkpoint"]).structure
                    for r in structural
                )
            )
