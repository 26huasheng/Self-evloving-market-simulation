import json
import tempfile
import unittest
from pathlib import Path
import torch
from sems_attention import NeuralAttentionPolicy
from sems.evaluation.sessions import evaluate
from sems.rsi import (
    Controller,
    History,
    RSIConfig,
    Diagnoser,
    Validator,
    ValidationResult,
    ImprovementProposal,
    ProposalAdapter,
    CandidatePolicy,
)


def fixture():
    s = {
        "session_id": "s1",
        "user_id": "u1",
        "scheduled_at_ms": 1,
        "status": "ok",
        "trajectory": [
            {"tool": "get_reviews", "asin": "p1"},
            {"tool": "STOP", "reason": "policy_stop"},
        ],
        "budget_used": 1,
        "decision": {
            "purchase": True,
            "product_id": "p1",
            "submit_rating": True,
            "rating": 4,
            "publish_review": True,
            "review_text": "good",
        },
    }
    sim = {
        "schema": "simulation_result_v1",
        "case_id": "case",
        "seed": 7,
        "budget": 3,
        "attention_model": "initial",
        "profile_version": "v1",
        "days": [{"sessions": [s]}],
    }
    truth = {"s1": {"user_id": "u1", "timestamp_ms": 1, "asin": "p1", "rating": 4, "text": "good"}}
    return sim, evaluate(sim, truth, ["p1", "p2"])


def policy():
    return NeuralAttentionPolicy(["STOP", "get_reviews"], seed=7)


def weights(p):
    return {k: v.clone() for k, v in p.state_dict().items()}


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def evaluator(report):
    def evaluate_policy(p, conditions):
        return json.loads(json.dumps(report))

    return evaluate_policy


class RSITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sim, self.report = fixture()
        self.h = History(self.tmp.name)
        self.c = Controller(self.h, RSIConfig(max_rounds=3, seed=7))
        self.c.initialize(policy())

    def test_diagnosis_and_default_pending(self):
        d = Diagnoser().diagnose(self.sim, self.report)
        self.assertTrue(d.sufficient)
        self.assertEqual(d.query_statistics["total_queries"], 1)
        self.assertIsNone(d.metrics["JSD_full"]["value"])
        self.assertEqual(d.budget_stop["stop_reasons"]["policy_stop"], 1)
        r = self.c.run_round(self.sim, self.report)
        self.assertEqual(r.validation["status"], "pending")
        self.assertFalse(r.candidate["metadata"]["trained"])
        self.assertEqual(self.h.current.version_id, "initial")

    def test_accept_reject_resume_and_rollback(self):
        self.c.validator = Validator(
            evaluator(self.report),
            lambda a, b: ValidationResult("accepted", "explicit offline test"),
        )
        r = self.c.run_round(self.sim, self.report)
        self.assertEqual(self.h.current.version_id, r.candidate["version_id"])
        before = weights(self.h.restore(self.h.current.version_id))
        self.c.validator = Validator(
            evaluator(self.report),
            lambda a, b: ValidationResult("rejected", "explicit offline test"),
        )
        self.c.run_round(self.sim, self.report)
        self.assertTrue(same(before, weights(self.h.restore(self.h.current.version_id))))
        h = History(self.tmp.name)
        self.assertEqual(len(h.data["records"]), 2)
        h.rollback("initial")
        self.assertEqual(h.current.version_id, "initial")
        with self.assertRaises(ValueError):
            h.rollback("round-0002-00")

    def test_modified_candidate_never_overwrites_current(self):
        class TestTrainer:
            def train(
                self, policy, diagnosis, simulation, evaluation, config, candidate_version, history
            ):
                with torch.no_grad():
                    policy.temperature.add_(1)
                path = history.save_policy(candidate_version, policy)
                return CandidatePolicy(
                    candidate_version,
                    history.current.version_id,
                    "training",
                    path,
                    policy.get_evolvable_state()["config"],
                    metadata={"test_only": True},
                )

        self.c.trainer = TestTrainer()
        original = weights(self.h.restore("initial"))
        self.c.validator = Validator(
            evaluator(self.report), lambda a, b: ValidationResult("rejected", "offline test")
        )
        record = self.c.run_round(self.sim, self.report)
        self.assertTrue(same(original, weights(self.h.restore("initial"))))
        self.assertFalse(same(original, weights(self.h.restore(record.candidate["version_id"]))))
        self.c.validator = Validator(
            evaluator(self.report), lambda a, b: ValidationResult("accepted", "offline test")
        )
        accepted = self.c.run_round(self.sim, self.report)
        self.assertEqual(self.h.current.version_id, accepted.candidate["version_id"])
        self.assertFalse(same(original, weights(self.h.restore(self.h.current.version_id))))

    def test_structure_and_isolation(self):
        proposal = ImprovementProposal(
            "attention", "config", {"example": 1}, "test only", ["fixed tools"], "round-0001-00"
        )
        self.c.proposer = ProposalAdapter(proposal)
        self.c.validator = Validator(
            evaluator(self.report), lambda a, b: ValidationResult("rejected", "test")
        )
        original = weights(self.h.restore("initial"))
        r = self.c.run_round(self.sim, self.report, path="structure")
        self.assertEqual(r.candidate["kind"], "structure")
        self.assertFalse(r.candidate["metadata"]["proposal_applied"])
        self.assertTrue(same(original, weights(self.h.restore("initial"))))

    def test_exception_and_missing_metrics(self):
        original = Path(self.h.current.checkpoint).read_bytes()

        def fail(*args):
            raise RuntimeError("test failure")

        self.c.validator = Validator(fail)
        with self.assertRaises(RuntimeError):
            self.c.run_round(self.sim, self.report)
        self.assertEqual(Path(self.h.current.checkpoint).read_bytes(), original)
        self.assertEqual(History(self.tmp.name).current.version_id, "initial")
        missing = {"metrics": {"x": {"value": None, "status": "insufficient_data"}}}
        self.c.validator = Validator(
            evaluator(missing), lambda a, b: ValidationResult("accepted", "wrong")
        )
        r = self.c.run_round(self.sim, self.report)
        self.assertEqual(r.validation["status"], "pending")

    def test_seed_reproducibility_and_paths(self):
        with tempfile.TemporaryDirectory() as other:
            c = Controller(History(other), RSIConfig(max_rounds=3, seed=7))
            c.initialize(policy())
            a = self.c.run_round(self.sim, self.report)
            b = c.run_round(self.sim, self.report)
            self.assertEqual(a.diagnosis, b.diagnosis)
            self.assertEqual(a.validation, b.validation)
            self.assertTrue(
                same(
                    weights(self.h.restore(a.candidate["version_id"])),
                    weights(c.history.restore(b.candidate["version_id"])),
                )
            )
        simpath = Path(self.tmp.name) / "sim.json"
        simpath.write_text(json.dumps(self.sim), encoding="utf-8")
        r = self.c.run_round(simpath, self.report)
        self.assertEqual(r.simulation_ref, str(simpath))


if __name__ == "__main__":
    unittest.main()
