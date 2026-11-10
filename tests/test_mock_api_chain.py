import json
from pathlib import Path
import tempfile
import unittest

from sems.rsi.diagnoser import Diagnoser
from sems.rsi.history import History
from sems_attention import NeuralAttentionPolicy


class MockChainRegressions(unittest.TestCase):
    def test_checkpoint_manifest_and_reload_on_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            history = History(directory)
            history.initialize(NeuralAttentionPolicy(["STOP", "get_reviews"], seed=7), "initial")
            restored = History(directory)
            self.assertEqual(restored.current.version_id, "initial")
            self.assertEqual(restored.restore("initial").tool_names, ("STOP", "get_reviews"))
            self.assertEqual(
                json.loads((Path(directory) / "history.json").read_text(encoding="utf-8"))[
                    "current"
                ]["version_id"],
                "initial",
            )

    def test_diagnoser_accepts_native_event_diagnostics(self):
        simulation = {
            "days": [
                {
                    "sessions": [
                        {"user_id": "u", "trajectory": [{"tool": "STOP", "reason": "policy_stop"}]}
                    ]
                }
            ]
        }
        for diagnostics in ([], [{"session_id": "failed", "reason": "invalid_decision"}]):
            evaluation = {
                "metrics": {"TVD": {"value": 0.5, "status": "ok"}},
                "diagnostics": diagnostics,
            }
            diagnosis = Diagnoser().diagnose(simulation, evaluation)
            self.assertTrue(diagnosis.sufficient)
            self.assertEqual(diagnosis.products, [])


if __name__ == "__main__":
    unittest.main()
