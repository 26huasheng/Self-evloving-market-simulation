"""Offline integration checks; test-only encoders and profile fixtures."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from sems.api_client import SEMSAPIClient
from sems.decision import DecisionModel, OpenAICompatibleClient
from sems.simulation_runner import SimulationRunner, save_result, validate_decision

CASE = Path(__import__("os").environ["SEMS_CASE_PATH"])


class Embeddings:
    def encode(self, text):
        return np.ones(384, dtype="float32") * 0.01

    def mean(self, texts):
        return self.encode("")


class Profiles:
    def load(self, user_id, history):
        assert history
        return {
            "user_id": user_id,
            "narrative": "Test fixture for integration.",
            "preferences": [],
            "uncertainties": [],
        }


class SimulationTests(unittest.TestCase):
    def test_conditioned_session_after_stop(self):
        class Conditioned:
            def decide_review_event(self, agent_id, profile, observation, correction=None):
                return {
                    "selected_asin": observation["shelf"][0]["product_id"],
                    "rating_probabilities": [0, 0, 0, 0.25, 0.75],
                    "rating": 5,
                    "review_text": "test review",
                }

        runner = SimulationRunner(
            CASE,
            Embeddings(),
            Conditioned(),
            Profiles(),
            seed=4,
            budget=3,
            session_mode="review_event_conditioned",
            min_read_queries=1,
        )
        result = runner.run(max_sessions=1)
        row = next(s for d in result["days"] for s in d["sessions"])
        self.assertEqual(row["status"], "ok")
        self.assertEqual(len(row["write_actions"]), 3)
        self.assertEqual(row["decision"]["rating_probabilities"], [0, 0, 0, 0.25, 0.75])
        self.assertTrue(
            any(step["tool"] not in ("STOP", "sort_shelf") for step in row["trajectory"])
        )
        self.assertEqual(row["trajectory"][-1]["tool"], "STOP")

    def test_scripted_stop_and_query_contrast(self):
        class Conditioned:
            def decide_review_event(self, agent_id, profile, observation, correction=None):
                return {
                    "selected_asin": observation["shelf"][0]["product_id"],
                    "rating_probabilities": [0, 0, 0, 0, 1],
                    "rating": 5,
                    "review_text": "test",
                }

        for depth in (0, 2, 4):

            def selector(obs, space, step):
                if step >= depth:
                    return next(a for a in space.actions if a.is_stop)
                return next(
                    a
                    for a in space.actions
                    if not a.is_stop and a.tool_name == "get_product_details"
                )

            result = SimulationRunner(
                CASE,
                Embeddings(),
                Conditioned(),
                Profiles(),
                seed=4,
                budget=6,
                session_mode="review_event_conditioned",
                action_selector=selector,
            ).run(max_sessions=1)
            row = next(s for day in result["days"] for s in day["sessions"])
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["budget_used"], depth)
            self.assertEqual(len(row["observation"]["tool_results"]), depth)
            self.assertEqual(len(row["write_actions"]), 3)

    def test_configured_budget_and_two_queries(self):
        class Conditioned:
            def decide_review_event(self, agent_id, profile, observation, correction=None):
                return {
                    "selected_asin": observation["shelf"][0]["product_id"],
                    "rating_probabilities": [0, 0, 0, 0, 1],
                    "rating": 5,
                    "review_text": "test",
                }

        for budget in (6, 12, 20):
            result = SimulationRunner(
                CASE,
                Embeddings(),
                Conditioned(),
                Profiles(),
                seed=4,
                budget=budget,
                session_mode="review_event_conditioned",
                min_read_queries=2,
                min_query_kinds=("read", "shelf"),
            ).run(max_sessions=1)
            row = next(s for day in result["days"] for s in day["sessions"])
            calls = [x for x in row["trajectory"] if x["tool"] != "STOP"]
            self.assertEqual(row["status"], "ok")
            self.assertGreaterEqual(len(calls), 2)
            self.assertLessEqual(len(calls), budget)
            self.assertEqual(row["budget_used"], sum(step["cost"] for step in calls))
            self.assertGreaterEqual(sum(step["cost"] for step in calls), 2)
            self.assertEqual(
                row["trajectory"][-1]["budget_before"], row["trajectory"][-1]["budget_after"]
            )

    def test_conditioned_format_repair_and_failure(self):
        class Decision:
            def __init__(self, repair):
                self.calls = 0
                self.repair = repair

            def decide_review_event(self, agent_id, profile, observation, correction=None):
                self.calls += 1
                return {
                    "selected_asin": observation["shelf"][0]["product_id"]
                    if correction and self.repair
                    else "invalid",
                    "rating_probabilities": [0, 0, 0, 0, 1],
                    "rating": 5,
                    "review_text": "test",
                }

        for repair in (True, False):
            decision = Decision(repair)
            result = SimulationRunner(
                CASE,
                Embeddings(),
                decision,
                Profiles(),
                seed=4,
                budget=2,
                session_mode="review_event_conditioned",
                decision_repairs=1,
            ).run(max_sessions=1)
            row = next(s for d in result["days"] for s in d["sessions"])
            self.assertEqual(decision.calls, 2)
            self.assertEqual(row["status"], "ok" if repair else "decision_failed")
            if repair:
                self.assertEqual(row["decision_format_repairs"], 1)
            else:
                self.assertEqual(row["write_actions"], [])

    def test_protocol_and_mock_api(self):
        calls = []

        def response(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            calls.append(payload)
            content = {
                "purchase": True,
                "product_id": payload["observation"]["shelf"][0]["product_id"],
                "submit_rating": True,
                "rating": 4,
                "publish_review": True,
                "review_text": "test review",
                "rationale": "test",
            }
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
                model="gpt-4.1-mini",
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))],
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"SEMS_API_KEY": "test-only"}),
        ):
            client = SEMSAPIClient(
                max_requests=4, max_retries=0, usage_log=Path(tmp) / "usage.jsonl"
            )
            client._client.chat.completions.create = Mock(side_effect=response)
            decision = DecisionModel(OpenAICompatibleClient(client))
            runner = SimulationRunner(CASE, Embeddings(), decision, Profiles(), seed=4)
            result = runner.run(max_sessions=2)
            rows = [r for day in result["days"] for r in day["sessions"]]
            self.assertEqual([r["status"] for r in rows], ["ok", "ok"])
            self.assertEqual(client.usage_summary()["actual_requests"], 2)
            self.assertTrue(all(len(r["write_actions"]) == 3 for r in rows))
            self.assertTrue(
                all(
                    r["budget_used"] == sum(step.get("cost", 0) for step in r["trajectory"])
                    for r in rows
                )
            )
            self.assertTrue(all("action_mask" in step for r in rows for step in r["trajectory"]))
            self.assertTrue(
                all(
                    "review_id" not in json.dumps(p) and "ground_truth" not in json.dumps(p)
                    for p in calls
                )
            )
            path = Path(tmp) / "result.json"
            save_result(result, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), result)
            self.assertEqual(runner.run(max_sessions=2), result)
            self.assertIsNot(rows[0]["observation"], rows[1]["observation"])
            with self.assertRaises(ValueError):
                validate_decision(
                    {
                        "purchase": True,
                        "product_id": "outside",
                        "submit_rating": False,
                        "rating": None,
                        "publish_review": False,
                        "review_text": "",
                        "rationale": "",
                    },
                    {"shelf"},
                )
            with self.assertRaises(ValueError):
                validate_decision(
                    {
                        "purchase": True,
                        "product_id": "shelf",
                        "submit_rating": True,
                        "rating": True,
                        "publish_review": False,
                        "review_text": "",
                        "rationale": "",
                    },
                    {"shelf"},
                )


if __name__ == "__main__":
    unittest.main()
