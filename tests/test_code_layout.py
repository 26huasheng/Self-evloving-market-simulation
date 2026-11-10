"""Integration contracts for the supported entry points and representation choice."""

import ast
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from sems.cli.evaluate import decision_mode_for, main as evaluate_main
from sems.cli.simulate import build_parser, main as simulate_main
from sems.decision import DecisionModel
from sems.offline import FixtureEmbeddings, OfflineProfiles
from sems.simulation_runner import SimulationRunner
from sems_attention import NeuralAttentionPolicy, TrendAttentionPolicy
from sems.rsi.proposer import StructuralExecutor, StructuralProposal

ROOT = Path(__file__).resolve().parents[1]
CASE = Path(__import__("os").environ["SEMS_CASE_PATH"])


@pytest.mark.parametrize("mode", ["purchase", "review_event"])
def test_supported_commands_keep_protocols_separate(tmp_path, mode):
    torch.set_num_threads(1)
    result_path = tmp_path / "simulation.json"
    with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
        simulate_main(
            [
                "--case",
                str(CASE),
                "--mode",
                "offline",
                "--decision-mode",
                mode,
                "--max-sessions",
                "2",
                "--query-budget",
                "1",
                "--output",
                str(result_path),
            ]
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["attention_representation"] == "trend_v3"
        assert result["decision_mode"] == mode
        assert result["test_only"] is True
        rows = [row for day in result["days"] for row in day["sessions"]]
        assert len(rows) == 2 and all(row["status"] == "ok" for row in rows)
        report_dir = tmp_path / "evaluation"
        evaluate_main(
            [
                "--case",
                str(CASE),
                "--simulation",
                str(result_path),
                "--output-dir",
                str(report_dir),
                "--judge",
                "fake",
                "--max-reviews",
                "1",
            ]
        )
    report = json.loads((report_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert report["configuration"]["decision_mode"] == mode
    expected = "review_event_evaluation_v1" if mode == "review_event" else "session_evaluation_v1"
    assert report["schema"] == expected
    assert report["api_usage"]["actual_requests"] == 0


def test_purchase_can_decline_without_writing_events():
    class DecliningClient:
        def complete_json(self, system, payload):
            return {
                "purchase": False,
                "product_id": None,
                "submit_rating": False,
                "rating": None,
                "publish_review": False,
                "review_text": "",
                "rationale": "No purchase.",
            }

    result = SimulationRunner(
        CASE, FixtureEmbeddings(), DecisionModel(DecliningClient()), OfflineProfiles(), budget=0
    ).run(max_sessions=1)
    row = next(row for day in result["days"] for row in day["sessions"])
    assert row["status"] == "ok" and row["write_actions"] == []
    assert not any(day["committed_actions"] for day in result["days"])


def test_modes_and_old_checkpoints_are_rejected_before_paid_client(tmp_path):
    assert (
        build_parser()
        .parse_args(["--case", str(CASE), "--output", str(tmp_path / "default.json")])
        .attention_version
        == "trend_v3"
    )
    checkpoint = tmp_path / "old.pt"
    NeuralAttentionPolicy(["STOP", "get_reviews"]).save(checkpoint)
    with patch("sems.cli.simulate.SEMSAPIClient", side_effect=AssertionError("paid client opened")):
        with pytest.raises(SystemExit):
            simulate_main(
                [
                    "--case",
                    str(CASE),
                    "--output",
                    str(tmp_path / "bad.json"),
                    "--profile-cache",
                    str(tmp_path),
                    "--mode",
                    "real",
                    "--max-sessions",
                    "1",
                    "--policy",
                    str(checkpoint),
                ]
            )
        with pytest.raises(SystemExit):
            simulate_main(
                [
                    "--case",
                    str(CASE),
                    "--output",
                    str(tmp_path / "conflict.json"),
                    "--decision-mode",
                    "purchase",
                    "--session-mode",
                    "review_event_conditioned",
                ]
            )
    with pytest.raises(ValueError, match="consistent"):
        decision_mode_for({"decision_mode": "purchase", "session_mode": "review_event_conditioned"})
    output = tmp_path / "v2.json"
    simulate_main(
        [
            "--case",
            str(CASE),
            "--policy",
            str(checkpoint),
            "--attention-version",
            "state_v2",
            "--max-sessions",
            "1",
            "--query-budget",
            "0",
            "--output",
            str(output),
        ]
    )
    assert json.loads(output.read_text(encoding="utf-8"))["attention_representation"] == "state_v2"


def test_old_structure_proposal_cannot_redefine_public_trend_environment():
    policy = TrendAttentionPolicy(["STOP", "get_reviews"])
    with pytest.raises(ValueError, match="state_v2-only"):
        StructuralExecutor().apply(policy, StructuralProposal("candidate"))


def test_every_core_module_imports_without_experiment_dependency():
    for package in ("sems", "sems_attention", "sems_tools"):
        for path in (ROOT / "src" / package).rglob("*.py"):
            module = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
            if module.endswith(".__init__"):
                module = module[:-9]
            loaded = importlib.import_module(module)
            assert Path(loaded.__file__).resolve().is_relative_to(ROOT / "src")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("experiments"), path
                if isinstance(node, ast.Import):
                    assert not any(alias.name.startswith("experiments") for alias in node.names), (
                        path
                    )
