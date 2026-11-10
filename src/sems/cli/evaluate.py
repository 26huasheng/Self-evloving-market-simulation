"""Evaluate a saved run using the evaluator matching its Decision protocol."""

import argparse
import json
from pathlib import Path

from sems.api_client import SEMSAPIClient
from sems.evaluation import review_events, sessions
from sems.evaluation.review_judge import ReviewEventJudge
from sems.simulation_time import build_schedule
from sems_tools import CaseStore

SESSION_MODES = {"ordinary": "purchase", "review_event_conditioned": "review_event"}


def decision_mode_for(result):
    legacy = result.get("session_mode")
    recorded = result.get("decision_mode")
    if legacy is not None and legacy not in SESSION_MODES:
        raise ValueError("unknown recorded session mode")
    inferred = SESSION_MODES.get(legacy)
    mode = recorded or inferred
    if mode not in ("purchase", "review_event") or (recorded and inferred and recorded != inferred):
        raise ValueError("simulation is missing a consistent Decision mode")
    return mode


class FakeJudgeClient:
    """Explicit synthetic scores for testing report generation only."""

    def __init__(self):
        self.hits = 0
        self.models = {"judge": "offline_fake"}

    def record_cache_hit(self, *args):
        self.hits += 1

    def complete_json(self, role, system, payload, **kwargs):
        value = {name: 3 for name in sessions.JUDGE_NAMES}
        if "reasons" in system:
            value["reasons"] = {name: "Offline test score." for name in sessions.JUDGE_NAMES}
        return value, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument(
        "--decision-mode",
        choices=("purchase", "review_event"),
        help="Optional assertion; mismatch is rejected",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge", choices=("off", "fake", "api"), default="off")
    parser.add_argument("--max-reviews", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--unit", choices=("event", "user"), default="event", help="Review-event evaluation unit"
    )
    parser.add_argument("--facts", type=Path)
    args = parser.parse_args(argv)
    if args.max_reviews < 0 or (args.judge != "off" and args.max_reviews < 1):
        parser.error("enabled judge requires --max-reviews >= 1")
    result = json.loads(args.simulation.read_text(encoding="utf-8"))
    mode = decision_mode_for(result)
    if args.decision_mode and args.decision_mode != mode:
        parser.error("requested Decision mode differs from the saved simulation")
    facts = json.loads(args.facts.read_text(encoding="utf-8")) if args.facts else {}
    case = CaseStore(args.case)
    client = None
    try:
        if result.get("case_id") != case.case_id:
            raise ValueError("simulation and case IDs do not match")
        _, truth = build_schedule(case)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.judge == "api":
            client = SEMSAPIClient(
                max_requests=args.max_reviews * 3, usage_log=args.output_dir / "api_usage.jsonl"
            )
        elif args.judge == "fake":
            client = FakeJudgeClient()
        judge = None
        if client:
            cache = args.output_dir / (
                "review_event_judge_cache" if mode == "review_event" else "session_judge_cache"
            )
            judge = (
                ReviewEventJudge(client, cache)
                if mode == "review_event"
                else sessions.Judge(client, client.models["judge"], cache)
            )
        if mode == "review_event":
            report = review_events.evaluate_review_events(
                result,
                truth,
                case.asins,
                judge=judge,
                t0_facts=facts,
                judge_max=args.max_reviews,
                unit=args.unit,
            )
        else:
            report = sessions.evaluate(
                result,
                truth,
                case.asins,
                judge=judge,
                judge_max=args.max_reviews,
                seed=args.seed,
                product_facts=facts,
            )
        report.setdefault("configuration", {}).update(decision_mode=mode, judge_mode=args.judge)
        if args.judge == "fake":
            report["test_only"] = True
            for name in sessions.JUDGE_NAMES:
                report["metrics"][name]["comparability"] = "Synthetic offline Judge score only"
        usage = (
            client.usage_summary()
            if args.judge == "api"
            else {
                "actual_requests": 0,
                "successes": 0,
                "failures": 0,
                "retries": 0,
                "cache_hits": getattr(client, "hits", 0),
            }
        )
        report["api_usage"] = usage
        (args.output_dir / "evaluation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / "api_usage.json").write_text(
            json.dumps(usage, indent=2), encoding="utf-8"
        )
        lines = [
            "# Evaluation",
            "",
            f"Decision mode: `{mode}`; Judge: `{args.judge}`.",
            "",
            "Ground truth records review events, not independently verified purchase times.",
            "",
            "| Metric | Value | Status |",
            "|---|---:|---|",
        ]
        for name, metric in report["metrics"].items():
            lines.append(f"| {name} | {metric['value']} | {metric['status']} |")
        (args.output_dir / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "EVALUATION",
            {"decision_mode": mode, "coverage": report["coverage"], "output": str(args.output_dir)},
        )
    finally:
        case.close()
        if args.judge == "api" and client:
            client.report_usage(args.output_dir / "api_usage.json")


if __name__ == "__main__":
    main()
