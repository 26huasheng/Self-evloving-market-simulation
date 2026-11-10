"""Run SEMS with explicit purchase/review-event protocols and trend Attention."""

import argparse
from pathlib import Path

from sems.api_client import SEMSAPIClient
from sems.decision import DecisionModel, OpenAICompatibleClient
from sems.embeddings import TextEmbeddings
from sems.offline import FixtureEmbeddings, OfflineDecision, OfflineProfiles
from sems.simulation_runner import CachedProfiles, SimulationRunner, save_result

DECISION_MODES = {"purchase": "ordinary", "review_event": "review_event_conditioned"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="External Case ZIP or directory")
    parser.add_argument("--mode", choices=("offline", "real"), default="offline")
    parser.add_argument("--decision-mode", choices=tuple(DECISION_MODES))
    parser.add_argument(
        "--session-mode",
        choices=tuple(DECISION_MODES.values()),
        help="Legacy alias for --decision-mode",
    )
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--max-api-requests", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy")
    parser.add_argument("--attention-mode", choices=("greedy", "sample"), default="sample")
    parser.add_argument("--attention-version", choices=("state_v2", "trend_v3"), default="trend_v3")
    parser.add_argument("--query-budget", type=int, default=6)
    parser.add_argument("--min-read-queries", type=int, default=0)
    parser.add_argument(
        "--profile-cache", type=Path, help="External validated four-stage profile cache"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSON path outside the source tree"
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    session_mode = (
        DECISION_MODES[args.decision_mode]
        if args.decision_mode
        else args.session_mode or "ordinary"
    )
    if args.session_mode and args.decision_mode and session_mode != args.session_mode:
        parser.error("--decision-mode and --session-mode disagree")
    if args.max_sessions is not None and args.max_sessions < 1:
        parser.error("--max-sessions must be positive")
    if args.query_budget < 0 or not 0 <= args.min_read_queries <= args.query_budget:
        parser.error("query budget and minimum reads are inconsistent")
    if args.mode == "real" and (not args.max_sessions or not args.policy or not args.profile_cache):
        parser.error("real mode requires positive --max-sessions, --policy and --profile-cache")
    # Check the representation before opening any paid client.
    if args.policy:
        from sems_attention import NeuralAttentionPolicy

        if NeuralAttentionPolicy.load(args.policy).representation != args.attention_version:
            parser.error(
                "checkpoint representation differs; select its --attention-version explicitly"
            )
    client = None
    profiles = None
    try:
        if args.mode == "real":
            client = SEMSAPIClient(
                max_requests=args.max_api_requests,
                max_retries=0,
                usage_log=args.output.parent / "api_usage.jsonl",
            )
            decision = DecisionModel(OpenAICompatibleClient(client))
            profiles = CachedProfiles(args.profile_cache, client.models["profile"], client=client)
            embeddings = TextEmbeddings()
        else:
            decision, profiles, embeddings = (
                OfflineDecision(),
                OfflineProfiles(),
                FixtureEmbeddings(),
            )
        result = SimulationRunner(
            args.case,
            embeddings,
            decision,
            profiles,
            seed=args.seed,
            policy_path=args.policy,
            budget=args.query_budget,
            min_read_queries=args.min_read_queries,
            session_mode=session_mode,
            mode=args.attention_mode,
            attention_version=args.attention_version,
        ).run(args.max_sessions)
        result["test_only"] = args.mode == "offline"
        save_result(result, args.output)
        rows = [session for day in result["days"] for session in day["sessions"]]
        print(
            "SIMULATION",
            {
                "days": len(result["days"]),
                "sessions": len(rows),
                "completed": sum(s["status"] == "ok" for s in rows),
                "failed": sum(s["status"] != "ok" for s in rows),
                "output": str(args.output),
            },
        )
    finally:
        if client:
            client.report_usage(args.output.parent / "api_usage.json")
        else:
            print(
                "API_USAGE",
                {
                    "actual_requests": 0,
                    "successes": 0,
                    "failures": 0,
                    "retries": 0,
                    "profile_cache_hits": getattr(profiles, "hits", 0),
                },
            )


if __name__ == "__main__":
    main()
