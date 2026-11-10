"""Multi-consumer simulation over the existing market, Attention and Decision."""

import hashlib
import json
import inspect
from collections import defaultdict
from pathlib import Path

import torch
from sems_attention import AttentionRunner, NeuralAttentionPolicy
from sems_attention.profiles.four_stage import PROMPT_VERSION, history_for_user, profile_input
from sems_attention.profiles.cache import CachedProfiles
from sems_tools import MarketEnvironment
from .simulation_time import Clock, build_schedule
from .protocol import validate_decision, validate_review_event


class SimulationRunner:
    def __init__(
        self,
        case_path,
        embeddings,
        decision,
        profiles,
        *,
        seed=0,
        budget=6,
        policy_path=None,
        mode="greedy",
        session_ids=None,
        sample_callback=None,
        session_mode="ordinary",
        min_read_queries=0,
        decision_repairs=1,
        min_query_kinds=("read",),
        action_selector=None,
        resume_days=None,
        day_callback=None,
        session_callback=None,
        fail_fast=False,
        attention_version=None,
    ):
        self.case_path = case_path
        self.embeddings = embeddings
        self.decision = decision
        self.profiles = profiles
        self.seed = seed
        self.budget = budget
        self.policy_path = policy_path
        self.mode = mode
        self.session_ids = set(session_ids) if session_ids is not None else None
        self.sample_callback = sample_callback
        self.session_mode = session_mode
        self.min_read_queries = min_read_queries
        self.decision_repairs = decision_repairs
        self.min_query_kinds = ("read",)
        self.action_selector = action_selector
        self.resume_days = list(resume_days or [])
        self.day_callback = day_callback
        self.session_callback = session_callback
        self.fail_fast = fail_fast
        if session_mode not in ("ordinary", "review_event_conditioned"):
            raise ValueError("invalid session mode")
        if type(decision_repairs) is not int or not 0 <= decision_repairs <= 2:
            raise ValueError("invalid repair limit")
        if type(budget) is not int or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        if mode not in ("greedy", "sample"):
            raise ValueError("invalid attention mode")
        if type(min_read_queries) is not int or not 0 <= min_read_queries <= budget:
            raise ValueError("minimum reads must be within query budget")
        if attention_version not in (None, "state_v2", "trend_v3"):
            raise ValueError("unknown attention version")
        self.attention_version = attention_version

    def run(self, max_sessions=None, max_days=None):
        if max_sessions is not None and (type(max_sessions) is not int or max_sessions < 1):
            raise ValueError("max_sessions must be positive")
        market = MarketEnvironment(
            self.case_path, {"simulation_mode": True, "query_budget": self.budget}
        )
        try:
            clock = Clock(market.data.t0_ms, market.data.end_ms)
            sessions, _truth = build_schedule(market.data)
            if max_days is not None and (
                type(max_days) is not int or not 1 <= max_days <= len(clock)
            ):
                raise ValueError("max_days must be within the case window")
            if max_days is not None and len(self.resume_days) > max_days:
                raise ValueError("saved days exceed requested window")
            # Ground truth never crosses the scheduler boundary into a consumer.
            del _truth
            if self.session_ids is not None:
                sessions = [s for s in sessions if s.session_id in self.session_ids]
            if (
                self.session_ids is not None
                and {s.session_id for s in sessions} != self.session_ids
            ):
                raise ValueError("unknown requested session IDs")
            if max_sessions is not None:
                sessions = sessions[:max_sessions]
            by_day = defaultdict(list)
            for item in sessions:
                by_day[item.simulation_day].append(item)
            prototype = market.open_session("__policy_init__", "__policy_init__")
            if self.policy_path:
                policy = NeuralAttentionPolicy.load(self.policy_path)
            elif self.attention_version in (None, "trend_v3"):
                from sems_attention.trend_policy import TrendAttentionPolicy

                policy = TrendAttentionPolicy.from_registry(prototype.registry, self.seed)
            else:
                policy = NeuralAttentionPolicy.from_registry(prototype.registry, self.seed)
            if (
                self.attention_version is not None
                and self.attention_version != policy.representation
            ):
                raise ValueError("checkpoint and requested attention representation differ")
            if policy.representation == "trend_v3":
                market.enable_public_trends()
            policy.generator.manual_seed(self.seed)
            policy.eval()
            model_version = (
                (
                    "checkpoint_sha256:"
                    + hashlib.sha256(Path(self.policy_path).read_bytes()).hexdigest()
                )
                if self.policy_path
                else f"untrained_seed_{self.seed}"
            )
            attention = AttentionRunner(
                policy,
                self.embeddings,
                mode=self.mode,
                sample_callback=self.sample_callback,
                min_read_queries=self.min_read_queries,
                min_query_kinds=self.min_query_kinds,
                action_selector=self.action_selector,
            )
            result = {
                "schema": "simulation_result_v1",
                "case_id": market.data.case_id,
                "seed": self.seed,
                "budget": self.budget,
                "session_mode": self.session_mode,
                "min_read_queries": self.min_read_queries,
                "min_query_kinds": list(self.min_query_kinds),
                "attention_model": model_version,
                "profile_version": PROMPT_VERSION,
                "days": [],
            }
            result["decision_mode"] = (
                "review_event" if self.session_mode == "review_event_conditioned" else "purchase"
            )
            result["attention_mode"] = self.mode
            result["attention_representation"] = policy.representation
            resume_signature = {
                "case_id": market.data.case_id,
                "seed": self.seed,
                "budget": self.budget,
                "session_mode": self.session_mode,
                "min_read_queries": self.min_read_queries,
                "mode": self.mode,
                "attention_model": model_version,
                "profile_version": PROMPT_VERSION,
                "session_ids": [s.session_id for s in sessions],
            }
            if policy.representation != "state_v2":
                resume_signature["attention_representation"] = policy.representation
            if self.resume_days:
                from datetime import datetime

                for index, saved_day in enumerate(self.resume_days):
                    if saved_day.get("resume_signature") != resume_signature:
                        raise ValueError(
                            "resume configuration missing or different; use a new run directory"
                        )
                    if (
                        saved_day["simulation_day"] != index
                        or saved_day["end_ms"] != clock.day(index).end_ms
                    ):
                        raise ValueError("invalid saved day sequence")
                    for event in saved_day["committed_actions"]:
                        restored = dict(event)
                        restored["visible_at"] = datetime.fromisoformat(restored["visible_at"])
                        market.pending_events.append(restored)
                    if market.commit_day(saved_day["end_ms"]) != saved_day["committed_count"]:
                        raise ValueError("saved market commit count mismatch")
                    result["days"].append(saved_day)
                if self.mode == "sample" and self.resume_days:
                    saved_state = self.resume_days[-1].get("attention_rng_state")
                    if saved_state is None:
                        raise ValueError("sample-mode resume requires saved Attention RNG state")
                    policy.generator.set_state(torch.tensor(saved_state, dtype=torch.uint8))
            vectors = {}
            for day in clock:
                if max_days is not None and day.index >= max_days:
                    break
                if day.index < len(self.resume_days):
                    continue
                daily = {
                    "simulation_day": day.index,
                    "start_ms": day.start_ms,
                    "end_ms": day.end_ms,
                    "sessions": [],
                    "resume_signature": resume_signature,
                }
                if policy.representation == "trend_v3":
                    daily["public_market_context"] = market.public_market_context()
                for scheduled in by_day[day.index]:
                    row = {
                        **scheduled.to_dict(),
                        "profile_version": PROMPT_VERSION,
                        "attention_model": model_version,
                    }
                    pending_start = len(market.pending_events)
                    try:
                        history = history_for_user(market.data, scheduled.user_id)
                        profile = self.profiles.load(scheduled.user_id, history)
                        row["profile_type"] = profile.get("profile_type", "formal_four_stage")
                        if scheduled.user_id not in vectors:
                            vectors[scheduled.user_id] = profile_input(
                                profile, history, self.embeddings
                            )
                        session = market.open_session(scheduled.user_id, scheduled.session_id)
                        with torch.no_grad():
                            execution = attention.run(session, profile, vectors[scheduled.user_id])
                        observation = execution["decision_observation"]
                        row.update(
                            trajectory=execution["trajectory"],
                            observation=observation,
                            stop_reason=execution["trajectory"][-1]["reason"],
                            budget_used=self.budget - session.remaining,
                        )
                        shelf = {p["product_id"] for p in observation["shelf"]}
                        if self.session_mode == "review_event_conditioned":
                            correction = None
                            for attempt in range(self.decision_repairs + 1):
                                try:
                                    raw = self.decision.decide_review_event(
                                        scheduled.user_id, profile, observation, correction
                                    )
                                    decision, actions = validate_review_event(raw, shelf)
                                    break
                                except (ValueError, TypeError, KeyError) as exc:
                                    correction = (
                                        "The prior response did not satisfy the schema: "
                                        + str(exc)
                                        + ". Return a corrected JSON object with all required fields."
                                    )
                                    if attempt == self.decision_repairs:
                                        raise ValueError("decision_failed: " + str(exc)) from exc
                            row["decision_format_repairs"] = attempt
                        else:
                            correction = None
                            method = (
                                getattr(self.decision, "decide_purchase", None)
                                or self.decision.decide_simulation
                            )
                            supports_correction = (
                                "correction" in inspect.signature(method).parameters
                            )
                            for attempt in range(self.decision_repairs + 1):
                                try:
                                    decision = method(
                                        scheduled.user_id,
                                        profile,
                                        observation,
                                        **(
                                            {"correction": correction}
                                            if supports_correction
                                            else {}
                                        ),
                                    )
                                    actions = validate_decision(decision, shelf)
                                    break
                                except (ValueError, TypeError, KeyError) as exc:
                                    correction = (
                                        "The prior response did not satisfy the schema: "
                                        + str(exc)
                                        + ". Return a corrected JSON object."
                                    )
                                    if attempt == self.decision_repairs:
                                        raise ValueError("decision_failed: " + str(exc)) from exc
                            row["decision_format_repairs"] = attempt
                        row["decision"] = decision
                        receipts = []
                        for action in actions:
                            receipt = session.registry.call(action["tool"], **action["args"])
                            if receipt["status"] != "ok":
                                raise ValueError("write rejected: " + str(receipt.get("error")))
                            receipts.append({"action": action, "receipt": receipt})
                        row.update(status="ok", write_actions=receipts)
                        row["information_diagnostics"] = {
                            "initial_budget": self.budget,
                            "remaining_budget": session.remaining,
                            "paid_reads": row["budget_used"],
                            "immediate_stop": execution["trajectory"][0]["tool"] == "STOP",
                            "profile_type": row["profile_type"],
                            "profile_narrative_characters": len(profile.get("narrative", "")),
                            "review_products_observed": sum(
                                bool(v) for v in execution["neural_observation"].reviews.values()
                            ),
                            "queried_tools": [
                                s["tool"] for s in execution["trajectory"] if s["tool"] != "STOP"
                            ],
                        }
                    except Exception as exc:
                        del market.pending_events[pending_start:]
                        row.update(
                            status="decision_failed"
                            if str(exc).startswith("decision_failed:")
                            else "failed",
                            error_type=type(exc).__name__,
                            error=str(exc),
                            write_actions=[],
                        )
                        if self.fail_fast:
                            daily["sessions"].append(row)
                            raise RuntimeError(
                                "full-case session failed at "
                                + scheduled.session_id
                                + ": "
                                + str(exc)
                            ) from exc
                    daily["sessions"].append(row)
                    if self.session_callback:
                        self.session_callback(row)
                pending = [
                    {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in e.items()}
                    for e in market.pending_events
                ]
                affected = {e["asin"] for e in market.pending_events}
                before = {a: market.rating(a) for a in affected}
                daily["committed_count"] = market.commit_day(day.end_ms)
                daily["committed_actions"] = pending
                daily["market_changes"] = {
                    a: {
                        "rating_before": before[a],
                        "rating_after": market.rating(a)
                        if not market.simulation_finished
                        else None,
                    }
                    for a in affected
                }
                if self.mode == "sample":
                    daily["attention_rng_state"] = policy.generator.get_state().tolist()
                result["days"].append(daily)
                if self.day_callback:
                    self.day_callback(daily, result)
            return result
        finally:
            market.close()


def save_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
