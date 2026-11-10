"""Multi-step neural query loop over the existing MarketSession."""

from .actions import ActionSpace
from .observation import Observation
from copy import deepcopy
import torch


class AttentionRunner:
    def __init__(
        self,
        policy,
        embeddings,
        max_steps=None,
        mode="greedy",
        sample_callback=None,
        min_read_queries=0,
        min_query_kinds=("read",),
        action_selector=None,
    ):
        self.policy = policy
        self.embeddings = embeddings
        self.max_steps = max_steps
        self.mode = mode
        self.sample_callback = sample_callback
        self.min_read_queries = min_read_queries
        self.min_query_kinds = ("read",)
        self.action_selector = action_selector
        if not self.min_query_kinds or not set(self.min_query_kinds) <= {"read", "shelf"}:
            raise ValueError("invalid minimum query kinds")
        if type(min_read_queries) is not int or min_read_queries < 0:
            raise ValueError("invalid minimum read queries")
        if mode not in ("greedy", "sample"):
            raise ValueError("invalid attention mode")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
            raise ValueError("invalid step limit")

    def run(self, session, profile, profile_x, observation=None):
        if self.min_read_queries > session.remaining:
            raise ValueError("minimum reads exceed remaining budget")
        observation = observation or Observation(
            session.initial_observation(),
            session.remaining,
            session.market.budget,
            session.market.current_time.isoformat(),
            profile,
        )
        trajectory = []
        limit = (
            self.max_steps
            if self.max_steps is not None
            else session.remaining + session.max_free_sorts
        )
        for step in range(max(0, limit) + 1):
            space = ActionSpace(session.registry, observation)
            reads = sum(
                1
                for x in trajectory
                if x.get("tool") != "STOP"
                and session.registry.schema()[x["tool"]]["kind"] in self.min_query_kinds
                and x.get("result_status") == "ok"
            )
            if reads < self.min_read_queries and any(
                session.registry.schema().get(a.tool_name, {}).get("kind") in self.min_query_kinds
                for a in space.actions
            ):
                space.actions = [a for a in space.actions if not a.is_stop]
                space.tools = [t for t in space.tools if t != "STOP"]
                if session.remaining <= self.min_read_queries - reads:
                    space.actions = [
                        a
                        for a in space.actions
                        if session.registry.schema().get(a.tool_name, {}).get("kind")
                        in self.min_query_kinds
                    ]
                    space.tools = list(dict.fromkeys(a.tool_name for a in space.actions))
            mask = [a.as_dict() for a in space.actions]
            before_observation = deepcopy(observation.decision_observation())
            if step >= limit or session.remaining == 0:
                trajectory.append(
                    {
                        "step": step,
                        "tool": "STOP",
                        "reason": "budget_exhausted" if session.remaining == 0 else "step_limit",
                        "budget_before": session.remaining,
                        "budget_after": session.remaining,
                        "action_mask": mask,
                        "observation_before": before_observation,
                    }
                )
                break
            if self.action_selector is None:
                action, info = self.policy.select_action(
                    profile_x, observation, self.embeddings, space, self.mode
                )
            else:
                action = self.action_selector(observation, space, step)
                if not space.contains(action):
                    raise ValueError("scripted action is masked")
                forward = self.policy.forward(profile_x, observation, self.embeddings, space)
                info = {
                    "tool_probability": None,
                    "product_probability": None,
                    "parameter_probability": None,
                    "action_log_prob": None,
                    "forward": forward,
                }

            if self.sample_callback is not None:
                self.sample_callback(profile_x, observation, space, action, info)
            trend_diagnostics = {}
            if self.policy.representation == "trend_v3":
                trend_diagnostics = {
                    "public_trend_features": info["forward"]["public_trend_features"]
                    .detach()
                    .tolist(),
                    "query_state_before": info["forward"]["query_state"].detach().tolist(),
                }
            if action.is_stop:
                trajectory.append(
                    {
                        "step": step,
                        "tool": "STOP",
                        "reason": "scripted_stop" if self.action_selector else "policy_stop",
                        "tool_probability": info["tool_probability"],
                        "tool_probabilities_before": dict(
                            zip(
                                self.policy.tool_names,
                                info["forward"]["tool_probs"].detach().tolist(),
                            )
                        ),
                        "action_log_prob": info["action_log_prob"],
                        "budget_before": session.remaining,
                        "budget_after": session.remaining,
                        "action_mask": mask,
                        "observation_before": before_observation,
                    }
                )
                trajectory[-1].update(trend_diagnostics)
                break
            if not space.contains(action):
                raise ValueError("policy selected masked action")
            before = session.remaining
            title = observation.cards[action.asin]["name"] if action.asin is not None else None
            tool_probs_before = dict(
                zip(self.policy.tool_names, info["forward"]["tool_probs"].detach().tolist())
            )
            result = session.registry.call(action.tool_name, **action.arguments)
            observation.update(action, result)
            updated = self.policy.encode(profile_x, observation, self.embeddings)
            after_space = ActionSpace(session.registry, observation)
            tool_probs_after = (
                self.policy.forward(profile_x, observation, self.embeddings, after_space)[
                    "tool_probs"
                ]
                .detach()
                .tolist()
            )
            trajectory.append(
                {
                    "step": step,
                    "market_time": session.market.current_time.isoformat(),
                    "tool": action.tool_name,
                    "asin": action.asin,
                    "product_title": title,
                    "parameters": action.parameters,
                    "tool_call": {"name": action.tool_name, "arguments": action.arguments},
                    "tool_probability": info["tool_probability"],
                    "product_probability": info["product_probability"],
                    "parameter_probability": info["parameter_probability"],
                    "tool_probabilities_before": tool_probs_before,
                    "tool_probabilities_after": dict(zip(self.policy.tool_names, tool_probs_after)),
                    "action_log_prob": info["action_log_prob"],
                    "budget_before": before,
                    "budget_after": session.remaining,
                    "cost": result.get("cost", 0),
                    "result_status": result["status"],
                    "result": result,
                    "observed_fields": list((result.get("data") or {}).keys()),
                    "lambda_norm": float(info["forward"]["lambda_matrix"].detach().norm()),
                    "environment_norm": float(info["forward"]["e"].detach().norm()),
                }
            )
            trajectory[-1]["lambda_gate_after"] = (
                self.policy.lambda_layer.gate(updated["e"]).detach().tolist()
            )
            trajectory[-1]["action_mask"] = mask
            trajectory[-1]["observation_before"] = before_observation
            trajectory[-1]["observation_after"] = deepcopy(observation.decision_observation())
            if trend_diagnostics:
                trend_diagnostics.update(
                    query_state_after=updated["query_state"].detach().tolist(),
                    environment_delta_after_query=float(
                        torch.linalg.vector_norm(
                            updated["e"].detach() - info["forward"]["e"].detach()
                        )
                    ),
                    lambda_delta_after_query=float(
                        torch.linalg.vector_norm(
                            updated["lambda_matrix"].detach()
                            - info["forward"]["lambda_matrix"].detach()
                        )
                    ),
                )
                trajectory[-1].update(trend_diagnostics)
        return {
            "neural_observation": observation,
            "decision_observation": observation.decision_observation(),
            "trajectory": trajectory,
        }
