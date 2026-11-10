"""Synthetic test fixtures only; no experiment entry point."""

from datetime import datetime, timezone

from types import SimpleNamespace

import numpy as np

import torch

from sems_attention import AttentionRunner, Observation, TrendAttentionPolicy

from sems_attention.actions import ActionSpace, ToolAction

from sems_tools.public_trends import TREND_NAMES, TREND_SCHEMA

NOW = datetime(2025, 1, 15, tzinfo=timezone.utc)


class Embeddings:
    """Small deterministic fixture; the real-Case validator uses actual MiniLM."""

    def encode(self, text):
        return np.full(384, (sum(text.encode()) % 31) / 31, dtype=np.float32)

    def mean(self, texts):
        return (
            np.mean([self.encode(text) for text in texts], axis=0)
            if texts
            else np.zeros(384, dtype=np.float32)
        )


def bulletin(rising=False):
    return {
        "schema": TREND_SCHEMA,
        "as_of": NOW.isoformat(),
        "features": {
            name: {
                "value": float(rising) if name == "review_activity_change" else 0.0,
                "known": True,
                "source": "synthetic_controlled_test",
            }
            for name in TREND_NAMES
        },
    }


def shelf(rising=False):
    return {
        "products": [
            {
                "asin": "a",
                "name": "Mouse",
                "new_price": {"value": 10, "state": "known", "source": "observed"},
            }
        ],
        "public_market_context": bulletin(rising),
    }


def review_result(step):
    return {
        "status": "ok",
        "remaining": 6 - step,
        "cost": 1,
        "data": {"reviews": [{"id": str(step), "text": "known review", "rating": 4}]},
    }


def observation(rising=False, steps=0):
    obs = Observation(shelf(rising), 6, 6, NOW.isoformat(), {})
    for step in range(1, steps + 1):
        obs.update(ToolAction("get_reviews", "a", {}), review_result(step))
    return obs


class Session:
    def __init__(self, rising):
        self.rising = rising
        self.remaining = 6
        self.max_free_sorts = 0
        self.market = SimpleNamespace(budget=6, current_time=NOW)
        self.registry = self

    def initial_observation(self):
        return shelf(self.rising)

    def schema(self):
        return {"get_reviews": {"kind": "read"}}

    def actions(self):
        return [
            {"tool": "STOP", "args": {}, "cost": 0, "allowed": True},
            {
                "tool": "get_reviews",
                "args": {"asin": "a"},
                "cost": 1,
                "allowed": self.remaining > 0,
            },
        ]

    def call(self, name, **args):
        if name != "get_reviews" or self.remaining <= 0:
            raise ValueError("invalid synthetic query")
        self.remaining -= 1
        return review_result(6 - self.remaining)


def train_demo():
    torch.set_num_threads(1)
    policy = TrendAttentionPolicy(["STOP", "get_reviews"], seed=7)
    embeddings = Embeddings()
    profile_x = np.ones(776, dtype=np.float32)
    examples = []
    for rising in (False, True):
        for step in range(6):
            obs = observation(rising, step)
            label = 0 if step >= (5 if rising else 3) else 1
            examples.append((obs, ActionSpace(Session(rising), obs), label))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    for epoch in range(300):
        optimizer.zero_grad()
        loss = torch.stack(
            [
                -policy.forward(profile_x, obs, embeddings, space)["tool_probs"][label].log()
                for obs, space, label in examples
            ]
        ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5)
        optimizer.step()
        if float(loss.detach()) < 0.01:
            break
    policy.eval()
    records = []
    for rising in (False, True):
        session = Session(rising)
        with torch.no_grad():
            run = AttentionRunner(policy, embeddings, mode="greedy", min_read_queries=0).run(
                session, {}, profile_x
            )
        records.append(
            {
                "rising": rising,
                "paid_reads": 6 - session.remaining,
                "stop_reason": run["trajectory"][-1]["reason"],
                "trajectory": run["trajectory"],
            }
        )
    report = {
        "test_only": True,
        "api_calls": 0,
        "training_source": "invented_synthetic_3_vs_5_labels",
        "interpretation": "Capacity and plumbing only; not human behavior, held-out prediction or real-Case improvement.",
        "budget": 6,
        "min_read_queries": 0,
        "epochs": epoch + 1,
        "loss": float(loss.detach()),
        "records": records,
    }
    if [record["paid_reads"] for record in records] != [3, 5]:
        raise AssertionError("synthetic capacity demonstration did not converge")
    return policy, report
