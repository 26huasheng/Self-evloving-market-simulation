"""Summarize existing session evaluator reports and simulation trajectories."""

from collections import Counter, defaultdict
from .types import DiagnosisReport


class Diagnoser:
    def diagnose(self, simulation, evaluation):
        metrics = evaluation.get("metrics", {})
        rows = [s for day in simulation.get("days", []) for s in day.get("sessions", [])]
        tools, stops, users = (
            Counter(),
            Counter(),
            defaultdict(lambda: {"sessions": 0, "queries": 0, "budget_used": 0.0}),
        )
        budget = []
        for row in rows:
            trajectory = row.get("trajectory", [])
            user = users[str(row.get("user_id", "unknown"))]
            user["sessions"] += 1
            for step in trajectory:
                tool = step.get("tool")
                if tool == "STOP":
                    stops[step.get("reason", "unknown")] += 1
                elif tool:
                    tools[tool] += 1
                    user["queries"] += 1
            if isinstance(row.get("budget_used"), (int, float)):
                budget.append(row["budget_used"])
                user["budget_used"] += row["budget_used"]
        usable = [
            name
            for name, m in metrics.items()
            if isinstance(m, dict) and m.get("status") == "ok" and m.get("value") is not None
        ]
        reasons = []
        if not usable:
            reasons.append("no available evaluation metrics")
        if not any(row.get("trajectory") for row in rows):
            reasons.append("no query trajectories")
        diagnostics = evaluation.get("diagnostics") or {}
        products = diagnostics.get("products", []) if isinstance(diagnostics, dict) else []
        return DiagnosisReport(
            metrics=metrics,
            query_statistics={
                "sessions": len(rows),
                "tool_calls": dict(tools),
                "total_queries": sum(tools.values()),
            },
            budget_stop={
                "stop_reasons": dict(stops),
                "mean_budget_used": sum(budget) / len(budget) if budget else None,
                "budget_samples": len(budget),
            },
            consumers=dict(users),
            products=products,
            sufficient=not reasons,
            reasons=reasons,
        )
