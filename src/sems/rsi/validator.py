"""Validation runs with separate policy instances and explicit shared conditions."""

from copy import deepcopy
from .types import ValidationResult


class Validator:
    def __init__(self, evaluate=None, judge=None):
        self.evaluate, self.judge = evaluate, judge

    def validate(self, current, candidate, conditions):
        baseline = self.evaluate(current.clone(), deepcopy(conditions)) if self.evaluate else None
        proposed = self.evaluate(candidate.clone(), deepcopy(conditions)) if self.evaluate else None
        if self.judge is None or baseline is None or proposed is None:
            return ValidationResult(
                "pending",
                "No explicit validation criterion and paired evaluation",
                dict(conditions),
                baseline,
                proposed,
            )

        def available(report):
            return {
                k
                for k, v in report.get("metrics", {}).items()
                if isinstance(v, dict)
                and v.get("status") == "ok"
                and isinstance(v.get("value"), (int, float))
                and not isinstance(v.get("value"), bool)
            }

        if not (available(baseline) & available(proposed)):
            return ValidationResult(
                "pending", "No comparable available metrics", dict(conditions), baseline, proposed
            )
        decision = self.judge(baseline, proposed)
        if not isinstance(decision, ValidationResult):
            raise TypeError("judge must return ValidationResult")
        decision.conditions = dict(conditions)
        decision.baseline_report = baseline
        decision.candidate_report = proposed
        return decision
