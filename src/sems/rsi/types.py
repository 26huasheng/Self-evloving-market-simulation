"""Serializable RSI contracts; simulation and evaluation retain their native schemas."""

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


@dataclass
class RSIConfig:
    max_rounds: int = 1
    seed: int = 0
    stop_on_pending: bool = True
    stop_on_rejection: bool = False

    def __post_init__(self):
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be nonnegative")


@dataclass
class TrainingConfig:
    seed: int = 0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyVersion:
    version_id: str
    checkpoint: str
    config: dict[str, Any] = field(default_factory=dict)
    verified: bool = False


@dataclass
class DiagnosisReport:
    metrics: dict[str, Any]
    query_statistics: dict[str, Any]
    budget_stop: dict[str, Any]
    consumers: dict[str, Any]
    products: list[dict[str, Any]]
    sufficient: bool
    reasons: list[str]


@dataclass
class ImprovementProposal:
    target: str
    change_type: str
    changes: dict[str, Any]
    expected_effect: str
    constraints: list[str]
    candidate_version: str


@dataclass
class CandidatePolicy:
    version_id: str
    parent_version: str
    kind: Literal["training", "structure"]
    checkpoint: str
    config: dict[str, Any] = field(default_factory=dict)
    proposal: dict[str, Any] | None = None
    optimizer_checkpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    status: Literal["accepted", "rejected", "pending"]
    reason: str
    conditions: dict[str, Any] = field(default_factory=dict)
    baseline_report: dict[str, Any] | None = None
    candidate_report: dict[str, Any] | None = None

    def __post_init__(self):
        if self.status not in ("accepted", "rejected", "pending"):
            raise ValueError("invalid validation status")
        if not self.reason:
            raise ValueError("validation reason required")


@dataclass
class EvolutionRecord:
    round: int
    current: dict[str, Any]
    candidate: dict[str, Any]
    diagnosis: dict[str, Any]
    validation: dict[str, Any]
    simulation_ref: str | None
    evaluation_ref: str | None
    config: dict[str, Any]
    api_usage: dict[str, Any]


def serialize(value):
    return asdict(value)
