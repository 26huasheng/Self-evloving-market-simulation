"""Structured proposal protocol; never executes source changes."""

from typing import Protocol
from .types import CandidatePolicy, serialize


class Proposer(Protocol):
    def propose(self, policy, diagnosis, candidate_version, history) -> CandidatePolicy: ...


def validate_proposal(proposal):
    if proposal.target != "attention":
        raise ValueError("only attention may be modified")
    if proposal.change_type not in ("config", "structure"):
        raise ValueError("invalid change type")
    if (
        not isinstance(proposal.changes, dict)
        or not proposal.changes
        or not proposal.expected_effect.strip()
        or not proposal.constraints
    ):
        raise ValueError("incomplete proposal")
    if not proposal.candidate_version:
        raise ValueError("candidate version required")
    # This stage cannot apply structural or policy configuration modifications.
    return True


class ProposalAdapter:
    def __init__(self, proposal):
        self.proposal = proposal

    def propose(self, policy, diagnosis, candidate_version, history):
        validate_proposal(self.proposal)
        if self.proposal.candidate_version != candidate_version:
            raise ValueError("candidate version mismatch")
        path = history.save_policy(candidate_version, policy)
        return CandidatePolicy(
            candidate_version,
            history.current.version_id,
            "structure",
            path,
            config=policy.get_evolvable_state()["config"],
            proposal=serialize(self.proposal),
            metadata={"proposal_applied": False},
        )


from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralProposal:
    """Closed allowlist: only observation-derived query context is supported."""

    candidate_version: str
    feature: str = "query_context_v1"
    target: str = "attention.environment_encoder"
    expected_effect: str = "Represent review and query coverage in the environment encoding."


class StructuralExecutor:
    def apply(self, policy, proposal):
        import torch
        from sems_attention import NeuralAttentionPolicy

        if policy.representation != "state_v2":
            raise ValueError(
                "query_context_v1 is a state_v2-only structural proposal; trend_v3 keeps query progress separate from public e"
            )
        if (
            proposal.target != "attention.environment_encoder"
            or proposal.feature != "query_context_v1"
            or not proposal.candidate_version
        ):
            raise ValueError("unsupported structural proposal")
        if policy.structure:
            raise ValueError("structure already present")
        candidate = NeuralAttentionPolicy(policy.tool_names, structure={"query_context_v1": True})
        source = policy.state_dict()
        target = candidate.state_dict()
        extra = set(target) - set(source)
        if extra != {"environment_encoder.condition_adapter.weight"} or set(source) - set(target):
            raise ValueError("unexpected structure migration")
        for k, v in source.items():
            if v.shape != target[k].shape:
                raise ValueError("incompatible migration: " + k)
            target[k] = v.detach().clone()
        candidate.load_state_dict(target, strict=True)
        return candidate


class DeterministicStructuralProposer:
    def propose(self, diagnosis, candidate_version):
        return StructuralProposal(candidate_version=candidate_version)
