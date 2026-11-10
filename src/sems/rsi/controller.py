"""Recoverable outer control loop; no simulation or decision internals are changed."""

from pathlib import Path
import json
from .diagnoser import Diagnoser
from .trainer import NoOpTrainer
from .validator import Validator
from .types import RSIConfig, TrainingConfig, EvolutionRecord, serialize

ZERO_USAGE = {
    "actual_requests": 0,
    "successes": 0,
    "failures": 0,
    "retries": 0,
    "cache_hits": 0,
    "input_tokens": None,
    "output_tokens": None,
}


def _read(value):
    return (
        json.loads(Path(value).read_text(encoding="utf-8"))
        if isinstance(value, (str, Path))
        else value
    )


class Controller:
    def __init__(
        self, history, config=None, diagnoser=None, trainer=None, proposer=None, validator=None
    ):
        self.history = history
        self.config = config or RSIConfig()
        self.diagnoser = diagnoser or Diagnoser()
        self.trainer = trainer or NoOpTrainer()
        self.proposer = proposer
        self.validator = validator or Validator()

    def initialize(self, policy, version="initial", config=None):
        return self.history.initialize(policy, version, config)

    def run_round(
        self,
        simulation,
        evaluation,
        *,
        path="training",
        simulation_ref=None,
        evaluation_ref=None,
        conditions=None,
        training_config=None,
    ):
        if not self.history.current:
            raise ValueError("initialize first")
        if len(self.history.data["records"]) >= self.config.max_rounds:
            raise StopIteration("maximum rounds reached")
        if path not in ("training", "structure"):
            raise ValueError("unknown improvement path")
        if path == "structure" and self.proposer is None:
            raise ValueError("proposer required")
        simulation_ref = simulation_ref or (
            str(simulation) if isinstance(simulation, (str, Path)) else None
        )
        evaluation_ref = evaluation_ref or (
            str(evaluation) if isinstance(evaluation, (str, Path)) else None
        )
        simulation = _read(simulation)
        evaluation = _read(evaluation)
        current = self.history.current
        diagnosis = self.diagnoser.diagnose(simulation, evaluation)
        number = len(self.history.data["records"]) + 1
        # Include a deterministic suffix when an interrupted attempt left an orphan checkpoint.
        attempt = 0
        while True:
            version = f"round-{number:04d}-{attempt:02d}"
            if not (self.history.directory / (version + ".pt")).exists():
                break
            attempt += 1
        working = self.history.restore(current.version_id)
        if path == "training":
            candidate = self.trainer.train(
                working,
                diagnosis,
                simulation,
                evaluation,
                training_config or TrainingConfig(seed=self.config.seed + number - 1),
                version,
                self.history,
            )
        else:
            candidate = self.proposer.propose(working, diagnosis, version, self.history)
        if (
            candidate.version_id != version
            or candidate.parent_version != current.version_id
            or candidate.kind != path
        ):
            raise ValueError("invalid candidate identity")
        expected = (self.history.directory / (version + ".pt")).resolve()
        if Path(candidate.checkpoint).resolve() != expected:
            raise ValueError("candidate checkpoint outside isolated version")
        proposed = __import__(
            "sems_attention", fromlist=["NeuralAttentionPolicy"]
        ).NeuralAttentionPolicy.load(candidate.checkpoint)
        baseline = self.history.restore(current.version_id)
        shared = dict(conditions or {})
        shared.setdefault("seed", self.config.seed + number - 1)
        result = self.validator.validate(baseline, proposed, shared)
        if candidate.metadata.get("proposal_applied") is False and result.status == "accepted":
            raise ValueError("unapplied structural proposal cannot be accepted")
        usage = dict(ZERO_USAGE)
        usage.update(evaluation.get("api_usage") or {})
        record = EvolutionRecord(
            number,
            serialize(current),
            serialize(candidate),
            serialize(diagnosis),
            serialize(result),
            simulation_ref,
            evaluation_ref,
            serialize(self.config),
            usage,
        )
        self.history.record(record)
        return record

    def run(self, simulation, evaluation, *, path="training", conditions=None, stop=None):
        records = []
        while len(self.history.data["records"]) < self.config.max_rounds:
            if stop and stop(self.history):
                break
            record = self.run_round(simulation, evaluation, path=path, conditions=conditions)
            records.append(record)
            if record.validation["status"] == "pending" and self.config.stop_on_pending:
                break
            if record.validation["status"] == "rejected" and self.config.stop_on_rejection:
                break
        return records

    def rollback(self, version):
        self.history.rollback(version)


class AutoEvolutionController:
    """Outer multi-round driver; all model changes remain isolated until validation.

    Pipeline implements train(policy, label, batch_count), diagnose(train_result),
    and compare(baseline, candidate, label). Both compare reports must be native
    serialized metric reports; compare returns a ValidationResult.
    """

    def __init__(
        self,
        history,
        pipeline,
        *,
        rounds=2,
        numeric_batches=2,
        structural_batches=1,
        should_stop=None,
    ):
        self.history = history
        self.pipeline = pipeline
        self.rounds = rounds
        self.numeric_batches = numeric_batches
        self.structural_batches = structural_batches
        self.should_stop = should_stop or (lambda: False)

    def _phase(self, policy, path, round_no, diagnosis=None, proposal=None):
        from dataclasses import asdict
        import os

        current = self.history.current
        index = len(self.history.data["records"]) + 1
        attempt = 0
        while True:
            version = f"auto-{index:04d}-{path}-{attempt:02d}"
            checkpoint_path = self.history.directory / (version + ".pt")
            if not checkpoint_path.exists():
                break
            if version not in self.history.data[
                "versions"
            ] and self.pipeline.has_completed_training(version):
                break
            attempt += 1
        trained, train_log = self.pipeline.train(
            policy, version, self.numeric_batches if path == "training" else self.structural_batches
        )
        checkpoint = (
            str(checkpoint_path)
            if checkpoint_path.exists()
            else self.history.save_policy(version, trained)
        )
        result = self.pipeline.compare(self.history.restore(current.version_id), trained, version)
        candidate = {
            "version_id": version,
            "parent_version": current.version_id,
            "kind": path,
            "checkpoint": checkpoint,
            "config": trained.get_evolvable_state()["config"],
            "proposal": asdict(proposal) if proposal else None,
            "optimizer_checkpoint": train_log.get("optimizer_checkpoint"),
            "metadata": {"auto_round": round_no, "train_log": train_log},
        }
        record = EvolutionRecord(
            index,
            serialize(current),
            candidate,
            diagnosis or {},
            serialize(result),
            train_log.get("simulation_ref"),
            train_log.get("evaluation_ref"),
            {"auto_round": round_no, "path": path},
            self.pipeline.usage(),
        )
        self.history.record(record)
        return result, train_log

    def run(self):
        from .proposer import StructuralExecutor, DeterministicStructuralProposer
        from dataclasses import asdict

        if not self.history.current:
            raise ValueError("initialize history first")
        completed = {
            r.get("config", {}).get("auto_round")
            for r in self.history.data["records"]
            if r.get("config", {}).get("path") == "round_complete"
        }
        # Round completion is reconstructed from the pipeline's durable journal.
        start = self.pipeline.completed_rounds()
        for round_no in range(start + 1, self.rounds + 1):
            if self.should_stop():
                break
            current = self.history.restore(self.history.current.version_id)
            numerical, nlog = self._phase(current, "training", round_no)
            diagnosis = self.pipeline.diagnose(nlog)
            self.pipeline.record_event(
                {
                    "round": round_no,
                    "phase": "numerical",
                    "status": numerical.status,
                    "reason": numerical.reason,
                    "diagnosis": diagnosis,
                }
            )
            if self.should_stop():
                break
            current = self.history.restore(self.history.current.version_id)
            if not current.structure and self.pipeline.should_propose(diagnosis):
                proposal = DeterministicStructuralProposer().propose(
                    diagnosis, f"structure-round-{round_no}"
                )
                candidate = StructuralExecutor().apply(current, proposal)
                verdict, slog = self._phase(candidate, "structure", round_no, diagnosis, proposal)
                self.pipeline.record_event(
                    {
                        "round": round_no,
                        "phase": "structure",
                        "status": verdict.status,
                        "reason": verdict.reason,
                        "proposal": asdict(proposal),
                        "train": slog,
                    }
                )
            self.pipeline.finish_round(round_no)
        return self.history.current
