"""Isolated Attention training: no-op, behavior cloning and policy-gradient updates."""

from typing import Protocol
from .types import CandidatePolicy


class Trainer(Protocol):
    def train(
        self, policy, diagnosis, simulation, evaluation, config, candidate_version, history
    ) -> CandidatePolicy: ...


class NoOpTrainer:
    def train(self, policy, diagnosis, simulation, evaluation, config, candidate_version, history):
        # Future trainers may use policy.named_parameters(), and persist optimizer state
        # separately via history.save_optimizer_state(). The supplied policy is isolated.
        path = history.save_policy(candidate_version, policy)
        return CandidatePolicy(
            candidate_version,
            history.current.version_id,
            "training",
            path,
            config=policy.get_evolvable_state()["config"],
            metadata={"trainer": "noop", "trained": False, "seed": config.seed},
        )


class BehaviorCloningTrainer:
    """Local supervised training on legal demonstration snapshots, not consumer labels.

    Samples contain profile_x, observation, action_space, action and embeddings;
    snapshot creation must not include future ground truth. Each optimizer step
    recomputes the hierarchical probabilities under the current policy.
    """

    def __init__(self, samples, embeddings):
        self.samples = list(samples)
        self.embeddings = embeddings
        self.log = None

    @staticmethod
    def action_log_prob(policy, sample, embeddings):
        action, space = sample["action"], sample["action_space"]
        if not space.contains(action):
            raise ValueError("demonstration action is masked")
        x, obs = sample["profile_x"], sample["observation"]
        first = policy.forward(x, obs, embeddings, space)
        tool = policy.tool_names.index(action.tool_name)
        logp = first["tool_probs"][tool].log()
        if action.is_stop:
            return logp
        second = policy.forward(x, obs, embeddings, space, action.tool_name)
        if action.asin not in second["targets"]:
            raise ValueError("invalid demonstration product")
        logp = logp + second["product_probs"][second["targets"].index(action.asin)].log()
        third = policy.forward(x, obs, embeddings, space, action.tool_name, action.asin)
        choices = [a.key() for a in third["actions"]]
        return logp + third["parameter_probs"][choices.index(action.key())].log()

    def train(self, policy, diagnosis, simulation, evaluation, config, candidate_version, history):
        import math
        import torch

        if not self.samples:
            raise ValueError("no demonstration samples")
        options = config.options
        epochs = int(options.get("epochs", 4))
        learning_rate = float(options.get("learning_rate", 0.001))
        stop_weight = float(options.get("stop_weight", 1.0))
        if (
            epochs < 1
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
            or not math.isfinite(stop_weight)
            or not 0 < stop_weight <= 1
        ):
            raise ValueError("invalid training config")
        weights = torch.tensor(
            [stop_weight if sample["action"].is_stop else 1.0 for sample in self.samples],
            dtype=torch.float32,
        )
        torch.manual_seed(config.seed)
        before = {k: v.detach().clone() for k, v in policy.named_parameters()}
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        losses, gradients = [], {}
        policy.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = (
                torch.stack(
                    [
                        -self.action_log_prob(policy, sample, self.embeddings)
                        for sample in self.samples
                    ]
                )
                * weights
            ).sum() / weights.sum()
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            for name, parameter in policy.named_parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise ValueError("non-finite gradient")
                    gradients[name] = max(gradients.get(name, 0.0), float(parameter.grad.norm()))
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            final_loss = float(
                (
                    torch.stack(
                        [-self.action_log_prob(policy, s, self.embeddings) for s in self.samples]
                    )
                    * weights
                ).sum()
                / weights.sum()
            )
        changes = {
            name: float((parameter.detach() - before[name]).norm())
            for name, parameter in policy.named_parameters()
        }
        if not any(value > 0 for value in changes.values()):
            raise RuntimeError("training produced no parameter update")
        policy.eval()
        path = history.save_policy(candidate_version, policy)
        optimizer_path = history.save_optimizer_state(candidate_version, optimizer.state_dict())
        self.log = {
            "loss_before_steps": losses,
            "loss_after": final_loss,
            "gradient_norm_max": gradients,
            "parameter_delta_norm": changes,
            "samples": len(self.samples),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "stop_weight": stop_weight,
            "seed": config.seed,
            "demonstration_type": "synthetic heuristic, not consumer behavior",
        }
        return CandidatePolicy(
            candidate_version,
            history.current.version_id,
            "training",
            path,
            config=policy.get_evolvable_state()["config"],
            optimizer_checkpoint=optimizer_path,
            metadata={"trainer": "behavior_cloning", "trained": True, "training_log": self.log},
        )


class ReinforceTrainer:
    """One on-policy update using freshly sampled states and an explicit baseline."""

    def __init__(self, trajectories, embeddings, source_version):
        self.trajectories = list(trajectories)
        self.embeddings = embeddings
        self.source_version = source_version
        self.log = None

    def train(self, policy, diagnosis, simulation, evaluation, config, candidate_version, history):
        import torch

        if not self.trajectories:
            raise ValueError("no feedback trajectories")
        if any(t["policy_version"] != self.source_version for t in self.trajectories):
            raise ValueError("off-policy trajectory")
        if history.current.version_id != self.source_version:
            raise ValueError("source version does not match current policy")
        before = {k: v.detach().clone() for k, v in policy.named_parameters()}
        lr = float(config.options.get("learning_rate", 1e-4))
        baseline = float(config.options.get("baseline", 0.5))
        if not (0 < lr < 1) or not torch.isfinite(torch.tensor(baseline)):
            raise ValueError("invalid optimizer configuration")
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        opt.zero_grad()
        losses = []
        for trajectory in self.trajectories:
            if not trajectory["samples"]:
                raise ValueError("trajectory has no sampled decisions")
            logp = torch.stack(
                [
                    BehaviorCloningTrainer.action_log_prob(policy, s, self.embeddings)
                    for s in trajectory["samples"]
                ]
            ).sum()
            losses.append(-logp * (float(trajectory["reward"]) - baseline))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("invalid feedback loss")
        loss.backward()
        gradients = {
            k: float(v.grad.norm()) for k, v in policy.named_parameters() if v.grad is not None
        }
        if not all(torch.isfinite(v.grad).all() for v in policy.parameters() if v.grad is not None):
            raise ValueError("non-finite gradient")
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5)
        opt.step()
        changes = {k: float((v.detach() - before[k]).norm()) for k, v in policy.named_parameters()}
        if not any(v > 0 for v in changes.values()):
            raise RuntimeError("no parameter update")
        path = history.save_policy(candidate_version, policy)
        opt_path = history.save_optimizer_state(candidate_version, opt.state_dict())
        self.log = {
            "loss": float(loss.detach()),
            "baseline": baseline,
            "learning_rate": lr,
            "rewards": [t["reward"] for t in self.trajectories],
            "source_version": self.source_version,
            "gradient_norm": gradients,
            "parameter_delta_norm": changes,
            "one_on_policy_update": True,
        }
        return CandidatePolicy(
            candidate_version,
            self.source_version,
            "training",
            path,
            config=policy.get_evolvable_state()["config"],
            optimizer_checkpoint=opt_path,
            metadata={"trainer": "reinforce", "training_log": self.log},
        )


class OnPolicyBatchOptimizer:
    """Continuous on-policy batches; no update is made for constant rewards."""

    def __init__(self, policy, learning_rate=1e-4):
        import torch

        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def update(self, trajectories, embeddings):
        import torch
        from .reward import centered_advantages

        rewards = [float(t["reward"]) for t in trajectories]
        try:
            baseline, advantages = centered_advantages(rewards)
        except ValueError as exc:
            return {"updated": False, "reason": str(exc), "rewards": rewards}
        before = {k: v.detach().clone() for k, v in self.policy.named_parameters()}
        self.optimizer.zero_grad()
        losses = []
        for t, adv in zip(trajectories, advantages):
            if not t["samples"]:
                raise ValueError("empty policy trajectory")
            logp = torch.stack(
                [
                    BehaviorCloningTrainer.action_log_prob(self.policy, s, embeddings)
                    for s in t["samples"]
                ]
            ).sum()
            losses.append(-logp * adv)
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("non-finite policy loss")
        loss.backward()
        grads = {
            k: float(v.grad.norm()) for k, v in self.policy.named_parameters() if v.grad is not None
        }
        if not all(
            torch.isfinite(v.grad).all() for v in self.policy.parameters() if v.grad is not None
        ):
            raise ValueError("non-finite gradient")
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
        self.optimizer.step()
        delta = {
            k: float((v.detach() - before[k]).norm()) for k, v in self.policy.named_parameters()
        }
        return {
            "updated": any(x > 0 for x in delta.values()),
            "loss": float(loss.detach()),
            "rewards": rewards,
            "baseline": baseline,
            "advantages": advantages,
            "gradient_norm": grads,
            "parameter_delta_norm": delta,
        }


class ClippedBehaviorBatchOptimizer:
    """Score-function updates from a saved behavior policy with bounded IS ratios.

    A full market rollout is collected before optimization. Later mini-batches
    are therefore off-policy; their recorded action probabilities are used to
    correct the trajectory score. Clipping is explicit and reported.
    """

    def __init__(self, policy, learning_rate=5e-5, ratio_clip=2.0):
        import torch

        if not 1 <= ratio_clip <= 10:
            raise ValueError("invalid importance ratio clip")
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.ratio_clip = ratio_clip

    def update(self, trajectories, embeddings):
        import math, torch
        from .reward import centered_advantages

        rewards = [float(t["reward"]) for t in trajectories]
        try:
            baseline, advantages = centered_advantages(rewards)
        except ValueError as exc:
            return {"updated": False, "reason": str(exc), "rewards": rewards}
        before = {k: v.detach().clone() for k, v in self.policy.named_parameters()}
        self.optimizer.zero_grad()
        terms = []
        ratios = []
        clipped = 0
        for trajectory, adv in zip(trajectories, advantages):
            samples = trajectory["samples"]
            behavior = trajectory["sampled_log_prob"]
            if not samples or len(samples) != len(behavior):
                raise ValueError("invalid recorded behavior trajectory")
            current = torch.stack(
                [
                    BehaviorCloningTrainer.action_log_prob(self.policy, s, embeddings)
                    for s in samples
                ]
            ).sum()
            log_ratio = current.detach() - sum(float(x) for x in behavior)
            ratio = float(torch.exp(log_ratio.clamp(-20, 20)))
            bounded = max(1 / self.ratio_clip, min(self.ratio_clip, ratio))
            clipped += int(bounded != ratio)
            ratios.append(ratio)
            terms.append(-bounded * adv * current)
        loss = torch.stack(terms).mean()
        if not torch.isfinite(loss):
            raise ValueError("non-finite policy loss")
        loss.backward()
        gradients = {
            k: float(v.grad.norm()) for k, v in self.policy.named_parameters() if v.grad is not None
        }
        if not all(
            torch.isfinite(v.grad).all() for v in self.policy.parameters() if v.grad is not None
        ):
            raise ValueError("non-finite gradient")
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
        self.optimizer.step()
        delta = {
            k: float((v.detach() - before[k]).norm()) for k, v in self.policy.named_parameters()
        }
        return {
            "updated": any(v > 0 for v in delta.values()),
            "loss": float(loss.detach()),
            "rewards": rewards,
            "baseline": baseline,
            "advantages": advantages,
            "importance_ratios": ratios,
            "ratio_clip": self.ratio_clip,
            "clipped_trajectories": clipped,
            "gradient_norm": gradients,
            "parameter_delta_norm": delta,
        }
