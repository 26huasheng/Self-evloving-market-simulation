"""Hierarchical differentiable Tool, Product, Parameter routing."""

import copy, hashlib, json, math, warnings
import torch
from torch import nn
from .actions import ActionSpace
from .profile_encoder import ProfileEncoder
from .environment_encoder import EnvironmentEncoder
from .lambda_layer import DynamicLambda
from .observation import FEATURE_SCHEMA

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_VERSION = "attention_policy_v2"


class NeuralAttentionPolicy(nn.Module):
    representation = "state_v2"

    def __init__(self, tool_names, seed=0, structure=None):
        super().__init__()
        torch.manual_seed(seed)
        self.tool_names = tuple(tool_names)
        self.structure = dict(structure or {})
        if self.structure not in ({}, {"query_context_v1": True}):
            raise ValueError("unsupported structure")
        if not self.tool_names or self.tool_names[0] != "STOP":
            raise ValueError("tool names must start with STOP")
        self.profile_encoder = ProfileEncoder()
        self.environment_encoder = EnvironmentEncoder(self.structure)
        self.lambda_layer = DynamicLambda()
        self.W = nn.Parameter(torch.zeros(64))
        self.norm = nn.LayerNorm(64)
        self.tool_embedding = nn.Embedding(len(self.tool_names), 64)
        self.tool_head = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, len(self.tool_names))
        )
        self.tool_bias = nn.Parameter(torch.zeros(len(self.tool_names)))
        self.product_head = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1))
        self.parameter_head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.generator = torch.Generator().manual_seed(seed)

    @staticmethod
    def from_registry(registry, seed=0):
        return NeuralAttentionPolicy(
            ["STOP"] + [n for n, s in registry.schema().items() if s["kind"] in ("read", "shelf")],
            seed,
        )

    def _param_vector(self, action):
        # Fixed semantic parameter tokens; ASIN and pagination position are excluded.
        vector = torch.zeros(64)
        for k, v in sorted(action.parameters.items()):
            digest = hashlib.sha256(f"{k}={v}".encode()).digest()
            vector[int.from_bytes(digest[:2], "little") % 64] += 1
        return vector

    def encode(self, profile_x, observation, embeddings):
        p = self.profile_encoder(torch.as_tensor(profile_x, dtype=torch.float32))
        h, e, raw = self.environment_encoder(observation, embeddings)
        lambda_p = self.lambda_layer(e, p)
        z = self.norm(self.W + lambda_p)
        return {
            "p": p,
            "H": h,
            "e": e,
            "z": z,
            "lambda_p": lambda_p,
            "lambda_matrix": self.lambda_layer.matrix(e),
            "product_input": raw,
        }

    def forward(self, profile_x, observation, embeddings, space, tool=None, asin=None):
        enc = self.encode(profile_x, observation, embeddings)
        z = self.routing_state(enc, observation)
        h = enc["H"]
        device = z.device
        mask = torch.tensor(
            [name in space.tools for name in self.tool_names], dtype=torch.bool, device=device
        )
        temperature = self.temperature.clamp_min(0.05)
        logits = self.tool_head(z) + self.tool_bias
        probs = torch.softmax((logits / temperature).masked_fill(~mask, -torch.inf), 0)
        out = {
            **enc,
            "tools": self.tool_names,
            "tool_logits": logits,
            "tool_mask": mask,
            "tool_probs": probs,
        }
        if tool is None or tool == "STOP":
            return out
        ti = self.tool_names.index(tool)
        tv = self.tool_embedding(torch.tensor(ti, device=device))
        targets = space.products(tool)
        if targets == [None]:
            product_logits = torch.zeros(1, device=device)
            product_mask = torch.ones(1, dtype=torch.bool, device=device)
            product_probs = torch.ones(1, device=device)
        else:
            indices = [observation.order.index(a) for a in targets]
            pv = h[indices]
            product_logits = self.product_head(
                torch.cat([z.expand(len(targets), -1), tv.expand(len(targets), -1), pv], 1)
            ).squeeze(-1)
            product_mask = torch.ones(len(targets), dtype=torch.bool, device=device)
            product_probs = torch.softmax(product_logits / temperature, 0)
        out.update(
            targets=targets,
            product_logits=product_logits,
            product_mask=product_mask,
            product_probs=product_probs,
        )
        if asin not in targets:
            return out
        actions = space.parameters(tool, asin)
        selected = (
            torch.zeros(64, device=device) if asin is None else h[observation.order.index(asin)]
        )
        if len(actions) == 1 and not actions[0].parameters:
            out.update(
                actions=actions,
                parameter_logits=torch.zeros(1, device=device),
                parameter_mask=torch.ones(1, dtype=torch.bool, device=device),
                parameter_probs=torch.ones(1, device=device),
            )
            return out
        rows = torch.stack([self._param_vector(a).to(device) for a in actions])
        n = len(actions)
        param_logits = self.parameter_head(
            torch.cat([z.expand(n, -1), tv.expand(n, -1), selected.expand(n, -1), rows], 1)
        ).squeeze(-1)
        param_mask = torch.ones(n, dtype=torch.bool, device=device)
        out.update(
            actions=actions,
            parameter_logits=param_logits,
            parameter_mask=param_mask,
            parameter_probs=torch.softmax(param_logits / temperature, 0),
        )
        return out

    def routing_state(self, encoded, observation):
        return encoded["z"]

    def _choose(self, probs, mode):
        return (
            int(torch.multinomial(probs.detach(), 1, generator=self.generator))
            if mode == "sample"
            else int(torch.argmax(probs))
        )

    def select_action(self, profile_x, observation, embeddings, space, mode="greedy"):
        if mode not in ("greedy", "sample"):
            raise ValueError("invalid mode")
        first = self.forward(profile_x, observation, embeddings, space)
        ti = self._choose(first["tool_probs"], mode)
        tool = self.tool_names[ti]
        tp = first["tool_probs"][ti]
        if tool == "STOP":
            return space.actions[0], {
                "tool_probability": float(tp.detach()),
                "product_probability": 1.0,
                "parameter_probability": 1.0,
                "action_log_prob": float(tp.log().detach()),
                "log_prob_tensor": tp.log(),
                "forward": first,
            }
        second = self.forward(profile_x, observation, embeddings, space, tool)
        pi = self._choose(second["product_probs"], mode)
        asin = second["targets"][pi]
        pp = second["product_probs"][pi]
        third = self.forward(profile_x, observation, embeddings, space, tool, asin)
        ai = self._choose(third["parameter_probs"], mode)
        ap = third["parameter_probs"][ai]
        lp = tp.log() + pp.log() + ap.log()
        return third["actions"][ai], {
            "tool_probability": float(tp.detach()),
            "product_probability": float(pp.detach()),
            "parameter_probability": float(ap.detach()),
            "action_log_prob": float(lp.detach()),
            "log_prob_tensor": lp,
            "forward": third,
        }

    def get_evolvable_state(self):
        config = {
            "tools": self.tool_names,
            "feature_schema": FEATURE_SCHEMA,
            "embedding_model": MODEL_ID,
            "model_version": MODEL_VERSION,
        }
        if self.structure:
            config["structure"] = dict(self.structure)
        return {"config": config, "state_dict": copy.deepcopy(self.state_dict())}

    def load_evolvable_state(self, payload):
        expected = self.get_evolvable_state()["config"]
        config = dict(payload["config"])
        incoming = dict(payload["state_dict"])
        version = config.pop("model_version", None)
        expected.pop("model_version")
        if config != expected:
            raise ValueError("incompatible policy config")
        if version is None:
            # The unversioned v1 format had exactly these two extra projections.
            for key, shape in (("W_e.weight", (64, 64)), ("W_q.weight", (64, 16))):
                value = incoming.get(key)
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != shape
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"invalid legacy parameter {key}")
                del incoming[key]
            warnings.warn(
                "Migrating unversioned Attention checkpoint: dropping W_e.weight and W_q.weight; behavior changes",
                UserWarning,
                stacklevel=2,
            )
        elif version != MODEL_VERSION:
            raise ValueError(f"unsupported policy version: {version}")
        own = self.state_dict()
        if set(own) != set(incoming):
            raise ValueError("parameter keys mismatch")
        for k, v in incoming.items():
            if v.shape != own[k].shape or not torch.isfinite(v).all():
                raise ValueError(f"invalid parameter {k}")
        self.load_state_dict(incoming)

    def clone(self):
        p = NeuralAttentionPolicy(self.tool_names, structure=self.structure)
        p.load_evolvable_state(self.get_evolvable_state())
        return p

    def save(self, path):
        torch.save(self.get_evolvable_state(), path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["config"].get("model_version") == "attention_policy_trend_v3":
            from .trend_policy import TrendAttentionPolicy

            p = TrendAttentionPolicy(payload["config"]["tools"])
        else:
            p = cls(payload["config"]["tools"], structure=payload["config"].get("structure"))
        p.load_evolvable_state(payload)
        return p
