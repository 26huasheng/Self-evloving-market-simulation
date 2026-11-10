"""Versioned v3 policy. No implicit migration of the v2 environment meaning."""

import copy
import torch
from torch import nn
from .policy import NeuralAttentionPolicy, MODEL_ID
from .observation import FEATURE_SCHEMA
from .trend_encoder import TrendEnvironmentEncoder, trend_features
from sems_tools.public_trends import TREND_SCHEMA

MODEL_VERSION = "attention_policy_trend_v3"


class TrendAttentionPolicy(NeuralAttentionPolicy):
    representation = "trend_v3"

    def __init__(self, tool_names, seed=0):
        super().__init__(tool_names, seed)
        self.environment_encoder = TrendEnvironmentEncoder()
        # Learned routing fusion, shared by Tool/Product/Parameter heads. STOP
        # remains a normal policy action; no forced reads or heat multiplier.
        self.search_fusion = nn.Sequential(nn.Linear(144, 64), nn.GELU(), nn.LayerNorm(64))

    def routing_state(self, encoded, observation):
        q = torch.as_tensor(
            observation.query_state(), dtype=encoded["z"].dtype, device=encoded["z"].device
        )
        return self.search_fusion(torch.cat([encoded["z"], encoded["H"].mean(0), q]))

    def encode(self, profile_x, observation, embeddings):
        result = super().encode(profile_x, observation, embeddings)
        result["public_trend_features"] = trend_features(observation)
        result["query_state"] = torch.as_tensor(observation.query_state())
        return result

    def get_evolvable_state(self):
        return {
            "config": {
                "tools": self.tool_names,
                "feature_schema": FEATURE_SCHEMA,
                "embedding_model": MODEL_ID,
                "model_version": MODEL_VERSION,
                "trend_schema": TREND_SCHEMA,
                "query_schema": "query_state_16_v1",
            },
            "state_dict": copy.deepcopy(self.state_dict()),
        }

    def load_evolvable_state(self, payload):
        if payload["config"] != self.get_evolvable_state()["config"]:
            raise ValueError(
                "incompatible trend_v3 checkpoint; v2 weights cannot be silently reinterpreted"
            )
        incoming = payload["state_dict"]
        own = self.state_dict()
        if set(incoming) != set(own):
            raise ValueError("parameter keys mismatch")
        for key, value in incoming.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != own[key].shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError("invalid parameter " + key)
        self.load_state_dict(incoming)

    def clone(self):
        policy = TrendAttentionPolicy(self.tool_names)
        policy.load_evolvable_state(self.get_evolvable_state())
        return policy

    @classmethod
    def from_registry(cls, registry, seed=0):
        return cls(
            ["STOP"] + [n for n, s in registry.schema().items() if s["kind"] in ("read", "shelf")],
            seed,
        )
