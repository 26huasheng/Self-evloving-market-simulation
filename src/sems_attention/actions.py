"""Adapter over the authoritative ToolRegistry action list."""

from dataclasses import dataclass, field
import json


@dataclass(frozen=True)
class ToolAction:
    tool_name: str
    asin: str | None = None
    parameters: dict = field(default_factory=dict)

    @property
    def is_stop(self):
        return self.tool_name == "STOP"

    @property
    def arguments(self):
        return ({"asin": self.asin} if self.asin is not None else {}) | self.parameters

    def as_dict(self):
        return {"tool_name": self.tool_name, "asin": self.asin, "parameters": self.parameters}

    def key(self):
        return self.tool_name, self.asin, json.dumps(self.parameters, sort_keys=True)


class ActionSpace:
    def __init__(self, registry, observation=None):
        self.registry = registry
        self.actions = []
        seen = (
            {
                (
                    r["action"]["tool_name"],
                    r["action"]["asin"],
                    json.dumps(r["action"]["parameters"], sort_keys=True),
                )
                for r in observation.results
            }
            if observation is not None
            else set()
        )
        for row in registry.actions():
            if not row["allowed"]:
                continue
            args = row["args"]
            action = ToolAction(
                row["tool"], args.get("asin"), {k: v for k, v in args.items() if k != "asin"}
            )
            if action.tool_name != "get_reviews" and action.key() in seen:
                continue
            self.actions.append(action)
        self.tools = list(dict.fromkeys(a.tool_name for a in self.actions))

    def products(self, tool):
        return list(dict.fromkeys(a.asin for a in self.actions if a.tool_name == tool))

    def parameters(self, tool, asin):
        return [a for a in self.actions if a.tool_name == tool and a.asin == asin]

    def contains(self, action):
        return action.key() in {a.key() for a in self.actions}
