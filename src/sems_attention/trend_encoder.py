"""v3 separates public temporal context from privately acquired product facts."""

import math
import torch
from torch import nn
from sems_tools.public_trends import TREND_SCHEMA, TREND_NAMES


def trend_features(observation):
    context = observation.initial_shelf.get("public_market_context")
    if not isinstance(context, dict) or context.get("schema") != TREND_SCHEMA:
        raise ValueError("trend_v3 requires an explicit public market bulletin")
    if context.get("as_of") != observation.market_time:
        raise ValueError("public market bulletin is stale or future-dated")
    values = []
    masks = []
    for name in TREND_NAMES:
        item = context.get("features", {}).get(name)
        if not isinstance(item, dict) or type(item.get("known")) is not bool:
            raise ValueError("invalid public trend feature: " + name)
        value = item.get("value")
        if item["known"]:
            if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("invalid public trend value: " + name)
            values.append(float(value))
            masks.append(1.0)
        else:
            if value is not None:
                raise ValueError("unknown public trend must have null value")
            values.append(0.0)
            masks.append(0.0)
    return torch.tensor(values + masks, dtype=torch.float32)


class TrendEnvironmentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.product = nn.Sequential(nn.Linear(1168, 128), nn.GELU(), nn.Linear(128, 64))
        self.trend = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 64), nn.LayerNorm(64))

    def forward(self, observation, embeddings):
        rows = []
        for asin in observation.order:
            title, detail, reviews = observation.product_texts(asin)
            rows.append(
                torch.cat(
                    [
                        torch.as_tensor(embeddings.encode(title)),
                        torch.as_tensor(embeddings.encode(detail)) if detail else torch.zeros(384),
                        torch.as_tensor(embeddings.mean(reviews)),
                        torch.as_tensor(observation.product_features(asin)),
                    ]
                )
            )
        raw = torch.stack(rows)
        return self.product(raw), self.trend(trend_features(observation)), raw
